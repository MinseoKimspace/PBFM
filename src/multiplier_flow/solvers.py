"""PGS labels/baseline and structure-preserving Euler for contact CFM."""
from __future__ import annotations

from itertools import combinations

import torch

from .problem import converged, gap, value


@torch.no_grad()
def pgs(problem, start, tolerance=1e-7, max_sweeps=10000):
    """Projected coordinate descent on Q; negative multiplier increments allowed."""
    if tolerance <= 0 or max_sweeps < 1:
        raise ValueError("Positive tolerance and sweep budget required")
    lam = start.clone()
    diagonal = problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-30)
    sweeps = torch.zeros(len(lam), dtype=torch.long, device=lam.device)
    updates = torch.zeros_like(sweeps)
    for _ in range(max_sweeps):
        active = ~converged(problem, lam, tolerance)
        if not active.any():
            break
        g = gap(problem, lam)
        for i in range(lam.shape[1]):
            enabled = active & problem["mask"][:, i]
            increment = ((lam[:, i] - g[:, i] / diagonal[:, i]).clamp_min(0) - lam[:, i]) * enabled
            lam[:, i] += increment
            g += problem["D"][:, :, i] * increment[:, None]
            updates += enabled.long()
        sweeps += active.long()
    return dict(final=lam, sweeps=sweeps, contact_evals=updates,
                converged=converged(problem, lam, tolerance))


@torch.no_grad()
def active_set_solution(problem, max_contacts=12, tolerance=1e-8):
    """Independent tiny-QP oracle: enumerate active sets and check KKT.

    CPU float64 only; not a scalable baseline. Redundant rows are handled by
    least squares. Lambda may be nonunique; compare decoded primal positions.
    """
    if len(problem["p"]) != 1 or problem["p"].device.type != "cpu" or problem["p"].dtype != torch.float64:
        raise ValueError("Oracle expects one CPU float64 problem")
    count = int(problem["mask"].sum())
    if count > max_contacts:
        raise ValueError("Active-set enumeration restricted to tiny problems")
    d, c = problem["D"][0, :count, :count], problem["c"][0, :count]
    for size in range(count + 1):
        for active in combinations(range(count), size):
            lam = torch.zeros_like(problem["c"])
            if active:
                index = torch.tensor(active)
                block = d[index[:, None], index[None, :]]
                solution = torch.linalg.lstsq(block, -c[index], driver="gelsd").solution
                lam[0, index] = solution
            g = gap(problem, lam)
            if ((lam >= -tolerance).all() and (g >= -tolerance).all()
                    and (lam * g).abs().max() <= tolerance):
                return lam.clamp_min(0)
    raise RuntimeError("No KKT solution found; problem may be infeasible or badly scaled")


@torch.no_grad()
def run_cfm(model, problem, start, calls, guarded=False, max_backtracks=12):
    """Euler on [0,1], evaluated as a nonnegative convex combination.

    u=(endpoint-lambda)/(1-tau), alpha=h/(1-tau) <= 1, hence
    lambda_next=(1-alpha)*lambda+alpha*endpoint. No division of the field near
    tau=1 or post-integration clipping. The projection is INSIDE the model.
    Raw means no EXTRA Q guard or PGS finish, not an unconstrained output head.
    Guarded halves h until Q does not increase; it may fail to finish.
    NFE counts endpoint evaluations; backtracking reuses the same endpoint.
    """
    if not isinstance(calls, int) or calls < 1 or max_backtracks < 0:
        raise ValueError("Positive integer calls and nonnegative backtrack limit required")
    if not torch.isfinite(start).all() or (start < 0).any():
        raise ValueError("CFM starts must be finite and nonnegative")
    lam, elapsed = start.clone(), start.new_zeros(len(start))
    nfe = torch.zeros(len(start), dtype=torch.long, device=start.device)
    backtracks, interventions = torch.zeros_like(nfe), torch.zeros_like(nfe)
    failed = torch.zeros(len(start), dtype=torch.bool, device=start.device)
    next_h = torch.full_like(elapsed, 1.0 / calls)
    accepted_steps = torch.zeros_like(nfe)
    minimum_h = torch.full_like(elapsed, float("inf"))
    failure_code = torch.zeros_like(nfe)  # 0=complete, 1=guard, 2=budget, 3=nonfinite, 4=negative endpoint
    for _ in range(calls * 64):
        active = (elapsed < 1) & ~failed
        if not active.any():
            break
        remaining = 1 - elapsed
        h = (torch.minimum(remaining, next_h) if guarded else
             ((accepted_steps + 1).to(lam.dtype) / calls).clamp_max(1) - elapsed)
        endpoint = model.endpoint(lam, elapsed, problem)
        nfe += active.long()
        nonfinite = ~torch.isfinite(endpoint).all(-1)
        negative = (endpoint < 0).any(-1)
        invalid = active & (nonfinite | negative)
        failed |= invalid
        failure_code = torch.where(invalid, torch.where(nonfinite, 3, 4), failure_code)
        pending = active & ~invalid
        old_q = value(problem, lam)
        for retry in range(max_backtracks + 1 if guarded else 1):
            alpha = (h / remaining.clamp_min(torch.finfo(lam.dtype).tiny)).clamp(0, 1)[:, None]
            candidate = (1 - alpha) * lam + alpha * endpoint
            candidate_q = value(problem, candidate)
            acceptable = torch.isfinite(candidate).all(-1) & torch.isfinite(candidate_q)
            if guarded:
                allowance = 32 * torch.finfo(lam.dtype).eps * old_q.abs().clamp_min(1e-8)
                acceptable &= (candidate >= 0).all(-1) & (candidate_q <= old_q + allowance)
            accept = pending & acceptable
            lam = torch.where(accept[:, None], candidate, lam)
            elapsed = torch.where(accept, (elapsed + h).clamp_max(1), elapsed)
            accepted_steps += accept.long()
            minimum_h = torch.where(accept, torch.minimum(minimum_h, h), minimum_h)
            next_h = torch.where(accept, h, next_h)
            pending &= ~accept
            if not pending.any():
                break
            if retry == 0:
                interventions += pending.long()
            if retry < max_backtracks and guarded:
                backtracks += pending.long()
                h = torch.where(pending, h / 2, h)
        failed |= pending
        failure_code = torch.where(pending, 1 if guarded else 3, failure_code)
    failure_code = torch.where((elapsed < 1) & ~failed, 2, failure_code)
    failed |= elapsed < 1
    return dict(final=lam, completed=~failed, time=elapsed, nfe=nfe,
                neural_evals=nfe * model.neural_evaluations,
                backtracks=backtracks, interventions=interventions,
                accepted_steps=accepted_steps, failure_code=failure_code,
                min_accepted_h=torch.where(accepted_steps > 0, minimum_h, 0))
