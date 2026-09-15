"""PGS reference and a shared, differentiable Euler update for contact CFM."""
from __future__ import annotations

from itertools import combinations

import torch

from .problem import converged, gap, value


def advance_endpoint(state, endpoint, time, step_size):
    """Euler for u=(endpoint-state)/(1-time), without dividing the field.

    The training integrator and the diagnostic inference driver both use this
    update. The caller supplies 0 <= step_size <= 1-time. Clamping only guards
    clock roundoff; it does not project the predicted state or solve the QP.
    """
    remaining = (1 - time).clamp_min(torch.finfo(state.dtype).tiny)
    fraction = (step_size / remaining).clamp(0, 1).reshape(-1, 1)
    return (1 - fraction) * state + fraction * endpoint


def cfm_step(model, problem, state, time, next_time):
    """One autograd-preserving step; geometry and the original QP stay fixed."""
    time = torch.as_tensor(time, dtype=state.dtype, device=state.device).expand(len(state))
    next_time = torch.as_tensor(next_time, dtype=state.dtype, device=state.device).expand(len(state))
    endpoint = model.endpoint(state, time, problem)
    return advance_endpoint(state, endpoint, time, next_time - time)


def integrate_cfm(model, problem, start, calls, *, start_step=0, end_step=None,
                  return_trajectory=False):
    """Integrate any contiguous part of the ORIGINAL uniform [0,1] clock.

    `calls` always means the full-interval budget. For example, calls=4 and
    start_step=2 performs two calls at t=.5,.75, rather than restarting at zero.
    No state is detached. Use torch.no_grad() outside this function for eval.
    A returned trajectory includes the supplied initial state as its first row.
    Raw inference and training have no early stop, guard, or hidden PGS finish.
    """
    end_step = calls if end_step is None else end_step
    if (type(calls) is not int or calls < 1 or type(start_step) is not int
            or type(end_step) is not int or not 0 <= start_step <= end_step <= calls):
        raise ValueError("Require integer 0 <= start_step <= end_step <= calls, calls > 0")
    if start.shape != problem["c"].shape:
        raise ValueError("Multiplier state must match the padded contact shape")
    state = start
    trajectory = [state] if return_trajectory else None
    for step in range(start_step, end_step):
        state = cfm_step(model, problem, state, step / calls, (step + 1) / calls)
        if trajectory is not None:
            trajectory.append(state)
    return torch.stack(trajectory) if trajectory is not None else state


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
            candidate = advance_endpoint(lam, endpoint, elapsed, h)
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
