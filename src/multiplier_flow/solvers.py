"""Independent dual PGS, accurate finite-time reference, and learned-map guard."""
from __future__ import annotations

from itertools import combinations
import math

import torch

from .problem import converged, field, gap, value


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
def reference(problem, start, duration, rtol=1e-5, atol=1e-8, max_steps=50000):
    """Adaptive step-doubled SSP RK3. dt<=1 preserves lambda>=0.

    Full duration is integrated even after reaching projection tolerance.
    Labels are finite-time flow endpoints, not PGS equilibria. Returns actual
    field evaluation/rejection counts, including local-error checking work.
    """
    duration = torch.as_tensor(duration, dtype=start.dtype, device=start.device).expand(len(start)).clone()
    if (not torch.isfinite(start).all() or (start < 0).any()
            or not torch.isfinite(duration).all() or (duration < 0).any()
            or min(rtol, atol) <= 0):
        raise ValueError("Invalid nonnegative start/duration or reference tolerance")
    lam, elapsed = start.clone(), torch.zeros_like(duration)
    step = torch.minimum(torch.full_like(duration, 0.5), duration)
    nfe = torch.zeros_like(duration, dtype=torch.long)
    rejected = torch.zeros_like(nfe)

    def rk3(x, h):
        first = x + h[:, None] * field(problem, x)
        second = 0.75 * x + 0.25 * (first + h[:, None] * field(problem, first))
        return x / 3 + (2 / 3) * (second + h[:, None] * field(problem, second))

    for _ in range(max_steps):
        active = elapsed < duration
        if not active.any():
            return dict(final=lam, nfe=nfe, rejected=rejected, time=elapsed)
        h = torch.minimum(step, duration - elapsed)
        full = rk3(lam, h)
        half = rk3(rk3(lam, h / 2), h / 2)
        scale = atol + rtol * torch.maximum(lam.abs(), half.abs())
        error = ((half - full).abs() / (7 * scale)).amax(-1)
        if not torch.isfinite(error).all() or not torch.isfinite(half).all():
            raise FloatingPointError("Nonfinite reference integration")
        accepted = active & (error <= 1)
        lam = torch.where(accepted[:, None], half, lam)
        elapsed = torch.where(accepted, torch.minimum(elapsed + h, duration), elapsed)
        nfe += 9 * active.long()
        rejected += (active & ~accepted).long()
        factor = (0.9 * error.clamp_min(1e-12).pow(-1 / 4)).clamp(0.2, 2.0)
        step = (h * factor).clamp_max(1.0)
        if ((step <= torch.finfo(start.dtype).eps * elapsed.abs().clamp_min(1)) & active & ~accepted).any():
            raise RuntimeError("Reference step underflow; inspect tolerances and precision")
    raise RuntimeError("Reference budget exhausted; do not train on partial labels")


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
def run_map(model, problem, start, total_time, calls, guarded=False, max_backtracks=12):
    """Fixed total-time evaluation. Guard retries F at a shorter h, not a scaled
    displacement. Accepted times still sum to total_time. No PGS/GD fallback.
    """
    if not math.isfinite(total_time) or total_time <= 0 or calls < 1:
        raise ValueError("Positive time and call budget required")
    lam, elapsed = start.clone(), start.new_zeros(len(start))
    nfe = torch.zeros(len(start), dtype=torch.long, device=start.device)
    backtracks, interventions = torch.zeros_like(nfe), torch.zeros_like(nfe)
    failed = torch.zeros(len(start), dtype=torch.bool, device=start.device)
    nominal = total_time / calls
    next_h = torch.full_like(elapsed, nominal)
    accepted_steps = torch.zeros_like(nfe)
    minimum_h = torch.full_like(elapsed, float("inf"))
    failure_code = torch.zeros_like(nfe)  # 0=complete, 1=retry limit, 2=outer budget
    for _ in range(calls * 64):
        active = (elapsed < total_time) & ~failed
        if not active.any():
            break
        h = torch.minimum(total_time - elapsed, next_h)
        pending = active.clone()
        old_q = value(problem, lam)
        for retry in range(max_backtracks + 1 if guarded else 1):
            candidate = model(lam, h, problem)
            nfe += pending.long()
            finite = (torch.isfinite(candidate).all(-1) & (candidate >= 0).all(-1)
                      & torch.isfinite(value(problem, candidate)))
            acceptable = finite
            if guarded:
                # Floating-point allowance only, not a substitute for KKT tests.
                allowance = 32 * torch.finfo(lam.dtype).eps * old_q.abs().clamp_min(1e-8)
                acceptable = acceptable & (value(problem, candidate) <= old_q + allowance)
            accept = pending & acceptable
            lam = torch.where(accept[:, None], candidate, lam)
            elapsed = torch.where(accept, (elapsed + h).clamp_max(total_time), elapsed)
            accepted_steps += accept.long()
            minimum_h = torch.where(accept, torch.minimum(minimum_h, h), minimum_h)
            # Reuse the last safe clock instead of re-rejecting the nominal h.
            next_h = torch.where(accept, h, next_h)
            pending &= ~accept
            if not pending.any():
                break
            if retry == 0:
                interventions += pending.long()
            backtracks += pending.long()
            h = torch.where(pending, h / 2, h)
        failed |= pending
        failure_code = torch.where(pending, 1, failure_code)
    failure_code = torch.where((elapsed < total_time) & ~failed, 2, failure_code)
    failed |= elapsed < total_time
    return dict(final=lam, completed=~failed, time=elapsed, nfe=nfe,
                backtracks=backtracks, interventions=interventions,
                accepted_steps=accepted_steps, failure_code=failure_code,
                min_accepted_h=torch.where(accepted_steps > 0, minimum_h, 0))
