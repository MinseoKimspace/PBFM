"""Adaptive Euler over tau in [0,1]; checks accepted nodes, not whole segments."""
from dataclasses import dataclass
import math

import torch

from .physics import energy_and_violation, geometry, valid_positions


@dataclass(frozen=True)
class SolverConfig:
    steps: int = 32
    max_steps: int = 256
    max_backtracks: int = 16
    min_step: float = 1e-8
    max_displacement: float = 0.15
    tolerance: float = 0.001
    armijo: float = 0.0001

    def __post_init__(self):
        if self.steps < 1 or self.max_steps < 1 or self.max_backtracks < 0:
            raise ValueError("Positive step counts and nonnegative backtrack budget required")
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Solver parameters must be finite")
        if min(self.min_step, self.max_displacement) <= 0 or self.tolerance < 0 or not 0 < self.armijo < 1:
            raise ValueError("Invalid step bounds, tolerance or Armijo coefficient")


@torch.no_grad()
def integrate(field, z, radius, physics, cfg, record=False, stop_on_tolerance=True):
    """No fallback and no velocity rescaling: rejected h is reduced in tau too.

    nfe/energy_evals are logical per-world evaluations. field_calls records
    actual vectorized calls (which also execute inactive rows). History holds
    pre-step states and the *accepted* dt; failed/inactive rows have dt=0.
    Below contact tolerance only, a descent displacement no larger than one
    coordinate ULP may advance time if computed energy does not increase.
    Its actual tangent is retained and accepted_roundoff counts this event.
    """
    z = z.detach().clone()
    batch = z.shape[0]
    current = geometry(z, radius, physics)
    tau = z.new_zeros(batch)
    failed = torch.zeros(batch, dtype=torch.bool, device=z.device)
    code = torch.zeros(batch, dtype=torch.long, device=z.device)
    nfe, backtracks = torch.zeros_like(code), torch.zeros_like(code)
    accepted_roundoff = torch.zeros_like(code)
    energy_evals = torch.ones_like(code)
    initially_solved = current["max_violation"] <= cfg.tolerance
    first_tolerance_time = torch.where(initially_solved, tau, -torch.ones_like(tau))
    first_tolerance_nfe = torch.where(initially_solved, nfe, -torch.ones_like(nfe))
    first_tolerance_backtracks = first_tolerance_nfe.clone()
    first_tolerance_energy_evals = torch.where(initially_solved, energy_evals, -torch.ones_like(code))
    field_calls = 0
    history = dict(z=[], tau=[], u=[], dt=[], energy=[], max_violation=[])
    for iteration in range(cfg.max_steps):
        active = ~failed & (tau < 1.0)
        if iteration:
            if not active.any():
                break
            current = geometry(z, radius, physics)
            energy_evals += active
        if stop_on_tolerance:
            active &= current["max_violation"] > cfg.tolerance
        if not active.any():
            break
        old_z, old_tau = z.clone(), tau.clone()
        u = field(z, tau)
        field_calls += 1
        nfe += active
        if u.shape != z.shape:
            raise ValueError("field must return [B,N,2]")
        finite = torch.isfinite(u).all(dim=(1, 2))
        bad = active & ~finite
        failed |= bad
        code[bad] = 1
        u = torch.where((active & finite)[:, None, None], u, 0.0)
        norm = torch.linalg.vector_norm(u, dim=-1).amax(dim=1)
        slope = (current["gradient"] * u.flatten(1)).sum(dim=1)
        stationary = (current["energy"] == 0) & (norm == 0)
        bad = active & finite & ~stationary & ((slope >= 0) | ~torch.isfinite(slope))
        failed |= bad
        code[bad] = 2
        pending = active & ~failed
        h = torch.minimum(torch.full_like(tau, 1.0 / cfg.steps), 1.0 - tau)
        h = torch.minimum(h, cfg.max_displacement / norm.clamp_min(torch.finfo(z.dtype).tiny))
        floor = torch.minimum(torch.full_like(tau, cfg.min_step), 1.0 - tau)
        bad = pending & (h < floor)
        failed |= bad
        code[bad] = 3
        pending &= ~bad
        accepted_dt = torch.zeros_like(tau)
        toward = torch.where(u >= 0, torch.full_like(u, float("inf")),
                             torch.full_like(u, -float("inf")))
        resolution = (torch.nextafter(old_z, toward) - old_z).abs()
        for trial in range(cfg.max_backtracks + 1):
            if not pending.any():
                break
            candidate = old_z + torch.where(pending, h, 0.0)[:, None, None] * u
            valid = valid_positions(candidate)
            safe_candidate = torch.where(valid[:, None, None], candidate, old_z)
            value, candidate_violation = energy_and_violation(safe_candidate, radius, physics)
            energy_evals += pending
            moved = (candidate != old_z).any(dim=(1, 2))
            decrease = ((value < current["energy"]) & moved &
                        (value <= current["energy"] + cfg.armijo * h * slope))
            subresolution = ((h[:, None, None] * u).abs() <= resolution).all(dim=(1, 2))
            roundoff = ((current["max_violation"] <= cfg.tolerance) & (slope < 0) &
                        subresolution & (value <= current["energy"]))
            accept = pending & valid & torch.isfinite(value) & (decrease | stationary | roundoff)
            reached = accept & (first_tolerance_time < 0) & (candidate_violation <= cfg.tolerance)
            first_tolerance_time = torch.where(reached, (old_tau + h).clamp_max(1.0), first_tolerance_time)
            first_tolerance_nfe = torch.where(reached, nfe, first_tolerance_nfe)
            first_tolerance_backtracks = torch.where(reached, backtracks, first_tolerance_backtracks)
            first_tolerance_energy_evals = torch.where(reached, energy_evals, first_tolerance_energy_evals)
            accepted_roundoff += accept & roundoff & ~decrease
            z = torch.where(accept[:, None, None], candidate, z)
            accepted_dt = torch.where(accept, h, accepted_dt)
            pending &= ~accept
            if trial < cfg.max_backtracks:
                backtracks += pending
                h = torch.where(pending, h * 0.5, h)
                bad = pending & (h < floor)
                failed |= bad
                code[bad] = 3
                pending &= ~bad
        failed |= pending
        code[pending] = 4
        tau = (tau + accepted_dt).clamp_max(1.0)
        if record:
            history["z"].append(old_z)
            history["tau"].append(old_tau)
            history["u"].append(torch.where((accepted_dt > 0)[:, None, None], u, 0.0))
            history["dt"].append(accepted_dt)
            history["energy"].append(current["energy"])
            history["max_violation"].append(current["max_violation"])
    final_geometry = geometry(z, radius, physics)
    energy_evals += 1
    converged = final_geometry["max_violation"] <= cfg.tolerance
    completed = ~failed & ((tau >= 1.0) | (converged & stop_on_tolerance))
    budget_failed = ~failed & ~completed
    failed |= budget_failed
    code[budget_failed] = 5
    labels = ("none", "nonfinite_field", "non_descent_or_zero_field", "min_step",
              "line_search_failed", "max_steps")
    result = dict(final=z, time=tau, converged=converged, completed=completed, failed=failed,
                  nfe=nfe, backtracks=backtracks, energy_evals=energy_evals,
                  accepted_roundoff=accepted_roundoff,
                  first_tolerance_time=first_tolerance_time,
                  first_tolerance_nfe=first_tolerance_nfe,
                  first_tolerance_backtracks=first_tolerance_backtracks,
                  first_tolerance_energy_evals=first_tolerance_energy_evals,
                  field_calls=field_calls, executed_field_rows=field_calls * batch,
                  failure_reason=[labels[index] for index in code.cpu().tolist()])
    if record:
        result["history"] = {key: torch.stack(values) if values else
                             z.new_zeros((0, batch, *z.shape[1:]) if key in ("z", "u") else (0, batch))
                             for key, values in history.items()}
    return result
