"""Native mass-weighted nonlinear Gauss-Seidel PBD, not a tau ODE baseline.

Each violated contact receives its full linearized position correction; circle
normals are recomputed when that contact is visited. Sweeps repeat to tolerance
or an explicit budget exhaustion. There is no gain, line search, or fallback.
"""
import math

import torch

from .physics import energy_and_violation, geometry, valid_positions


@torch.no_grad()
def integrate_pbd(z, radius, physics, max_sweeps=32, tolerance=0.001, record=False):
    """Common evaluation adapter plus sweeps/contact_evals/contact_updates.

    nfe and time are zero because no learned field or tau integration is used.
    energy_evals counts whole-world residual checks; contact_evals separately
    counts visited individual pair/wall constraints. History records the actual
    pre-sweep states and full sweep displacement, not an invented ODE tangent.
    """
    if not isinstance(max_sweeps, int) or isinstance(max_sweeps, bool) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    z = z.detach().clone()
    initial = geometry(z, radius, physics)
    batch, nodes, _ = z.shape
    inverse_mass = initial["inverse_mass"]
    device = z.device
    zeros = torch.zeros(batch, dtype=torch.long, device=device)
    sweeps, contact_evals, contact_updates = zeros.clone(), zeros.clone(), zeros.clone()
    energy_evals = torch.ones_like(zeros)
    failed = torch.zeros(batch, dtype=torch.bool, device=device)
    code = zeros.clone()
    value, violation = initial["energy"], initial["max_violation"]
    converged = violation <= tolerance
    first_tolerance_sweep = torch.where(converged, zeros, -torch.ones_like(zeros))
    first_tolerance_contact_evals = first_tolerance_sweep.clone()
    first_tolerance_energy_evals = torch.where(converged, energy_evals, -torch.ones_like(zeros))
    history = dict(z=[], u=[], tau=[], dt=[], energy=[], max_violation=[], sweep_index=[])
    pairs = torch.triu_indices(nodes, nodes, 1, device=device).T.tolist()

    for _ in range(max_sweeps):
        active = ~failed & ~converged
        if not active.any():
            break
        previous = z.clone()
        previous_value, previous_violation = value.clone(), violation.clone()
        previous_sweeps = sweeps.clone()
        sweeps += active
        for i, j in pairs:
            running = active & ~failed
            contact_evals += running
            difference = z[:, i] - z[:, j]
            distance = torch.linalg.vector_norm(difference, dim=-1)
            invalid = running & ((distance <= 1e-10) | ~torch.isfinite(distance))
            failed |= invalid
            code[invalid] = 1
            running &= ~invalid
            safe_distance = torch.where(running, distance, torch.ones_like(distance))
            normal = difference / safe_distance.unsqueeze(-1)
            residual = (distance - radius[:, i] - radius[:, j] + physics.slop).clamp_max(0.0)
            amount = torch.where(running, -residual / (inverse_mass[:, i] + inverse_mass[:, j]), 0.0)
            correction = amount.unsqueeze(-1) * normal
            z[:, i] += inverse_mass[:, i, None] * correction
            z[:, j] -= inverse_mass[:, j, None] * correction
            contact_updates += running & (residual < 0)
        for wall in range(3):
            for i in range(nodes):
                running = active & ~failed
                contact_evals += running
                axis = 1 if wall == 0 else 0
                if wall == 0:
                    gap = z[:, i, 1] - radius[:, i] - physics.y_ground
                elif wall == 1:
                    gap = z[:, i, 0] - radius[:, i] + physics.xy_limit
                else:
                    gap = physics.xy_limit - z[:, i, 0] - radius[:, i]
                residual = (gap + physics.slop).clamp_max(0.0)
                displacement = torch.where(running, -residual, 0.0)
                z[:, i, axis] += displacement if wall < 2 else -displacement
                contact_updates += running & (residual < 0)
        invalid = active & ~valid_positions(z)
        failed |= invalid
        code[invalid] = 1
        # Failed worlds retain the last valid pre-sweep configuration.
        z = torch.where(failed[:, None, None], previous, z)
        value, violation = energy_and_violation(z, radius, physics)
        energy_evals += active
        converged = violation <= tolerance
        reached = active & ~failed & converged & (first_tolerance_sweep < 0)
        first_tolerance_sweep = torch.where(reached, sweeps, first_tolerance_sweep)
        first_tolerance_contact_evals = torch.where(reached, contact_evals, first_tolerance_contact_evals)
        first_tolerance_energy_evals = torch.where(reached, energy_evals, first_tolerance_energy_evals)
        if record:
            history["z"].append(previous)
            history["u"].append(z - previous)
            history["tau"].append(z.new_zeros(batch))
            history["dt"].append(active.to(z.dtype))
            history["energy"].append(previous_value)
            history["max_violation"].append(previous_violation)
            history["sweep_index"].append(previous_sweeps)
    exhausted = ~failed & ~converged
    labels = ("none", "invalid_contact_configuration")
    result = dict(final=z, converged=converged, failed=failed, completed=~failed,
                  budget_exhausted=exhausted,
                  time=z.new_zeros(batch), nfe=zeros.clone(), backtracks=zeros.clone(),
                  accepted_roundoff=zeros.clone(), energy_evals=energy_evals,
                  field_calls=0, executed_field_rows=0, sweeps=sweeps,
                  contact_evals=contact_evals, contact_updates=contact_updates,
                  first_tolerance_sweep=first_tolerance_sweep,
                  first_tolerance_contact_evals=first_tolerance_contact_evals,
                  first_tolerance_energy_evals=first_tolerance_energy_evals,
                  failure_reason=[labels[index] for index in code.cpu().tolist()],
                  termination_reason=["invalid_contact_configuration" if bad else
                                      "max_sweeps" if budget else "tolerance"
                                      for bad, budget in zip(failed.cpu().tolist(), exhausted.cpu().tolist())])
    if record:
        result["history"] = {key: torch.stack(values) if values else z.new_zeros(
            (0, batch, nodes, 2) if key in ("z", "u") else (0, batch))
            for key, values in history.items()}
    return result
