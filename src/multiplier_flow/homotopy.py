"""Non-learning, fixed-normal Newton-homotopy diagnostics (CPU float64).

The original QP/position anchor never changes. Time t is solver progress, not
physical time. No arc-length clock, neural model, clipping, or PGS finish.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

import torch


@dataclass
class Homotopy:
    D: torch.Tensor
    c: torch.Tensor
    start: torch.Tensor
    r0: torch.Tensor
    mu0: float
    mu_min: float
    schedule: str
    kappa: float
    gap_scale: float

    @classmethod
    def create(cls, D, c, *, initialization="diagonal_scaled", schedule="geometric",
               gap_scale=0.1, mu0_factor=1.0, mu_min_ratio=1e-6,
               epsilon_ratio=1e-6, kappa=4.0):
        """Scale mu as gap^2 / mean(D_ii), lambda as gap / D_ii.

        diagonal_scaled cancels only the diagonal/barrier mismatch. It is NOT
        a globally centered or necessarily primal-feasible initial state.
        isolated_root additionally includes c in each scalar barrier solve.
        """
        if (D.ndim != 2 or c.ndim != 1 or D.shape != (len(c), len(c)) or not len(c)
                or D.dtype != torch.float64 or c.dtype != D.dtype
                or D.device.type != "cpu" or c.device != D.device
                or not torch.isfinite(D).all() or not torch.isfinite(c).all()
                or not torch.allclose(D, D.T, atol=1e-12, rtol=1e-10)
                or not (D.diag() > 0).all()):
            raise ValueError("Expected a nonempty symmetric CPU float64 contact matrix")
        if float(torch.linalg.eigvalsh(D).min()) < -1e-10 * max(1., float(D.abs().max())):
            raise ValueError("Contact matrix must be positive semidefinite")
        values = (gap_scale, mu0_factor, mu_min_ratio, epsilon_ratio, kappa)
        if not all(isfinite(x) and x > 0 for x in values) or mu_min_ratio >= 1:
            raise ValueError("Positive finite scales/kappa and 0 < mu_min_ratio < 1 required")
        if schedule not in ("linear", "geometric"):
            raise ValueError("schedule must be linear or geometric")
        diagonal = D.diag()
        mu0 = mu0_factor * gap_scale ** 2 / float(diagonal.mean())
        if not isfinite(mu0) or not isfinite(mu0 * mu_min_ratio) or mu0 * mu_min_ratio <= 0:
            raise ValueError("Barrier scale overflow/underflow")
        if initialization == "epsilon":
            start = epsilon_ratio * gap_scale / diagonal
        elif initialization == "diagonal_scaled":
            start = (mu0 / diagonal).sqrt()
        elif initialization == "isolated_root":
            discriminant = (c.square() + 4 * diagonal * mu0).sqrt()
            # Stable positive quadratic root, including well-separated contacts.
            start = torch.empty_like(c)
            positive = c >= 0
            start[positive] = 2 * mu0 / (discriminant[positive] + c[positive])
            start[~positive] = (discriminant[~positive] - c[~positive]) / (2 * diagonal[~positive])
        else:
            raise ValueError("Unknown homotopy initialization")
        if not torch.isfinite(start).all() or not (start > 0).all():
            raise ValueError("Initialization overflow/underflow; choose well-scaled parameters")
        r0 = D @ start + c - mu0 / start
        return cls(D.clone(), c.clone(), start, r0, mu0, mu0 * mu_min_ratio,
                   schedule, kappa, gap_scale)

    def mu(self, t):
        if not 0 <= t <= 1:
            raise ValueError("Homotopy time must lie in [0, 1]")
        if self.schedule == "geometric":
            rate = log(self.mu_min / self.mu0)
            value = self.mu_min if t == 1 else self.mu0 * exp(rate * t)
            return value, rate * value
        return self.mu0 + t * (self.mu_min - self.mu0), self.mu_min - self.mu0

    def residual(self, lam, t):
        if lam.shape != self.c.shape or not torch.isfinite(lam).all() or not (lam > 0).all():
            raise ValueError("Homotopy multipliers must remain finite and strictly positive")
        mu, _ = self.mu(t)
        return self.D @ lam + self.c - mu / lam - (1 - t) * self.r0

    def system(self, lam, t):
        H = self.residual(lam, t)
        mu, mu_dot = self.mu(t)
        A = self.D + torch.diag(mu / lam.square())
        H_t = -mu_dot / lam + self.r0
        return A, -H_t - self.kappa * H, H, H_t

    def field(self, lam, t):
        A, rhs, _, _ = self.system(lam, t)
        return torch.linalg.solve(A, rhs)


def positive_step(lam, direction, requested=1.0, fraction=0.99):
    """Fraction-to-boundary only; does not promise energy/residual descent."""
    if not 0 < fraction < 1:
        raise ValueError("Boundary fraction must lie in (0, 1)")
    negative = direction < 0
    limit = fraction * float((-lam[negative] / direction[negative]).min()) if negative.any() else requested
    return min(requested, limit)


def solve_center(flow, t, start, tolerance=1e-9, max_iterations=80):
    """Newton root solve for a reference path point, NOT a cheap Euler step.

    Each factorization/solve and line-search residual evaluation is counted.
    This routine is never called by euler() or recovery().
    """
    lam = start.clone()
    solves = backtracks = evaluations = 0
    reason = "iteration_budget"
    for _ in range(max_iterations):
        H = flow.residual(lam, t)
        evaluations += 1
        if float(H.abs().max()) <= tolerance * flow.gap_scale:
            reason = "converged"
            break
        mu, _ = flow.mu(t)
        direction = torch.linalg.solve(flow.D + torch.diag(mu / lam.square()), -H)
        solves += 1
        h = positive_step(lam, direction)
        old = float(H.norm())
        for attempt in range(40):
            candidate = lam + h * direction
            if torch.isfinite(candidate).all() and (candidate > 0).all():
                new = flow.residual(candidate, t)
                evaluations += 1
                if float(new.norm()) <= (1 - 1e-4 * h) * old:
                    lam = candidate
                    break
            h *= .5
            backtracks += 1
        else:
            reason = "line_search_failed"
            break
    error = float(flow.residual(lam, t).abs().max()) / flow.gap_scale
    return dict(final=lam, completed=error <= tolerance, reason="converged" if error <= tolerance else reason,
                linear_solves=solves, backtracks=backtracks, residual_evals=evaluations,
                normalized_H=error)


def reference_path(flow, points=129, tolerance=1e-9):
    """Uniform-t root solves expose late motion; there is no arc-length remap."""
    if points < 3:
        raise ValueError("At least three reference points required")
    times, states = [0.0], [flow.start.clone()]
    totals = dict(linear_solves=0, backtracks=0, residual_evals=0)
    result = dict(completed=True, reason="converged", normalized_H=0.)
    for i in range(1, points):
        t = i / (points - 1)
        result = solve_center(flow, t, states[-1], tolerance)
        for key in totals:
            totals[key] += result[key]
        times.append(t)
        states.append(result["final"])
        if not result["completed"]:
            break
    return dict(times=times, states=torch.stack(states), **totals,
                completed=result["completed"] and len(times) == points,
                reason=result["reason"], normalized_H=result["normalized_H"])


def euler(flow, calls, *, guarded=False, max_nfe_multiplier=64, min_step=1e-12):
    """Full analytic field: ideal-target diagnostic, not learned-model evidence.

    Raw uses exactly the requested uniform grid or explicitly fails positivity.
    Guarded may take MORE calls to finish [0,1]; no hidden catch-up/PGS finish.
    """
    if calls < 1 or max_nfe_multiplier < 1 or min_step <= 0:
        raise ValueError("Positive integration budgets required")
    lam, t = flow.start.clone(), 0.0
    states, times = [lam.clone()], [t]
    nfe = interventions = 0
    reason = "nfe_budget"
    for _ in range(calls * max_nfe_multiplier if guarded else calls):
        if t >= 1:
            reason = "completed"
            break
        direction = flow.field(lam, t)
        nfe += 1
        if not torch.isfinite(direction).all():
            reason = "nonfinite_field"
            break
        h = min(1 / calls, 1 - t) if guarded else (len(times) / calls - t)
        if guarded:
            limited = positive_step(lam, direction, h)
            interventions += int(limited < h)
            h = limited
        if h < min_step:
            reason = "step_underflow"
            break
        candidate = lam + h * direction
        if (not torch.isfinite(candidate).all() or not (candidate > 0).all()
                or not torch.isfinite(.5 * candidate @ (flow.D @ candidate) + flow.c @ candidate)):
            reason = "nonpositive_or_nonfinite_state"
            break
        lam, t = candidate, min(1., t + h)
        if 1 - t < 1e-14:
            t = 1.
        states.append(lam.clone())
        times.append(t)
    return dict(final=lam, completed=t == 1., reason="completed" if t == 1. else reason,
                nfe=nfe, linear_solves=nfe, interventions=interventions, elapsed=t,
                times=times, states=torch.stack(states))


def recovery(flow, start, t0=.5, duration=.25, steps=128, log_perturbation=.35):
    """RK4 off-path recovery, with the ORIGINAL r0 frozen.

    Exact ODE identity: H(t)=exp(-kappa*(t-t0))*H(t0). This says nothing about
    monotone physical energy or penetration. RK stages never clip lambda.
    """
    if not 0 <= t0 < t0 + duration <= 1 or steps < 1 or log_perturbation <= 0:
        raise ValueError("Invalid recovery interval/perturbation")
    signs = torch.where(torch.arange(len(start)) % 2 == 0, 1., -1.).to(start)
    lam = start * (log_perturbation * signs).exp()
    initial_H = flow.residual(lam, t0)
    times, errors, expected = [t0], [float(initial_H.norm())], [float(initial_H.norm())]
    nfe, reason = 0, "completed"
    h = duration / steps
    for i in range(steps):
        t = t0 + i * h
        stages = []
        try:
            for fraction, multiplier in ((0., 0.), (.5, .5), (.5, .5), (1., 1.)):
                point = lam if not stages else lam + multiplier * h * stages[-1]
                stages.append(flow.field(point, t + fraction * h))
                nfe += 1
            candidate = lam + h * (stages[0] + 2*stages[1] + 2*stages[2] + stages[3]) / 6
            current_H = flow.residual(candidate, t + h)
        except (ValueError, torch.linalg.LinAlgError):
            reason = "invalid_rk_stage"
            break
        if not torch.isfinite(current_H).all():
            reason = "nonfinite_residual"
            break
        lam = candidate
        times.append(t + h)
        errors.append(float(current_H.norm()))
        expected.append(float(initial_H.norm()) * exp(-flow.kappa*(t+h-t0)))
    target = initial_H * exp(-flow.kappa * (times[-1] - t0))
    final_H = flow.residual(lam, times[-1])
    relative = float((final_H - target).norm()) / max(float(initial_H.norm()), 1e-30)
    A, rhs, H, H_t = flow.system(start * (log_perturbation * signs).exp(), t0)
    direction = torch.linalg.solve(A, rhs)
    identity_error = float((A @ direction + H_t + flow.kappa * H).abs().max()) / flow.gap_scale
    return dict(completed=len(times) == steps + 1, reason=reason,
                nfe=nfe, linear_solves=nfe + 1, relative_decay_error=relative,
                differential_identity_error=identity_error, times=times,
                H_norm=errors, expected_H_norm=expected,
                final_multiplier=lam.tolist(), r0_recomputed=False)
