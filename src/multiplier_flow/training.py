"""CFM supervision plus optional differentiable endpoint and recovery losses.

These losses keep the original QP and teacher endpoint fixed. Only complete
integrations to t=1 receive a physical loss; intermediate FM states need not
satisfy KKT. No teacher trajectory, new velocity target, or physical-frame
unrolling is introduced here.
"""
from math import isfinite

import torch

from .model import cfm_loss
from .problem import gap, position_error
from .solvers import integrate_cfm


def validate_objectives(settings):
    """Reject invalid weights/budgets before preparing data or training."""
    for name in ("inner_rollout", "recovery"):
        options = settings.get(name, {})
        for key, default in (("weight", 0.0), ("residual_weight", 1.0), ("position_weight", 1.0)):
            value = float(options.get(key, default))
            if not isfinite(value) or value < 0:
                raise ValueError(f"train.{name}.{key} must be finite and nonnegative")
        if float(options.get("weight", 0.0)) > 0:
            if not any(float(options.get(key, 1.0)) > 0 for key in ("residual_weight", "position_weight")):
                raise ValueError(f"train.{name} must enable at least one physical loss")
    calls = settings.get("inner_rollout", {}).get("calls", [1, 2, 4])
    if not isinstance(calls, (list, tuple)) or not calls or any(type(k) is not int or k < 1 for k in calls):
        raise ValueError("inner_rollout.calls must be a nonempty list of positive integers")
    recovery = settings.get("recovery", {})
    if type(recovery.get("calls", 4)) is not int or recovery.get("calls", 4) < 2:
        raise ValueError("recovery.calls must be an integer >= 2")
    amplitude = float(recovery.get("amplitude", .1))
    if not isfinite(amplitude) or amplitude < 0:
        raise ValueError("recovery.amplitude must be finite and nonnegative")
    kinds = recovery.get("kinds", ["independent", "correlated"])
    if not kinds or any(kind not in ("independent", "correlated") for kind in kinds):
        raise ValueError("recovery.kinds must contain independent and/or correlated")


def physical_endpoint_loss(problem, prediction, target, length_scale, options):
    """Dimensionless KKT and mass-weighted physical position error.

    R_eta/eta has gap units; divide by length_scale to retain absolute error
    magnitude. Position MSE uses physical displacement, so redundant multiplier
    representations are not penalized for representing the same position.
    """
    residual = prediction - (prediction - problem["eta"] * gap(problem, prediction)).clamp_min(0)
    normalized = residual / (problem["eta"] * length_scale)
    valid = problem["mask"]
    residual_loss = ((normalized.square() * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
    position_loss = position_error(problem, prediction, target).mean() / length_scale ** 2
    total = (float(options.get("residual_weight", 1.0)) * residual_loss
             + float(options.get("position_weight", 1.0)) * position_loss)
    return total, residual_loss, position_loss


def training_objective(model, problem, source, target, tau, settings, generator):
    """Sample one inner budget per update; leave every unrolled state attached.

    The independent CPU RNG makes rollout sampling reproducible without changing
    the CFM times/data order between ablations. Perturbation randomness is an
    exogenous training input; the model's prefix still receives suffix gradients.
    """
    cfm, endpoint_mse = cfm_loss(model, problem, source, target, tau)
    total = cfm
    zero = cfm.new_zeros(())
    terms = dict(cfm=cfm, endpoint_estimate_mse=endpoint_mse,
                 inner=zero, inner_residual=zero, inner_position=zero,
                 recovery=zero, recovery_residual=zero, recovery_position=zero)
    costs = dict(inner_calls=0, recovery_calls=0)
    inner = settings.get("inner_rollout", {})
    if float(inner.get("weight", 0.0)) > 0:
        budgets = inner.get("calls", [1, 2, 4])
        calls = budgets[int(torch.randint(len(budgets), (), generator=generator))]
        endpoint = integrate_cfm(model, problem, torch.zeros_like(source), calls)
        loss, residual, position = physical_endpoint_loss(problem, endpoint, target, model.length_scale, inner)
        total = total + float(inner["weight"]) * loss
        terms.update(inner=loss, inner_residual=residual, inner_position=position)
        costs["inner_calls"] = calls
    recovery = settings.get("recovery", {})
    if float(recovery.get("weight", 0.0)) > 0:
        from .recovery import perturb_state

        calls = int(recovery.get("calls", 4))
        split_step = int(torch.randint(1, calls, (), generator=generator))
        kinds = recovery.get("kinds", ["independent", "correlated"])
        kind = kinds[int(torch.randint(len(kinds), (), generator=generator))]
        state = integrate_cfm(model, problem, torch.zeros_like(source), calls, end_step=split_step)
        perturbed, _ = perturb_state(problem, state, kind=kind,
            amplitude=float(recovery.get("amplitude", .1)), length_scale=model.length_scale,
            generator=generator)
        endpoint = integrate_cfm(model, problem, perturbed, calls, start_step=split_step)
        loss, residual, position = physical_endpoint_loss(problem, endpoint, target, model.length_scale, recovery)
        total = total + float(recovery["weight"]) * loss
        terms.update(recovery=loss, recovery_residual=residual, recovery_position=position)
        costs["recovery_calls"] = calls  # Prefix plus suffix; no extra solve is hidden.
    terms["total"] = total
    return total, {name: value.detach() for name, value in terms.items()}, costs
