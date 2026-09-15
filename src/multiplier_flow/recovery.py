"""Fixed-clock perturbation diagnostics for the frozen contact QP.

The comparison measures the extra error caused by a perturbation, as well as
absolute solver error. A small gain alone does not imply an accurate solution.
"""
from __future__ import annotations

import math
from time import perf_counter

import torch

from .model import LocalProjection
from .problem import (contact_endpoint, converged, gap, make_problem, move, pack,
                      position_error, residuals, select)
from .solvers import integrate_cfm, pgs


def perturb_state(problem, state, *, kind, amplitude, length_scale, generator):
    """Return an orthant-valid state and its *actual* multiplier change.

    Nonnegative multipliers do not imply QP feasibility or convergence.

    ``amplitude * length_scale`` is the requested mass-weighted position RMS.
    Noise is sampled with a CPU generator, then moved to the state's device.
    Correlated noise is smoothed over physical contact coupling, not slot order.

    Each multiplier change is bounded by eight times the requested length
    divided by its contact diagonal. We first clip this bounded candidate to
    the nonnegative orthant, then shrink its actual displacement to the desired
    RMS. If clipping or a near-null direction leaves too little displacement,
    the returned perturbation stays smaller; no unbounded normalization occurs.
    Callers must measure the returned change rather than assume its amplitude.
    """
    if kind not in ("independent", "correlated"):
        raise ValueError("Perturbation kind must be independent or correlated")
    if not math.isfinite(amplitude) or amplitude < 0 or not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("Finite nonnegative amplitude and positive length scale required")
    if generator.device.type != "cpu":
        raise ValueError("Perturbations require a CPU random generator")
    if state.shape != problem["c"].shape or not torch.isfinite(state).all() or (state < 0).any():
        raise ValueError("Perturbation states must be finite, nonnegative, and match the problem")
    mask = problem["mask"]
    if (state[~mask] != 0).any():
        raise ValueError("Padded multipliers must be zero")
    if amplitude == 0:
        return state.clone(), torch.zeros_like(state)

    noise = torch.randn(state.shape, generator=generator, dtype=state.dtype, device="cpu").to(state.device)
    noise = noise * mask
    if kind == "correlated":
        # Positive averaging gives smooth errors along chains/stacks. Distinct
        # contact components remain independent, including padded contacts.
        adjacency = problem["D"].abs() * mask[:, :, None] * mask[:, None, :]
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(torch.finfo(state.dtype).tiny)
        for _ in range(3):
            noise = 0.5 * (noise + (adjacency @ noise[..., None]).squeeze(-1))
    noise = noise / noise.abs().amax(-1, keepdim=True).clamp_min(1)
    diagonal = problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
    requested_rms = amplitude * length_scale
    candidate = (state + 8 * requested_rms * noise / diagonal).clamp_min(0) * mask
    delta = candidate - state
    # Clamp before sqrt so a physically null direction has finite gradients.
    candidate_rms = position_error(problem, candidate, state).clamp_min(torch.finfo(state.dtype).tiny).sqrt()
    shrink = (requested_rms / candidate_rms).clamp_max(1)
    perturbed = state + shrink[:, None] * delta
    # Returning the actual subtraction includes rounding and boundary clipping.
    return perturbed, perturbed - state


def recovery_gain(problem, clean_start, perturbed_start, clean_final, perturbed_final,
                  denominator_epsilon=1e-10):
    """One-scene physical amplification; null perturbations have no ratio."""
    if len(clean_start) != 1 or not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("Gain expects one scene and a positive denominator threshold")
    before = float(position_error(problem, perturbed_start, clean_start).clamp_min(0).sqrt()[0])
    after = float(position_error(problem, perturbed_final, clean_final).clamp_min(0).sqrt()[0])
    valid = math.isfinite(before) and math.isfinite(after) and before > denominator_epsilon
    return dict(actual_perturbation_rms=_finite(before), endpoint_difference_rms=_finite(after),
                gain=after / before if valid else None)


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _timed(device, operation):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, perf_counter() - begin


def _state_metrics(problem, state, optimum, tolerance, completed=True):
    valid = bool(torch.isfinite(state).all() and (state >= 0).all())
    completed = bool(completed and valid)
    stats = {name: _finite(value[0]) for name, value in residuals(problem, state).items()}
    stats.update(projection_position_mse=_finite(position_error(problem, state, optimum)[0]),
                 completed=completed, success=bool(completed and converged(problem, state, tolerance)[0]))
    return stats


def _trajectory_metrics(problem, trajectory, optimum, tolerance, calls, start_step):
    return [dict(step=start_step + i, tau=(start_step + i) / calls,
                 **_state_metrics(problem, state, optimum, tolerance))
            for i, state in enumerate(trajectory)]


def _flow_trial(model, problem, start, optimum, tolerance, calls, start_step, device):
    trajectory, seconds = _timed(device, lambda: integrate_cfm(
        model, problem, start, calls, start_step=start_step, return_trajectory=True))
    completed = bool(torch.isfinite(trajectory).all() and (trajectory >= 0).all())
    metrics = _state_metrics(problem, trajectory[-1], optimum, tolerance, completed)
    metrics.update(seconds=seconds, nfe=calls - start_step,
                   neural_evals=(calls - start_step) * model.neural_evaluations,
                   trajectory=_trajectory_metrics(problem, trajectory, optimum, tolerance, calls, start_step))
    return trajectory[-1], metrics


def _pgs_trial(problem, start, optimum, tolerance, max_sweeps, device):
    result, seconds = _timed(device, lambda: pgs(problem, start, tolerance, max_sweeps))
    metrics = _state_metrics(problem, result["final"], optimum, tolerance)
    metrics.update(seconds=seconds, sweeps=int(result["sweeps"][0]),
                   contact_evals=int(result["contact_evals"][0]),
                   converged=bool(result["converged"][0]),
                   initial=_state_metrics(problem, start, optimum, tolerance))
    return metrics


def _stratified_indices(names, limit):
    if limit < 1:
        raise ValueError("Recovery max_scenes must be positive to bound diagnostic cost")
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(name.rsplit("_", 1)[0], []).append(index)
    selected = []
    for offset in range(max(map(len, groups.values()), default=0)):
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
                if len(selected) == limit:
                    return selected
    return selected


def _start_steps(settings, calls):
    if "start_steps" in settings:
        steps = settings["start_steps"]
        if any(not isinstance(step, int) or not 0 <= step < calls for step in steps):
            raise ValueError("Recovery start_steps must lie in [0, calls)")
    else:
        fractions = settings.get("start_fractions", [0.5])
        if any(not math.isfinite(fraction) or not 0 <= fraction < 1 for fraction in fractions):
            raise ValueError("Recovery start_fractions must lie in [0, 1)")
        steps = [min(calls - 1, int(calls * fraction)) for fraction in fractions]
    if not steps:
        raise ValueError("Recovery requires at least one perturbation step")
    return sorted(set(steps))


def _mean_and_max(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return dict(mean=sum(values) / len(values) if values else None,
                max=max(values) if values else None, finite_count=len(values))


def _summaries(rows):
    """Group repeated trials without treating null or failed gains as zero."""
    groups = {}
    for row in rows:
        key = f"{row['method']}_k{row['calls']}_{row.get('kind', 'invalid_prefix')}"
        groups.setdefault(key, []).append(row)
    report = {}
    for key, trials in groups.items():
        gains = [row["gain"] for row in trials if "gain" in row]
        summary = dict(trials=len(trials), invalid_prefix_count=sum("failure" in row for row in trials),
                       gain=_mean_and_max(gains), valid_gain_count=sum(value is not None for value in gains),
                       null_gain_count=sum(value is None for value in gains))
        for suffix in ("clean", "perturbed"):
            results = [row[suffix] for row in trials if suffix in row]
            stats = dict(attempted=len(results), completed_count=sum(row["completed"] for row in results),
                         success_count=sum(row["success"] for row in results))
            for metric in ("projected_gradient", "penetration", "complementarity", "projection_position_mse"):
                stats[metric] = _mean_and_max(row[metric] for row in results)
            summary[suffix] = stats
        report[key] = summary
    return report


def _cooperative_case(model, calls_list, tolerance, device):
    """Three coupled contacts: initially satisfied neighbors must also move.

    The explicit optimum is used only for reporting errors. Both learned and
    local solvers receive the original QP and zero start, without oracle inputs.
    """
    jacobian = torch.tensor([[1., -1., 0., 0.], [0., 1., -1., 0.],
                             [0., 0., 1., -1.]], device=device)
    problem = pack([make_problem(torch.zeros(4, device=device), jacobian,
                                torch.ones(4, device=device),
                                torch.tensor([0., -0.2, 0.], device=device))])
    optimum = torch.tensor([[0.1, 0.2, 0.1]], device=device)
    start = torch.zeros_like(optimum)
    rows = []
    for method, solver in (("cfm", model), ("local", LocalProjection())):
        for calls in sorted(set([1, *calls_list])):
            final, stats = _flow_trial(solver, problem, start, optimum, tolerance, calls, 0, device)
            rows.append(dict(method=method, calls=calls, final_multiplier=[_finite(x) for x in final[0]],
                             final_gap=[_finite(x) for x in gap(problem, final)[0]], **stats))
    exact_endpoint = contact_endpoint(problem, optimum)
    return dict(scope="Analytic coupled-contact diagnostic; optimum appears only in error metrics",
                initial_gap=problem["c"][0].tolist(), initially_satisfied=[True, False, True],
                optimum_multiplier=optimum[0].tolist(),
                exact_coupling_head_gap=gap(problem, exact_endpoint)[0].tolist(), rows=rows)


@torch.no_grad()
def recovery_evaluation(model, split, config, device):
    """Bounded, stratified clean/perturbed suffix comparison on a fixed QP.

    Configure under ``evaluation.recovery``. Defaults use 12 scenes, K=2/4,
    a halfway perturbation, and independent/correlated noise of amplitude 0.1.
    Model and local trials reuse the same seed for each scene/schedule/noise
    setting. Their states (and therefore clipping) can differ. PGS starts from
    those same states and runs to tolerance; it has no artificial FM clock.
    Reported timings are individual diagnostic calls, not a speed benchmark.
    """
    settings = config["evaluation"].get("recovery", {})
    if not settings.get("enabled", True):
        return dict(enabled=False)
    device = torch.device(device)
    calls_list = settings.get("calls", [2, 4])
    if not calls_list or any(not isinstance(calls, int) or calls < 1 for calls in calls_list):
        raise ValueError("Recovery calls must be a nonempty list of positive integers")
    indices = _stratified_indices(split["names"], int(settings.get("max_scenes", 12)))
    kinds = settings.get("kinds", ["independent", "correlated"])
    amplitudes = settings.get("amplitudes", [0.1])
    if not kinds or not amplitudes:
        raise ValueError("Recovery requires perturbation kinds and amplitudes")
    seed = int(settings.get("seed", int(config.get("seed", 0)) + 30000))
    tolerance = float(config["evaluation"]["tolerance"])
    max_sweeps = int(settings.get("pgs_max_sweeps", config["reference"]["max_sweeps"]))
    length_scale = float(config["model"]["length_scale"])
    denominator_epsilon = float(settings.get("denominator_epsilon", 1e-10))
    if not math.isfinite(tolerance) or tolerance <= 0 or max_sweeps < 1:
        raise ValueError("Recovery requires a finite positive tolerance and positive PGS budget")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("Recovery denominator_epsilon must be finite and positive")
    rows = []
    was_training = getattr(model, "training", None)
    if was_training is not None:
        model.eval()
    try:
        cooperative = _cooperative_case(model, calls_list, tolerance, device)
        for index in indices:
            problem = move(select(split["problem"], slice(index, index + 1)), device, torch.float32)
            optimum = split["optimum"][index:index + 1].to(device=device, dtype=torch.float32)
            for method, solver in (("cfm", model), ("local", LocalProjection())):
                for calls in calls_list:
                    for start_step in _start_steps(settings, calls):
                        prefix, prefix_seconds = _timed(device, lambda: integrate_cfm(
                            solver, problem, torch.zeros_like(optimum), calls,
                            end_step=start_step, return_trajectory=True))
                        start = prefix[-1]
                        if not torch.isfinite(prefix).all() or (prefix < 0).any():
                            rows.append(dict(name=split["names"][index], method=method, calls=calls,
                                             start_step=start_step, completed=False, failure="invalid_prefix"))
                            continue
                        clean_final, clean = _flow_trial(
                            solver, problem, start, optimum, tolerance, calls, start_step, device)
                        pgs_clean = _pgs_trial(problem, start, optimum, tolerance, max_sweeps, device)
                        prefix_metrics = _trajectory_metrics(problem, prefix, optimum, tolerance, calls, 0)
                        for kind_index, kind in enumerate(kinds):
                            for amplitude_index, amplitude in enumerate(amplitudes):
                                # No method term: both solvers get matching random draws.
                                trial_seed = seed + index * 1000003 + calls * 10007 + start_step * 503
                                trial_seed += kind_index * 37 + amplitude_index
                                perturbed, delta = perturb_state(
                                    problem, start, kind=kind, amplitude=float(amplitude),
                                    length_scale=length_scale,
                                    generator=torch.Generator(device="cpu").manual_seed(trial_seed))
                                final, disturbed = _flow_trial(
                                    solver, problem, perturbed, optimum, tolerance, calls, start_step, device)
                                rows.append(dict(
                                    name=split["names"][index], method=method, calls=calls,
                                    start_step=start_step, start_time=start_step / calls,
                                    remaining_calls=calls - start_step, kind=kind, amplitude=float(amplitude),
                                    requested_perturbation_rms=float(amplitude) * length_scale,
                                    multiplier_delta_max=float(delta.abs().max()),
                                    **recovery_gain(problem, start, perturbed, clean_final, final, denominator_epsilon),
                                    prefix_seconds=prefix_seconds, prefix_trajectory=prefix_metrics,
                                    clean=clean, perturbed=disturbed, pgs_clean=pgs_clean,
                                    pgs_perturbed=_pgs_trial(problem, perturbed, optimum, tolerance, max_sweeps, device)))
    finally:
        if was_training is not None:
            model.train(was_training)
    return dict(enabled=True, scope="Frozen-QP perturbation; unchanged FM clock and remaining calls",
                seed=seed, scenes=[split["names"][index] for index in indices],
                norm="Mass-weighted RMS position displacement; null denominators have gain=null",
                amplitude_note="Requested physical RMS; clipping and bounded multiplier noise can reduce it",
                timing_note="One suffix/PGS solve per row; excludes setup, perturbation generation and metrics",
                denominator_epsilon=denominator_epsilon, summary=_summaries(rows),
                cooperative_case=cooperative, rows=rows)
