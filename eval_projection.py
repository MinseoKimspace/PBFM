"""Evaluate contact-flow solvers; --preflight needs neither training nor a dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.contact_flow.dynamics import free_position, finite_difference_state, make_projection_condition
from src.contact_flow.physics import FlowConfig, PhysicsConfig, geometry, make_baseline
from src.contact_flow.solver import SolverConfig, integrate


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite_number(value):
    """Never silently serialize JSON's nonstandard Infinity or NaN tokens."""
    value = float(value)
    return value if math.isfinite(value) else None


def linear_residual_diagnostic(steps):
    """Fixed, independent linear constraint: Newton displacement != ODE flow."""
    if steps < 1:
        raise ValueError("steps must be positive")
    return {"newton_unit_step_residual_ratio": 0.0,
            "continuous_unit_time_residual_ratio": math.exp(-1.0),
            "euler_unit_time_residual_ratio": (1.0 - 1.0 / steps) ** steps,
            "euler_steps": steps,
            "scope": "fixed linear constraint, nonsingular D; not a nonlinear contact projection"}


@torch.no_grad()
def fixed_contact_diagnostic(steps, selected_device=torch.device("cpu")):
    """Exercise the actual inverse field/integrator on one fixed ground contact."""
    physics = PhysicsConfig(slop=0.0, contact_margin=0.0)
    radius = torch.tensor([[0.5]], dtype=torch.float64, device=selected_device)
    initial = torch.tensor([[[0.0, 0.4]]], dtype=torch.float64, device=selected_device)
    solver = SolverConfig(steps=steps, max_steps=max(256, steps + 1),
                          max_displacement=1.0, tolerance=0.0)
    result = integrate(make_baseline("inverse", radius, physics), initial, radius,
                       physics, solver, stop_on_tolerance=False)
    ratio = geometry(result["final"], radius, physics)["max_violation"][0] / 0.1
    # The inverse baseline regularizes D by 1e-4 * max(diag(D)). Here D is scalar.
    gain = 1.0 / 1.0001 ** 2
    predicted = (1.0 - gain / steps) ** steps
    return {**linear_residual_diagnostic(steps),
            "actual_regularized_inverse_residual_ratio": finite_number(ratio),
            "predicted_regularized_euler_residual_ratio": predicted,
            "absolute_error": finite_number(abs(ratio - predicted)),
            "failed": bool(result["failed"][0])}


def _failure_histogram(result, failed=None):
    failed = result["failed"] if failed is None else failed
    reasons = result.get("failure_reason", ["unspecified"] * failed.numel())
    return dict(Counter(reason if reason != "none" else "nonfinite_or_incomplete_state"
                        for reason, is_failed in zip(reasons, failed.cpu().tolist()) if is_failed))


def _contact_metrics(proposal, final, radius, physics):
    before = geometry(proposal.double(), radius.double(), physics)
    after = geometry(final.double(), radius.double(), physics)
    active_pairs = (before["edge_j"] >= 0)[None, :] & (before["residual"] < 0)
    gaps = after["gap"][active_pairs].clamp_min(0)
    return {
        "energy_initial_mean": finite_number(before["energy"].mean()),
        "energy_final_mean": finite_number(after["energy"].mean()),
        "energy_increase_count": int((after["energy"] > before["energy"] + 1e-10).sum()),
        "penetration_mean": finite_number(after["penetration"].mean()),
        "penetration_max": finite_number(after["penetration"].max()),
        "max_violation": finite_number(after["max_violation"].max()),
        "correction_norm_mean": finite_number((final.double() - proposal.double()).flatten(1).norm(dim=1).mean()),
        "initial_active_pair_gap_mean": finite_number(gaps.mean()) if gaps.numel() else None,
        "initial_active_pair_count": gaps.numel(),
    }


def _cost_metrics(result):
    report = {"field_evaluations": int(result["nfe"].sum()),
              "vectorized_field_calls": int(result.get("field_calls", result["nfe"].max())),
              "executed_field_rows": int(result.get("executed_field_rows", result["nfe"].sum())),
              "backtracks": int(result["backtracks"].sum()),
              "energy_evaluations": int(result["energy_evals"].sum()),
              "tau_mean": finite_number(result["time"].double().mean())}
    for key in ("sweeps", "contact_evals", "contact_updates"):
        if key in result:
            report[key] = int(result[key].sum())
    for key in ("time", "nfe", "backtracks", "energy_evals", "sweep", "contact_evals"):
        name = "first_tolerance_" + key
        if name in result:
            values = torch.as_tensor(result[name]).cpu()
            report[name + "_per_world"] = values.tolist()
            reached = values >= 0
            report[name + "_mean_reached"] = finite_number(values[reached].double().mean()) if reached.any() else None
    if "sweeps" in result:
        report["cost_unit"] = "native PBD sweeps/contact evaluations; zero neural/ODE NFE is not zero work"
        report["tau_mean"] = None
    else:
        report["cost_unit"] = "ODE field/energy evaluations, including adaptive backtracking"
    return report


def _convergence_breakdown(proposal, radius, physics, tolerance, result, scene_types=None):
    before = geometry(proposal.double(), radius.double(), physics)
    violating = before["max_violation"] > tolerance
    finite = torch.isfinite(result["final"]).flatten(1).all(1)
    solved = result["converged"] & finite & ~result["failed"]

    def counts(mask):
        return {"samples": int(mask.sum()), "converged_count": int((mask & solved).sum()),
                "integration_failed_count": int((mask & result["failed"]).sum()),
                "initial_max_violation": finite_number(before["max_violation"][mask].max()) if mask.any() else None}

    report = {"initially_violating": counts(violating), "initially_solved": counts(~violating)}
    if scene_types is not None:
        if len(scene_types) != proposal.shape[0]:
            raise ValueError("scene_types must contain one label per physical world")
        report["scene_types"] = {str(label): counts(torch.tensor(
            [value == label for value in scene_types], device=proposal.device)) for label in sorted(set(scene_types))}
    return report


def pbd_integrator(max_sweeps):
    """Keep native mass-weighted contact sweeps separate from tau integration."""
    if max_sweeps < 1:
        raise ValueError("PBD sweep budget must be positive")

    def run(proposal, radius, physics, solver, record=False):
        from src.contact_flow.pbd import integrate_pbd
        return integrate_pbd(proposal, radius, physics, max_sweeps=max_sweeps,
                             tolerance=solver.tolerance, record=record)
    return run


def _solve(state, proposal, radius, field_factory, physics, solver, record=False,
           projection_integrator=None):
    if projection_integrator is not None:
        return projection_integrator(proposal, radius, physics, solver, record=record)
    return integrate(field_factory(state, proposal, radius), proposal, radius, physics, solver, record=record)


@torch.no_grad()
def evaluate_projection(state, radius, field_factory, physics, dynamics, solver, record=False,
                        projection_integrator=None, scene_types=None):
    """Run one physical free step plus one fixed-[0,1] refinement solve."""
    proposal = free_position(state, **dynamics)
    _sync(state.device)
    start = time.perf_counter()
    result = _solve(state, proposal, radius, field_factory, physics, solver, record,
                    projection_integrator)
    _sync(state.device)
    elapsed = time.perf_counter() - start
    finite = torch.isfinite(result["final"]).flatten(1).all(1)
    report = {"samples": state.shape[0], "wall_seconds": elapsed,
              "converged_count": int((result["converged"] & finite).sum()),
              "integration_failed_count": int(result["failed"].sum()),
              "failure_reasons": _failure_histogram(result),
              "nonfinite_count": int((~finite).sum()),
              "tolerance_miss_count": int((~result["converged"]).sum()),
              "budget_exhausted_count": int(result.get("budget_exhausted", torch.zeros_like(result["failed"])).sum()),
              "start_noise_std": 0.0,
              "convergence_breakdown": _convergence_breakdown(proposal, radius, physics,
                  solver.tolerance, result, scene_types),
              "metric_samples": int(finite.sum())}
    report.update(_cost_metrics(result))
    if finite.any():
        report.update(_contact_metrics(proposal[finite], result["final"][finite], radius[finite], physics))
    return report, result


@torch.no_grad()
def rollout_metrics(state, radius, field_factory, physics, dynamics, solver, steps,
                    projection_integrator=None):
    """Stop each failed world; never reset state, apply a fallback, or hide truncation."""
    if steps < 0:
        raise ValueError("rollout steps must be nonnegative")
    batch = state.shape[0]
    active = torch.ones(batch, dtype=torch.bool, device=state.device)
    completed = torch.zeros(batch, dtype=torch.long, device=state.device)
    current = state.clone()
    sums = {"kinetic_energy_change": 0.0, "correction_norm": 0.0,
            "penetration_sum": 0.0, "initial_active_pair_gap_sum": 0.0}
    max_penetration = 0.0
    gap_count = penetration_count = accepted = failed = nfe = backtracks = unconverged = 0
    field_calls = executed_rows = energy_evals = 0
    native_sweeps = contact_evals = contact_updates = budget_exhausted_steps = 0
    first_unconverged = torch.full_like(completed, -1)
    failure_reasons = Counter()
    _sync(state.device)
    start = time.perf_counter()
    for _ in range(steps):
        indices = active.nonzero(as_tuple=True)[0]
        if not indices.numel():
            break
        old, radii = current[indices], radius[indices]
        proposal = free_position(old, **dynamics)
        result = _solve(old, proposal, radii, field_factory, physics, solver,
                        projection_integrator=projection_integrator)
        nfe += int(result["nfe"].sum())
        field_calls += int(result.get("field_calls", result["nfe"].max()))
        executed_rows += int(result.get("executed_field_rows", result["nfe"].sum()))
        energy_evals += int(result["energy_evals"].sum())
        backtracks += int(result["backtracks"].sum())
        native_sweeps += int(result.get("sweeps", torch.tensor(0)).sum())
        contact_evals += int(result.get("contact_evals", torch.tensor(0)).sum())
        contact_updates += int(result.get("contact_updates", torch.tensor(0)).sum())
        budget_exhausted_steps += int(result.get("budget_exhausted", torch.zeros_like(result["failed"])).sum())
        next_state = finite_difference_state(old, result["final"], dynamics["time_step"])
        valid = (~result["failed"] & result["completed"] &
                 torch.isfinite(next_state).flatten(1).all(1))
        unresolved = valid & ~result["converged"]
        unconverged += int(unresolved.sum())
        new_unresolved = indices[unresolved & (first_unconverged[indices] < 0)]
        first_unconverged[new_unresolved] = completed[new_unresolved]
        active[indices[~valid]] = False
        failure_reasons.update(_failure_histogram(result, ~valid))
        failed += int((~valid).sum())
        if not valid.any():
            continue
        old, radii, proposal, next_state = old[valid], radii[valid], proposal[valid], next_state[valid]
        current[indices[valid]] = next_state
        completed[indices[valid]] += 1
        before = geometry(proposal.double(), radii.double(), physics)
        after = geometry(next_state[..., :2].double(), radii.double(), physics)
        mass = 1.0 / before["inverse_mass"]
        free_v = (proposal.double() - old[..., :2].double()) / dynamics["time_step"]
        final_v = next_state[..., 2:].double()
        delta_ke = 0.5 * (mass * (final_v.square().sum(-1) - free_v.square().sum(-1))).sum(-1)
        sums["kinetic_energy_change"] += float(delta_ke.sum())
        sums["correction_norm"] += float((next_state[..., :2].double() - proposal.double()).flatten(1).norm(dim=1).sum())
        penetration = after["penetration"]
        sums["penetration_sum"] += float(penetration.sum())
        penetration_count += penetration.numel()
        max_penetration = max(max_penetration, float(penetration.max()))
        contacts = (before["edge_j"] >= 0)[None, :] & (before["residual"] < 0)
        gaps = after["gap"][contacts].clamp_min(0)
        sums["initial_active_pair_gap_sum"] += float(gaps.sum())
        gap_count += gaps.numel()
        accepted += int(valid.sum())
    _sync(state.device)
    return {"requested_steps": steps, "samples": batch,
            "start_noise_std": 0.0,
            "completed_steps_per_world": completed.cpu().tolist(),
            "completed_world_count": int((completed == steps).sum()),
            "failed_world_count": failed, "accepted_physical_steps": accepted,
            "failure_reasons": dict(failure_reasons),
            "unconverged_projection_steps": unconverged,
            "budget_exhausted_projection_steps": budget_exhausted_steps,
            "first_unconverged_step_per_world": first_unconverged.cpu().tolist(),
            "metrics_scope": "finite completed solves, including tolerance misses; failed worlds stop without fallback",
            "field_evaluations": nfe, "backtracks": backtracks,
            "vectorized_field_calls": field_calls, "executed_field_rows": executed_rows,
            "energy_evaluations": energy_evals,
            "native_pbd_sweeps": native_sweeps, "contact_evals": contact_evals,
            "contact_updates": contact_updates,
            "wall_seconds": time.perf_counter() - start,
            "projection_kinetic_energy_change_mean": finite_number(sums["kinetic_energy_change"] / accepted) if accepted else None,
            "correction_norm_mean": finite_number(sums["correction_norm"] / accepted) if accepted else None,
            "penetration_mean": finite_number(sums["penetration_sum"] / penetration_count) if penetration_count else None,
            "penetration_max": finite_number(max_penetration) if accepted else None,
            "initial_active_pair_gap_mean": finite_number(sums["initial_active_pair_gap_sum"] / gap_count) if gap_count else None,
            "initial_active_pair_count": gap_count}


def _render(result, radius, path, physics, image_size=900):
    from data.box2d_render import render_state_image
    history = result.get("history", {})
    final = result["final"][0]
    if not torch.isfinite(final).all():
        return False
    if history and history["z"].shape[0]:
        valid = history["dt"][:, 0] > 0
        points = history["z"][valid, 0]
    else:
        points = final.new_empty(0, *final.shape)
    trajectory = torch.cat([points, final[None]], dim=0).cpu()
    try:
        canvas = render_state_image(final.cpu(), radius[0].cpu(), physics.xy_limit,
                                    physics.y_ground, image_size, trajectory=trajectory)
    except (OverflowError, ValueError):
        # A diverged finite state may exceed Pillow's coordinate range.
        # Preserve the numerical failure report rather than crashing during rendering.
        return False
    if canvas is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return True


def load_checkpoint(path, selected_device):
    checkpoint = torch.load(path, map_location=selected_device, weights_only=True)
    if checkpoint.get("format") != "contact_flow_v2":
        raise ValueError("Expected contact_flow_v2 checkpoint with explicit flow settings. "
                         "contact_flow_v1 and old projection checkpoints are incompatible; retrain with regenerated paths.")
    from src.contact_flow.model import ContactFlowNet
    model = ContactFlowNet(**checkpoint["model_kwargs"], flow=FlowConfig(**checkpoint["flow"])).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


def physical_preflight(physics, solver, methods, selected_device, gains=(1, 4, 16, 64),
                       normalizations=("none", "diagonal"), flow=None, pbd_sweeps=(4, 16, 64),
                       mobility_bounds=None):
    """Known small configurations, separate from random simulation starts."""
    cases = {"single_ground_contact": [[0.0, 0.4]],
             "fixed_normal_three_stack": [[0.0, 0.42], [0.0, 1.36], [0.0, 2.3]],
             "fixed_normal_five_stack": [[0.0, 0.42 + 0.94 * i] for i in range(5)],
             "changing_contact_chain": [[-0.4, 2.0], [0.4, 2.0], [1.42, 2.0]]}
    if not gains or not normalizations or not pbd_sweeps:
        raise ValueError("Preflight gain, normalization and native PBD budgets must be nonempty")
    flow = flow or FlowConfig()
    mobility_bounds = [flow.mobility_bound] if mobility_bounds is None else list(mobility_bounds)
    if not mobility_bounds:
        raise ValueError("Preflight mobility_bounds must be nonempty")
    controls = [(gain, normalization, bound, replace(flow, gain=float(gain),
                normalization=normalization, mobility_bound=float(bound)))
                for gain in gains for normalization in normalizations for bound in mobility_bounds]
    if min(pbd_sweeps) < 1:
        raise ValueError("Native PBD sweep budgets must be positive")
    report = {}
    for name, points in cases.items():
        z = torch.tensor([points], dtype=torch.float64, device=selected_device)
        z[..., 1] += physics.y_ground
        radius = z.new_full(z.shape[:2], 0.5)
        state = torch.cat([z, torch.zeros_like(z)], -1)
        experiments = []
        for method in methods:
            # Normalization acts on the PSD C parameterization only. Classical
            # gradient/Jacobi formulas have no C and must not appear twice.
            method_controls = controls if method in {"isotropic", "inverse"} else controls[::len(normalizations) * len(mobility_bounds)]
            for gain, normalization, bound, settings in method_controls:
                factory = lambda old, proposal, r, method=method, settings=settings: make_baseline(method, r, physics, flow=settings)
                suffix = f"{normalization}/cap{bound:g}" if method in {"isotropic", "inverse"} else "classical"
                label = f"{method}/gain{gain:g}/{suffix}"
                experiments.append((label, factory, None, settings, None))
        for sweeps in pbd_sweeps:
            experiments.append((f"pbd/sweeps{sweeps}", None, pbd_integrator(sweeps), None, sweeps))
        for label, factory, native, settings, sweeps in experiments:
            metrics, result = evaluate_projection(state, radius, factory, physics,
                {"time_step": 1.0, "gravity_y": 0.0, "linear_damping": 0.0}, solver, record=True,
                projection_integrator=native)
            before = geometry(z, radius, physics)["residual"] < 0
            changes = 0
            for position in torch.cat([result["history"]["z"], result["final"][None]], 0):
                after = geometry(position, radius, physics)["residual"] < 0
                changes += int((before != after).sum())
                before = after
            metrics["active_contact_changes"] = changes
            metrics["nominal_steps"] = solver.steps if native is None else None
            metrics["max_sweeps"] = sweeps
            metrics["flow"] = asdict(settings) if settings else None
            metrics["comparison_scope"] = "native Gauss-Seidel PBD sweeps" if native else "fixed unit-time energy ODE"
            report[f"{name}/{label}"] = metrics
    return report


def summarize_path_cache(path, physics, flow, expected_spec=None, temperature=None, weighting=None):
    """Read-only oracle diagnosis; never charge just the winning candidate."""
    from src.contact_flow.paths import path_summary
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if expected_spec is not None and cache.get("spec") != expected_spec:
        raise ValueError("Reference cache spec differs from checkpoint (pool/seed/dynamics/path settings); "
                         "set --path-cache to the cache used to train this checkpoint")
    container = cache
    if "buffers" in cache:
        if cache.get("format") != "contact_path_cache_v2":
            raise ValueError("eval.path_cache requires a v2 cache; regenerate old reference paths")
        container = cache["buffers"]
    buffers = {name: container[name] for name in ("train", "val") if name in container}
    if not buffers and "path_scores" in cache:
        buffers = {"paths": cache}
    if not buffers:
        raise ValueError("eval.path_cache must contain a contact-path buffer or train/val buffers")
    output = {"path": str(path), "scope": "same stored noisy start for all candidate paths; "
              "oracle is diagnostic, not a free or independently timed solver",
              "checkpoint_spec_verified": expected_spec is not None,
              "temperature_source": "checkpoint_epoch" if temperature is not None else "cache_generation_initial",
              "weighting_source": "checkpoint_epoch" if weighting is not None else "cache_generation_initial",
              "splits": {}}
    for split, buffer in buffers.items():
        metadata = buffer["metadata"]
        if metadata.get("physics") != asdict(physics) or metadata.get("flow") != asdict(flow):
            raise ValueError("Path cache physics/flow differs from evaluation; use a matching v2 cache")
        paths = metadata["paths"]
        output["splits"][split] = path_summary(buffer,
            paths["temperature"] if temperature is None else temperature,
            paths.get("weighting", "score") if weighting is None else weighting)
    return output


def main():
    from src.contact_flow.io import load_config, device, load_states
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_projection.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--path-cache", help="Matching training cache when selecting a different checkpoint")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--rollout-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    ev = config.get("eval", {})
    runtime = config.get("runtime", {})
    selected_device = device(args.device or runtime.get("device", "cpu"))
    cpu_threads = int(runtime.get("cpu_threads", 4))
    count, rollout_count = int(ev.get("count", 8)), int(ev.get("rollout_count", 2))
    budgets = [int(k) for k in ev.get("solver_steps", [4, 16])]
    rollout_steps = args.rollout_steps if args.rollout_steps is not None else (0 if args.preflight else int(ev.get("rollout_steps", 300)))
    image_size = int(config.get("render", {}).get("size", 900))
    if min(count, rollout_count, cpu_threads, int(ev.get("num_objects", 5))) < 1:
        parser.error("count, rollout_count, num_objects and cpu_threads must be positive")
    if not budgets or min(budgets) < 1:
        parser.error("solver_steps must be a nonempty list of positive budgets")
    if rollout_steps < 0 or image_size < 16:
        parser.error("rollout_steps must be nonnegative and render.size at least 16")
    if selected_device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    model = None
    objective = None
    physics_dict = config.get("physics", {})
    dynamics = config.get("dynamics", {})
    flow_dict = config.get("flow", {})
    solver_dict = config.get("solver", {})
    checkpoint_path = args.checkpoint or ev.get("checkpoint", "")
    if not args.preflight:
        if not checkpoint_path:
            parser.error("Set eval.checkpoint or --checkpoint; use --preflight for network-free validation.")
        model, checkpoint = load_checkpoint(checkpoint_path, selected_device)
        objective = checkpoint.get("objective", checkpoint.get("config", {}).get("train", {}).get("objective", "fm"))
        if objective not in {"fm", "energy"}:
            parser.error(f"Unknown checkpoint objective: {objective}")
        for key, supplied in (("physics", physics_dict), ("dynamics", dynamics), ("flow", flow_dict)):
            saved = checkpoint[key]
            if any(k not in saved or saved[k] != value for k, value in supplied.items()):
                parser.error(f"{key} differs from checkpoint; inference-only gain/normalization overrides are not supported.")
        physics_dict, dynamics = checkpoint["physics"], checkpoint["dynamics"]
        flow_dict = checkpoint["flow"]
        solver_dict = {**checkpoint["solver"], **solver_dict}
    else:
        dynamics = {"time_step": 1 / 60, "gravity_y": -9.8, "linear_damping": 0.1, **dynamics}
    physics = PhysicsConfig(**physics_dict)
    flow = FlowConfig(**flow_dict)
    solver = SolverConfig(**solver_dict)
    seed = int(runtime.get("seed", 42))
    state, radius, scene_types = load_states("" if args.preflight else ev.get("dataset", ""), ev.get("split", "val"),
                                count, seed, int(ev.get("num_objects", 5)),
                                physics=physics, return_scene_types=True)
    state, radius = state.to(selected_device), radius.to(selected_device)
    methods = list(ev.get("methods", ["gradient", "isotropic", "diagonal", "inverse", "jacobi"]))
    if any(name not in {"gradient", "isotropic", "diagonal", "inverse", "jacobi"} for name in methods):
        parser.error("methods must contain gradient, isotropic, diagonal, inverse and/or jacobi")
    if not methods and model is None:
        parser.error("preflight requires at least one baseline method")
    factories = {name: (lambda old, proposal, r, name=name: make_baseline(name, r, physics, flow=flow)) for name in methods}
    native_budgets = [int(value) for value in ev.get("pbd_sweeps", [4, 16, 64])]
    if any(value < 1 for value in native_budgets):
        parser.error("pbd_sweeps must contain positive sweep budgets")
    if model is not None:
        def learned(old, proposal, r):
            condition = make_projection_condition(old, proposal)
            return lambda z, tau: model(z, tau, r, condition, physics)
        factories[objective] = learned
    output_path = Path(config.get("output", {}).get("json", "runs/contact_eval/eval.json"))
    render_dir = Path(config.get("render", {}).get("dir", str(output_path.parent / "renders")))
    if args.preflight:
        output_path = output_path.with_name(output_path.stem + "_preflight" + (output_path.suffix or ".json"))
        render_dir = render_dir / "preflight"
    if output_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = output_path.with_name(output_path.stem + "_" + stamp + output_path.suffix)
        render_dir = render_dir / stamp
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"format": "contact_flow_eval_v2", "preflight": args.preflight,
              "checkpoint": str(checkpoint_path) if model is not None else None,
              "checkpoint_objective": objective,
              "device": str(selected_device), "seed": seed, "cpu_threads": cpu_threads,
              "output_json": str(output_path), "render_dir": str(render_dir), "render_size": image_size,
              "physics": asdict(physics), "dynamics": dynamics, "flow": asdict(flow),
              "solver": asdict(solver), "solver_steps": budgets,
              "native_pbd_sweep_budgets": native_budgets, "start_noise_std": 0.0,
              "rollout_steps": rollout_steps,
              "data_source": "procedural" if args.preflight or not ev.get("dataset") else ev["dataset"],
              "scene_types": scene_types,
              "linear_diagnostic": [fixed_contact_diagnostic(k, selected_device) for k in budgets],
              "note": "No Box2D endpoint MSE: the learned target is a contact-energy path. "
                      "Flow methods use fixed unit solver time, adaptive energy-decreasing Euler; native PBD uses contact sweeps. "
                      "No fallback, restart or rollout start noise. Learned flow settings come from its checkpoint.",
              "results": {}}
    if args.preflight:
        preflight = config.get("preflight", {})
        report["physical_preflight"] = physical_preflight(physics, replace(solver, steps=max(budgets)),
            methods, selected_device, gains=preflight.get("gains", [1, 4, 16, 64]),
            normalizations=preflight.get("normalizations", ["none", "diagonal"]), flow=flow,
            pbd_sweeps=preflight.get("pbd_sweeps", [4, 16, 64]),
            mobility_bounds=preflight.get("mobility_bounds", [flow.mobility_bound]))
    path_cache = args.path_cache if args.path_cache is not None else ev.get("path_cache")
    if path_cache and not args.preflight:
        if any(name not in checkpoint for name in ("reference_cache_spec", "temperature", "weighting")):
            parser.error("Checkpoint lacks reference cache/temperature provenance; use a newly trained v2 "
                         "checkpoint or disable eval.path_cache for solver-only evaluation")
        report["reference_path_diagnostics"] = summarize_path_cache(path_cache, physics, flow,
            expected_spec=checkpoint["reference_cache_spec"], temperature=checkpoint["temperature"],
            weighting=checkpoint["weighting"])
    for name, factory in factories.items():
        for budget in budgets:
            settings = replace(solver, steps=budget)
            label = f"{name}_k{budget}"
            metrics, result = evaluate_projection(state, radius, factory, physics, dynamics, settings,
                                                   record=True, scene_types=scene_types)
            metrics["rendered"] = _render(result, radius, render_dir / f"{label}.png", physics, image_size)
            if rollout_steps:
                take = min(rollout_count, state.shape[0])
                metrics["rollout"] = rollout_metrics(state[:take], radius[:take], factory, physics, dynamics, settings, rollout_steps)
            report["results"][label] = metrics
            print(f"{label}: {metrics['converged_count']}/{metrics['samples']} within tolerance, nfe={metrics['field_evaluations']}")
    for sweep_budget in native_budgets:
        native = pbd_integrator(sweep_budget)
        label = f"pbd_sweeps{sweep_budget}"
        metrics, result = evaluate_projection(state, radius, None, physics, dynamics, solver,
                                               record=True, projection_integrator=native, scene_types=scene_types)
        metrics["max_sweeps"] = sweep_budget
        metrics["completion_policy"] = "Finite completed sweep budget may miss tolerance; numerical failure stops a world."
        metrics["rendered"] = _render(result, radius, render_dir / f"{label}.png", physics, image_size)
        if rollout_steps:
            take = min(rollout_count, state.shape[0])
            metrics["rollout"] = rollout_metrics(state[:take], radius[:take], None, physics,
                dynamics, solver, rollout_steps, projection_integrator=native)
        report["results"][label] = metrics
        print(f"{label}: {metrics['converged_count']}/{metrics['samples']} within tolerance, "
              f"sweeps={metrics.get('sweeps', 0)}, contact_evals={metrics.get('contact_evals', 0)}")
    payload = json.dumps(report, indent=2, allow_nan=False)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
