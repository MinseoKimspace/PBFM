"""PGS versus D+PGS on large circle piles: matched QPs and physical rollouts."""
from __future__ import annotations

import math
from pathlib import Path
import random
import statistics

import torch

from src.contact_flow.dynamics import free_position, finite_difference_state
from src.contact_flow.physics import PhysicsConfig
from .benchmark import distribution as scalar_summary, runtime_environment
from .evaluation import timed, write_json
from .problem import converged, decode, residuals
from .rollout import render_motion
from .solvers import pgs, run_cfm, run_hybrid
from .stress_problem import SCENES, circle_gaps, compact_problem, large_scene


def validate_stress(config):
    options = config["stress"]
    for key, minimum in (("sizes", 4), ("seeds", 0), ("hybrid_calls", 1)):
        values = options[key]
        if (not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values)
                or any(type(x) is not int or x < minimum for x in values)):
            raise ValueError(f"stress.{key} must contain distinct integers >= {minimum}")
    if (not options["scenes"] or len(set(options["scenes"])) != len(options["scenes"])
            or any(x not in SCENES for x in options["scenes"])):
        raise ValueError(f"stress.scenes must be a nonempty subset of {SCENES}")
    if options["mode"] not in ("snapshot", "rollout"):
        raise ValueError("stress.mode must be snapshot or rollout")
    if options["pgs_backend"] not in ("sparse_cpu", "torch"):
        raise ValueError("stress.pgs_backend must be sparse_cpu or torch")
    for key in ("steps", "max_sweeps", "timing_repeats", "image_size", "frame_stride"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"stress.{key} must be a positive integer")
    for key in ("tolerance", "radius", "max_speed", "max_position"):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"stress.{key} must be finite and positive")
    if not math.isfinite(options["spacing_gap"]) or options["spacing_gap"] < 0:
        raise ValueError("stress.spacing_gap must be nonnegative")
    if type(options["render"]) is not bool:
        raise ValueError("stress.render must be boolean")


def memory_start(device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def memory_end(device, initial):
    if initial is None:
        return dict(cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None,
                    cuda_additional_peak_bytes=None)
    peak = torch.cuda.max_memory_allocated(device)
    return dict(cuda_peak_allocated_bytes=peak,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                cuda_additional_peak_bytes=max(0, peak-initial))


def prepare_problem(proposal, radius, physics, eta_fraction, options):
    problem, geo, info = compact_problem(proposal, radius, physics, eta_fraction)
    backend = None
    if options["pgs_backend"] == "sparse_cpu":
        from .stress_pgs import SparsePGS
        backend = SparsePGS(problem)
    return problem, geo, info, backend


def solve(problem, model, calls, options, backend=None):
    start = torch.zeros_like(problem["c"])
    if backend is not None:
        if calls == 0:
            return backend.solve(start, options["tolerance"], options["max_sweeps"])
        prefix = run_cfm(model, problem, start, calls, collect_diagnostics=False)
        if bool(prefix["completed"].all()):
            finish = backend.solve(prefix["final"], options["tolerance"], options["max_sweeps"])
        else:
            finish = dict(final=prefix["final"], sweeps=torch.zeros(1, dtype=torch.long, device=start.device),
                          contact_evals=torch.zeros(1, dtype=torch.long, device=start.device),
                          converged=torch.zeros(1, dtype=torch.bool, device=start.device))
        return dict(prefix, final=finish["final"], sweeps=finish["sweeps"], contact_evals=finish["contact_evals"],
                    converged=finish["converged"], cfm_final=prefix["final"], cfm_completed=prefix["completed"],
                    cfm_converged=finish["converged"] & (finish["sweeps"] == 0))
    if calls == 0:
        return pgs(problem, start, options["tolerance"], options["max_sweeps"])
    return run_hybrid(model, problem, start, calls, options["tolerance"], options["max_sweeps"])


def solver_metrics(problem, result, tolerance):
    completed = bool(result.get("completed", torch.ones(1, dtype=torch.bool)).all())
    finite = bool(torch.isfinite(result["final"]).all())
    stats = residuals(problem, result["final"])
    successful = finite and completed and bool(converged(problem, result["final"], tolerance).all())
    def scalar(key):
        value = float(stats[key][0])
        return value if math.isfinite(value) else None
    row = dict(solver_completed=completed, solver_converged=successful,
               pgs_budget_exhausted=completed and not successful,
               projected_gradient=scalar("projected_gradient"),
               linear_penetration=scalar("penetration"), negative_multiplier=scalar("negative_multiplier"),
               nfe=int(result.get("nfe", torch.zeros(1)).sum()),
               sweeps=int(result["sweeps"].sum()), contact_evals=int(result["contact_evals"].sum()))
    if "cfm_final" in result:
        row.update(cfm_completed=bool(result["cfm_completed"].all()),
                   cfm_converged=bool(result["cfm_converged"].all()),
                   cfm_projected_gradient=float(residuals(problem, result["cfm_final"])["projected_gradient"][0]))
    return row


def geometric_metrics(position, radius, physics, near):
    actual = circle_gaps(position, radius, physics)
    violation = (-actual["gap"]-physics.slop).clamp_min(0)
    return dict(geometric_penetration=float(violation.max()),
                new_violations_outside_frozen_contacts=int(((violation > 0) & ~near).sum()))


def warmup_pipeline(model, initial, radius, config, calls):
    """Prime geometry, imports/JIT, solver and FD without advancing the scene."""
    physics = PhysicsConfig(**config["physics"])
    proposal = free_position(initial[None], **config["dynamics"])[0]
    problem, _, _, backend = prepare_problem(proposal, radius, physics,
        config["reference"]["eta_fraction"], config["stress"])
    result = solve(problem, model, calls, config["stress"], backend)
    if torch.isfinite(result["final"]).all():
        position = decode(problem, result["final"]).reshape(1, len(radius), 2)
        finite_difference_state(initial[None], position, config["dynamics"]["time_step"])


@torch.no_grad()
def benchmark_snapshot(model, initial, radius, config, device, seed):
    """Every method starts from zero on the exact same frozen QP, batch size 1."""
    settings = config["stress"]
    physics = PhysicsConfig(**config["physics"])
    initial, radius = initial.to(device), radius.to(device)
    proposal = free_position(initial[None], **config["dynamics"])[0]
    # Do not charge the first baseline for lazy imports and cold geometry kernels.
    primed = prepare_problem(proposal, radius, physics, config["reference"]["eta_fraction"], settings)
    del primed
    memory = memory_start(device)
    (problem, before, info, backend), setup_seconds = timed(device, lambda: prepare_problem(
        proposal, radius, physics, config["reference"]["eta_fraction"], settings))
    report = dict(**info, setup_seconds=setup_seconds, setup_memory=memory_end(device, memory), methods={})
    methods = {"pgs": 0, **{f"hybrid_k{k}": k for k in settings["hybrid_calls"]}}
    timings = {name: [] for name in methods}
    peaks = {name: [] for name in methods}
    predictions = {}
    # One untimed warmup per method; rotate the timed order across repetitions.
    rng = random.Random(seed)
    for repeat in range(-1, settings["timing_repeats"]):
        order = list(methods)
        rng.shuffle(order)
        for name in order:
            if report["methods"].get(name, {}).get("failure"):
                continue
            memory = memory_start(device)
            try:
                result, seconds = timed(device, lambda: solve(problem, model, methods[name], settings, backend))
                peak = memory_end(device, memory)
                if repeat >= 0:
                    timings[name].append(seconds)
                    peaks[name].append(peak)
                    metrics = solver_metrics(problem, result, settings["tolerance"])
                    position = decode(problem, result["final"]).reshape(-1, 2)
                    if torch.isfinite(position).all():
                        try:
                            metrics.update(geometric_metrics(position, radius, physics, before["near"]))
                        except ValueError:
                            metrics["geometric_penetration"] = None
                    else:
                        metrics["geometric_penetration"] = None
                    report["methods"][name] = metrics
                    predictions[name] = position.cpu()
                del result
            except torch.cuda.OutOfMemoryError:
                report["methods"][name] = dict(failure="cuda_out_of_memory", solver_converged=False)
                torch.cuda.empty_cache()
    for name, row in report["methods"].items():
        if row.get("failure"):
            continue
        seconds = statistics.median(timings[name])
        row.update(solver_seconds=seconds, timing_seconds=timings[name],
                   setup_plus_solver_seconds=setup_seconds+seconds, warmups=1)
        row["memory"] = {key: max(p[key] for p in peaks[name]) if peaks[name][0][key] is not None else None
                         for key in peaks[name][0]}
    baseline = report["methods"]["pgs"]
    for name, row in report["methods"].items():
        comparable = row["solver_converged"] and baseline["solver_converged"]
        row["speedup_vs_pgs"] = baseline["solver_seconds"]/row["solver_seconds"] if comparable else None
        row["setup_inclusive_speedup_vs_pgs"] = (
            baseline["setup_plus_solver_seconds"]/row["setup_plus_solver_seconds"] if comparable else None)
        if name in predictions and "pgs" in predictions:
            row["position_rmse_vs_pgs"] = float((predictions[name]-predictions["pgs"]).square().mean().sqrt())
    return report, predictions


def summarize_rollout(frames, failure, requested_steps):
    accepted = [row for row in frames if row["advanced"]]
    return dict(requested_steps=requested_steps, attempted_steps=len(frames), completed_steps=len(accepted),
                completed=failure is None and len(accepted) == requested_steps,
                converged_attempts=sum(row["solver_converged"] for row in frames),
                total_attempt_seconds=sum(row["simulation_seconds"] for row in frames),
                total_pgs_sweeps=sum(row["sweeps"] for row in frames),
                total_nfe=sum(row["nfe"] for row in frames),
                solver_seconds=scalar_summary([row["solver_seconds"] for row in frames]),
                simulation_seconds=scalar_summary([row["simulation_seconds"] for row in frames]),
                contacts=scalar_summary([row["contacts"] for row in frames]),
                pgs_sweeps=scalar_summary([row["sweeps"] for row in frames]),
                geometric_penetration=scalar_summary([row["geometric_penetration"] for row in accepted]))


@torch.no_grad()
def simulate_stress(model, initial, radius, config, device, calls, progress=None):
    """Own physical trajectory; failed FM/PGS frames never advance or fall back."""
    settings = config["stress"]
    physics = PhysicsConfig(**config["physics"])
    state, radius = initial.to(device), radius.to(device)
    states, frames, failure = [initial.cpu().clone()], [], None
    try:
        warmup_pipeline(model, state, radius, config, calls)
    except torch.cuda.OutOfMemoryError:
        failure = dict(frame=1, reason="cuda_out_of_memory_in_warmup", unmeasured_failed_attempt=True)
        torch.cuda.empty_cache()
        return dict(states=torch.stack(states), frames=frames, failure=failure,
                    requested_steps=settings["steps"], completed_steps=0,
                    summary=summarize_rollout(frames, failure, settings["steps"]))
    for frame in range(1, settings["steps"]+1):
        memory = memory_start(device)
        try:
            proposal, dynamics_seconds = timed(device, lambda: free_position(state[None], **config["dynamics"])[0])
            (problem, before, info, backend), setup_seconds = timed(device, lambda: prepare_problem(
                proposal, radius, physics, config["reference"]["eta_fraction"], settings))
            result, solver_seconds = timed(device, lambda: solve(problem, model, calls, settings, backend))
            metrics = solver_metrics(problem, result, settings["tolerance"])
            row = dict(frame=frame, physical_time=frame*config["dynamics"]["time_step"], **info, **metrics,
                       dynamics_seconds=dynamics_seconds, setup_seconds=setup_seconds, solver_seconds=solver_seconds,
                       simulation_seconds=dynamics_seconds+setup_seconds+solver_seconds,
                       advanced=False, phase="stress")
            if not metrics["solver_converged"]:
                failure = dict(frame=frame, reason="incomplete_fm" if not metrics["solver_completed"] else "pgs_budget_exhausted")
                row["memory"] = memory_end(device, memory)
                frames.append(row)
                break
            def update():
                position = decode(problem, result["final"]).reshape(1, len(radius), 2)
                return finite_difference_state(state[None], position, config["dynamics"]["time_step"])[0]
            next_state, update_seconds = timed(device, update)
            row["simulation_seconds"] += update_seconds
            row["dynamics_seconds"] += update_seconds
            row["memory"] = memory_end(device, memory)
            if not torch.isfinite(next_state).all():
                failure = dict(frame=frame, reason="nonfinite_endpoint")
                frames.append(row)
                break
            try:
                row.update(geometric_metrics(next_state[:, :2], radius, physics, before["near"]))
            except ValueError:
                failure = dict(frame=frame, reason="coincident_endpoint")
                frames.append(row)
                break
            row["max_speed"] = float(next_state[:, 2:].norm(dim=-1).max())
            row["advanced"] = True
            frames.append(row)
            state = next_state
            states.append(state.cpu().clone())
            # Release the previous dense QP before constructing the next frame.
            del problem, before, result, backend
            if row["max_speed"] > settings["max_speed"] or float(state[:, :2].abs().max()) > settings["max_position"]:
                failure = dict(frame=frame, reason="diagnostic_bound_exceeded_no_clipping")
                break
            if frame % 10 == 0:
                print(f"  {'PGS' if calls == 0 else f'Hybrid K={calls}'} frame {frame}/{settings['steps']} "
                      f"contacts={row['contacts']} sweeps={row['sweeps']}", flush=True)
                if progress:
                    progress(frames)
        except torch.cuda.OutOfMemoryError:
            failure = dict(frame=frame, reason="cuda_out_of_memory", unmeasured_failed_attempt=True)
            torch.cuda.empty_cache()
            break
    return dict(states=torch.stack(states), frames=frames, failure=failure,
                requested_steps=settings["steps"], completed_steps=len(states)-1,
                summary=summarize_rollout(frames, failure, settings["steps"]))


def write_stress_table(report, path):
    snapshot = report["mode"] == "snapshot"
    lines = ["# Large-circle PGS / D+PGS experiment", "", report["comparison_scope"],
             "Speedup uses solver time in snapshot mode, complete simulation time in rollout mode.", "",
             "| Scene | N | Seed | Method | Success | PGS sweeps | Solver ms | Setup + solver/total ms | Speedup |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for case in report["cases"]:
        for name, row in case.get("methods", {}).items():
            if snapshot:
                success = row["solver_converged"]
                sweeps, seconds = row.get("sweeps"), row.get("solver_seconds")
                total = row.get("setup_plus_solver_seconds")
                speedup = row.get("speedup_vs_pgs")
            else:
                summary = row["summary"]
                success = f"{summary['completed_steps']}/{summary['requested_steps']} frames"
                sweeps, seconds = summary["pgs_sweeps"]["mean"], summary["solver_seconds"]["mean"]
                total = summary["simulation_seconds"]["mean"]
                speedup = row.get("complete_rollout_speedup_vs_pgs")
            time = f"{seconds*1000:.3f}" if seconds is not None else "n/a"
            speed = f"{speedup:.3f}" if speedup is not None else "n/a"
            total_time = f"{total*1000:.3f}" if total is not None else "n/a"
            sweep_text = f"{sweeps:.2f}" if sweeps is not None else "n/a"
            lines.append(f"| {case['scene']} | {case['circles']} | {case['seed']} | {name} | {success} | {sweep_text} | {time} | {total_time} | {speed} |")
        if case.get("failure"):
            lines.append(f"\nCase failure: {case['name']}: {case['failure']}\n")
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")


@torch.no_grad()
def evaluate_stress(model, config, device, output, metadata=None):
    validate_stress(config)
    if model.head_type != "analytic" or model.communication != "global":
        raise ValueError("Stress comparison requires D's global analytic model")
    model.eval()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    settings = config["stress"]
    physics = PhysicsConfig(**config["physics"])
    snapshot = settings["mode"] == "snapshot"
    report = dict(format="multiplier_stress_v1", mode=settings["mode"], config=config, metadata=metadata,
        environment=runtime_environment(device, model), cases=[],
        comparison_scope=("Same frozen QP, same zero start, same tolerance and PGS cap; batch size 1."
                          if snapshot else "Same initial scene; each method evolves its own QPs. No timing ratio for incomplete trajectories."),
        pgs_backend=settings["pgs_backend"],
        solver_definition="Sequential PGS versus original D FM map followed by the identical PGS backend; no fallback. sparse_cpu uses compiled CSR updates in original contact order; torch uses the original dense device implementation.",
        geometry_definition="All pair distances; near-contact Jacobian only; original gap/slop/normals/order. Dense contact D and global attention retained.",
        eta_definition="eta_fraction / maximum absolute row sum of D (safe spectral upper bound), shared by all methods. Changes residual step, not the QP or D analytic field.",
        timing_scope="Synchronized GPU wall time. CSR construction/QP transfer is setup; CPU/GPU multiplier transfers are inside solve. Snapshot: setup priming, one solver warmup, shuffled method order, median repeats. Rollout: full first-frame pipeline warmup, single pass. Excludes metrics/rendering/IO and warmup imports/compilation.",
        accuracy_scope="FP32 KKT at configured tolerance plus true circle penetration; position RMSE versus PGS is agreement, not an independent exact-reference error.",
        memory_scope="CUDA allocator peaks, includes resident model/QP; additional peak relative to operation start. CPU peak memory not measured.",
        limitations="These PGS backends are not a complete native physics engine. No friction, restitution, CCD, relinearization or temporal warm start.")

    def save():
        write_json(output/"stress.json", report)
        write_stress_table(report, output/"stress.md")

    for count in settings["sizes"]:
        for kind in settings["scenes"]:
            for seed in settings["seeds"]:
                name = f"{kind}_n{count}_seed{seed}"
                print(f"Stress {name} ({settings['mode']})", flush=True)
                case = dict(name=name, scene=kind, circles=count, seed=seed, methods={})
                report["cases"].append(case)
                initial, radius = large_scene(kind, count, seed, physics,
                    radius=settings["radius"], spacing_gap=settings["spacing_gap"])
                torch.save(dict(initial=initial, radius=radius), output/f"{name}_initial.pt")
                try:
                    if snapshot:
                        summary, predictions = benchmark_snapshot(model, initial, radius, config, device, seed)
                        case.update(summary)
                        torch.save(predictions, output/f"{name}_positions.pt")
                    else:
                        runs = {}
                        for method, calls in [("pgs", 0)]+[(f"hybrid_k{k}", k) for k in settings["hybrid_calls"]]:
                            def progress(frames, method=method):
                                write_json(output/f"{name}_{method}_progress.json", dict(frames=frames, complete=False))
                            run = simulate_stress(model, initial, radius, config, device, calls, progress)
                            runs[method] = run
                            torch.save(dict(states=run["states"], radius=radius), output/f"{name}_{method}_trajectory.pt")
                            case["methods"][method] = {key: value for key, value in run.items() if key != "states"}
                            base = runs["pgs"]["summary"]
                            both_complete = base["completed"] and run["summary"]["completed"]
                            case["methods"][method]["complete_rollout_speedup_vs_pgs"] = (
                                base["total_attempt_seconds"]/run["summary"]["total_attempt_seconds"] if both_complete else None)
                            save()
                        if settings["render"]:
                            rendering = dict(settings, time_step=config["dynamics"]["time_step"])
                            for k in settings["hybrid_calls"]:
                                method = f"hybrid_k{k}"
                                render_motion({key: runs[key] for key in ("pgs", method)}, radius, physics, rendering,
                                    output/f"{name}_k{k}.gif", labels={"pgs": "PGS", method: f"D K={k} + PGS"})
                except torch.cuda.OutOfMemoryError:
                    case["failure"] = "cuda_out_of_memory_during_case_setup"
                    torch.cuda.empty_cache()
                save()
    return report
