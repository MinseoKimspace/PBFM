"""Numerical preflight and common finite-time/projection/cost measurements."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch

from src.contact_flow.physics import PhysicsConfig
from .data import release_problem
from .problem import (circle_problem, converged, decode, field, gap,
                      move, pack, position_error, relinearize, residuals, select)
from .solvers import active_set_solution, pgs, reference, run_map


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def timed(device, operation):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, perf_counter() - begin


def summarize(problem, lam, optimum, tolerance, target=None):
    stats = residuals(problem, lam)
    report = {name: float(value.mean()) for name, value in stats.items()}
    report.update(success_count=int(converged(problem, lam, tolerance).sum()),
                  samples=len(lam), max_penetration=float(stats["penetration"].max()),
                  max_projected_gradient=float(stats["projected_gradient"].max()),
                  projection_position_mse=float(position_error(problem, lam, optimum).mean()))
    if target is not None:
        report["finite_time_map_position_mse"] = float(position_error(problem, lam, target).mean())
    return report


def scene_indices(names, limit=0):
    """Zero means ALL. A limit uses round-robin groups, never a prefix slice."""
    if limit < 0:
        raise ValueError("max_scenes must be nonnegative (0=all)")
    if not limit or limit >= len(names):
        return torch.arange(len(names))
    groups = {}
    for i, name in enumerate(names):
        groups.setdefault(name.rsplit("_", 1)[0], []).append(i)
    order = [group[k] for k in range(max(map(len, groups.values())))
             for group in groups.values() if k < len(group)]
    return torch.tensor(order[:limit])


def group_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["name"].rsplit("_", 1)[0], []).append(row)
    return {name: dict(samples=len(items), success_count=sum(x["success_count"] for x in items),
                       projected_gradient=sum(x["projected_gradient"] for x in items)/len(items),
                       max_penetration=max(x["max_penetration"] for x in items),
                       projection_position_mse=sum(x["projection_position_mse"] for x in items)/len(items))
            for name, items in groups.items()}


def evaluation_start(problem, optimum, mode):
    if mode == "zero":
        return torch.zeros_like(optimum)
    if mode == "over":
        start = 1.7 * optimum + .01 / problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
    elif mode == "mixed":
        factor = torch.where(torch.arange(optimum.shape[1], device=optimum.device) % 2 == 0, .4, 1.8)
        start = optimum * factor
    else:
        raise ValueError("start mode must be zero, over or mixed")
    return start * problem["mask"]


@torch.no_grad()
def solver_validation(model, split, config, device):
    """Cheap ALL-world raw-map diagnostic; no ODE regeneration/guard/renderer."""
    settings = config["evaluation"]
    report, all_rows = {}, []
    for mode in settings["start_modes"]:
        rows = []
        for index in torch.arange(len(split["names"])).split(config["train"]["batch_size"]):
            problem = move(select(split["problem"], index), device, torch.float32)
            optimum = split["optimum"][index].to(device=device, dtype=torch.float32)
            result = run_map(model, problem, evaluation_start(problem, optimum, mode),
                             settings["total_time"], config["train"].get("validation_calls", 8))
            for j, context in enumerate(index):
                rows.append(dict(name=split["names"][int(context)], **summarize(
                    select(problem, slice(j, j+1)), result["final"][j:j+1],
                    optimum[j:j+1], settings["tolerance"])))
        report[mode] = group_summary(rows)
        all_rows.extend(x for x in rows if not x["name"].startswith("free_flight"))
    report["nontrivial"] = dict(samples=len(all_rows),
        success_rate=sum(x["success_count"] for x in all_rows)/max(1, len(all_rows)),
        mean_projected_gradient=sum(x["projected_gradient"] for x in all_rows)/max(1, len(all_rows)))
    return report


def render_case(problem, start, final, radius, config, path, title):
    if not len(radius):
        return False  # Abstract halfspaces are not circles; do not misrender them.
    from data.box2d_render import render_projection_comparison
    count = 2 * len(radius)
    initial = decode(problem, start)[0, :count].reshape(-1, 2).detach().cpu()
    endpoint = decode(problem, final)[0, :count].reshape(-1, 2).detach().cpu()
    return render_projection_comparison(
        initial, endpoint, radius.cpu(), Path(path), config["physics"]["xy_limit"],
        config["physics"]["y_ground"], config["evaluation"]["image_size"],
        title=title, final_label="finite-time endpoint (no trajectory shown)")


def relinearization_check():
    """Changing/adding/removing normals must not implicitly move primal state."""
    anchor = torch.zeros(2, dtype=torch.float64)
    position = torch.tensor([0.2, 0.1], dtype=torch.float64)
    result = relinearize(anchor, position, torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64),
                         torch.ones_like(anchor), torch.tensor([0.05, 0.05], dtype=torch.float64),
                         ["x", "diagonal"], torch.tensor([0.1, 0.1], dtype=torch.float64),
                         ["diagonal", "new"])
    problem = result["problem"]
    expected_gaps = problem["c"] + problem["J"] @ (position - anchor)
    passed = (torch.equal(position, result["position"]) and torch.equal(anchor, problem["p"])
              and torch.allclose(expected_gaps, torch.full_like(anchor, 0.05))
              and result["multiplier_guess"].tolist() == [0.1, 0.0])
    return dict(passed=bool(passed), primal_jump=0.0,
                unresolved_stationarity_norm=float(result["stationarity_mismatch"].norm()),
                scope="Bookkeeping only; mismatch must be resolved by a future nonlinear primal-dual solver")


def preflight(config, output):
    """No neural checkpoint needed. Analytic answers + independent active sets."""
    physics, ref = PhysicsConfig(**config["physics"]), config["reference"]
    definitions = []
    radius = torch.tensor([0.5], dtype=torch.float64)
    for mode in ("under", "solved", "over"):
        p = torch.tensor([[0., physics.y_ground + 0.4]], dtype=torch.float64)
        definitions.append((f"floor_{mode}", circle_problem(p, radius, physics, ref["eta_fraction"]), radius, mode))
    definitions.append(("contact_release", release_problem(eta_fraction=ref["eta_fraction"]), torch.empty(0), "sequential"))
    definitions.append(("mixed_release", release_problem(eta_fraction=ref["eta_fraction"]), torch.empty(0), "mixed"))
    for size in config["preflight"]["stack_sizes"]:
        p = torch.stack((torch.zeros(size), 0.42 + 0.94 * torch.arange(size)), -1).double()
        p[:, 1] += physics.y_ground
        r = torch.full((size,), 0.5, dtype=torch.float64)
        definitions.append((f"stack_{size}", circle_problem(p, r, physics, ref["eta_fraction"]), r, "under"))
    problem = pack([item[1] for item in definitions])
    qp, qp_seconds = timed(torch.device("cpu"), lambda: pgs(
        problem, torch.zeros_like(problem["c"]), ref["qp_tolerance"], ref["max_sweeps"]))
    start = torch.zeros_like(qp["final"])
    oracle_errors = []
    for i, (_, single, _, mode) in enumerate(definitions):
        oracle = active_set_solution(pack([single]))
        oracle_errors.append(float(position_error(pack([single]), qp["final"][i:i+1, :oracle.shape[1]], oracle)))
        if mode == "solved":
            start[i] = qp["final"][i]
        elif mode == "over":
            start[i] = 2 * qp["final"][i]
        elif mode == "sequential":
            start[i, :2] = start.new_tensor([0.1, 0.1])
        elif mode == "mixed":
            start[i, :2] = start.new_tensor([0.2, 0.03])
    slope = (gap(problem, start) * field(problem, start)).sum(-1)
    rebound = relinearization_check()
    report = dict(scope="Frozen-normal dual QP; no learned model or PBD finish",
                  relinearization=rebound, independent_oracle_position_mse=oracle_errors,
                  initial_Q_directional_derivative=slope.tolist(), cases={},
                  pgs=dict(seconds=qp_seconds, sweeps=qp["sweeps"].tolist(),
                           contact_evals=qp["contact_evals"].tolist(), tolerance=ref["qp_tolerance"]))
    last_success = False
    for duration in config["preflight"]["times"]:
        result, seconds = timed(torch.device("cpu"), lambda: reference(
            problem, start, duration, ref["rtol"], ref["atol"], ref["max_steps"]))
        for i, (name, _, r, _) in enumerate(definitions):
            item = summarize(select(problem, slice(i, i+1)), result["final"][i:i+1],
                             qp["final"][i:i+1], config["evaluation"]["tolerance"])
            item.update(nfe=int(result["nfe"][i]), rejected=int(result["rejected"][i]),
                        reference_time=float(duration))
            report["cases"].setdefault(name, {})[str(duration)] = item
            if config["evaluation"]["render"] and duration == max(config["preflight"]["times"]):
                item["rendered"] = render_case(select(problem, slice(i, i+1)), start[i:i+1],
                    result["final"][i:i+1], r, config, Path(output).parent / "renders" / f"{name}.png", name)
        report.setdefault("batched_reference_seconds", {})[str(duration)] = seconds
        if duration == max(config["preflight"]["times"]):
            last_success = bool(converged(problem, result["final"], config["evaluation"]["tolerance"]).all())
    report["passed"] = bool(qp["converged"].all() and max(oracle_errors) < 1e-12
                            and (slope <= 1e-10).all() and rebound["passed"] and last_success)
    report["cost_note"] = "Reference timings are batched CPU float64; not a neural speed claim"
    write_json(output, report)
    return report


@torch.no_grad()
def evaluate(model, split, config, device, output, metadata=None):
    ref, settings = config["reference"], config["evaluation"]
    contexts = scene_indices(split["names"], int(settings["max_scenes"]))
    names = [split["names"][int(i)] for i in contexts]
    count = len(contexts)
    if count < 1 or settings.get("timing_repeats", 3) < 1:
        raise ValueError("Positive evaluation scene count and timing repeats required")
    problem64 = select(split["problem"], contexts)
    problem = move(problem64, device, torch.float32)
    optimum = split["optimum"][contexts].to(device=device, dtype=torch.float32)
    modes = settings["start_modes"]
    if any(mode not in {"zero", "over", "mixed"} for mode in modes):
        raise ValueError("Evaluation start_modes must be zero, over or mixed")
    report = dict(scope="Frozen contacts, fixed total time; NO hidden finishing solver",
                  device=str(device), total_time=settings["total_time"], scenes=names,
                  guard_version="reuse_accepted_h_v2",
                  modes={}, timings_include="Solver calls/guards, exclude cached geometry setup and data generation",
                  cost_units="NFE counts logical per-world evaluations; seconds measure the complete batch",
                  config=config, metadata=metadata or {})
    for mode in modes:
        start64 = evaluation_start(problem64, split["optimum"][contexts], mode)
        label = reference(problem64, start64, settings["total_time"], ref["rtol"], ref["atol"], ref["max_steps"])["final"]
        start, target = start64.to(device=device, dtype=torch.float32), label.to(device=device, dtype=torch.float32)
        label_quality = summarize(problem64, label, split["optimum"][contexts], settings["tolerance"])
        rows = {}
        # Warm up the same-device kernels before measuring any method.
        model(start, start.new_full((count,), 1.0), problem)
        operations = {
            "reference": lambda: reference(problem, start, settings["total_time"],
                                             max(ref["rtol"], 1e-4), max(ref["atol"], 1e-7), ref["max_steps"]),
            "pgs": lambda: pgs(problem, start, settings["tolerance"], ref["max_sweeps"]),
        }
        for calls in settings["calls"]:
            for guarded in (False, True):
                name = f"map_k{calls}_{'guarded' if guarded else 'raw'}"
                operations[name] = lambda k=calls, safe=guarded: run_map(
                    model, problem, start, settings["total_time"], k, safe, settings["max_backtracks"])
        for name, operation in operations.items():
            operation()  # Warm up EACH solver, not just the network.
            measurements = [timed(device, operation) for _ in range(settings.get("timing_repeats", 3))]
            result = measurements[-1][0]
            seconds = sum(item[1] for item in measurements) / len(measurements)
            row = summarize(problem, result["final"], optimum, settings["tolerance"], target)
            row["seconds"] = seconds
            row["timing_repeats"] = len(measurements)
            for key in ("nfe", "rejected", "sweeps", "contact_evals", "backtracks", "interventions"):
                if key in result:
                    row[key] = int(result[key].sum())
            row["completed_count"] = int(result["completed"].sum()) if "completed" in result else count
            row["per_scene"] = [dict(name=names[i], **summarize(
                select(problem, slice(i, i+1)), result["final"][i:i+1], optimum[i:i+1],
                settings["tolerance"], target[i:i+1])) for i in range(count)]
            for i, scene in enumerate(row["per_scene"]):
                for key in ("completed", "time", "nfe", "backtracks", "interventions",
                            "accepted_steps", "min_accepted_h", "failure_code"):
                    if key in result:
                        scene[key] = result[key][i].item()
            row["groups"] = group_summary(row["per_scene"])
            rows[name] = row
            if settings["render"] and mode == "zero":
                render_case(select(problem, slice(0, 1)), start[:1], result["final"][:1],
                            split["radii"][int(contexts[0])], config, Path(output).parent / "renders" / f"{name}.png", name)
        report["modes"][mode] = rows
        report.setdefault("float64_label_quality", {})[mode] = label_quality
    write_json(output, report)
    return report
