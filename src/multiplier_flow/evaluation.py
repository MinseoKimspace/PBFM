"""QP-label preflight and CFM projection, convergence and cost diagnostics."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from src.contact_flow.physics import PhysicsConfig
from .benchmark import (accuracy_summary, budget_comparison, reference_errors,
                        runtime_environment, timing_summary, validate_budgets,
                        work_summary, write_budget_table)
from .data import release_problem
from .model import CHECKPOINT_FORMAT, SOLVER_DESCRIPTION, ConditionalField, LocalProjection
from .problem import (circle_problem, contact_endpoint, converged, decode, field, gap,
                      move, pack, position_error, relinearize, residuals, select)
from .solvers import active_set_solution, pgs, run_cfm


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


def summarize(problem, lam, optimum, tolerance):
    stats = residuals(problem, lam)
    report = {name: float(value.mean()) for name, value in stats.items()}
    report.update(success_count=int(converged(problem, lam, tolerance).sum()),
                  samples=len(lam), max_penetration=float(stats["penetration"].max()),
                  max_projected_gradient=float(stats["projected_gradient"].max()),
                  projection_position_mse=float(position_error(problem, lam, optimum).mean()))
    return report


def projection_summary(result, index=None):
    """Reduce accepted-step projection diagnostics, preserving their units.

    Projection is part of direct-head inference. These counters measure its
    magnitude; they are not additional convergence criteria or solver work.
    """
    report = {}
    for key in ("clipped_entries", "projection_steps", "projection_l1",
                "projection_max", "min_unprojected_multiplier"):
        if key not in result:
            continue
        values = result[key] if index is None else result[key][index:index+1]
        if key == "projection_max":
            value = values.max()
        elif key == "min_unprojected_multiplier":
            value = values.min()
        else:
            value = values.sum()
        report[key] = int(value) if key in ("clipped_entries", "projection_steps") else float(value)
    if "projection_steps" in report and "accepted_steps" in result:
        steps = result["accepted_steps"]
        accepted = int((steps if index is None else steps[index:index+1]).sum())
        report["projection_step_fraction"] = report["projection_steps"] / max(1, accepted)
    return report


def collect_projection_diagnostics(model, problem, start, calls, result,
                                   guarded=False, max_backtracks=12):
    """Repeat a deterministic direct solve outside timing to collect counters.

    The timed solve still evaluates the actual field, projection and safety
    checks. Only reductions used for reporting move to this separate pass.
    """
    diagnostic = run_cfm(model, problem, start, calls, guarded, max_backtracks,
                         collect_diagnostics=True)
    if (not torch.equal(result["final"], diagnostic["final"])
            or not torch.equal(result["completed"], diagnostic["completed"])):
        raise RuntimeError("Diagnostic replay changed the solver result; timings and diagnostics must describe the same deterministic solve")
    for key in ("clipped_entries", "projection_steps", "projection_l1",
                "projection_max", "min_unprojected_multiplier"):
        result[key] = diagnostic[key]


def start_mode_description(mode):
    if mode == "zero":
        return "Primary solver evaluation: zero initialization, no reference endpoint input"
    return "Diagnostic only: initialization is constructed using the reference endpoint"


def aggregate_projection_rows(rows):
    """Aggregate the same counters across scenes or physical frames."""
    report = {}
    for key in ("clipped_entries", "projection_steps", "projection_l1",
                "projection_max", "min_unprojected_multiplier"):
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        report[key] = (max(values) if key == "projection_max" else
                       min(values) if key == "min_unprojected_multiplier" else sum(values))
    if "projection_steps" in report:
        accepted = sum(row.get("accepted_steps", 0) for row in rows)
        report["projection_step_fraction"] = report["projection_steps"] / max(1, accepted)
    return report


def unclipped_summary(model, problem, start, calls, tolerance, optimum=None):
    """Untimed direct-field counterfactual; never replaces the projected result."""
    result = run_cfm(model, problem, start, calls, project_state=False)
    stats = residuals(problem, result["final"])
    completed = result["completed"]
    success = completed & converged(problem, result["final"], tolerance)
    report = dict(scope="Diagnostic only: direct Euler without multiplier projection; excluded from primary performance",
                  samples=len(start), completed_count=int(completed.sum()),
                  success_count=int(success.sum()),
                  max_penetration=float(stats["penetration"].max()),
                  max_projected_gradient=float(stats["projected_gradient"].max()),
                  max_negative_multiplier=float(stats["negative_multiplier"].max()),
                  **projection_summary(result))
    if optimum is not None:
        report["projection_position_mse"] = float(position_error(problem, result["final"], optimum).mean())
    report["per_scene"] = [dict(completed=bool(completed[i]), success=bool(success[i]),
        failure_code=int(result["failure_code"][i]), accepted_steps=int(result["accepted_steps"][i]),
        time=float(result["time"][i]),
        projected_gradient=float(stats["projected_gradient"][i]),
        penetration=float(stats["penetration"][i]),
        negative_multiplier=float(stats["negative_multiplier"][i]),
        **projection_summary(result, i)) for i in range(len(start))]
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
                       projection_position_mse=sum(x["projection_position_mse"] for x in items)/len(items),
                       **aggregate_projection_rows(items))
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
    """ALL-world raw CFM diagnostic, grouped so release counts cannot dominate."""
    budgets = config["train"]["validation_calls"]
    if isinstance(budgets, list):
        if not budgets or any(type(k) is not int or k < 1 for k in budgets) or len(set(budgets)) != len(budgets):
            raise ValueError("Validation budgets must be distinct positive integers")
        reports = {}
        for calls in budgets:
            single = deepcopy(config)
            single["train"]["validation_calls"] = calls
            reports[str(calls)] = solver_validation(model, split, single, device)
        return dict(by_calls=reports, selection="Equal weight per NFE budget and scene/source group; includes reference-derived diagnostic starts for historical comparability",
            balanced=dict(groups=sum(report["balanced"]["groups"] for report in reports.values()),
                success_rate=sum(report["balanced"]["success_rate"] for report in reports.values())/len(reports),
                mean_projected_gradient=sum(report["balanced"]["mean_projected_gradient"] for report in reports.values())/len(reports)),
            primary_zero=dict(groups=sum(report["primary_zero"]["groups"] for report in reports.values()),
                success_rate=sum(report["primary_zero"]["success_rate"] for report in reports.values())/len(reports),
                mean_projected_gradient=sum(report["primary_zero"]["mean_projected_gradient"] for report in reports.values())/len(reports)))
    settings = config["evaluation"]
    report, all_rows = {}, []
    for mode in settings["start_modes"]:
        rows = []
        for index in torch.arange(len(split["names"])).split(config["train"]["batch_size"]):
            problem = move(select(split["problem"], index), device, torch.float32)
            optimum = split["optimum"][index].to(device=device, dtype=torch.float32)
            result = run_cfm(model, problem, evaluation_start(problem, optimum, mode),
                             config["train"]["validation_calls"])
            for j, context in enumerate(index):
                row = dict(name=split["names"][int(context)], **summarize(
                    select(problem, slice(j, j+1)), result["final"][j:j+1],
                    optimum[j:j+1], settings["tolerance"]))
                row["completed"] = bool(result["completed"][j])
                row["success_count"] *= int(row["completed"])
                row["accepted_steps"] = int(result["accepted_steps"][j])
                row.update(projection_summary(result, j))
                rows.append(row)
        report[mode] = group_summary(rows)
        all_rows.extend(x for x in rows if not x["name"].startswith("free_flight"))
    report["nontrivial"] = dict(samples=len(all_rows),
        success_rate=sum(x["success_count"] for x in all_rows)/max(1, len(all_rows)),
        mean_projected_gradient=sum(x["projected_gradient"] for x in all_rows)/max(1, len(all_rows)))
    groups = [stats for mode in settings["start_modes"] for name, stats in report[mode].items()
              if not name.startswith("free_flight")]
    report["balanced"] = dict(groups=len(groups),
        success_rate=sum(g["success_count"]/g["samples"] for g in groups)/max(1, len(groups)),
        mean_projected_gradient=sum(g["projected_gradient"] for g in groups)/max(1, len(groups)))
    zero_groups = [stats for name, stats in report.get("zero", {}).items()
                   if not name.startswith("free_flight")]
    report["primary_zero"] = dict(groups=len(zero_groups),
        success_rate=sum(g["success_count"]/g["samples"] for g in zero_groups)/max(1, len(zero_groups)),
        mean_projected_gradient=sum(g["projected_gradient"] for g in zero_groups)/max(1, len(zero_groups)))
    report["start_mode_roles"] = {mode: start_mode_description(mode) for mode in settings["start_modes"]}
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
        title=title, final_label="projection endpoint (no trajectory shown)")


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
                  head_type=config["model"].get("head_type", "analytic"),
                  relinearization=rebound, independent_oracle_position_mse=oracle_errors,
                  initial_Q_directional_derivative=slope.tolist(), cases={},
                  pgs=dict(seconds=qp_seconds, sweeps=qp["sweeps"].tolist(),
                           contact_evals=qp["contact_evals"].tolist(), tolerance=ref["qp_tolerance"]))
    all_success = True
    for i, (name, _, r, _) in enumerate(definitions):
        single = select(problem, slice(i, i+1))
        result = pgs(single, start[i:i+1], ref["qp_tolerance"], ref["max_sweeps"])
        item = summarize(single, result["final"], qp["final"][i:i+1], config["evaluation"]["tolerance"])
        # The canonical endpoint is the zero-start QP solution. Its straight
        # source-to-target path lies in the nonnegative multiplier orthant.
        path = torch.stack([torch.lerp(start[i], qp["final"][i], tau) for tau in (0., .25, .5, .75, 1.)])
        item["path_nonnegative"] = bool((path >= 0).all())
        item["reference_sweeps"] = int(result["sweeps"][0])
        all_success &= bool(result["converged"].all()) and item["projection_position_mse"] < 1e-10
        if config["evaluation"]["render"]:
            item["rendered"] = render_case(single, start[i:i+1], result["final"], r, config,
                Path(output).parent / "renders" / f"{name}.png", name)
        report["cases"][name] = item
    report["passed"] = bool(qp["converged"].all() and max(oracle_errors) < 1e-12
                            and (slope <= 1e-10).all() and rebound["passed"] and all_success)
    # Verify the NEW head before training, not just the old label generator.
    # An exact coupling rate must reproduce the same CFM straight path, even
    # when a multiplier needs to decrease. This is a representability check,
    # NOT a claim about an untrained network's coupled-contact accuracy.
    # The analytic baseline keeps its exact isolated-contact property. Direct
    # heads are not expected to solve any contact before learning.
    analytic_settings = dict(config["model"], head_type="analytic")
    net = ConditionalField(**analytic_settings).double()
    scalar = select(problem, slice(0, 3))
    with torch.no_grad():
        net.head.bias.fill_(100.)  # Isolated contact cannot depend on this head.
        isolated = run_cfm(net, scalar, start[:3], 4)
        isolated_error = float(position_error(scalar, isolated["final"], qp["final"][:3]).max())
    exact_rate = qp["final"] - start
    matching_error, endpoint_error = [], []
    for tau in (0., .5, .99):
        times = start.new_full((len(start),), tau)
        state = torch.lerp(start, qp["final"], tau)
        # Inject the known coupling rate only for this non-learning check.
        endpoint = contact_endpoint(problem, state + (1 - tau) * exact_rate)
        matching_error.append(float(((endpoint-state)/(1-tau)-exact_rate).abs().max()))
        endpoint_error.append(float(position_error(problem, endpoint, qp["final"]).max()))
    report["structured_head"] = dict(
        isolated_untrained_position_mse=isolated_error,
        exact_coupling_field_max_error=max(matching_error),
        exact_coupling_endpoint_position_mse=max(endpoint_error),
        passed=isolated_error < 1e-12 and max(matching_error) < 1e-4 and max(endpoint_error) < 1e-10,
        scope="Isolated analytic solve + exact-coupling representability; not learned performance")
    report["passed"] &= report["structured_head"]["passed"]
    if config["model"].get("head_type", "analytic") == "direct":
        class ExactVelocity:
            head_type = "direct"
            neural_evaluations = 1

            def __call__(self, state, tau, fixed_problem):
                return exact_rate * fixed_problem["mask"]

        # A constant signed CFM target integrates the straight path exactly.
        # This checks the deployed projected update, not untrained accuracy.
        direct_errors, clipping = [], []
        for calls in (1, 4):
            result = run_cfm(ExactVelocity(), problem, start, calls)
            direct_errors.append(float((result["final"] - qp["final"]).abs().max()))
            clipping.append(projection_summary(result))
        direct_net = ConditionalField(**config["model"]).double()
        with torch.no_grad():
            direct_net.head.weight.zero_()
            direct_net.head.bias.fill_(-1.)
            signed_output = direct_net(start, torch.zeros(len(start), dtype=start.dtype), problem)
        negative_output = bool((signed_output[problem["mask"]] < 0).any())
        padding_zero = bool((signed_output[~problem["mask"]] == 0).all())
        report["direct_head"] = dict(
            controlled_velocity_has_positive=bool((exact_rate > 0).any()),
            controlled_velocity_has_negative=bool((exact_rate < 0).any()),
            exact_velocity_endpoint_max_error=max(direct_errors),
            signed_neural_output=negative_output, padded_output_zero=padding_zero,
            projection_by_calls=dict(zip(("1", "4"), clipping)),
            scope="Controlled signed velocity and projected straight-path integration; not untrained solver accuracy")
        report["direct_head"]["passed"] = bool(
            (exact_rate > 0).any() and (exact_rate < 0).any()
            and max(direct_errors) < 1e-10 and negative_output and padding_zero)
        report["passed"] &= report["direct_head"]["passed"]
    report["cost_note"] = "PGS labels and independent oracle use CPU float64; no CFM speed claim"
    write_json(output, report)
    return report



@torch.no_grad()
def evaluate(model, split, config, device, output, metadata=None):
    ref, settings = config["reference"], config["evaluation"]
    budgets = validate_budgets(settings)
    position_tolerance = float(settings.get("position_tolerance", settings["tolerance"]))
    model.eval()
    contexts = scene_indices(split["names"], int(settings["max_scenes"]))
    names = [split["names"][int(i)] for i in contexts]
    count = len(contexts)
    if count < 1 or settings["timing_repeats"] < 1:
        raise ValueError("Positive evaluation scene count and timing repeats required")
    problem64 = select(split["problem"], contexts)
    problem = move(problem64, device, torch.float32)
    optimum64 = split["optimum"][contexts]
    optimum = optimum64.to(device=device, dtype=torch.float32)
    direct = getattr(model, "head_type", "analytic") == "direct"
    report = dict(scope=getattr(model, "solver_description", SOLVER_DESCRIPTION),
                  model_format=getattr(model, "checkpoint_format", CHECKPOINT_FORMAT),
                  head_type=getattr(model, "head_type", "analytic"),
                  device=str(device), tau_interval=[0., 1.],
                  integrator="projected explicit Euler" if direct else "convex-combination Euler", scenes=names,
                  raw_definition=("Signed direct velocity; project multipliers after every Euler step; no Q guard or finish"
                                  if direct else "Analytic contact projection is in the head; no EXTRA guard/finish"),
                  local_definition="Analytic local endpoint with zero neural coupling and the same tau schedule; not native Jacobi",
                  guard_version="cfm_euler_Q_nonnegative_v1", modes={},
                  start_mode_roles={mode: start_mode_description(mode) for mode in settings["start_modes"]},
                  primary_start_mode="zero",
                  timings_include="Solver including neural features, updates/projection and optional guard; exclude geometry setup, labels, projection-counter reductions and unclipped diagnostics",
                  cost_units="NFE is field/endpoint evaluations; neural_evals excludes local ablation; seconds measure the complete batch",
                  projection_diagnostics="Direct-head state projection on accepted steps only (not analytic endpoint ReLU): clipped_entries counts valid negative coordinates; projection_steps counts steps with clipping; projection_l1 sums corrections; projection_max is the largest correction",
                  config=config, metadata=metadata or {},
                  environment=runtime_environment(device, model),
                  fixed_budget=dict(budgets=budgets,
                      cfm="Exactly K raw Euler field evaluations on [0,1]; no PGS finish",
                      pgs="Exactly K sequential full-contact sweeps; no early stopping",
                      comparison_scope="Identical frozen QPs and initial states; equal iteration counts, not equal compute",
                      reference="Original float64 cached PGS endpoints; no reference input for zero-start",
                      position_accuracy="Mass-weighted per-coordinate position RMSE <= position_tolerance; separate from KKT success",
                      objective_gap="Signed Q(prediction)-Q(reference), recomputed on original float64 QP",
                      multiplier_error="Diagnostic only; redundant constraints can make multipliers nonunique"),
                  fixed_budget_comparison={},
                  float64_label_quality=summarize(problem64, optimum64, optimum64, settings["tolerance"]))
    for mode in settings["start_modes"]:
        start = evaluation_start(problem, optimum, mode)
        rows = {}
        raw_budgets = {}
        diagnostic_specs = {}
        operations = {"pgs": lambda: pgs(problem, start, settings["tolerance"], ref["max_sweeps"])}
        for calls in budgets:
            operations[f"pgs_k{calls}_fixed"] = lambda k=calls: pgs(
                problem, start, settings["tolerance"], k, stop_at_tolerance=False)
            for guarded in ([False, True] if settings.get("guarded", False) else [False]):
                local_name = f"local_k{calls}_{'guarded' if guarded else 'raw'}"
                operations[local_name] = lambda k=calls, safe=guarded: run_cfm(
                    LocalProjection(), problem, start, k, safe, settings["max_backtracks"],
                    collect_diagnostics=False)
                name = f"cfm_k{calls}_{'guarded' if guarded else 'raw'}"
                operations[name] = lambda k=calls, safe=guarded: run_cfm(
                    model, problem, start, k, safe, settings["max_backtracks"],
                    collect_diagnostics=False)
                if direct:
                    diagnostic_specs[name] = (calls, guarded)
                if direct and not guarded:
                    raw_budgets[name] = calls
        for name, operation in operations.items():
            operation()
            measurements = [timed(device, operation) for _ in range(settings["timing_repeats"])]
            result = measurements[-1][0]
            if name in diagnostic_specs:
                calls, guarded = diagnostic_specs[name]
                collect_projection_diagnostics(model, problem, start, calls, result,
                                               guarded, settings["max_backtracks"])
            row = summarize(problem, result["final"], optimum, settings["tolerance"])
            row["seconds"] = sum(item[1] for item in measurements) / len(measurements)
            row["timing_repeats"] = len(measurements)
            row["timing"] = timing_summary([item[1] for item in measurements], count)
            row["cost"] = work_summary(result, problem, model if name.startswith("cfm_") else None)
            errors = reference_errors(problem64, result["final"], optimum64, model.length_scale)
            completed = result.get("completed", torch.ones(count, dtype=torch.bool, device=device))
            row["converged_count"] = row["success_count"]
            row["success_count"] = int((completed & converged(problem, result["final"], settings["tolerance"])).sum())
            row["completed_count"] = int(completed.sum())
            for key in ("nfe", "neural_evals", "sweeps", "contact_evals", "backtracks", "interventions", "accepted_steps"):
                if key in result:
                    row[key] = int(result[key].sum())
            row.update(projection_summary(result))
            row["per_scene"] = []
            for i in range(count):
                scene = dict(name=names[i], **summarize(select(problem, slice(i, i+1)),
                    result["final"][i:i+1], optimum[i:i+1], settings["tolerance"]))
                scene["converged_count"] = scene["success_count"]
                scene["success_count"] *= int(completed[i])
                scene["completed"] = bool(completed[i])
                for key in ("time", "nfe", "neural_evals", "sweeps", "contact_evals", "backtracks", "interventions", "accepted_steps", "min_accepted_h", "failure_code"):
                    if key in result:
                        scene[key] = result[key][i].item()
                scene.update(projection_summary(result, i))
                scene.update({key: float(values[i]) for key, values in errors.items()})
                row["per_scene"].append(scene)
            row["accuracy"] = accuracy_summary(row["per_scene"], position_tolerance)
            row["groups"] = group_summary(row["per_scene"])
            for group, stats in row["groups"].items():
                members = [scene for scene in row["per_scene"] if scene["name"].rsplit("_", 1)[0] == group]
                stats["accuracy"] = accuracy_summary(members, position_tolerance)
            if name in raw_budgets:
                diagnostic = unclipped_summary(model, problem, start, raw_budgets[name],
                                               settings["tolerance"], optimum)
                for scene_name, scene in zip(names, diagnostic["per_scene"]):
                    scene["name"] = scene_name
                row["unclipped_diagnostic"] = diagnostic
            rows[name] = row
            if settings["render"] and mode == "zero" and not name.startswith("pgs_k"):
                render_case(select(problem, slice(0, 1)), start[:1], result["final"][:1],
                            split["radii"][int(contexts[0])], config, Path(output).parent / "renders" / f"{name}.png", name)
        report["modes"][mode] = rows
        report["fixed_budget_comparison"][mode] = budget_comparison(rows, budgets)
    if settings.get("recovery", {}).get("enabled", False):
        from .recovery import recovery_evaluation
        report["recovery"] = recovery_evaluation(model, split, config, device)
    write_json(output, report)
    write_budget_table(report, Path(output).with_suffix(".md"))
    return report
