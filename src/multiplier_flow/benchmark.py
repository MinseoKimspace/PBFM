"""Untimed accuracy/work accounting for fixed-budget contact solver comparisons.

NFE and PGS sweeps are different units. Work counters describe this padded
implementation; they are not FLOP estimates or claims of equal compute.
"""
from __future__ import annotations

import math
import platform
import statistics
from pathlib import Path

import torch

from .problem import position_error, value


def validate_budgets(settings):
    calls = settings["calls"]
    if (not isinstance(calls, (list, tuple)) or not calls
            or any(type(k) is not int or k < 1 for k in calls)
            or len(set(calls)) != len(calls)):
        raise ValueError("evaluation.calls must contain distinct positive integers")
    for key in ("tolerance", "position_tolerance"):
        number = float(settings.get(key, settings["tolerance"]))
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"evaluation.{key} must be finite and positive")
    if type(settings["timing_repeats"]) is not int or settings["timing_repeats"] < 1:
        raise ValueError("evaluation.timing_repeats must be a positive integer")
    return list(calls)


def distribution(values):
    """Equal weight per QP, with explicit accounting for nonfinite diagnostics."""
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten().cpu()
    finite = tensor[torch.isfinite(tensor)]
    result = dict(samples=len(tensor), finite_count=len(finite),
                  nonfinite_count=len(tensor) - len(finite))
    if not len(finite):
        return dict(result, mean=None, median=None, p95=None, max=None)
    return dict(result, mean=float(finite.mean()), median=float(finite.quantile(.5)),
                p95=float(finite.quantile(.95)), max=float(finite.max()))


def reference_errors(problem64, prediction, reference64, length_scale):
    """Score the FP32 solver output against the original FP64 label/problem.

    Position error is mass-weighted per coordinate. Multiplier RMSE is only a
    diagnostic: redundant contacts can encode identical positions differently.
    The signed objective difference is not clipped to hide rounding differences.
    """
    prediction64 = prediction.detach().to(device=reference64.device, dtype=torch.float64)
    rmse = position_error(problem64, prediction64, reference64).sqrt()
    mask = problem64["mask"]
    multiplier_mse = ((prediction64 - reference64).square() * mask).sum(-1)
    multiplier_mse /= mask.sum(-1).clamp_min(1)
    return dict(position_rmse=rmse, position_normalized_rmse=rmse / length_scale,
                multiplier_rmse=multiplier_mse.sqrt(),
                objective_gap=value(problem64, prediction64) - value(problem64, reference64))


def accuracy_summary(rows, position_tolerance):
    count = len(rows)
    within = sum(math.isfinite(row["position_rmse"])
                 and row["position_rmse"] <= position_tolerance
                 and row["completed"] for row in rows)
    success = sum(row["success_count"] for row in rows)
    return dict(samples=count, success_count=success, success_rate=success / count,
                position_tolerance=position_tolerance,
                position_within_tolerance_count=within,
                position_within_tolerance_rate=within / count,
                **{name: distribution([row[name] for row in rows]) for name in (
                    "projected_gradient", "penetration", "position_rmse",
                    "position_normalized_rmse", "multiplier_rmse", "objective_gap")})


def timing_summary(seconds, samples):
    median = statistics.median(seconds)
    return dict(batch_seconds=list(seconds), mean_seconds=statistics.mean(seconds),
                median_seconds=median, min_seconds=min(seconds), max_seconds=max(seconds),
                stdev_seconds=statistics.pstdev(seconds),
                amortized_median_seconds_per_qp=median / samples,
                qps_per_second=samples / median,
                warmup_runs=1,
                scope="Complete solver batch; excludes geometry, decode, labels and reporting. Per-QP time is amortized throughput, not single-QP latency.")


def work_summary(result, problem, model=None):
    """Logical work plus actual padded neural-token/attention-score counts."""
    batch, padded_contacts = problem["mask"].shape
    counts = problem["mask"].sum(-1)
    totals = {key: int(result[key].sum()) if key in result else 0 for key in (
        "nfe", "neural_evals", "sweeps", "contact_evals", "backtracks", "accepted_steps")}
    field_calls = int(result["nfe"].max()) if "nfe" in result else 0
    neural_calls = int(result["neural_evals"].max()) if "neural_evals" in result else 0
    batch_sweeps = int(result["sweeps"].max()) if "sweeps" in result else 0
    attention = getattr(model, "attention", ())
    # Each implemented attention layer computes dense scores before masking.
    attention_heads = sum(layer.heads for layer in attention)
    return dict(logical_totals=totals,
                logical_mean_per_qp={key: total / batch for key, total in totals.items()},
                valid_contacts=distribution(counts), padded_contacts=padded_contacts,
                executed_batch_field_calls=field_calls,
                executed_batch_neural_calls=neural_calls,
                executed_batch_pgs_sweeps=batch_sweeps,
                padded_pgs_coordinate_visits=batch_sweeps * batch * padded_contacts,
                padded_neural_token_evaluations=neural_calls * batch * padded_contacts,
                dense_attention_score_entries=neural_calls * batch * padded_contacts ** 2 * attention_heads,
                parameter_count=sum(p.numel() for p in model.parameters()) if model is not None else 0,
                note="Counts, not FLOPs: NFE includes features/head/update; a PGS sweep sequentially visits all contacts. Padded counts include masked work.")


def runtime_environment(device, model):
    return dict(device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                platform=platform.platform(), python_version=platform.python_version(),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                cpu_threads=torch.get_num_threads(), solver_dtype="float32",
                reference_metric_dtype="float64", matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                model_parameters=sum(p.numel() for p in model.parameters()))


def budget_comparison(methods, budgets):
    """Paired results on identical QPs/initial states, never guarded runs."""
    comparison = {}
    for budget in budgets:
        cfm, baseline = methods[f"cfm_k{budget}_raw"], methods[f"pgs_k{budget}_fixed"]
        pairs = list(zip(cfm["per_scene"], baseline["per_scene"]))
        if any(a["name"] != b["name"] for a, b in pairs) or len(cfm["per_scene"]) != len(baseline["per_scene"]):
            raise ValueError("Fixed-budget comparisons must use identical ordered scenes")
        comparison[str(budget)] = dict(
            cfm_method=f"cfm_k{budget}_raw", pgs_method=f"pgs_k{budget}_fixed",
            success_rate_difference=cfm["accuracy"]["success_rate"] - baseline["accuracy"]["success_rate"],
            cfm_only_success=sum(bool(a["success_count"]) and not b["success_count"] for a, b in pairs),
            pgs_only_success=sum(bool(b["success_count"]) and not a["success_count"] for a, b in pairs),
            cfm_lower_projected_gradient=sum(a["projected_gradient"] < b["projected_gradient"] for a, b in pairs),
            cfm_lower_position_rmse=sum(a["position_rmse"] < b["position_rmse"] for a, b in pairs),
            equal_budget_pgs_over_cfm_time=baseline["timing"]["median_seconds"] / cfm["timing"]["median_seconds"],
            timing_note="Equal iteration budget only; this ratio does not establish equal accuracy.")
    return comparison


def write_budget_table(report, output):
    """Human-readable companion to the detailed JSON; one row per method/K."""
    lines = ["# Fixed-budget solver comparison", "",
             "CFM: K field calls on [0,1]. PGS fixed: exactly K full contact sweeps, without early stopping.",
             "PGS converged: separate tolerance-stopped baseline. K is not equal compute.",
             "Position RMSE is mass-weighted per coordinate against the original FP64 reference.",
             "Times measure the complete solver batch; per-QP time is amortized throughput, not latency.",
             "Residual/RMSE distributions weight each QP equally. Inspect JSON groups for hard scenes.",
             f"KKT tolerance: {report['config']['evaluation']['tolerance']:g}.", ""]
    for mode, methods in report["modes"].items():
        lines.extend([f"## Start: {mode}", "", report["start_mode_roles"][mode], "",
                      "| K | Method | Solved % | PG mean | PG p95 | Position RMSE mean | RMSE p95 | Position within tolerance % | NFE/QP | Sweeps/QP | PGS contact updates/QP | Batch median ms | Amortized us/QP |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        names = [(k, name) for k in report["fixed_budget"]["budgets"] for name in
                 (f"cfm_k{k}_raw", f"pgs_k{k}_fixed", f"local_k{k}_raw")]
        names.append(("until tolerance", "pgs"))
        for k, name in names:
            row = methods[name]
            acc, timing, cost = row["accuracy"], row["timing"], row["cost"]["logical_mean_per_qp"]
            def number(value):
                return "n/a" if value is None else f"{value:.6g}"
            lines.append(f"| {k} | {name} | {100 * acc['success_rate']:.2f} | "
                         f"{number(acc['projected_gradient']['mean'])} | {number(acc['projected_gradient']['p95'])} | "
                         f"{number(acc['position_rmse']['mean'])} | {number(acc['position_rmse']['p95'])} | "
                         f"{100 * acc['position_within_tolerance_rate']:.2f} | {cost['nfe']:.2f} | "
                         f"{cost['sweeps']:.2f} | {cost['contact_evals']:.2f} | "
                         f"{1000 * timing['median_seconds']:.3f} | "
                         f"{1e6 * timing['amortized_median_seconds_per_qp']:.3f} |")
        lines.extend(["", f"Position tolerance: {methods['pgs']['accuracy']['position_tolerance']:g}.", ""])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
