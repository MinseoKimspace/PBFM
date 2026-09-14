"""Compare initialization scales and mu clocks without training/changing CFM."""
from __future__ import annotations

from math import isfinite
from time import perf_counter

import torch

from src.contact_flow.physics import PhysicsConfig
from .data import release_problem
from .evaluation import summarize, write_json
from .homotopy import Homotopy, euler, recovery, reference_path
from .problem import circle_problem, converged, make_problem, pack
from .solvers import active_set_solution, pgs


def cases(config):
    physics = PhysicsConfig(**config["physics"])
    eta = config["reference"]["eta_fraction"]
    for name, gap in (("isolated_penetrating", -.1), ("isolated_separated", .05), ("isolated_touching", 0.)):
        yield name, pack([make_problem(torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.tensor([gap], dtype=torch.float64), eta)])
    yield "contact_release", pack([release_problem(eta_fraction=eta)])
    yield "redundant_contacts", pack([make_problem(torch.zeros(1, dtype=torch.float64),
        torch.ones(2, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
        torch.full((2,), -.1, dtype=torch.float64), eta)])
    for size in config["homotopy_preflight"]["stack_sizes"]:
        p = torch.stack((torch.zeros(size, dtype=torch.float64),
                        physics.y_ground + .42 + .94 * torch.arange(size, dtype=torch.float64)), -1)
        yield f"stack_{size}", pack([circle_problem(p, torch.full((size,), .5, dtype=torch.float64), physics, eta)])


def movement(times, states, flow, bins=8):
    """Diagnostic only. All samples retain their ORIGINAL uniform solver clock."""
    scale = flow.gap_scale / flow.D.diag()
    lengths = ((states[1:] - states[:-1]) / scale).norm(dim=-1)
    allocation = [0.] * bins
    for left, right, distance in zip(times[:-1], times[1:], lengths.tolist()):
        # Split intervals at bin boundaries (especially coarse Euler intervals).
        for i in range(bins):
            overlap = max(0., min(right, (i+1)/bins) - max(left, i/bins))
            allocation[i] += distance * overlap / max(right-left, 1e-30)
    total = float(lengths.sum())
    return dict(metric="Euclidean norm of delta_lambda / (gap_scale / D_ii)",
                bin_edges=[i/bins for i in range(bins+1)],
                movement_fraction=[x/total if total else 0. for x in allocation],
                max_bin_fraction=max(allocation)/total if total else 0.,
                last_bin_fraction=allocation[-1]/total if total else 0.,
                path_length=total, net_displacement=float(((states[-1]-states[0])/scale).norm()),
                note="Sampled path lengths, not a proof of low curvature or a new clock")


def elapsed(operation):
    begin = perf_counter()
    result = operation()
    return result, perf_counter() - begin


def validate_settings(config):
    settings = config["homotopy_preflight"]
    if (not settings["stack_sizes"] or any(not isinstance(n, int) or not 1 <= n <= 10 for n in settings["stack_sizes"])
            or not settings["calls"] or any(not isinstance(k, int) or k < 1 for k in settings["calls"])
            or not settings["initializations"] or not settings["schedules"] or not settings["mu0_factors"]):
        raise ValueError("Nonempty experiment grids required; independent oracle supports stacks <=10")
    for name in ("reference_tolerance", "tolerance", "recovery_relative_tolerance"):
        if not isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"Positive finite {name} required")
    if settings["reference_points"] < 3 or settings["recovery_steps"] < 1:
        raise ValueError("Positive reference/recovery budgets required")
    return settings


@torch.no_grad()
def preflight(config, output):
    settings = validate_settings(config)
    ref = config["reference"]
    report = dict(format="multiplier_homotopy_preflight_v1", config=config,
        scope="No network/training/cache; fixed-normal QP, original anchor, CPU float64",
        clock="t in [0,1]; geometric or linear mu; no arc-length remapping",
        reference_method="Damped Newton root solves of H=0 at uniform t; all solves/line searches counted",
        endpoint="Positive mu_min: perturbed KKT, never claimed to be the exact QP solution",
        guard="Positivity only; guarded runs can exceed K calls; no clipping or PGS fallback",
        cost_note="Single CPU timings include Python overhead; not a learned speedup claim. Oracle is diagnostic only.",
        cases={}, summary={})
    began = perf_counter()
    for name, problem in cases(config):
        zero = torch.zeros_like(problem["c"])
        target, oracle_seconds = elapsed(lambda: active_set_solution(problem))
        baseline, seconds = elapsed(lambda: pgs(problem, zero, settings["tolerance"], ref["max_sweeps"]))
        case = dict(contacts=len(problem["c"][0]), original_proposal=problem["p"][0].tolist(),
            D=problem["D"][0].tolist(), c=problem["c"][0].tolist(),
            initially_solved=bool(converged(problem, zero, settings["tolerance"]).all()),
            oracle_seconds=oracle_seconds,
            oracle=summarize(problem, target, target, settings["tolerance"]),
            pgs_zero=dict(**summarize(problem, baseline["final"], target, settings["tolerance"]),
                          seconds=seconds, sweeps=int(baseline["sweeps"][0]),
                          contact_evals=int(baseline["contact_evals"][0])), variants={})
        for factor in settings["mu0_factors"]:
            for initialization in settings["initializations"]:
                for schedule in settings["schedules"]:
                    key = f"{initialization}/{schedule}/mu0x{factor:g}"
                    flow, init_seconds = elapsed(lambda: Homotopy.create(problem["D"][0], problem["c"][0],
                        initialization=initialization, schedule=schedule, mu0_factor=factor,
                        gap_scale=settings["gap_scale"], mu_min_ratio=settings["mu_min_ratio"],
                        epsilon_ratio=settings["epsilon_ratio"], kappa=settings["kappa"]))
                    path, path_seconds = elapsed(lambda: reference_path(flow, settings["reference_points"],
                                                                          settings["reference_tolerance"]))
                    initial = summarize(problem, flow.start[None], target, settings["tolerance"])
                    final = summarize(problem, path["states"][-1:], target, settings["tolerance"])
                    same_start, same_seconds = elapsed(lambda: pgs(problem, flow.start[None], settings["tolerance"], ref["max_sweeps"]))
                    recovery_index = (len(path["times"]) - 1) // 2
                    t0 = path["times"][recovery_index]
                    rec = dict(completed=False, reason="reference_path_failed")
                    if path["completed"]:
                        rec, recovery_seconds = elapsed(lambda: recovery(flow, path["states"][recovery_index],
                            t0=t0, duration=.25, steps=settings["recovery_steps"],
                            log_perturbation=settings["log_perturbation"]))
                        rec["seconds"] = recovery_seconds
                    rec["passed"] = bool(rec["completed"] and
                        rec["relative_decay_error"] <= settings["recovery_relative_tolerance"] and
                        rec["differential_identity_error"] <= 1e-8)
                    row = dict(initialization_seconds=init_seconds, mu0=flow.mu0, mu_min=flow.mu_min,
                        initial_multiplier=flow.start.tolist(), r0=flow.r0.tolist(),
                        initial_H_max=float(flow.residual(flow.start, 0.).abs().max()),
                        initial_actual_gap=(flow.c + flow.D @ flow.start).tolist(),
                        initial_metrics=initial,
                        reference=dict(completed=path["completed"], reason=path["reason"],
                            seconds=path_seconds, total_seconds=init_seconds+path_seconds,
                            linear_solves=path["linear_solves"], residual_evals=path["residual_evals"],
                            backtracks=path["backtracks"], normalized_H=path["normalized_H"],
                            final_metrics=final, times=path["times"], multipliers=path["states"].tolist(),
                            mu=[flow.mu(t)[0] for t in path["times"]],
                            movement=movement(path["times"], path["states"], flow)),
                        recovery=rec,
                        pgs_same_initialization=dict(seconds=same_seconds, total_seconds=init_seconds+same_seconds,
                            sweeps=int(same_start["sweeps"][0]), contact_evals=int(same_start["contact_evals"][0]),
                            **summarize(problem, same_start["final"], target, settings["tolerance"])),
                        euler={})
                    for calls in settings["calls"]:
                        for guarded in (False, True):
                            result, run_seconds = elapsed(lambda: euler(flow, calls, guarded=guarded,
                                max_nfe_multiplier=settings["guard_max_nfe_multiplier"]))
                            metrics = summarize(problem, result["final"][None], target, settings["tolerance"])
                            row["euler"][f"k{calls}_{'positive_guard' if guarded else 'raw'}"] = dict(
                                requested_calls=calls, completed=result["completed"], reason=result["reason"],
                                success=bool(result["completed"] and metrics["success_count"]),
                                actual_nfe=result["nfe"], linear_solves=result["linear_solves"],
                                interventions=result["interventions"], elapsed_solver_time=result["elapsed"],
                                seconds=run_seconds, total_seconds=init_seconds+run_seconds,
                                normalized_H=float(flow.residual(result["final"], result["elapsed"]).abs().max())/flow.gap_scale,
                                final_multiplier=result["final"].tolist(), metrics=metrics,
                                movement=movement(result["times"], result["states"], flow))
                    row["reference_ready"] = bool(path["completed"] and final["success_count"] and rec["passed"])
                    case["variants"][key] = row
        report["cases"][name] = case
        print(f"  homotopy: {name} ({case['contacts']} contacts), {perf_counter()-began:.1f}s", flush=True)
    # Never hide trivial scenes or contact-count failures in one pooled score.
    for key in next(iter(report["cases"].values()))["variants"]:
        all_items = [(name, c["variants"][key]) for name, c in report["cases"].items()]
        items = [(name, row) for name, row in all_items if not report["cases"][name]["initially_solved"]]
        budgets = {}
        for label in items[0][1]["euler"]:
            budgets[label] = dict(solved=sum(row["euler"][label]["success"] for _, row in items),
                scenes=len(items), failures=[name for name, row in items if not row["euler"][label]["success"]],
                initially_solved_failures=[name for name, row in all_items
                    if report["cases"][name]["initially_solved"] and not row["euler"][label]["success"]],
                actual_nfe=sum(row["euler"][label]["actual_nfe"] for _, row in items),
                interventions=sum(row["euler"][label]["interventions"] for _, row in items))
        report["summary"][key] = dict(nontrivial_scenes=len(items),
            reference_ready=sum(row["reference_ready"] for _, row in items),
            all_reference_ready=all(row["reference_ready"] for _, row in all_items),
            reference_failures=[name for name, row in all_items if not row["reference_ready"]], budgets=budgets)
    report["reference_ready_variants"] = [key for key, row in report["summary"].items()
        if not key.startswith("epsilon/") and "/geometric/" in key
        and row["all_reference_ready"]]
    report["budget_ready_variants"] = {key: [label for label, value in report["summary"][key]["budgets"].items()
        if label.endswith("_raw") and value["solved"] == value["scenes"] and not value["initially_solved_failures"]]
        for key in report["reference_ready_variants"]}
    report["ready_for_learning"] = any(report["budget_ready_variants"].values())
    report["elapsed_seconds"] = perf_counter() - began
    write_json(output, report)
    return report
