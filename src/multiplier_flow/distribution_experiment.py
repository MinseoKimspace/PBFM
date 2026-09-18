"""Training and conditional-distribution evaluation using the unchanged D model."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import torch

from .benchmark import distribution as scalar_distribution, runtime_environment
from .distribution import (FORMAT, build_distribution_split, conditional_distribution_metrics,
                           integer_list, sample_batch, specification, target_samples, validate_distribution)
from .evaluation import timed, write_json
from .model import ConditionalField, LocalProjection, cfm_loss
from .problem import converged, move, position_error, residuals, select
from .solvers import integrate_cfm, pgs, run_cfm, run_hybrid
from .training import physical_endpoint_loss


def distribution_loss(model, problem, source, target, tau, settings, rng):
    cfm, _ = cfm_loss(model, problem, source, target, tau)
    inner = settings["inner_rollout"]
    total, calls = cfm, 0
    terms = {"cfm": cfm.detach()}
    if inner["weight"] > 0:
        calls = inner["calls"][int(torch.randint(len(inner["calls"]), (), generator=rng))]
        # Match the same deployment prior used by CFM; never force zero-start.
        endpoint = integrate_cfm(model, problem, source, calls)
        physical, kkt, position = physical_endpoint_loss(
            problem, endpoint, target, model.length_scale, inner)
        total = total + inner["weight"]*physical
        terms.update(inner=physical.detach(), inner_kkt=kkt.detach(), inner_position=position.detach())
    terms["total"] = total.detach()
    return total, terms, calls


def aggregate_rows(rows):
    keys = ("success_rate", "completed_rate", "position_within_tolerance_rate", "position_rmse_mean",
            "pg_mean", "sliced_w1", "marginal_w1", "variance_ratio", "group_total_relative_error",
            "mean_coordinate_bias")
    return {key: sum(row[key] for row in rows)/len(rows) for key in keys}


def score_predictions(split, indices, prediction, reference, completed, settings, seed):
    problem = select(split["problem"], indices)
    stats = residuals(problem, prediction)
    success = converged(problem, prediction, settings["tolerance"]) & completed
    rmse = position_error(problem, prediction, split["canonical"][indices]).sqrt()
    samples = settings["samples_per_qp"]
    rows = []
    for i, name in enumerate(split["names"]):
        take = slice(i*samples, (i+1)*samples)
        row = dict(name=name, **split["descriptions"][i], samples=samples,
            success_rate=float(success[take].double().mean()),
            completed_rate=float(completed[take].double().mean()),
            position_within_tolerance_rate=float(((rmse[take] <= settings["position_tolerance"]) & completed[take]).double().mean()),
            position_rmse_mean=float(rmse[take].mean()), position_rmse_max=float(rmse[take].max()),
            pg_mean=float(stats["projected_gradient"][take].mean()),
            pg_max=float(stats["projected_gradient"][take].max()),
            **conditional_distribution_metrics(prediction[take], reference[take],
                split["group_index"][i], split["group_totals"][i], settings["projections"], seed+i))
        rows.append(row)
    return dict(balanced=aggregate_rows(rows), per_qp=rows,
        projected_gradient=scalar_distribution(stats["projected_gradient"]),
        position_rmse=scalar_distribution(rmse),
        failed_samples=int((~completed).sum()), unsuccessful_samples=int((~success).sum()))


@torch.no_grad()
def evaluate_distribution(model, split, config, selected_device, *, calls=None, samples=None,
                          baselines=True, measure_time=True, output=None, checkpoint_info=None):
    validate_distribution(config)
    model.eval()
    settings = dict(config["distribution"]["evaluation"])
    if samples is not None:
        settings["samples_per_qp"] = samples
    budgets = list(settings["calls"] if calls is None else calls)
    integer_list(budgets, "evaluation calls")
    if type(settings["samples_per_qp"]) is not int or settings["samples_per_qp"] < 2:
        raise ValueError("Distribution evaluation requires at least two samples per QP")
    if output is not None:
        output = Path(output)
        if output.suffix.lower() != ".json":
            raise ValueError("Distribution output must be a .json path")
        if any(output.with_suffix(suffix).exists() for suffix in (".json", ".md", ".pt", ".png")):
            raise FileExistsError(f"Refusing to overwrite {output} or companions; choose --output")
    seed = config["seed"]+500000
    indices = torch.arange(len(split["names"])).repeat_interleave(settings["samples_per_qp"])
    problem, source, _ = sample_batch(split, indices, config,
        torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed+1))
    # Evaluation always measures the intended uniform solution distribution,
    # including when the comparison model was trained on a single point.
    reference = target_samples(split, indices, torch.Generator().manual_seed(seed+2))
    independent = target_samples(split, indices, torch.Generator().manual_seed(seed+3))
    report = dict(format=FORMAT, scope="Conditional multiplier allocation on synthetic redundant chain QPs; unique decoded position",
        spec=specification(config), environment=runtime_environment(selected_device, model),
        target_definition="Independent uniform simplex allocations per duplicated contact group; analytically exact KKT labels",
        source_definition="Independent Uniform(0, source_scale * length_scale / D_ii); QP inputs only",
        distribution_metrics="Within each QP; empirical sliced W1 uses all returned states, including failed ones. Check validity alongside diversity.",
        timing_scope="Sum of per-chunk solver medians, including FM and optional PGS; excludes data/setup/metrics/rendering. Not single-sample latency.",
        checkpoint=checkpoint_info, split=split["split"], evaluation=settings,
        samples_per_qp=settings["samples_per_qp"], qps=len(split["names"]), calls=budgets,
        tolerance=settings["tolerance"], position_tolerance=settings["position_tolerance"],
        methods={})
    predictions = {"reference": reference}
    ones = torch.ones(len(indices), dtype=torch.bool)
    for name, prediction in (("reference_mc", independent), ("collapsed_exact", split["canonical"][indices]),
                              ("source", source)):
        report["methods"][name] = score_predictions(split, indices, prediction, reference, ones, settings, seed+4)
        predictions[name] = prediction
    methods = [(f"cfm_k{k}_raw", "cfm", k) for k in budgets]
    if baselines:
        methods.insert(0, ("pgs", "pgs", 0))
        methods.extend((f"local_k{k}", "local", k) for k in budgets)
        if settings["hybrid"]:
            methods.extend((f"cfm_k{k}_hybrid", "hybrid", k) for k in budgets)
    for name, kind, budget in methods:
        prediction = torch.empty_like(source)
        completed = torch.empty(len(indices), dtype=torch.bool)
        nfe = torch.zeros(len(indices), dtype=torch.long)
        sweeps = torch.zeros_like(nfe)
        chunk_timings = []
        for chunk in torch.arange(len(indices)).split(settings["batch_size"]):
            qp = move(select(problem, chunk), selected_device, torch.float32)
            start = source[chunk].to(device=selected_device, dtype=torch.float32)

            def operation():
                if kind == "pgs":
                    return pgs(qp, start, settings["tolerance"], config["reference"]["max_sweeps"])
                if kind == "hybrid":
                    return run_hybrid(model, qp, start, budget, settings["tolerance"], config["reference"]["max_sweeps"])
                if kind == "local":
                    return run_cfm(LocalProjection(), qp, start, budget, collect_diagnostics=False)
                return run_cfm(model, qp, start, budget, collect_diagnostics=False)

            if measure_time:
                operation()
                measurements = [timed(selected_device, operation) for _ in range(settings["timing_repeats"])]
                result = measurements[-1][0]
                chunk_timings.append([seconds for _, seconds in measurements])
            else:
                result = operation()
            prediction[chunk] = result["final"].double().cpu()
            completed[chunk] = result.get("completed", torch.ones(len(chunk), dtype=torch.bool, device=selected_device)).cpu()
            for key, values in (("nfe", nfe), ("sweeps", sweeps)):
                if key in result:
                    values[chunk] = result[key].cpu()
        row = score_predictions(split, indices, prediction, reference, completed, settings, seed+4)
        row["work"] = dict(total_nfe=int(nfe.sum()), total_pgs_sweeps=int(sweeps.sum()),
                           pgs_sweeps=scalar_distribution(sweeps))
        if measure_time:
            seconds = sum(statistics.median(values) for values in chunk_timings)
            row["timing"] = dict(chunk_seconds=chunk_timings, total_chunk_median_seconds=seconds,
                                  amortized_seconds_per_sample=seconds/len(indices), warmups_per_chunk=1)
        report["methods"][name] = row
        predictions[name] = prediction
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(names=split["names"], group_index=split["group_index"], group_totals=split["group_totals"],
                        problem=split["problem"], spec=specification(config), split=split["split"],
                        samples_per_qp=settings["samples_per_qp"], predictions=predictions), output.with_suffix(".pt"))
        write_json(output, report)
        write_distribution_table(report, output.with_suffix(".md"))
        if settings["render"]:
            render_distribution(report, split, predictions, output.with_suffix(".png"))
    return report


def write_distribution_table(report, path):
    lines = ["# Conditional solution distribution", "",
             "Synthetic redundant contact QPs. Different multipliers represent the same optimal position.",
             "Metrics are computed within each QP, then averaged. W1 uses all samples; inspect KKT validity too.",
             "reference_mc is the independent finite-sample reference floor; collapsed_exact solves KKT but has no diversity.", "",
             "| Method | KKT % | Position % | Sliced W1 | Variance / target | Mean position RMSE | Solver ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["methods"].items():
        a = row["balanced"]
        time = row.get("timing", {}).get("total_chunk_median_seconds")
        ms = f"{1000*time:.2f}" if time is not None else "n/a"
        lines.append(f"| {name} | {100*a['success_rate']:.2f} | {100*a['position_within_tolerance_rate']:.2f} | "
                     f"{a['sliced_w1']:.5g} | {a['variance_ratio']:.4f} | {a['position_rmse_mean']:.5g} | {ms} |")
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")


def render_distribution(report, split, predictions, path):
    """Render outside the solver process, keeping plotting runtimes isolated.

    Some Windows Conda installations ship different OpenMP libraries with
    PyTorch and Matplotlib/NumPy. A fresh plotting process avoids loading both.
    Only JSON data is sent to the renderer, never executable shell text.
    """
    n = report["samples_per_qp"]
    group = split["group_index"][0]
    coordinates = torch.where(group == 0)[0][:2]
    scale = split["group_totals"][0, 0]
    name = f"cfm_k{max(report['calls'])}_raw"
    payload = dict(report=report, method=name, path=str(Path(path).resolve()),
        reference=(predictions["reference"][:n, coordinates]/scale).tolist(),
        generated=(predictions[name][:n, coordinates]/scale).tolist())
    renderer = Path(__file__).with_name("distribution_plot.py")
    subprocess.run([sys.executable, str(renderer)], input=json.dumps(payload), text=True, check=True)


def load_distribution_model(path, config, selected_device):
    checkpoint = torch.load(path, map_location=selected_device, weights_only=True)
    if checkpoint.get("format") != FORMAT or checkpoint.get("spec") != specification(config):
        raise ValueError("Distribution checkpoint format/spec differs; use its original variant and data/model settings")
    model = ConditionalField(**config["model"]).to(selected_device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def train_distribution(config, selected_device, *, max_updates=None):
    validate_distribution(config)
    torch.set_num_threads(config["cpu_threads"])
    settings = config["distribution"]["train"]
    optimizer_settings = config["train"]
    budget = settings["max_updates"] if max_updates is None else max_updates
    if type(budget) is not int or budget < 1:
        raise ValueError("Positive max_updates required")
    run = Path(config["outdir"])/"cfm"
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {run}; choose a new --outdir")
    training = build_distribution_split(config, "train")
    validation = build_distribution_split(config, "val")
    torch.manual_seed(config["seed"])
    model = ConditionalField(**config["model"]).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_settings["lr"], weight_decay=optimizer_settings["weight_decay"])
    generators = [torch.Generator().manual_seed(config["seed"]+offset) for offset in (100000, 200000, 300000, 400000, 600000)]
    context_rng, source_rng, target_rng, tau_rng, inner_rng = generators
    history, best_cfm, best_distribution = [], float("inf"), (float("inf"), float("inf"))
    totals, count, field_calls = {}, 0, 0
    began = perf_counter()
    run.mkdir(parents=True, exist_ok=True)
    write_json(run/"experiment.json", dict(spec=specification(config), config=config,
        labels=f"Analytic {config['distribution']['target']} targets; no PGS labels or trajectories",
        model="Unchanged D ConditionalField", inner_source="Same random source prior as CFM/deployment",
        selection="Validation only: highest raw KKT success, then lowest conditional sliced W1"))
    for update in range(1, budget+1):
        model.train()
        indices = torch.randint(len(training["names"]), (optimizer_settings["batch_size"],), generator=context_rng)
        problem, source, target = sample_batch(training, indices, config, source_rng, target_rng)
        problem = move(problem, selected_device, torch.float32)
        source, target = (x.to(device=selected_device, dtype=torch.float32) for x in (source, target))
        tau = torch.rand(len(indices), generator=tau_rng)*(1-optimizer_settings.get("tau_min_remaining", .001))
        tau[torch.rand(len(indices), generator=tau_rng) < optimizer_settings["tau_zero_fraction"]] = 0
        loss, terms, inner_calls = distribution_loss(model, problem, source, target, tau.to(selected_device), settings, inner_rng)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite distribution loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_settings["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.)+float(value)
        count += 1
        field_calls += len(indices)*(1+inner_calls)
        if update % settings["validate_every"] and update != budget:
            continue
        model.eval()
        with torch.no_grad():
            val_indices = torch.arange(len(validation["names"])).repeat_interleave(4)
            qp, a, b = sample_batch(validation, val_indices, config,
                torch.Generator().manual_seed(config["seed"]+700000), torch.Generator().manual_seed(config["seed"]+700001))
            val_loss, _ = cfm_loss(model, move(qp, selected_device, torch.float32), a.to(selected_device).float(),
                b.to(selected_device).float(), torch.full((len(a),), .5, device=selected_device))
        diagnostic = evaluate_distribution(model, validation, config, selected_device,
            calls=[settings["validation_calls"]], samples=settings["validation_samples"], baselines=False, measure_time=False)
        stats = diagnostic["methods"][f"cfm_k{settings['validation_calls']}_raw"]["balanced"]
        row = dict(updates=update, train={key: value/count for key, value in totals.items()}, val_cfm=float(val_loss),
                   validation=stats, reference_mc=diagnostic["methods"]["reference_mc"]["balanced"],
                   elapsed_seconds=perf_counter()-began, training_field_evaluations=field_calls)
        if not all(math.isfinite(value) for value in [float(val_loss), *stats.values()]):
            raise FloatingPointError("Nonfinite validation metrics")
        history.append(row)
        checkpoint = dict(format=FORMAT, spec=specification(config), config=config, updates=update,
                          model=model.state_dict(), optimizer=optimizer.state_dict(), validation=row,
                          selection_metric="last update")
        torch.save(checkpoint, run/"last.pt")
        if float(val_loss) < best_cfm:
            best_cfm = float(val_loss)
            torch.save(dict(checkpoint, selection_metric="fixed validation CFM loss"), run/"best.pt")
        score = (-stats["success_rate"], stats["sliced_w1"])
        if score < best_distribution:
            best_distribution = score
            torch.save(dict(checkpoint, selection_metric="raw validation KKT success, then conditional sliced W1"), run/"best_distribution.pt")
        write_json(run/"history.json", history)
        print(f"[{update}/{budget}] cfm={row['train']['cfm']:.5g} val_KKT={stats['success_rate']:.3f} "
              f"SW1={stats['sliced_w1']:.5g} variance={stats['variance_ratio']:.3f}", flush=True)
        totals, count = {}, 0
    return dict(checkpoint=str(run/"best_distribution.pt"), updates=budget)
