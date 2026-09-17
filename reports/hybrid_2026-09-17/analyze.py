"""Audit the supplied hybrid evaluation and create compact analysis artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def summarize_method(method):
    rows = method["per_scene"]
    return {
        "accuracy": method["accuracy"],
        "timing": method["timing"],
        "cost": method["cost"],
        "hybrid": method.get("hybrid"),
        "sweep_histogram": dict(sorted(Counter(r["sweeps"] for r in rows).items())),
        "position_misses": [
            {k: r[k] for k in ("name", "position_rmse", "sweeps")}
            for r in rows if r["position_rmse"] > method["accuracy"]["position_tolerance"]
        ],
    }


def audit(report):
    count = len(report["scenes"])
    checked_methods = 0
    prefix_residual_differences = []
    for mode, methods in report["modes"].items():
        for name, method in methods.items():
            rows = method["per_scene"]
            assert [r["name"] for r in rows] == report["scenes"], (mode, name)
            assert sum(r["success_count"] for r in rows) == method["success_count"]
            assert sum(r["completed"] for r in rows) == method["completed_count"]
            close(statistics.median(method["timing"]["batch_seconds"]),
                  method["timing"]["median_seconds"])
            for metric in ("projected_gradient", "position_rmse", "objective_gap"):
                assert all(math.isfinite(r[metric]) for r in rows)
                close(statistics.mean(r[metric] for r in rows), method["accuracy"][metric]["mean"])
            for metric, total in method["cost"]["logical_totals"].items():
                assert sum(r.get(metric, 0) for r in rows) == total
            if name.startswith("cfm_k"):
                calls = int(name.split("_")[1][1:])
                assert all(r["nfe"] == calls and r["completed"] for r in rows)
            if name.startswith("pgs_k"):
                sweeps = int(name.split("_")[1][1:])
                assert all(r["sweeps"] == sweeps for r in rows)
            checked_methods += 1
        for calls in report["hybrid"]["calls"]:
            hybrid = methods[f"hybrid_k{calls}_pgs"]
            raw = methods[f"cfm_k{calls}_raw"]
            assert hybrid["success_count"] == methods["pgs"]["success_count"] == count
            assert hybrid["hybrid"]["cfm_failure_count"] == 0
            assert hybrid["hybrid"]["pgs_budget_exhausted_count"] == 0
            for r, prefix in zip(hybrid["per_scene"], raw["per_scene"]):
                assert r["nfe"] == calls and r["cfm_completed"]
                # Hybrid prefix residuals use the whole batch; raw per-scene
                # residuals are recomputed with batch size one in float32.
                difference = abs(r["cfm_projected_gradient"] - prefix["projected_gradient"])
                prefix_residual_differences.append(difference)
                assert difference <= 1e-7, (mode, calls, r["name"], difference)
                close(r["cfm_position_rmse"], prefix["position_rmse"])
                assert r["cfm_converged"] == bool(prefix["success_count"])
                assert (r["sweeps"] == 0) == r["cfm_converged"]
                if not r["sweeps"]:
                    close(r["position_rmse"], r["cfm_position_rmse"])
            close(report["hybrid_comparison"][mode][str(calls)]["speedup_at_full_success"],
                  methods["pgs"]["timing"]["median_seconds"] / hybrid["timing"]["median_seconds"])
    return {"methods_checked": checked_methods, "rows_checked": checked_methods * count,
            "hybrid_prefix_rows_checked": count * len(report["modes"]) * len(report["hybrid"]["calls"]),
            "max_prefix_residual_reporting_difference": max(prefix_residual_differences),
            "all_checks_passed": True,
            "scope": "Saved aggregates, scene order, work counts, timings and prefix metrics; not an independent solver rerun."}


def make_figure(report, output):
    methods = report["modes"]["zero"]
    keys = ["pgs", "hybrid_k4_pgs", "hybrid_k8_pgs", "hybrid_k16_pgs"]
    labels = ["PGS", "FM4 + PGS", "FM8 + PGS", "FM16 + PGS"]
    colors = ["#64748b", "#087f8c", "#5c6bc0", "#bc6c25"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    fig.suptitle("D hybrid evaluation | zero start | 768 frozen contact QPs", fontsize=15)

    ax = axes[0, 0]
    medians = [1000 * methods[k]["timing"]["median_seconds"] for k in keys]
    ax.barh(labels, medians, color=colors, height=.6)
    for i, (key, median) in enumerate(zip(keys, medians)):
        t = methods[key]["timing"]
        ax.plot([1000*t["min_seconds"], 1000*t["max_seconds"]], [i, i], color="black", lw=2)
        ax.text(median + 5, i, f"{median:.2f} ms", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, 242)
    ax.set_title("A. Full solve time: all four solve 768/768")
    ax.set_xlabel("Batch median ms; black lines = min/max of 5 repeats")

    ax = axes[0, 1]
    grid = np.arange(99)
    for key, label, color in zip(keys, labels, colors):
        sweeps = np.array([r["sweeps"] for r in methods[key]["per_scene"]])
        ax.step(grid, [(sweeps <= s).mean()*100 for s in grid], where="post", label=label, color=color)
    ax.set_ylim(0, 103)
    ax.set_xlabel("PGS sweeps (hybrids first pay their FM cost)")
    ax.set_ylabel("QPs reaching the KKT tolerance (%)")
    ax.set_title("B. Distribution of remaining PGS work")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.2)

    ax = axes[1, 0]
    positions = np.arange(len(keys))
    means = [methods[k]["accuracy"]["position_rmse"]["mean"]*1e3 for k in keys]
    p95 = [methods[k]["accuracy"]["position_rmse"]["p95"]*1e3 for k in keys]
    ax.bar(positions-.17, means, width=.34, label="Mean", color="#087f8c")
    ax.bar(positions+.17, p95, width=.34, label="p95", color="#96b8be")
    ax.axhline(1, ls="--", color="#bc6c25", label="Position tolerance")
    ax.set_xticks(positions, labels, rotation=12)
    ax.set_ylabel("Position RMSE against FP64 reference (x 1e-3)")
    ax.set_title("C. Position accuracy is separate from KKT success")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    groups = [f"vertical_stack_n{n}" for n in (3, 5, 8, 10)]
    x = np.arange(len(groups))
    for offset, key, label, color in [(-.18, keys[0], labels[0], colors[0]), (.18, keys[1], labels[1], colors[1])]:
        values = [statistics.mean(r["sweeps"] for r in methods[key]["per_scene"]
                                  if r["name"].rsplit("_", 1)[0] == group) for group in groups]
        ax.bar(x+offset, values, width=.36, color=color, label=label)
        for xx, value in zip(x+offset, values):
            ax.text(xx, value+.7, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["n=3", "n=5", "n=8", "n=10"])
    ax.set_ylim(0, 65)
    ax.set_ylabel("Mean PGS sweeps per QP")
    ax.set_title("D. Vertical stacks: 43 QPs per size")
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    raw_bytes = args.input.read_bytes()
    report = json.loads(raw_bytes)
    output = Path(__file__).resolve().parent
    summary = {"source": str(args.input.resolve()), "sha256": hashlib.sha256(raw_bytes).hexdigest(),
               "validation": audit(report), "environment": report["environment"],
               "metadata": report["metadata"], "hybrid_settings": report["hybrid"],
               "modes": {}, "groups_zero": {}}
    keys = ["pgs"] + [f"hybrid_k{k}_pgs" for k in report["hybrid"]["calls"]]
    for mode, methods in report["modes"].items():
        summary["modes"][mode] = {key: summarize_method(methods[key]) for key in keys}
        base = methods["pgs"]["per_scene"]
        for key in keys[1:]:
            paired = list(zip(methods[key]["per_scene"], base))
            summary["modes"][mode][key]["paired"] = {
                "fewer_pgs_sweeps": sum(a["sweeps"] < b["sweeps"] for a, b in paired),
                "more_pgs_sweeps": sum(a["sweeps"] > b["sweeps"] for a, b in paired),
                "lower_position_rmse": sum(a["position_rmse"] < b["position_rmse"] for a, b in paired),
                "higher_position_rmse": sum(a["position_rmse"] > b["position_rmse"] for a, b in paired),
            }
    for group in report["modes"]["zero"]["pgs"]["groups"]:
        summary["groups_zero"][group] = {}
        for key in keys:
            method = report["modes"]["zero"][key]
            rows = [r for r in method["per_scene"] if r["name"].rsplit("_", 1)[0] == group]
            summary["groups_zero"][group][key] = {
                "accuracy": method["groups"][group]["accuracy"],
                "mean_pgs_sweeps": statistics.mean(r["sweeps"] for r in rows),
                "max_pgs_sweeps": max(r["sweeps"] for r in rows),
            }
    summary["comparisons"] = report["hybrid_comparison"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    make_figure(report, output / "comparison.png")
    print(json.dumps(summary["validation"]))
    print("Source SHA-256:", summary["sha256"])
    print("Artifacts:", output)


if __name__ == "__main__":
    main()
