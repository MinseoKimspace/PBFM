"""Standalone Matplotlib renderer. Reads a JSON payload from standard input."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(payload):
    report = payload["report"]
    budgets = sorted(report["calls"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.suptitle("Conditional solution distribution (synthetic redundant QPs)")
    for metric, ax, ylabel in (("success_rate", axes[0, 0], "KKT success fraction"),
                                ("sliced_w1", axes[0, 1], "Conditional empirical sliced W1")):
        for prefix, suffix, label in (("cfm", "_raw", "CFM"), ("local", "", "Analytic head only"),
                                      ("cfm", "_hybrid", "CFM + PGS")):
            keys = [f"{prefix}_k{k}{suffix}" for k in budgets]
            if all(key in report["methods"] for key in keys):
                ax.plot(budgets, [report["methods"][key]["balanced"][metric] for key in keys], "o-", label=label)
        for baseline, color in (("reference_mc", "green"), ("collapsed_exact", "gray"), ("pgs", "orange")):
            if baseline in report["methods"]:
                ax.axhline(report["methods"][baseline]["balanced"][metric], label=baseline, color=color, ls="--")
        ax.set(xlabel="Full-interval update budget K", ylabel=ylabel)
        ax.legend(fontsize=8)
    reference, generated = (np.asarray(payload[key]) for key in ("reference", "generated"))
    axes[1, 0].hist(reference[:, 0], bins=20, alpha=.5, label="reference")
    axes[1, 0].hist(generated[:, 0], bins=20, alpha=.5, label=payload["method"])
    axes[1, 0].set(xlabel="Multiplier / exact group total", ylabel="Samples", title="First QP, first duplicate group")
    axes[1, 0].legend()
    axes[1, 1].scatter(reference[:, 0], reference[:, 1], s=9, alpha=.4, label="reference")
    axes[1, 1].scatter(generated[:, 0], generated[:, 1], s=9, alpha=.4, label=payload["method"])
    axes[1, 1].set(xlabel="Normalized duplicate 1", ylabel="Normalized duplicate 2", title="First QP: two entries in the same group")
    axes[1, 1].legend()
    fig.savefig(payload["path"], dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    render(json.load(sys.stdin))
