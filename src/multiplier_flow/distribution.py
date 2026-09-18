"""Analytic conditional solution distributions for redundant contact QPs.

Each full-rank chain constraint is repeated r times. The sum of its multipliers
is uniquely determined, while their nonnegative allocation is free. Targets
are independent uniform simplex allocations, NOT numerical solver outputs.
Only ordinary QP tensors enter the unchanged D network; group labels and exact
totals are used exclusively for target sampling and evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import math
from pathlib import Path

import torch

from .experiment import resolve_experiment
from .problem import make_problem, pack, select


FORMAT = "multiplier_distribution_v1"
VARIANTS = ("D_distribution", "D_distribution_point")
SPLIT_SEEDS = {"train": 0, "val": 10000, "test": 20000}


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def integer_list(values, name, minimum=1):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(v) is not int or v < minimum for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must contain distinct integers >= {minimum}")


def resolve_distribution(config, variant="D_distribution", outdir=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown distribution variant: {variant}")
    # Reuse exactly D's model settings without adding a variant to the existing
    # single-endpoint training/evaluation CLIs and their shared pair cache.
    resolved = resolve_experiment(config, "D", outdir)
    resolved["variant"] = variant
    resolved["outdir"] = str(Path(resolved["outdir"]).parent / variant)
    resolved["distribution"] = deepcopy(config["distribution"])
    resolved["distribution"]["target"] = "point" if variant.endswith("_point") else "uniform_simplex"
    validate_distribution(resolved)
    return resolved


def validate_distribution(config):
    options = config["distribution"]
    data, train, evaluation = (options[k] for k in ("data", "train", "evaluation"))
    if options["target"] not in ("uniform_simplex", "point"):
        raise ValueError("Unknown distribution target")
    integer_list(data["base_contacts"], "distribution.data.base_contacts")
    integer_list(data["duplicates"], "distribution.data.duplicates", minimum=2)
    for split in SPLIT_SEEDS:
        positive_integer(data[f"{split}_per_setting"], f"{split}_per_setting")
    for key in ("inverse_mass_range", "total_multiplier_range"):
        values = data[key]
        if (not isinstance(values, (list, tuple)) or len(values) != 2
                or not all(math.isfinite(v) for v in values) or not 0 < values[0] <= values[1]):
            raise ValueError(f"{key} must be [positive minimum, maximum]")
    if not math.isfinite(options["source_scale"]) or options["source_scale"] <= 0:
        raise ValueError("distribution.source_scale must be positive")
    for key in ("max_updates", "validate_every", "validation_samples", "validation_calls"):
        positive_integer(train[key], f"distribution.train.{key}")
    positive_integer(config["train"]["batch_size"], "train.batch_size")
    for key in ("lr", "grad_clip"):
        if not math.isfinite(config["train"][key]) or config["train"][key] <= 0:
            raise ValueError(f"train.{key} must be positive")
    if not math.isfinite(config["train"]["weight_decay"]) or config["train"]["weight_decay"] < 0:
        raise ValueError("train.weight_decay must be nonnegative")
    if not 0 < config["train"].get("tau_min_remaining", .001) < 1:
        raise ValueError("train.tau_min_remaining must be in (0,1)")
    if not 0 <= config["train"]["tau_zero_fraction"] < 1:
        raise ValueError("train.tau_zero_fraction must be in [0,1)")
    inner = train["inner_rollout"]
    integer_list(inner["calls"], "distribution.train.inner_rollout.calls")
    for key in ("weight", "residual_weight", "position_weight"):
        if not math.isfinite(inner[key]) or inner[key] < 0:
            raise ValueError(f"distribution inner {key} must be nonnegative")
    integer_list(evaluation["calls"], "distribution.evaluation.calls")
    for key in ("samples_per_qp", "batch_size", "projections", "timing_repeats"):
        positive_integer(evaluation[key], f"distribution.evaluation.{key}")
    if min(evaluation["samples_per_qp"], train["validation_samples"]) < 2:
        raise ValueError("Distribution evaluation requires at least two samples per QP")
    if type(evaluation["hybrid"]) is not bool or type(evaluation["render"]) is not bool:
        raise ValueError("distribution evaluation hybrid/render must be boolean")
    for key in ("tolerance", "position_tolerance"):
        if not math.isfinite(evaluation[key]) or evaluation[key] <= 0:
            raise ValueError(f"distribution.evaluation.{key} must be positive")
    positive_integer(config["reference"]["max_sweeps"], "reference.max_sweeps")
    model = config["model"]
    if (model.get("head_type") != "analytic" or model.get("communication") != "global"
            or model.get("feature_version") != "residual"):
        raise ValueError("Distribution experiment requires D's global analytic residual model")


def specification(config):
    """Semantics checked when loading weights; evaluation budgets may change."""
    return dict(format=FORMAT, seed=config["seed"], data=deepcopy(config["distribution"]["data"]),
                target=config["distribution"]["target"], source_scale=config["distribution"]["source_scale"],
                eta_fraction=config["reference"]["eta_fraction"], model=deepcopy(config["model"]))


def build_distribution_split(config, split):
    settings = config["distribution"]["data"]
    rng = torch.Generator().manual_seed(config["seed"] + SPLIT_SEEDS[split])
    problems, groups, totals, names = [], [], [], []
    descriptions = []
    for n, copies in itertools.product(settings["base_contacts"], settings["duplicates"]):
        for index in range(settings[f"{split}_per_setting"]):
            low, high = settings["inverse_mass_range"]
            inverse_mass = low + (high-low)*torch.rand(n, generator=rng, dtype=torch.float64)
            low, high = settings["total_multiplier_range"]
            total = low + (high-low)*torch.rand(n, generator=rng, dtype=torch.float64)
            # Floor followed by relative contacts: d0>=..., di-d(i-1)>=...
            base_j = torch.eye(n, dtype=torch.float64)
            if n > 1:
                base_j[torch.arange(1, n), torch.arange(n-1)] = -1
            base_d = (base_j*inverse_mass) @ base_j.T
            gap = -(base_d @ total)
            order = torch.randperm(n*copies, generator=rng)
            group = torch.arange(n).repeat_interleave(copies)[order]
            problems.append(make_problem(torch.zeros(n, dtype=torch.float64),
                base_j.repeat_interleave(copies, dim=0)[order], inverse_mass,
                gap.repeat_interleave(copies)[order], config["reference"]["eta_fraction"]))
            groups.append(group)
            totals.append(total)
            names.append(f"chain_n{n}_duplicates{copies}_{index}")
            descriptions.append(dict(base_contacts=n, duplicates=copies))
    problem = pack(problems)
    group_index = torch.full(problem["c"].shape, -1, dtype=torch.long)
    group_totals = torch.zeros(len(problems), max(map(len, totals)), dtype=torch.float64)
    for i, (group, total) in enumerate(zip(groups, totals)):
        group_index[i, :len(group)] = group
        group_totals[i, :len(total)] = total
    split_data = dict(problem=problem, group_index=group_index, group_totals=group_totals,
                      names=names, descriptions=descriptions, split=split)
    split_data["canonical"] = target_samples(split_data, torch.arange(len(names)), rng, "point")
    return split_data


def target_samples(split, indices, rng, target="uniform_simplex"):
    """Independent Dirichlet(1,...,1) per group via normalized exponentials."""
    group = split["group_index"][indices]
    valid = group >= 0
    group = group.clamp_min(0)
    totals = split["group_totals"][indices]
    if target == "uniform_simplex":
        uniform = torch.rand(group.shape, generator=rng, dtype=totals.dtype)
        weights = -torch.log1p(-uniform.clamp_min(torch.finfo(totals.dtype).eps))
    elif target == "point":
        weights = torch.ones(group.shape, dtype=totals.dtype)
    else:
        raise ValueError(f"Unknown target: {target}")
    weights *= valid
    sums = torch.zeros_like(totals).scatter_add_(1, group, weights)
    return weights / sums.gather(1, group).clamp_min(1e-30) * totals.gather(1, group)


def source_samples(problem, rng, length_scale, source_scale):
    """Deployment prior uses only QP diagonal/scale, never analytic targets."""
    diagonal = problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
    noise = torch.rand(problem["c"].shape, dtype=problem["c"].dtype, generator=rng)
    return source_scale*length_scale/diagonal * noise * problem["mask"]


def sample_batch(split, indices, config, source_rng, target_rng):
    problem = select(split["problem"], indices)
    source = source_samples(problem, source_rng, config["model"]["length_scale"],
                            config["distribution"]["source_scale"])
    target = target_samples(split, indices, target_rng, config["distribution"]["target"])
    return problem, source, target


def conditional_distribution_metrics(prediction, reference, group_index, group_totals, projections, seed):
    """Compare distributions WITHIN one QP, normalized by its exact group totals.

    Equal-sized empirical sliced W1 probes the joint law. Samples outside the
    target support remain in this score; normalization never hides total error.
    A second independent reference sample supplies a finite-sample noise floor.
    """
    valid = group_index >= 0
    group = group_index[valid]
    scale = group_totals[group]
    x, y = prediction[:, valid].double()/scale, reference[:, valid].double()/scale
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Distribution metrics need equal sample counts >=2")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Distribution metrics require finite samples")
    rng = torch.Generator().manual_seed(seed)
    directions = torch.randn(x.shape[1], projections, dtype=torch.float64, generator=rng)
    directions /= directions.norm(dim=0, keepdim=True).clamp_min(1e-30)
    sliced = ((x @ directions).sort(dim=0).values - (y @ directions).sort(dim=0).values).abs().mean()
    marginal = (x.sort(dim=0).values-y.sort(dim=0).values).abs().mean()
    variance = x.var(dim=0, unbiased=False)
    copies = torch.bincount(group, minlength=len(group_totals)).double()
    # The exact marginal variance of a uniform r-simplex allocation.
    target_variance = (copies[group]-1)/(copies[group].square()*(copies[group]+1))
    group_sum = x.new_zeros(len(x), len(group_totals)).scatter_add_(1, group.expand(len(x), -1), x)
    present = copies > 0
    return dict(sliced_w1=float(sliced), marginal_w1=float(marginal),
                variance_ratio=float(variance.sum()/target_variance.sum()),
                group_total_relative_error=float((group_sum[:, present]-1).abs().mean()),
                mean_coordinate_bias=float((x.mean(0)-1/copies[group]).abs().mean()))
