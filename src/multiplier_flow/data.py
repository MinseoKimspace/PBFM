"""Shared finite-time segment labels. B and C read the exact same cache."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import math

import torch

from src.contact_flow.dynamics import free_position
from src.contact_flow.io import load_states
from src.contact_flow.physics import PhysicsConfig
from .problem import circle_problem, converged, make_problem, pack, residuals, select
from .solvers import pgs, reference


CACHE_FORMAT = "multiplier_segments_v2"
START_MODES = ("zero", "under", "over", "mixed", "near")


def sample_segments(problem, optimum, settings, rng):
    """Balanced starts, independently sampled clocks, and one horizon check/world.

    Never repeat J/D for the entire segment dataset. Those tensors are selected
    only in small reference/training batches. Anchor p is never perturbed.
    """
    per_scene = int(settings["segments_per_scene"])
    if per_scene < len(START_MODES):
        raise ValueError("Use >=5 segments to include all start types")
    indices = torch.arange(len(optimum)).repeat_interleave(per_scene)
    slot = torch.arange(len(indices)) % per_scene
    mode = slot % len(START_MODES)
    exact = optimum[indices]
    diagonal = problem["D"].diagonal(dim1=1, dim2=2)[indices].clamp_min(1e-12)
    random = torch.rand(exact.shape, generator=rng, dtype=exact.dtype)
    perturbation = (0.02 / diagonal) * random
    start = exact.clone()
    start[mode == 0] = 0
    start[mode == 1] *= (0.1 + 0.7 * random[mode == 1])
    start[mode == 2] = (1.1 + random[mode == 2]) * exact[mode == 2] + perturbation[mode == 2]
    start[mode == 3] = (0.2 + 2 * random[mode == 3]) * exact[mode == 3] + perturbation[mode == 3]
    # Both sides of the solution, including small positive inactive multipliers.
    near = mode == 4
    start[near] = ((0.95 + 0.1 * random[near]) * exact[near]
                   + 0.002 / diagonal[near] * (2 * random[near] - 1)).clamp_min(0)
    start *= problem["mask"][indices]
    low, high = map(float, settings["h_range"])
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Require finite 0 < h_min <= h_max")
    duration = torch.empty(len(indices), dtype=exact.dtype).uniform_(
        math.log(low), math.log(high), generator=rng).exp_()
    anchors = settings.get("time_anchors", [])
    fraction = float(settings.get("anchor_fraction", 0.5))
    if not 0 <= fraction <= 1 or any(not low <= t <= high for t in anchors):
        raise ValueError("Invalid time anchors/fraction")
    if anchors:
        use = torch.rand(len(indices), generator=rng) < fraction
        choice = torch.randint(len(anchors), (int(use.sum()),), generator=rng)
        duration[use] = torch.tensor(anchors, dtype=exact.dtype)[choice]
    horizon_check = slot == 0
    duration[horizon_check] = high
    return dict(context=indices, start=start, duration=duration,
                start_mode=mode, horizon_check=horizon_check)


def release_problem(a=0.1, b=0.3, eta_fraction=0.9):
    """x>=a, x+y>=b; for b>2a the first multiplier must be released."""
    return make_problem(torch.zeros(2, dtype=torch.float64),
                        torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64),
                        torch.ones(2, dtype=torch.float64),
                        torch.tensor([-a, -b], dtype=torch.float64), eta_fraction)


def build_split(config, split):
    began = perf_counter()
    settings, ref = config["data"], config["reference"]
    seed = int(config["seed"]) + {"train": 0, "val": 10000, "test": 20000}[split]
    rng = torch.Generator().manual_seed(seed)
    physics = PhysicsConfig(**config["physics"])
    problems, names, radii = [], [], []
    for size in settings[split + "_sizes"]:
        count = int(settings[split + "_per_size"])
        state, radius, kinds = load_states("", split, count, seed + size, size,
                                          physics=physics, return_scene_types=True)
        proposal = free_position(state.double(), **config["dynamics"])
        for i in range(count):
            problems.append(circle_problem(proposal[i], radius[i].double(), physics, ref["eta_fraction"]))
            names.append(f"{kinds[i]}_n{size}_{i}")
            radii.append(radius[i].double())
    for i in range(int(settings["release_count"])):
        a = float(0.05 + 0.1 * torch.rand((), generator=rng))
        b = a * float(2.5 + torch.rand((), generator=rng))
        problems.append(release_problem(a, b, ref["eta_fraction"]))
        names.append(f"release_{i}")
        radii.append(torch.empty(0, dtype=torch.float64))
    problem = pack(problems)
    chunk = int(settings.get("reference_batch_size", 128))
    if chunk < 1:
        raise ValueError("Positive reference_batch_size required")
    optimum = torch.zeros_like(problem["c"])
    for index in torch.arange(len(problems)).split(chunk):
        result = pgs(select(problem, index), optimum[index], ref["qp_tolerance"], ref["max_sweeps"])
        if not result["converged"].all():
            bad = index[~result["converged"]].tolist()
            raise RuntimeError(f"{split} QP oracle did not converge: {[names[i] for i in bad]}")
        optimum[index] = result["final"]
    segments = sample_segments(problem, optimum, settings, rng)
    target = torch.empty_like(segments["start"])
    nfe = torch.zeros(len(target), dtype=torch.long)
    rejected = torch.zeros_like(nfe)
    for index in torch.arange(len(target)).split(chunk):
        result = reference(select(problem, segments["context"][index]),
                           segments["start"][index], segments["duration"][index],
                           ref["rtol"], ref["atol"], ref["max_steps"])
        target[index], nfe[index], rejected[index] = result["final"], result["nfe"], result["rejected"]
        print(f"  {split}: {int(index[-1])+1}/{len(target)} segments, {perf_counter()-began:.1f}s", flush=True)
    return dict(problem=problem, names=names, radii=radii, optimum=optimum,
                **segments, target=target, reference_nfe=nfe, reference_rejected=rejected,
                preparation_seconds=perf_counter()-began)


def prepare(config, path):
    path = Path(path)
    spec = {key: config[key] for key in ("seed", "data", "physics", "dynamics", "reference")}
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        # Old caches remain readable for old-checkpoint evaluation, never rewritten.
        if cache.get("format") not in {CACHE_FORMAT, "multiplier_segments_v1"} or cache.get("spec") != spec:
            raise ValueError("Segment cache configuration differs; use a new output directory")
        return cache
    splits = {}
    parts = path.parent / (path.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        part = parts / f"{split}.pt"
        if part.exists():
            saved = torch.load(part, map_location="cpu", weights_only=True)
            if saved.get("spec") != spec or saved.get("format") != CACHE_FORMAT:
                raise ValueError("Partial cache differs; choose a new output directory")
            splits[split] = saved["split"]
        else:
            print(f"Preparing {split} finite-time segments (CPU float64)...", flush=True)
            splits[split] = build_split(config, split)
            temporary = part.with_suffix(".tmp")
            torch.save(dict(format=CACHE_FORMAT, spec=spec, split=splits[split]), temporary)
            temporary.replace(part)
    cache = dict(format=CACHE_FORMAT, spec=spec, splits=splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(cache, temporary)
    temporary.replace(path)
    return cache


def cache_summary(cache, tolerance):
    """Finite-time reference quality, including separate nontrivial scene groups."""
    report = {}
    for name, split in cache["splits"].items():
        horizon = split.get("horizon_check", split["start_mode"] == 0)
        indices = split["context"][horizon]
        selected = select(split["problem"], indices)
        success = converged(selected, split["target"][horizon], tolerance)
        groups = {}
        for i, context in enumerate(indices):
            group = split["names"][int(context)].rsplit("_", 1)[0]
            row = groups.setdefault(group, dict(scenes=0, solved=0))
            row["scenes"] += 1
            row["solved"] += int(success[i])
        report[name] = dict(scenes=len(split["names"]), segments=len(split["context"]),
                            horizon=float(split["duration"][horizon].max()),
                            horizon_solved=int(success.sum()), groups=groups,
                            horizon_failed_scenes=[split["names"][int(i)] for i in indices[~success]],
                            horizon_max_projected_gradient=float(residuals(
                                selected, split["target"][horizon])["projected_gradient"].max()),
                            start_counts={label: int((split["start_mode"] == i).sum())
                                          for i, label in enumerate(START_MODES)},
                            preparation_seconds=split.get("preparation_seconds"),
                            reference_nfe=int(split["reference_nfe"].sum()),
                            reference_rejected=int(split["reference_rejected"].sum()))
    return report


def batch(split, indices, device):
    selected = select(split["problem"], split["context"][indices])
    problem = {key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else None)
               for key, value in selected.items()}
    tensors = [split[key][indices].to(device=device, dtype=torch.float32)
               for key in ("start", "duration", "target")]
    return problem, *tensors
