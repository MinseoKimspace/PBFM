"""Shared finite-time segment labels. B and C read the exact same cache."""
from __future__ import annotations

from pathlib import Path

import torch

from src.contact_flow.dynamics import free_position
from src.contact_flow.io import load_states
from src.contact_flow.physics import PhysicsConfig
from .problem import circle_problem, converged, make_problem, pack, select
from .solvers import pgs, reference


CACHE_FORMAT = "multiplier_segments_v1"


def release_problem(a=0.1, b=0.3, eta_fraction=0.9):
    """x>=a, x+y>=b; for b>2a the first multiplier must be released."""
    return make_problem(torch.zeros(2, dtype=torch.float64),
                        torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64),
                        torch.ones(2, dtype=torch.float64),
                        torch.tensor([-a, -b], dtype=torch.float64), eta_fraction)


def build_split(config, split):
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
    optimum = pgs(problem, torch.zeros_like(problem["c"]), ref["qp_tolerance"], ref["max_sweeps"])
    if not optimum["converged"].all():
        bad = (~optimum["converged"]).nonzero().flatten().tolist()
        raise RuntimeError(f"{split} QP oracle did not converge: {[names[i] for i in bad]}")
    per_scene = int(settings["segments_per_scene"])
    if per_scene < 5:
        raise ValueError("Use >=5 segments to include zero/under/over/mixed/near-solved starts")
    indices = torch.arange(len(problems)).repeat_interleave(per_scene)
    exact = optimum["final"][indices]
    selected = select(problem, indices)
    diagonal = selected["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
    # Perturb multipliers, never the anchor p. Off-path starts get freshly solved
    # trajectories, not the tangent of an unrelated clean starting state.
    perturbation = (0.02 / diagonal) * torch.rand(exact.shape, generator=rng, dtype=exact.dtype)
    mode = torch.arange(len(indices)) % per_scene
    start = exact.clone()
    start[mode == 0] = 0
    start[mode == 1] *= 0.4
    start[mode == 2] = 1.7 * exact[mode == 2] + perturbation[mode == 2]
    mixed = mode == 3
    start[mixed] = exact[mixed] * (0.2 + 2 * torch.rand(exact[mixed].shape, generator=rng, dtype=exact.dtype)) + perturbation[mixed]
    start *= selected["mask"]
    low, high = map(float, settings["h_range"])
    if not 0 < low <= high:
        raise ValueError("Require 0 < h_min <= h_max")
    duration = torch.exp(torch.empty(len(indices), dtype=torch.float64).uniform_(
        torch.log(torch.tensor(low)).item(), torch.log(torch.tensor(high)).item(), generator=rng))
    duration[mode == 0] = high  # Every scene explicitly sees the common evaluation horizon.
    result = reference(selected, start, duration, ref["rtol"], ref["atol"], ref["max_steps"])
    return dict(problem=problem, names=names, radii=radii, optimum=optimum["final"],
                context=indices, start=start, duration=duration, target=result["final"],
                reference_nfe=result["nfe"], reference_rejected=result["rejected"],
                start_mode=mode)


def prepare(config, path):
    path = Path(path)
    spec = {key: config[key] for key in ("seed", "data", "physics", "dynamics", "reference")}
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache.get("format") != CACHE_FORMAT or cache.get("spec") != spec:
            raise ValueError("Segment cache configuration differs; use a new output directory")
        return cache
    splits = {}
    for split in ("train", "val", "test"):
        print(f"Preparing {split} finite-time segments (CPU float64)...", flush=True)
        splits[split] = build_split(config, split)
    cache = dict(format=CACHE_FORMAT, spec=spec, splits=splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


def cache_summary(cache, tolerance):
    """Finite-time reference quality, including separate nontrivial scene groups."""
    report = {}
    for name, split in cache["splits"].items():
        horizon = split["start_mode"] == 0
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
