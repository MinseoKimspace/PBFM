"""Source-to-solution pairs for CFM. No numerical trajectory integration."""
from __future__ import annotations

from pathlib import Path
import os
from time import perf_counter

import torch

from src.contact_flow.dynamics import free_position
from src.contact_flow.io import load_states
from src.contact_flow.physics import PhysicsConfig
from .problem import circle_problem, converged, make_problem, move, pack, residuals, select
from .solvers import pgs


CACHE_FORMAT = "multiplier_cfm_pairs_v1"
START_MODES = ("zero", "under", "over", "mixed", "near", "solved")


def sample_pairs(problem, optimum, settings, rng):
    """Keep the original anchor p fixed; every source targets its QP optimum."""
    count = int(settings["sources_per_scene"])
    if count < len(START_MODES):
        raise ValueError("Use >=6 sources per scene to include every start mode")
    context = torch.arange(len(optimum)).repeat_interleave(count)
    mode = (torch.arange(len(context)) % count) % len(START_MODES)
    target = optimum[context]
    diagonal = problem["D"].diagonal(dim1=1, dim2=2)[context].clamp_min(1e-12)
    random = torch.rand(target.shape, generator=rng, dtype=target.dtype)
    source = target.clone()
    source[mode == 0] = 0
    source[mode == 1] *= (0.1 + 0.7 * random[mode == 1])
    source[mode == 2] = ((1.1 + random[mode == 2]) * target[mode == 2]
                         + .02 / diagonal[mode == 2] * random[mode == 2])
    source[mode == 3] = ((.2 + 2 * random[mode == 3]) * target[mode == 3]
                         + .02 / diagonal[mode == 3] * random[mode == 3])
    near = mode == 4
    source[near] = ((.95 + .1 * random[near]) * target[near]
                    + .002 / diagonal[near] * (2 * random[near] - 1)).clamp_min(0)
    source *= problem["mask"][context]
    return dict(context=context, source=source, target=target, start_mode=mode)


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
    # Include gravity-sized corrections/resting-floor precision. These random
    # contexts are not the held-out dynamic rollout trajectories.
    for i in range(int(settings["floor_count"])):
        r = .4 + .2 * torch.rand(1, generator=rng, dtype=torch.float64)
        depth = 10. ** (-5 + 4.3 * float(torch.rand((), generator=rng)))
        if i % 4 == 0:
            depth = -depth
        p = torch.tensor([[0., physics.y_ground + float(r[0]) - physics.slop - depth]], dtype=torch.float64)
        problems.append(circle_problem(p, r, physics, ref["eta_fraction"]))
        names.append(f"floor_{i}")
        radii.append(r)
    for i in range(int(settings["release_count"])):
        a = float(.05 + .1 * torch.rand((), generator=rng))
        b = a * float(2.5 + torch.rand((), generator=rng))
        problems.append(release_problem(a, b, ref["eta_fraction"]))
        names.append(f"release_{i}")
        radii.append(torch.empty(0, dtype=torch.float64))
    problem = pack(problems)
    chunk = int(settings["reference_batch_size"])
    if chunk < 1:
        raise ValueError("Positive reference batch size required")
    optimum = torch.zeros_like(problem["c"])
    sweeps = torch.zeros(len(problems), dtype=torch.long)
    for index in torch.arange(len(problems)).split(chunk):
        result = pgs(select(problem, index), optimum[index], ref["qp_tolerance"], ref["max_sweeps"])
        if not result["converged"].all():
            bad = index[~result["converged"]].tolist()
            raise RuntimeError(f"{split} QP labels did not converge: {[names[i] for i in bad]}")
        optimum[index], sweeps[index] = result["final"], result["sweeps"]
        print(f"  {split}: {int(index[-1])+1}/{len(problems)} solved contexts, {perf_counter()-began:.1f}s", flush=True)
    return dict(problem=problem, names=names, radii=radii, optimum=optimum,
                **sample_pairs(problem, optimum, settings, rng), reference_sweeps=sweeps,
                preparation_seconds=perf_counter()-began)


def prepare(config, path):
    """Reuse a complete cache or acquire exclusive ownership of preparation.

    Ablation variants share this path. A second writer fails clearly instead
    of replacing another process's partial cache. Completed caches are atomic
    and remain readable while training variants run concurrently.
    """
    path = Path(path)
    if path.exists():
        return _prepare_cache(config, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Another process is preparing {path}; finish --prepare-only before starting variants. "
            f"If preparation crashed, confirm no writer is active before removing {lock_path}."
        ) from error
    try:
        with lock:
            lock.write(f"pid={os.getpid()}\n")
        return _prepare_cache(config, path)
    finally:
        lock_path.unlink()


def _prepare_cache(config, path):
    """Atomic cache writes; caller owns the preparation lock for a new cache."""
    path = Path(path)
    spec = {key: config[key] for key in ("seed", "data", "physics", "dynamics", "reference")}
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache.get("format") != CACHE_FORMAT or cache.get("spec") != spec:
            raise ValueError("Expected matching CFM solution-pair cache; use a new outdir. Old B/C caches are not compatible.")
        return cache
    splits = {}
    parts = path.parent / (path.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        part = parts / f"{split}.pt"
        if part.exists():
            saved = torch.load(part, map_location="cpu", weights_only=True)
            if saved.get("spec") != spec or saved.get("format") != CACHE_FORMAT:
                raise ValueError("Partial CFM cache differs; choose a new outdir")
            splits[split] = saved["split"]
        else:
            print(f"Preparing {split} converged QP endpoints (CPU float64)...", flush=True)
            splits[split] = build_split(config, split)
            temporary = part.with_suffix(".tmp")
            torch.save(dict(format=CACHE_FORMAT, spec=spec, split=splits[split]), temporary)
            temporary.replace(part)
    cache = dict(format=CACHE_FORMAT, spec=spec, splits=splits)
    temporary = path.with_suffix(".tmp")
    torch.save(cache, temporary)
    temporary.replace(path)
    return cache


def cache_summary(cache, tolerance):
    report = {}
    for name, split in cache["splits"].items():
        success = converged(split["problem"], split["optimum"], tolerance)
        groups = {}
        for i, label in enumerate(split["names"]):
            row = groups.setdefault(label.rsplit("_", 1)[0], dict(scenes=0, solved=0))
            row["scenes"] += 1
            row["solved"] += int(success[i])
        report[name] = dict(scenes=len(success), pairs=len(split["context"]),
                            solved=int(success.sum()), groups=groups,
                            max_projected_gradient=float(residuals(split["problem"], split["optimum"])["projected_gradient"].max()),
                            start_counts={label: int((split["start_mode"] == i).sum()) for i, label in enumerate(START_MODES)},
                            preparation_seconds=split["preparation_seconds"],
                            reference_sweeps=int(split["reference_sweeps"].sum()))
    return report


def batch(split, indices, device):
    problem = move(select(split["problem"], split["context"][indices]), device, torch.float32)
    return (problem, *(split[key][indices].to(device=device, dtype=torch.float32) for key in ("source", "target")))
