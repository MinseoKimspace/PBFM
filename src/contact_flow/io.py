"""Small configuration/state adapters. Box2D endpoints are never training labels."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix.lower() == ".json":
            result = json.load(stream)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configuration requires PyYAML: pip install pyyaml (JSON also supported)") from exc
            result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    return result


def device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return torch.device(name)


def load_states(path: str, split: str, count: int, seed: int,
                num_objects: int = 5, *, physics=None, return_scene_types=False):
    if count <= 0 or num_objects <= 0:
        raise ValueError("count and num_objects must be positive")
    rng = torch.Generator().manual_seed(seed)
    if path:
        data = torch.load(path, map_location="cpu", weights_only=True)[split]
        # Deliberately do not read target, rollout_target or target-derived statistics.
        source, radius = data["source"].float(), data["radius"].float()
        if source.ndim != 3 or source.shape[-1] != 4 or radius.shape != source.shape[:2]:
            raise ValueError("Expected source[S,N,4] and radius[S,N]")
        if not len(source):
            raise ValueError(f"Empty input split: {split}")
        index = torch.randperm(len(source), generator=rng)[:count]
        labels = data.get("scene_type", ["dataset"] * len(source))
        if len(labels) != len(source):
            raise ValueError("scene_type must contain one label per source")
        scene_types = [str(labels[int(i)]) for i in index]
        source, radius = source[index].contiguous(), radius[index].contiguous()
    else:
        # Feasible stacks with varied incoming velocities, oblique stacks and free flight.
        radius = 0.45 + 0.1 * torch.rand(count, num_objects, generator=rng)
        source = torch.zeros(count, num_objects, 4)
        height = torch.zeros(count)
        for i in range(num_objects):
            source[:, i, 1] = height + radius[:, i] + 0.002
            height = source[:, i, 1] + radius[:, i]
        source[..., 0] = (torch.rand(count, 1, generator=rng) - 0.5) * 4
        source[1::3, :, 0] += torch.arange(num_objects) * 0.12
        source[..., 2] = 0.4 * (torch.rand(count, num_objects, generator=rng) - 0.5)
        source[..., 3] = -2 - 3 * torch.rand(count, num_objects, generator=rng)
        source[2::3, :, 1] += 2.0
        source[2::3, :, 3] = -1.0
        scene_types = [("vertical_stack", "oblique_stack", "free_flight")[i % 3]
                       for i in range(count)]
        if physics is not None:
            source[..., 1] += physics.y_ground
            # Translate complete scenes only; do not distort particle gaps.
            left = (source[..., 0] - radius).amin(1)
            right = (source[..., 0] + radius).amax(1)
            if ((right - left) >= 2 * physics.xy_limit).any():
                raise ValueError("Procedural scene is wider than the configured domain")
            lower, upper = -physics.xy_limit - left, physics.xy_limit - right
            shift = torch.maximum(torch.minimum(torch.zeros_like(lower), upper), lower)
            source[..., 0] += shift[:, None]
    if not torch.isfinite(source).all() or not torch.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError("Input states/radii must be finite; radii must be positive")
    return (source, radius, scene_types) if return_scene_types else (source, radius)
