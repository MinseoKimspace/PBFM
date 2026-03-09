from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Tuple
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.physics_energy import PhysicsEnergyLoss


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)!r}")
    return data


def _get_nested(config: Mapping[str, Any], *keys: str, default: Any) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)
    render_enabled = bool(_get_nested(cfg, "render", "enabled", default=False))
    default_render_dir = str(_get_nested(cfg, "render", "dir", default="")) if render_enabled else ""

    parser = argparse.ArgumentParser(
        description="Generate random circle states, relax with physics energy, and save dataset splits."
    )
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--output", type=str, default=str(_get_nested(cfg, "dataset", "output", default="data/relaxed_circles.pt")))
    parser.add_argument("--train-samples", type=int, default=int(_get_nested(cfg, "dataset", "train_samples", default=8192)))
    parser.add_argument("--val-samples", type=int, default=int(_get_nested(cfg, "dataset", "val_samples", default=2048)))
    parser.add_argument("--test-samples", type=int, default=int(_get_nested(cfg, "dataset", "test_samples", default=2048)))
    parser.add_argument("--num-objects", type=int, default=int(_get_nested(cfg, "dataset", "num_objects", default=8)))
    parser.add_argument("--radius-min", type=float, default=float(_get_nested(cfg, "dataset", "radius_min", default=0.05)))
    parser.add_argument("--radius-max", type=float, default=float(_get_nested(cfg, "dataset", "radius_max", default=0.15)))
    parser.add_argument("--xy-limit", type=float, default=float(_get_nested(cfg, "dataset", "xy_limit", default=1.0)))

    parser.add_argument("--relax-steps", type=int, default=int(_get_nested(cfg, "relax", "steps", default=100)))
    parser.add_argument("--relax-lr", type=float, default=float(_get_nested(cfg, "relax", "lr", default=0.05)))
    parser.add_argument("--relax-batch-size", type=int, default=int(_get_nested(cfg, "relax", "batch_size", default=512)))

    parser.add_argument("--gravity-weight", type=float, default=float(_get_nested(cfg, "physics", "gravity_weight", default=0.2)))
    parser.add_argument("--ground-weight", type=float, default=float(_get_nested(cfg, "physics", "ground_weight", default=0.2)))
    parser.add_argument("--collision-weight", type=float, default=float(_get_nested(cfg, "physics", "collision_weight", default=0.2)))
    parser.add_argument("--y-ground", type=float, default=float(_get_nested(cfg, "physics", "y_ground", default=0.0)))

    parser.add_argument("--seed", type=int, default=int(_get_nested(cfg, "runtime", "seed", default=42)))
    parser.add_argument(
        "--device",
        type=str,
        default=str(_get_nested(cfg, "runtime", "device", default="auto")),
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--render-dir", type=str, default=default_render_dir)
    parser.add_argument(
        "--render-split",
        type=str,
        default=str(_get_nested(cfg, "render", "split", default="train")),
        choices=["train", "val", "test"],
    )
    parser.add_argument("--render-count", type=int, default=int(_get_nested(cfg, "render", "count", default=6)))
    parser.add_argument("--render-size", type=int, default=int(_get_nested(cfg, "render", "size", default=1024)))
    return parser.parse_args(remaining)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_random_batch(
    num_samples: int,
    num_objects: int,
    radius_min: float,
    radius_max: float,
    xy_limit: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    radius = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        radius_min, radius_max, generator=g
    )
    x = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        -xy_limit, xy_limit, generator=g
    )
    y = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        0.0, xy_limit, generator=g
    )
    state = torch.stack((x, y), dim=-1).contiguous()
    return state, radius.contiguous()


def relax_states(
    state_init: torch.Tensor,
    radius: torch.Tensor,
    physics_loss: PhysicsEnergyLoss,
    device: torch.device,
    relax_steps: int,
    relax_lr: float,
    xy_limit: float,
) -> torch.Tensor:
    if relax_steps <= 0:
        return state_init.clone()

    state = state_init.to(device)
    radius = radius.to(device)

    for _ in range(relax_steps):
        state = state.detach().requires_grad_(True)
        loss = physics_loss(state, radius)
        grad = torch.autograd.grad(loss, state, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            state = state - relax_lr * grad
            state[..., 0].clamp_(-xy_limit, xy_limit)
            state[..., 1].clamp_(0.0, xy_limit)
    return state.detach().cpu()


def average_energy(
    state: torch.Tensor,
    radius: torch.Tensor,
    physics_loss: PhysicsEnergyLoss,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, state.size(0), batch_size):
            end = min(start + batch_size, state.size(0))
            s = state[start:end].to(device)
            r = radius[start:end].to(device)
            loss = float(physics_loss(s, r).item())
            n = end - start
            total += loss * n
            count += n
    return total / max(count, 1)


def generate_split(
    split_name: str,
    num_samples: int,
    num_objects: int,
    radius_min: float,
    radius_max: float,
    xy_limit: float,
    split_seed: int,
    relax_steps: int,
    relax_lr: float,
    relax_batch_size: int,
    physics_loss: PhysicsEnergyLoss,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    state_init, radius = make_random_batch(
        num_samples=num_samples,
        num_objects=num_objects,
        radius_min=radius_min,
        radius_max=radius_max,
        xy_limit=xy_limit,
        seed=split_seed,
    )

    relaxed_chunks: list[torch.Tensor] = []
    for start in range(0, num_samples, relax_batch_size):
        end = min(start + relax_batch_size, num_samples)
        chunk = relax_states(
            state_init=state_init[start:end],
            radius=radius[start:end],
            physics_loss=physics_loss,
            device=device,
            relax_steps=relax_steps,
            relax_lr=relax_lr,
            xy_limit=xy_limit,
        )
        relaxed_chunks.append(chunk)
    state_relaxed = torch.cat(relaxed_chunks, dim=0).contiguous()

    e_init = average_energy(state_init, radius, physics_loss, device=device, batch_size=relax_batch_size)
    e_relaxed = average_energy(state_relaxed, radius, physics_loss, device=device, batch_size=relax_batch_size)
    print(f"[{split_name}] samples={num_samples} energy_init={e_init:.6f} energy_relaxed={e_relaxed:.6f}")

    return {
        "state_init": state_init,
        "state_relaxed": state_relaxed,
        "radius": radius,
    }


def render_one(
    state: torch.Tensor,
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    image_size: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("PIL not available. Skipping render.")
        return

    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    color_cycle = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]

    def to_px(x: float, y: float) -> Tuple[float, float]:
        x_px = (x + xy_limit) / (2.0 * xy_limit) * (image_size - 1)
        y_px = (1.0 - (y / max(xy_limit, 1e-6))) * (image_size - 1)
        return x_px, y_px

    scale = (image_size - 1) / (2.0 * xy_limit)
    ground_y = to_px(0.0, 0.0)[1]
    draw.line([(0, ground_y), (image_size - 1, ground_y)], fill=(0, 0, 0), width=1)

    for i in range(state.size(0)):
        cx, cy = to_px(float(state[i, 0]), float(state[i, 1]))
        rr = float(radius[i]) * scale
        color = color_cycle[i % len(color_cycle)]
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color, width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_split_samples(
    split_data: Dict[str, torch.Tensor],
    render_dir: Path,
    split_name: str,
    count: int,
    xy_limit: float,
    image_size: int,
) -> None:
    count = min(count, split_data["state_init"].size(0))
    for idx in range(count):
        render_one(
            split_data["state_init"][idx],
            split_data["radius"][idx],
            render_dir / f"{split_name}_{idx:04d}_init.png",
            xy_limit=xy_limit,
            image_size=image_size,
        )
        render_one(
            split_data["state_relaxed"][idx],
            split_data["radius"][idx],
            render_dir / f"{split_name}_{idx:04d}_relaxed.png",
            xy_limit=xy_limit,
            image_size=image_size,
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    physics_loss = PhysicsEnergyLoss(
        gravity_weight=args.gravity_weight,
        ground_weight=args.ground_weight,
        collision_weight=args.collision_weight,
        y_ground=args.y_ground,
    ).to(device)
    physics_loss.eval()

    train = generate_split(
        split_name="train",
        num_samples=args.train_samples,
        num_objects=args.num_objects,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        xy_limit=args.xy_limit,
        split_seed=args.seed,
        relax_steps=args.relax_steps,
        relax_lr=args.relax_lr,
        relax_batch_size=args.relax_batch_size,
        physics_loss=physics_loss,
        device=device,
    )
    val = generate_split(
        split_name="val",
        num_samples=args.val_samples,
        num_objects=args.num_objects,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        xy_limit=args.xy_limit,
        split_seed=args.seed + 1,
        relax_steps=args.relax_steps,
        relax_lr=args.relax_lr,
        relax_batch_size=args.relax_batch_size,
        physics_loss=physics_loss,
        device=device,
    )
    test = generate_split(
        split_name="test",
        num_samples=args.test_samples,
        num_objects=args.num_objects,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        xy_limit=args.xy_limit,
        split_seed=args.seed + 2,
        relax_steps=args.relax_steps,
        relax_lr=args.relax_lr,
        relax_batch_size=args.relax_batch_size,
        physics_loss=physics_loss,
        device=device,
    )

    payload = {
        "meta": {
            "num_objects": args.num_objects,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
            "xy_limit": args.xy_limit,
            "relax_steps": args.relax_steps,
            "relax_lr": args.relax_lr,
            "gravity_weight": args.gravity_weight,
            "ground_weight": args.ground_weight,
            "collision_weight": args.collision_weight,
            "y_ground": args.y_ground,
            "seed": args.seed,
        },
        "train": train,
        "val": val,
        "test": test,
    }
    torch.save(payload, output_path)
    print(f"Saved dataset to {output_path}")

    if args.render_dir:
        render_dir = Path(args.render_dir)
        split_map = {"train": train, "val": val, "test": test}
        render_split_samples(
            split_data=split_map[args.render_split],
            render_dir=render_dir,
            split_name=args.render_split,
            count=args.render_count,
            xy_limit=args.xy_limit,
            image_size=args.render_size,
        )
        print(f"Saved renders to {render_dir}")


if __name__ == "__main__":
    main()
