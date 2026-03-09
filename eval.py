from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.losses.physics_energy import PhysicsEnergyLoss
from src.models.network import FlowVelocityNet
from src.paths.linear import sample_linear_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FM + physics residual checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--num-objects", type=int, default=None)
    parser.add_argument("--radius-min", type=float, default=None)
    parser.add_argument("--radius-max", type=float, default=None)
    parser.add_argument("--xy-limit", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_dataset(
    num_samples: int,
    num_objects: int,
    radius_min: float,
    radius_max: float,
    xy_limit: float,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    radius = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        radius_min, radius_max, generator=generator
    )
    x = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        -xy_limit, xy_limit, generator=generator
    )
    y_offset = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        0.0, xy_limit, generator=generator
    )
    y = radius + y_offset
    state = torch.stack((x, y), dim=-1).contiguous()
    return TensorDataset(state, radius.contiguous())


def combined_loss(
    model: torch.nn.Module,
    physics_loss_fn: PhysicsEnergyLoss,
    x1: torch.Tensor,
    radius: torch.Tensor,
    physics_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _, x_t, t_now, target_v = sample_linear_path(x1)
    t_start = torch.zeros_like(t_now)
    v_hat = model(x_t, t_start, t_now)
    fm_term = F.mse_loss(v_hat, target_v)

    t_state = t_now
    while t_state.dim() < x_t.dim():
        t_state = t_state.unsqueeze(-1)
    x1_hat = x_t + (1.0 - t_state) * v_hat

    physics_term = physics_loss_fn(x1_hat, radius)
    residual = physics_weight * physics_term
    total = fm_term + residual

    metrics = {
        "loss": float(total.detach().item()),
        "fm": float(fm_term.detach().item()),
        "physics": float(physics_term.detach().item()),
        "physics_residual": float(residual.detach().item()),
    }
    return total, metrics


def mean_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {}
    sums: Dict[str, float] = {}
    for row in metrics:
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    n = float(len(metrics))
    return {key: value / n for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    train_args = checkpoint.get("train_args", {})
    model_kwargs = checkpoint.get("model_kwargs", {"hidden_dim": 128, "time_dim": 64, "num_layers": 3})
    loss_kwargs = checkpoint.get(
        "loss_kwargs",
        {"gravity_weight": 0.2, "ground_weight": 0.2, "collision_weight": 0.2, "y_ground": 0.0},
    )
    physics_weight = float(checkpoint.get("physics_weight", train_args.get("physics_weight", 0.1)))

    num_objects = args.num_objects if args.num_objects is not None else int(train_args.get("num_objects", 8))
    val_samples = args.val_samples if args.val_samples is not None else int(train_args.get("val_samples", 2048))
    radius_min = args.radius_min if args.radius_min is not None else float(train_args.get("radius_min", 0.05))
    radius_max = args.radius_max if args.radius_max is not None else float(train_args.get("radius_max", 0.15))
    xy_limit = args.xy_limit if args.xy_limit is not None else float(train_args.get("xy_limit", 1.0))
    seed = args.seed if args.seed is not None else int(train_args.get("seed", 42)) + 1

    dataset = make_synthetic_dataset(
        num_samples=val_samples,
        num_objects=num_objects,
        radius_min=radius_min,
        radius_max=radius_max,
        xy_limit=xy_limit,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = FlowVelocityNet(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    physics_loss_fn = PhysicsEnergyLoss(**loss_kwargs).to(device)
    model.eval()

    rows: list[Dict[str, float]] = []
    with torch.no_grad():
        for state, radius in loader:
            x1 = state.to(device)
            r = radius.to(device)
            _, metrics = combined_loss(
                model=model,
                physics_loss_fn=physics_loss_fn,
                x1=x1,
                radius=r,
                physics_weight=physics_weight,
            )
            rows.append(metrics)

    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "num_batches": len(rows),
        "metrics": mean_metrics(rows),
    }
    print(json.dumps(report, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
