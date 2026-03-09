from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.losses.physics_energy import PhysicsEnergyLoss
from src.models.network import FlowVelocityNet
from src.paths.linear import sample_linear_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flow model with FM + weighted physics residual.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--num-objects", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--radius-min", type=float, default=0.05)
    parser.add_argument("--radius-max", type=float, default=0.15)
    parser.add_argument("--xy-limit", type=float, default=1.0)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)

    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--gravity-weight", type=float, default=0.2)
    parser.add_argument("--ground-weight", type=float, default=0.2)
    parser.add_argument("--collision-weight", type=float, default=0.2)
    parser.add_argument("--y-ground", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def run_epoch(
    model: torch.nn.Module,
    physics_loss_fn: PhysicsEnergyLoss,
    loader: DataLoader,
    device: torch.device,
    physics_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[Dict[str, float]] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for state, radius in loader:
            x1 = state.to(device)
            r = radius.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            loss, metrics = combined_loss(
                model=model,
                physics_loss_fn=physics_loss_fn,
                x1=x1,
                radius=r,
                physics_weight=physics_weight,
            )

            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            rows.append(metrics)

    return mean_metrics(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.outdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_synthetic_dataset(
        num_samples=args.train_samples,
        num_objects=args.num_objects,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        xy_limit=args.xy_limit,
        seed=args.seed,
    )
    val_ds = make_synthetic_dataset(
        num_samples=args.val_samples,
        num_objects=args.num_objects,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        xy_limit=args.xy_limit,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "time_dim": args.time_dim,
        "num_layers": args.num_layers,
    }
    loss_kwargs = {
        "gravity_weight": args.gravity_weight,
        "ground_weight": args.ground_weight,
        "collision_weight": args.collision_weight,
        "y_ground": args.y_ground,
    }

    model = FlowVelocityNet(**model_kwargs).to(device)
    physics_loss_fn = PhysicsEnergyLoss(**loss_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[Dict[str, float]] = []

    print(f"Training on device={device}, run_dir={run_dir}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            physics_loss_fn=physics_loss_fn,
            loader=train_loader,
            device=device,
            physics_weight=args.physics_weight,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            physics_loss_fn=physics_loss_fn,
            loader=val_loader,
            device=device,
            physics_weight=args.physics_weight,
            optimizer=None,
        )

        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", float("nan"))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_kwargs": model_kwargs,
            "loss_kwargs": loss_kwargs,
            "physics_weight": args.physics_weight,
            "train_args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"fm={val_metrics.get('fm', float('nan')):.6f} "
            f"physics={val_metrics.get('physics', float('nan')):.6f}"
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss": best_val,
                "history": history,
                "model_kwargs": model_kwargs,
                "loss_kwargs": loss_kwargs,
                "physics_weight": args.physics_weight,
                "train_args": vars(args),
            },
            f,
            indent=2,
        )

    print(f"Done. Best val loss: {best_val:.6f}")
    print(f"Saved checkpoints: {run_dir / 'best.pt'} and {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
