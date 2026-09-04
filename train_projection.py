from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import NextStepDataset
from src.losses.projection import ProjectionLoss, free_position
from src.models.projection import build_projection_model

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_config(path: str) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.")
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(row.keys() for row in rows))
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def projection_stats(
    dataset: NextStepDataset,
    time_step: float,
    gravity_y: float,
    linear_damping: float,
) -> tuple[list[float], list[float]]:
    proposal = free_position(dataset.source, time_step, gravity_y, linear_damping)
    correction = (dataset.target[..., :2] - proposal).reshape(-1, 2)
    mean_value = correction.mean(dim=0)
    std_value = correction.std(dim=0).clamp_min(1e-6)
    return mean_value.tolist(), std_value.tolist()


def run_epoch(
    model: torch.nn.Module,
    loss_fn: ProjectionLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> dict[str, float]:
    model.train(optimizer is not None)
    loss_fn.train(optimizer is not None)
    rows: list[dict[str, float]] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()

    with context:
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            radius = batch["radius"].to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            loss, metrics = loss_fn(model, source, target, radius)

            if optimizer is not None:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            rows.append(metrics)

    return mean(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train position-projection delta/FM/gradient-FM solver.")
    parser.add_argument("--config", default="configs/train_projection.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = int(get(cfg, "runtime", "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(str(get(cfg, "runtime", "device", "auto")))
    model_type = str(get(cfg, "model", "type", "grad_fm"))
    run_name = str(get(cfg, "runtime", "run_name", "")) or datetime.now().strftime(
        f"%Y%m%d-%H%M%S_projection_{model_type}"
    )
    run_dir = Path(str(get(cfg, "runtime", "outdir", "runs"))) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = str(get(cfg, "data", "dataset", "data/box2d_next_step.pt"))
    train_ds = NextStepDataset(dataset_path, split=str(get(cfg, "data", "train_split", "train")))
    val_ds = NextStepDataset(dataset_path, split=str(get(cfg, "data", "val_split", "val")))
    batch_size = int(get(cfg, "train", "batch_size", 128))
    num_workers = int(get(cfg, "train", "num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    time_step = float(get(cfg, "dynamics", "time_step", train_ds.meta.get("time_step", 1.0 / 60.0)))
    gravity_y = float(get(cfg, "dynamics", "gravity_y", train_ds.meta.get("gravity_y", -9.8)))
    linear_damping = float(get(cfg, "dynamics", "linear_damping", train_ds.meta.get("linear_damping", 0.0)))
    correction_mean, correction_std = projection_stats(train_ds, time_step, gravity_y, linear_damping)

    model_kwargs = {
        "hidden_dim": int(get(cfg, "model", "hidden_dim", 256)),
        "time_dim": int(get(cfg, "model", "time_dim", 64)),
        "num_layers": int(get(cfg, "model", "num_layers", 3)),
        "message_steps": int(get(cfg, "model", "message_steps", 4)),
        "contact_margin": float(get(cfg, "model", "contact_margin", 0.25)),
    }
    loss_kwargs = {
        "model_type": model_type,
        "time_step": time_step,
        "gravity_y": gravity_y,
        "linear_damping": linear_damping,
        "physics_weight": float(get(cfg, "loss", "physics_weight", 0.1)),
        "ground_weight": float(get(cfg, "loss", "ground_weight", 1.0)),
        "collision_weight": float(get(cfg, "loss", "collision_weight", 1.0)),
        "y_ground": float(get(cfg, "loss", "y_ground", train_ds.meta.get("y_ground", 0.0))),
        "slop": float(get(cfg, "loss", "slop", train_ds.meta.get("box2d_linear_slop", 0.005))),
        "contact_margin": float(get(cfg, "loss", "contact_margin", model_kwargs["contact_margin"])),
        "contact_loss_weight": float(get(cfg, "loss", "contact_loss_weight", 4.0)),
        "correction_threshold": float(get(cfg, "loss", "correction_threshold", 1e-4)),
        "position_noise_std": float(get(cfg, "loss", "position_noise_std", 0.01)),
        "velocity_noise_std": float(get(cfg, "loss", "velocity_noise_std", 0.05)),
        "correction_mean": correction_mean,
        "correction_std": correction_std,
    }

    model = build_projection_model(model_type, **model_kwargs).to(device)
    loss_fn = ProjectionLoss(**loss_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get(cfg, "train", "lr", 3e-4)),
        weight_decay=float(get(cfg, "train", "weight_decay", 1e-4)),
    )

    best_val = float("inf")
    history: list[dict[str, float]] = []
    epochs = int(get(cfg, "train", "epochs", 300))
    grad_clip = float(get(cfg, "train", "grad_clip", 1.0))
    print(f"Training projection_{model_type} on {device}, dataset={dataset_path}, run_dir={run_dir}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loss_fn, train_loader, device, optimizer, grad_clip)
        val_metrics = run_epoch(model, loss_fn, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_physics": val_metrics["physics"],
                "val_contact_fraction": val_metrics["contact_fraction"],
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model_type": f"projection_{model_type}",
            "projection_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_kwargs": model_kwargs,
            "loss_kwargs": loss_kwargs,
            "config": cfg,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, run_dir / "best.pt")

        main_key = "delta" if model_type == "delta" else "fm"
        print(
            f"[{epoch:03d}/{epochs:03d}] "
            f"train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} "
            f"{main_key}={val_metrics[main_key]:.6f} physics={val_metrics['physics']:.6f} "
            f"contact={val_metrics['contact_fraction']:.3f}"
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best_val, "history": history, "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
