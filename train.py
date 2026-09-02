from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import ProjectionTransitionDataset
from src.losses.combined import ProjectionFlowLoss
from src.models.network import FlowVelocityNet

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
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def run_epoch(
    model: torch.nn.Module,
    loss_fn: ProjectionFlowLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> dict[str, float]:
    model.train(optimizer is not None)
    rows: list[dict[str, float]] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()

    with context:
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            condition = batch["condition"].to(device)
            radius = batch["radius"].to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            loss, metrics = loss_fn(model, source, target, radius, condition)

            if optimizer is not None:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            rows.append(metrics)

    return mean(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train projection flow model.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = int(get(cfg, "runtime", "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(str(get(cfg, "runtime", "device", "auto")))
    run_name = str(get(cfg, "runtime", "run_name", "")) or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(str(get(cfg, "runtime", "outdir", "runs"))) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = str(get(cfg, "data", "dataset", "data/box2d_projection.pt"))
    train_ds = ProjectionTransitionDataset(dataset_path, split=str(get(cfg, "data", "train_split", "train")))
    val_ds = ProjectionTransitionDataset(dataset_path, split=str(get(cfg, "data", "val_split", "val")))
    batch_size = int(get(cfg, "train", "batch_size", 128))
    num_workers = int(get(cfg, "train", "num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_kwargs = {
        "hidden_dim": int(get(cfg, "model", "hidden_dim", 128)),
        "time_dim": int(get(cfg, "model", "time_dim", 64)),
        "num_layers": int(get(cfg, "model", "num_layers", 3)),
    }
    loss_kwargs = {
        "physics_weight": float(get(cfg, "loss", "physics_weight", 0.1)),
        "ground_weight": float(get(cfg, "loss", "ground_weight", 0.1)),
        "collision_weight": float(get(cfg, "loss", "collision_weight", 0.3)),
        "y_ground": float(get(cfg, "loss", "y_ground", 0.0)),
        "unroll_steps": int(get(cfg, "loss", "unroll_steps", 4)),
    }
    model = FlowVelocityNet(**model_kwargs).to(device)
    loss_fn = ProjectionFlowLoss(**loss_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get(cfg, "train", "lr", 1e-3)),
        weight_decay=float(get(cfg, "train", "weight_decay", 1e-4)),
    )

    best_val = float("inf")
    history: list[dict[str, float]] = []
    epochs = int(get(cfg, "train", "epochs", 30))
    grad_clip = float(get(cfg, "train", "grad_clip", 1.0))
    print(f"Training on {device}, dataset={dataset_path}, run_dir={run_dir}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loss_fn, train_loader, device, optimizer, grad_clip)
        val_metrics = run_epoch(model, loss_fn, val_loader, device)
        row = {"epoch": float(epoch), "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]}
        history.append(row)

        checkpoint = {
            "epoch": epoch,
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

        print(
            f"[{epoch:03d}/{epochs:03d}] "
            f"train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} "
            f"fm={val_metrics['fm']:.6f} physics={val_metrics['physics']:.6f}"
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best_val, "history": history, "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
