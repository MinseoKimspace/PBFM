from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import RelaxedCirclesDataset
from src.losses.combined import CombinedLoss
from src.models.network import FlowVelocityNet


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

    parser = argparse.ArgumentParser(description="Train flow model with FM + weighted physics residual.")
    parser.add_argument("--config", type=str, default=pre_args.config)

    parser.add_argument("--dataset", type=str, default=_get_nested(cfg, "data", "dataset", default="data/relaxed_circles.pt"))
    parser.add_argument("--train-split", type=str, default=_get_nested(cfg, "data", "train_split", default="train"))
    parser.add_argument("--val-split", type=str, default=_get_nested(cfg, "data", "val_split", default="val"))
    parser.add_argument("--epochs", type=int, default=int(_get_nested(cfg, "train", "epochs", default=30)))
    parser.add_argument("--batch-size", type=int, default=int(_get_nested(cfg, "train", "batch_size", default=128)))
    parser.add_argument("--lr", type=float, default=float(_get_nested(cfg, "train", "lr", default=1e-3)))
    parser.add_argument("--weight-decay", type=float, default=float(_get_nested(cfg, "train", "weight_decay", default=1e-4)))
    parser.add_argument("--grad-clip", type=float, default=float(_get_nested(cfg, "train", "grad_clip", default=1.0)))
    parser.add_argument("--num-workers", type=int, default=int(_get_nested(cfg, "train", "num_workers", default=0)))

    parser.add_argument("--hidden-dim", type=int, default=int(_get_nested(cfg, "model", "hidden_dim", default=128)))
    parser.add_argument("--time-dim", type=int, default=int(_get_nested(cfg, "model", "time_dim", default=64)))
    parser.add_argument("--num-layers", type=int, default=int(_get_nested(cfg, "model", "num_layers", default=3)))
    parser.add_argument("--num-heads", type=int, default=int(_get_nested(cfg, "model", "num_heads", default=8)))
    parser.add_argument("--dropout", type=float, default=float(_get_nested(cfg, "model", "dropout", default=0.0)))
    parser.add_argument("--ffn-mult", type=int, default=int(_get_nested(cfg, "model", "ffn_mult", default=4)))
    parser.add_argument(
        "--use-radius-condition",
        dest="use_radius_condition",
        action="store_true",
        default=bool(_get_nested(cfg, "model", "use_radius_condition", default=False)),
    )
    parser.add_argument("--no-radius-condition", dest="use_radius_condition", action="store_false")

    parser.add_argument("--physics-weight", type=float, default=float(_get_nested(cfg, "loss", "physics_weight", default=0.1)))
    parser.add_argument(
        "--physics-weight-min",
        type=float,
        default=float(_get_nested(cfg, "loss", "physics_weight_min", default=0.05)),
    )
    parser.add_argument(
        "--physics-weight-max",
        type=float,
        default=float(_get_nested(cfg, "loss", "physics_weight_max", default=2.0)),
    )
    parser.add_argument(
        "--physics-weight-eps",
        type=float,
        default=float(_get_nested(cfg, "loss", "physics_weight_eps", default=1e-8)),
    )
    parser.add_argument("--gravity-weight", type=float, default=float(_get_nested(cfg, "loss", "gravity_weight", default=0.1)))
    parser.add_argument("--ground-weight", type=float, default=float(_get_nested(cfg, "loss", "ground_weight", default=0.1)))
    parser.add_argument("--collision-weight", type=float, default=float(_get_nested(cfg, "loss", "collision_weight", default=0.3)))
    parser.add_argument("--collision-alpha", type=float, default=float(_get_nested(cfg, "loss", "collision_alpha", default=0.1)))
    parser.add_argument(
        "--collision-epsilon",
        type=float,
        default=float(_get_nested(cfg, "loss", "collision_epsilon", default=1e-5)),
    )
    parser.add_argument(
        "--collision-constant",
        type=float,
        default=float(_get_nested(cfg, "loss", "collision_constant", default=0.01)),
    )
    parser.add_argument("--y-ground", type=float, default=float(_get_nested(cfg, "loss", "y_ground", default=0.0)))

    parser.add_argument("--seed", type=int, default=int(_get_nested(cfg, "runtime", "seed", default=42)))
    parser.add_argument(
        "--device",
        type=str,
        default=str(_get_nested(cfg, "runtime", "device", default="auto")),
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--outdir", type=str, default=str(_get_nested(cfg, "runtime", "outdir", default="runs")))
    parser.add_argument("--run-name", type=str, default=str(_get_nested(cfg, "runtime", "run_name", default="")))
    return parser.parse_args(remaining)


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
    criterion: CombinedLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[Dict[str, float]] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x1 = batch["state"].to(device)
            r = batch["radius"].to(device)
            x0 = batch["state_init"].to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            loss, metrics = criterion(model, x0, x1, r)

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

    train_ds = RelaxedCirclesDataset(
        dataset_path=args.dataset,
        split=args.train_split,
        use_relaxed_state=True,
        return_init_state=True,
    )
    val_ds = RelaxedCirclesDataset(
        dataset_path=args.dataset,
        split=args.val_split,
        use_relaxed_state=True,
        return_init_state=True,
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
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "ffn_mult": args.ffn_mult,
        "use_radius_condition": args.use_radius_condition,
    }
    loss_kwargs = {
        "physics_weight": args.physics_weight,
        "physics_weight_min": args.physics_weight_min,
        "physics_weight_max": args.physics_weight_max,
        "physics_weight_eps": args.physics_weight_eps,
        "gravity_weight": args.gravity_weight,
        "ground_weight": args.ground_weight,
        "collision_weight": args.collision_weight,
        "collision_alpha": args.collision_alpha,
        "collision_epsilon": args.collision_epsilon,
        "collision_constant": args.collision_constant,
        "y_ground": args.y_ground,
    }

    model = FlowVelocityNet(**model_kwargs).to(device)
    criterion = CombinedLoss(**loss_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[Dict[str, float]] = []

    print(f"Training on device={device}, run_dir={run_dir}")
    print(
        f"Dataset={args.dataset}, train_split={args.train_split}, val_split={args.val_split}, "
        "path=state_init->state_relaxed"
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
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
            f"physics={val_metrics.get('physics', float('nan')):.6f} "
            f"fm/physics={val_metrics.get('fm_over_physics', float('nan')):.6f} "
            f"w_eff={val_metrics.get('physics_weight_eff', float('nan')):.6f} "
            f"res/fm={val_metrics.get('physics_residual_over_fm', float('nan')):.6f}"
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss": best_val,
                "history": history,
                "model_kwargs": model_kwargs,
                "loss_kwargs": loss_kwargs,
                "train_args": vars(args),
            },
            f,
            indent=2,
        )

    print(f"Done. Best val loss: {best_val:.6f}")
    print(f"Saved checkpoints: {run_dir / 'best.pt'} and {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
