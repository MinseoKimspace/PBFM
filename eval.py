from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable

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

    parser = argparse.ArgumentParser(description="Evaluate FM + physics residual checkpoint.")
    parser.add_argument("--config", type=str, default=pre_args.config)

    parser.add_argument("--checkpoint", type=str, default=str(_get_nested(cfg, "eval", "checkpoint", default="")))
    parser.add_argument("--dataset", type=str, default=str(_get_nested(cfg, "eval", "dataset", default="")))
    parser.add_argument("--split", type=str, default=str(_get_nested(cfg, "eval", "split", default="val")))
    parser.add_argument(
        "--use-init-state",
        dest="use_init_state",
        action="store_true",
        default=bool(_get_nested(cfg, "eval", "use_init_state", default=False)),
    )
    parser.add_argument("--use-relaxed-state", dest="use_init_state", action="store_false")

    parser.add_argument("--batch-size", type=int, default=int(_get_nested(cfg, "eval", "batch_size", default=256)))
    parser.add_argument("--num-workers", type=int, default=int(_get_nested(cfg, "eval", "num_workers", default=0)))
    parser.add_argument(
        "--device",
        type=str,
        default=str(_get_nested(cfg, "runtime", "device", default="auto")),
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--output-json", type=str, default=str(_get_nested(cfg, "output", "json", default="")))
    parser.add_argument("--render-dir", type=str, default=str(_get_nested(cfg, "render", "dir", default="")))
    parser.add_argument("--render-count", type=int, default=int(_get_nested(cfg, "render", "count", default=8)))
    parser.add_argument("--render-size", type=int, default=int(_get_nested(cfg, "render", "size", default=1024)))
    parser.add_argument("--sample-steps", type=int, default=int(_get_nested(cfg, "render", "sample_steps", default=64)))
    parser.add_argument("--render-seed", type=int, default=int(_get_nested(cfg, "render", "seed", default=1234)))
    return parser.parse_args(remaining)


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


def sample_flow_euler(model: torch.nn.Module, x0: torch.Tensor, steps: int) -> torch.Tensor:
    if steps < 1:
        raise ValueError(f"sample steps must be >= 1, got {steps}")
    x = x0
    dt = 1.0 / float(steps)
    for i in range(steps):
        t_now = torch.full(
            (x.size(0), 1),
            (i + 0.5) / float(steps),
            device=x.device,
            dtype=x.dtype,
        )
        t_start = torch.zeros_like(t_now)
        v_hat = model(x, t_start, t_now)
        x = x + dt * v_hat
    return x


def render_state(
    state: torch.Tensor,
    radius: torch.Tensor,
    xy_limit: float,
    image_size: int,
):
    from PIL import Image, ImageDraw

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

    def to_px(x: float, y: float) -> tuple[float, float]:
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

    return canvas


def render_model_samples(
    model: torch.nn.Module,
    dataset: RelaxedCirclesDataset,
    device: torch.device,
    render_dir: str,
    render_count: int,
    render_size: int,
    sample_steps: int,
    render_seed: int,
) -> None:
    try:
        from PIL import Image
    except Exception:
        print("PIL not available. Skipping render.")
        return

    out_dir = Path(render_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xy_limit = float(dataset.meta.get("xy_limit", 1.0))
    count = min(render_count, len(dataset))
    g = torch.Generator(device="cpu").manual_seed(render_seed)

    with torch.no_grad():
        for idx in range(count):
            item = dataset[idx]
            x1 = item["state"].unsqueeze(0).to(device)
            radius = item["radius"].unsqueeze(0).to(device)
            x0 = torch.randn(x1.shape, generator=g, dtype=x1.dtype, device="cpu").to(device)
            x_hat = sample_flow_euler(model, x0, steps=sample_steps)
            x_hat[..., 0].clamp_(-xy_limit, xy_limit)
            x_hat[..., 1].clamp_(0.0, xy_limit)

            img_x0 = render_state(x0[0].cpu(), radius[0].cpu(), xy_limit=xy_limit, image_size=render_size)
            img_pred = render_state(x_hat[0].cpu(), radius[0].cpu(), xy_limit=xy_limit, image_size=render_size)
            img_target = render_state(x1[0].cpu(), radius[0].cpu(), xy_limit=xy_limit, image_size=render_size)

            canvas = Image.new("RGB", (render_size * 3, render_size), (255, 255, 255))
            canvas.paste(img_x0, (0, 0))
            canvas.paste(img_pred, (render_size, 0))
            canvas.paste(img_target, (render_size * 2, 0))
            canvas.save(out_dir / f"{idx:04d}_x0_pred_target.png")

    print(f"Saved model renders to {out_dir}")


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("Checkpoint path is required. Pass --checkpoint or set eval.checkpoint in config.")
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    train_args = checkpoint.get("train_args", {})
    model_kwargs = checkpoint.get(
        "model_kwargs",
        {
            "hidden_dim": 128,
            "time_dim": 64,
            "num_layers": 3,
            "num_heads": 8,
            "dropout": 0.0,
            "ffn_mult": 4,
        },
    )
    loss_kwargs = checkpoint.get(
        "loss_kwargs",
        {
            "physics_weight": float(checkpoint.get("physics_weight", train_args.get("physics_weight", 0.1))),
            "gravity_weight": 0.2,
            "ground_weight": 0.2,
            "collision_weight": 0.2,
            "y_ground": 0.0,
        },
    )
    if "physics_weight" not in loss_kwargs:
        loss_kwargs["physics_weight"] = float(checkpoint.get("physics_weight", train_args.get("physics_weight", 0.1)))

    dataset_path = args.dataset if args.dataset else str(train_args.get("dataset", ""))
    if not dataset_path:
        raise ValueError("Dataset path is required. Pass --dataset or use a checkpoint saved from dataset-based training.")

    split = args.split if args.split else str(train_args.get("val_split", "val"))
    use_init_state = args.use_init_state or bool(train_args.get("use_init_state", False))
    use_relaxed_state = not use_init_state

    dataset = RelaxedCirclesDataset(
        dataset_path=dataset_path,
        split=split,
        use_relaxed_state=use_relaxed_state,
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
    criterion = CombinedLoss(**loss_kwargs).to(device)
    model.eval()

    rows: list[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch["state"].to(device)
            r = batch["radius"].to(device)
            _, metrics = criterion(model, x1, r)
            rows.append(metrics)

    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "split": split,
        "use_relaxed_state": use_relaxed_state,
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

    if args.render_dir:
        render_model_samples(
            model=model,
            dataset=dataset,
            device=device,
            render_dir=args.render_dir,
            render_count=args.render_count,
            render_size=args.render_size,
            sample_steps=args.sample_steps,
            render_seed=args.render_seed,
        )


if __name__ == "__main__":
    main()
