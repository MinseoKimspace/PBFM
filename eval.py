from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml
from torch.utils.data import DataLoader

try:
    from data.box2d_render import render_state_image
except ModuleNotFoundError:
    from box2d_render import render_state_image

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
    vector_overlay_default = bool(
        _get_nested(
            cfg,
            "render",
            "vector_overlay",
            default=_get_nested(cfg, "render", "vector_field", default=False),
        )
    )

    parser = argparse.ArgumentParser(description="Evaluate FM + physics residual checkpoint.")
    parser.add_argument("--config", type=str, default=pre_args.config)

    parser.add_argument("--checkpoint", type=str, default=str(_get_nested(cfg, "eval", "checkpoint", default="")))
    parser.add_argument("--dataset", type=str, default=str(_get_nested(cfg, "eval", "dataset", default="")))
    parser.add_argument("--split", type=str, default=str(_get_nested(cfg, "eval", "split", default="val")))
    parser.add_argument("--batch-size", type=int, default=int(_get_nested(cfg, "eval", "batch_size", default=256)))
    parser.add_argument("--num-workers", type=int, default=int(_get_nested(cfg, "eval", "num_workers", default=0)))
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
    parser.add_argument(
        "--render-trajectory",
        action=argparse.BooleanOptionalAction,
        default=bool(_get_nested(cfg, "render", "trajectory", default=True)),
    )
    parser.add_argument(
        "--render-vector-overlay",
        action=argparse.BooleanOptionalAction,
        default=vector_overlay_default,
    )
    parser.add_argument(
        "--render-vector-stride",
        type=int,
        default=int(_get_nested(cfg, "render", "vector_stride", default=8)),
    )
    parser.add_argument(
        "--render-vector-scale",
        type=float,
        default=float(_get_nested(cfg, "render", "vector_scale", default=4.0)),
    )
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


def sample_flow_euler(
    model: torch.nn.Module,
    x0: torch.Tensor,
    radius: torch.Tensor,
    steps: int,
    return_trajectory: bool = False,
    return_vector_field: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    if steps < 1:
        raise ValueError(f"sample steps must be >= 1, got {steps}")
    x = x0
    trajectory = [x.detach().clone()] if return_trajectory else None
    vector_field = [] if return_vector_field else None
    dt = 1.0 / float(steps)
    for i in range(steps):
        t_now = torch.full(
            (x.size(0), 1),
            (i + 0.5) / float(steps),
            device=x.device,
            dtype=x.dtype,
        )
        t_start = torch.zeros_like(t_now)
        v_hat = model(x, t_start, t_now, radius=radius)
        if vector_field is not None:
            vector_field.append(v_hat.detach().clone())
        x = x + dt * v_hat
        if trajectory is not None:
            trajectory.append(x.detach().clone())
    return x, trajectory, vector_field


def clamp_state_for_render(
    state: torch.Tensor,
    xy_limit: float,
    y_ground: float,
) -> torch.Tensor:
    y_top = y_ground + xy_limit
    state_render = state.detach().clone()
    state_render[..., 0].clamp_(-xy_limit, xy_limit)
    state_render[..., 1].clamp_(y_ground, y_top)
    return state_render


def stack_trajectory_for_render(
    trajectory: list[torch.Tensor] | None,
    xy_limit: float,
    y_ground: float,
) -> torch.Tensor | None:
    if trajectory is None:
        return None
    return torch.stack(
        [clamp_state_for_render(state[0], xy_limit=xy_limit, y_ground=y_ground) for state in trajectory],
        dim=0,
    )


def stack_vector_field_for_render(vector_field: list[torch.Tensor] | None) -> torch.Tensor | None:
    if vector_field is None:
        return None
    return torch.stack([step[0].detach().clone() for step in vector_field], dim=0)


def render_model_samples(
    model: torch.nn.Module,
    dataset: RelaxedCirclesDataset,
    device: torch.device,
    render_dir: str,
    render_count: int,
    render_size: int,
    sample_steps: int,
    render_seed: int,
    render_trajectory: bool,
    render_vector_overlay: bool,
    render_vector_stride: int,
    render_vector_scale: float,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print("PIL not available. Skipping render.")
        return

    out_dir = Path(render_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xy_limit = float(dataset.meta.get("xy_limit", 1.0))
    y_ground = float(dataset.meta.get("y_ground", 0.0))
    count = min(render_count, len(dataset))
    capture_trace = render_trajectory or render_vector_overlay

    with torch.no_grad():
        for idx in range(count):
            item = dataset[idx]
            x1 = item["state"].unsqueeze(0).to(device)
            radius = item["radius"].unsqueeze(0).to(device)
            x0 = item["state_init"].unsqueeze(0).to(device)
            x_hat, trajectory, vector_field = sample_flow_euler(
                model,
                x0,
                radius=radius,
                steps=sample_steps,
                return_trajectory=capture_trace,
                return_vector_field=render_vector_overlay,
            )

            x0_render = clamp_state_for_render(x0[0], xy_limit=xy_limit, y_ground=y_ground)
            x_hat_render = clamp_state_for_render(x_hat[0], xy_limit=xy_limit, y_ground=y_ground)
            x1_render = clamp_state_for_render(x1[0], xy_limit=xy_limit, y_ground=y_ground)
            trajectory_render = stack_trajectory_for_render(trajectory, xy_limit=xy_limit, y_ground=y_ground)
            vector_field_render = stack_vector_field_for_render(vector_field)

            img_x0 = render_state_image(
                x0_render.cpu(), radius[0].cpu(), xy_limit=xy_limit, y_ground=y_ground, image_size=render_size
            )
            img_pred = render_state_image(
                x_hat_render.cpu(),
                radius[0].cpu(),
                xy_limit=xy_limit,
                y_ground=y_ground,
                image_size=render_size,
                trajectory=(trajectory_render.cpu() if render_vector_overlay and trajectory_render is not None else None),
                vector_field=(
                    vector_field_render.cpu() if render_vector_overlay and vector_field_render is not None else None
                ),
                vector_stride=render_vector_stride,
                vector_scale=render_vector_scale,
                vector_dt=1.0 / float(sample_steps),
            )
            img_target = render_state_image(
                x1_render.cpu(), radius[0].cpu(), xy_limit=xy_limit, y_ground=y_ground, image_size=render_size
            )
            if img_x0 is None or img_pred is None or img_target is None:
                print("PIL not available. Skipping render.")
                return

            panel_w, panel_h = img_x0.size
            caption_h = 36
            canvas = Image.new("RGB", (panel_w * 3, panel_h + caption_h), (255, 255, 255))
            canvas.paste(img_x0, (0, caption_h))
            canvas.paste(img_pred, (panel_w, caption_h))
            canvas.paste(img_target, (panel_w * 2, caption_h))

            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
            labels = ("x0", "pred", "target")
            for col, label in enumerate(labels):
                x_left = col * panel_w
                x_center = x_left + panel_w * 0.5
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x_text = int(round(x_center - text_w * 0.5))
                y_text = int(round((caption_h - text_h) * 0.5))
                draw.text((x_text, y_text), label, fill=(0, 0, 0), font=font)

            canvas.save(out_dir / f"{idx:04d}_x0_pred_target.png")

            if render_trajectory and trajectory_render is not None:
                traj_dir = out_dir / f"{idx:04d}_trajectory"
                traj_dir.mkdir(parents=True, exist_ok=True)
                for step_idx, state_render in enumerate(trajectory_render):
                    traj_prefix = trajectory_render[: step_idx + 1] if render_vector_overlay else None
                    vector_prefix = None
                    if render_vector_overlay and vector_field_render is not None and step_idx > 0:
                        vector_prefix = vector_field_render[:step_idx]
                    img_step = render_state_image(
                        state_render.cpu(),
                        radius[0].cpu(),
                        xy_limit=xy_limit,
                        y_ground=y_ground,
                        image_size=render_size,
                        trajectory=(traj_prefix.cpu() if traj_prefix is not None else None),
                        vector_field=(vector_prefix.cpu() if vector_prefix is not None else None),
                        vector_stride=render_vector_stride,
                        vector_scale=render_vector_scale,
                        vector_dt=1.0 / float(sample_steps),
                    )
                    if img_step is None:
                        print("PIL not available. Skipping render.")
                        return
                    img_step.save(traj_dir / f"step_{step_idx:03d}.png")

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
            "use_radius_condition": False,
        },
    )
    loss_kwargs = checkpoint.get(
        "loss_kwargs",
        {
            "physics_weight": float(checkpoint.get("physics_weight", train_args.get("physics_weight", 0.1))),
            "physics_weight_min": float(train_args.get("physics_weight_min", args.physics_weight_min)),
            "physics_weight_max": float(train_args.get("physics_weight_max", args.physics_weight_max)),
            "physics_weight_eps": float(train_args.get("physics_weight_eps", args.physics_weight_eps)),
            "gravity_weight": 0.2,
            "ground_weight": 0.2,
            "collision_weight": 0.2,
            "collision_alpha": float(train_args.get("collision_alpha", args.collision_alpha)),
            "collision_epsilon": float(train_args.get("collision_epsilon", args.collision_epsilon)),
            "collision_constant": float(train_args.get("collision_constant", args.collision_constant)),
            "y_ground": 0.0,
        },
    )
    if "physics_weight" not in loss_kwargs:
        loss_kwargs["physics_weight"] = float(checkpoint.get("physics_weight", train_args.get("physics_weight", 0.1)))
    if "physics_weight_min" not in loss_kwargs:
        loss_kwargs["physics_weight_min"] = float(train_args.get("physics_weight_min", args.physics_weight_min))
    if "physics_weight_max" not in loss_kwargs:
        loss_kwargs["physics_weight_max"] = float(train_args.get("physics_weight_max", args.physics_weight_max))
    if "physics_weight_eps" not in loss_kwargs:
        loss_kwargs["physics_weight_eps"] = float(train_args.get("physics_weight_eps", args.physics_weight_eps))
    if "collision_alpha" not in loss_kwargs:
        loss_kwargs["collision_alpha"] = float(train_args.get("collision_alpha", args.collision_alpha))
    if "collision_epsilon" not in loss_kwargs:
        loss_kwargs["collision_epsilon"] = float(train_args.get("collision_epsilon", args.collision_epsilon))
    if "collision_constant" not in loss_kwargs:
        loss_kwargs["collision_constant"] = float(train_args.get("collision_constant", args.collision_constant))

    dataset_path = args.dataset if args.dataset else str(train_args.get("dataset", ""))
    if not dataset_path:
        raise ValueError("Dataset path is required. Pass --dataset or use a checkpoint saved from dataset-based training.")

    split = args.split if args.split else str(train_args.get("val_split", "val"))

    dataset = RelaxedCirclesDataset(
        dataset_path=dataset_path,
        split=split,
        use_relaxed_state=True,
        return_init_state=True,
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
            x0 = batch["state_init"].to(device)
            r = batch["radius"].to(device)
            _, metrics = criterion(model, x0, x1, r)
            rows.append(metrics)

    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "split": split,
        "path": "state_init->state_relaxed",
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
            render_trajectory=args.render_trajectory,
            render_vector_overlay=args.render_vector_overlay,
            render_vector_stride=args.render_vector_stride,
            render_vector_scale=args.render_vector_scale,
        )


if __name__ == "__main__":
    main()
