from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.box2d_render import render_transition_panel
from data.dataset import NextStepDataset
from src.losses.combined import DeltaLoss, NextStepFlowLoss
from src.models.network import DeltaVelocityNet, FlowVelocityNet

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


def stats_tensors(loss_kwargs: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(loss_kwargs["delta_mean"], dtype=torch.float32, device=device).view(1, 1, 4)
    std = torch.tensor(loss_kwargs["delta_std"], dtype=torch.float32, device=device).view(1, 1, 4)
    return mean, std.clamp_min(1e-6)


def unnormalize(delta: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return delta * std + mean


def sample_flow(
    model: torch.nn.Module,
    source: torch.Tensor,
    radius: torch.Tensor,
    steps: int,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
    tau_zero_for_single_step: bool = False,
) -> torch.Tensor:
    condition = source
    z = source
    dtau = 1.0 / float(steps)
    for step in range(steps):
        value = 0.0 if tau_zero_for_single_step and steps == 1 else (step + 0.5) * dtau
        tau = source.new_full((source.size(0), 1), value)
        dz = unnormalize(model(z, tau, radius, condition), delta_mean, delta_std)
        z = z + dtau * dz
    return z


def predict_delta(
    model: torch.nn.Module,
    source: torch.Tensor,
    radius: torch.Tensor,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
) -> torch.Tensor:
    return source + unnormalize(model(source, radius), delta_mean, delta_std)


def add_mse(
    totals: dict[str, float],
    counts: dict[str, int],
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    err = (pred - target).pow(2)
    values = {
        f"{prefix}_mse": err.mean(dim=(1, 2)),
        f"{prefix}_pos_mse": err[..., :2].mean(dim=(1, 2)),
        f"{prefix}_vel_mse": err[..., 2:].mean(dim=(1, 2)),
    }
    if mask is None:
        mask = torch.ones(pred.size(0), dtype=torch.bool, device=pred.device)
    count = int(mask.sum().item())
    if count == 0:
        return
    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + float(value[mask].sum().item())
        counts[key] = counts.get(key, 0) + count


def rollout_prediction(
    model: torch.nn.Module,
    model_type: str,
    initial: torch.Tensor,
    radius: torch.Tensor,
    steps: int,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
    ode_steps: int,
) -> torch.Tensor:
    state = initial
    trajectory = []
    for _ in range(steps):
        if model_type == "delta":
            state = predict_delta(model, state, radius, delta_mean, delta_std)
        else:
            state = sample_flow(model, state, radius, ode_steps, delta_mean, delta_std)
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)


def add_rollout_metrics(
    totals: dict[str, float],
    counts: dict[str, int],
    pred: torch.Tensor,
    target: torch.Tensor,
    radius: torch.Tensor,
    length: torch.Tensor,
    y_ground: float,
    slop: float,
) -> None:
    valid = torch.arange(pred.size(1), device=pred.device).unsqueeze(0) < length.unsqueeze(1)
    count = int(valid.sum().item())
    if count == 0:
        return

    err = (pred - target).pow(2)
    values = {
        "rollout_mse": err.mean(dim=(2, 3)),
        "rollout_pos_mse": err[..., :2].mean(dim=(2, 3)),
        "rollout_vel_mse": err[..., 2:].mean(dim=(2, 3)),
    }

    pos = pred[..., :2]
    ground_penetration = torch.relu(float(y_ground) + radius.unsqueeze(1) - pos[..., 1] - float(slop))
    pair_delta = pos.unsqueeze(3) - pos.unsqueeze(2)
    pair_dist = torch.linalg.norm(pair_delta, dim=-1)
    pair_radius = radius.unsqueeze(1).unsqueeze(3) + radius.unsqueeze(1).unsqueeze(2)
    pair_penetration = torch.relu(pair_radius - pair_dist - float(slop))
    eye = torch.eye(radius.size(1), dtype=torch.bool, device=pred.device).view(1, 1, radius.size(1), radius.size(1))
    pair_penetration = pair_penetration.masked_fill(eye, 0.0)
    pair_count = max(radius.size(1) * (radius.size(1) - 1), 1)

    values.update(
        {
            "rollout_ground_penetration_mean": ground_penetration.mean(dim=2),
            "rollout_ground_penetration_max": ground_penetration.amax(dim=2),
            "rollout_pair_penetration_mean": pair_penetration.sum(dim=(2, 3)) / float(pair_count),
            "rollout_pair_penetration_max": pair_penetration.amax(dim=(2, 3)),
        }
    )

    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + float(value[valid].sum().item())
        counts[key] = counts.get(key, 0) + count


def evaluate_rollouts(
    model: torch.nn.Module,
    model_type: str,
    dataset: NextStepDataset,
    device: torch.device,
    checkpoint: dict[str, Any],
    batch_size: int,
    ode_steps: int,
    requested_steps: int,
) -> dict[str, float]:
    if dataset.rollout_initial is None or dataset.rollout_target is None or dataset.rollout_radius is None:
        return {}

    delta_mean, delta_std = stats_tensors(checkpoint["loss_kwargs"], device)
    saved_steps = int(dataset.rollout_target.size(1))
    steps = min(saved_steps, requested_steps if requested_steps > 0 else saved_steps)
    y_ground = float(dataset.meta.get("y_ground", 0.0))
    slop = float(checkpoint["loss_kwargs"].get("slop", dataset.meta.get("box2d_linear_slop", 0.005)))
    lengths = dataset.rollout_length
    if lengths is None:
        lengths = torch.full((dataset.rollout_initial.size(0),), steps, dtype=torch.long)

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    total_rollouts = int(dataset.rollout_initial.size(0))
    batch_size = max(1, min(batch_size, total_rollouts))

    with torch.no_grad():
        for start in range(0, total_rollouts, batch_size):
            end = min(start + batch_size, total_rollouts)
            initial = dataset.rollout_initial[start:end].to(device)
            target = dataset.rollout_target[start:end, :steps].to(device)
            radius = dataset.rollout_radius[start:end].to(device)
            length = lengths[start:end].to(device).clamp_max(steps)
            pred = rollout_prediction(model, model_type, initial, radius, steps, delta_mean, delta_std, ode_steps)
            add_rollout_metrics(totals, counts, pred, target, radius, length, y_ground, slop)

    return {key: totals[key] / counts[key] for key in totals}


def render_predictions(
    model: torch.nn.Module,
    model_type: str,
    dataset: NextStepDataset,
    device: torch.device,
    checkpoint: dict[str, Any],
    render_dir: str,
    count: int,
    image_size: int,
    sample_steps: int,
) -> None:
    xy_limit = float(dataset.meta.get("xy_limit", 1.0))
    y_ground = float(dataset.meta.get("y_ground", 0.0))
    delta_mean, delta_std = stats_tensors(checkpoint["loss_kwargs"], device)
    out_dir = Path(render_dir)

    with torch.no_grad():
        for idx in range(min(count, len(dataset))):
            item = dataset[idx]
            source = item["source"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            radius = item["radius"].unsqueeze(0).to(device)
            if model_type == "delta":
                prediction = predict_delta(model, source, radius, delta_mean, delta_std)
            else:
                prediction = sample_flow(model, source, radius, sample_steps, delta_mean, delta_std)
            render_transition_panel(
                first=source[0].cpu(),
                second=prediction[0].cpu(),
                target=target[0].cpu(),
                radius=radius[0].cpu(),
                output_path=out_dir / f"{idx:04d}_{model_type}.png",
                xy_limit=xy_limit,
                y_ground=y_ground,
                image_size=image_size,
                labels=("source", "prediction", "target"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate next-step flow or delta checkpoint.")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_value = str(get(cfg, "eval", "checkpoint", ""))
    if not checkpoint_value:
        raise ValueError("Set eval.checkpoint in configs/eval.yaml.")
    checkpoint_path = Path(checkpoint_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type", "flow")
    dataset_path = str(get(cfg, "eval", "dataset", "")) or checkpoint["config"]["data"]["dataset"]
    split = str(get(cfg, "eval", "split", "val"))
    device = resolve_device(str(get(cfg, "runtime", "device", "auto")))

    dataset = NextStepDataset(dataset_path, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(get(cfg, "eval", "batch_size", 256)),
        shuffle=False,
        num_workers=int(get(cfg, "eval", "num_workers", 0)),
    )

    if model_type == "delta":
        model = DeltaVelocityNet(**checkpoint["model_kwargs"]).to(device)
        loss_fn = DeltaLoss(**checkpoint["loss_kwargs"]).to(device)
    else:
        model = FlowVelocityNet(**checkpoint["model_kwargs"]).to(device)
        loss_fn = NextStepFlowLoss(**checkpoint["loss_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss_fn.eval()
    delta_mean, delta_std = stats_tensors(checkpoint["loss_kwargs"], device)
    ode_steps = int(get(cfg, "eval", "ode_steps", checkpoint["loss_kwargs"].get("unroll_steps", 8)))
    rollout_steps = int(get(cfg, "eval", "rollout_steps", 0))

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    loss_total = 0.0
    loss_batches = 0

    with torch.no_grad():
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            radius = batch["radius"].to(device)
            dynamic = batch["dynamic"].to(device)

            _, metrics = loss_fn(model, source, target, radius)
            loss_total += metrics["loss"]
            loss_batches += 1

            if model_type == "delta":
                pred_delta = predict_delta(model, source, radius, delta_mean, delta_std)
                add_mse(totals, counts, "delta", pred_delta, target)
                add_mse(totals, counts, "delta_dynamic", pred_delta, target, dynamic)
                add_mse(totals, counts, "delta_resting", pred_delta, target, ~dynamic)
            else:
                pred_1 = sample_flow(model, source, radius, 1, delta_mean, delta_std, tau_zero_for_single_step=True)
                pred_ode = sample_flow(model, source, radius, ode_steps, delta_mean, delta_std)
                add_mse(totals, counts, "one_step", pred_1, target)
                add_mse(totals, counts, "one_step_dynamic", pred_1, target, dynamic)
                add_mse(totals, counts, "one_step_resting", pred_1, target, ~dynamic)
                add_mse(totals, counts, "ode", pred_ode, target)
                add_mse(totals, counts, "ode_dynamic", pred_ode, target, dynamic)
                add_mse(totals, counts, "ode_resting", pred_ode, target, ~dynamic)

    metrics = {key: totals[key] / counts[key] for key in totals}
    metrics.update(
        evaluate_rollouts(
            model=model,
            model_type=model_type,
            dataset=dataset,
            device=device,
            checkpoint=checkpoint,
            batch_size=int(get(cfg, "eval", "batch_size", 256)),
            ode_steps=ode_steps,
            requested_steps=rollout_steps,
        )
    )
    metrics["loss"] = loss_total / max(loss_batches, 1)
    report = {
        "checkpoint": str(checkpoint_path),
        "model_type": model_type,
        "dataset": dataset_path,
        "split": split,
        "device": str(device),
        "ode_steps": ode_steps,
        "rollout_steps": rollout_steps,
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2))

    output_json = str(get(cfg, "output", "json", ""))
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    render_dir = str(get(cfg, "render", "dir", ""))
    if render_dir:
        render_predictions(
            model,
            model_type,
            dataset,
            device,
            checkpoint,
            render_dir,
            int(get(cfg, "render", "count", 8)),
            int(get(cfg, "render", "size", 1024)),
            int(get(cfg, "render", "sample_steps", ode_steps)),
        )


if __name__ == "__main__":
    main()
