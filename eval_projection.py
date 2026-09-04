from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.box2d_render import render_transition_panel
from data.dataset import NextStepDataset
from src.losses.projection import ProjectionLoss, finite_difference_state, free_position, make_projection_condition
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


def parse_steps(value: Any) -> list[int]:
    if isinstance(value, list):
        steps = [int(item) for item in value]
    elif isinstance(value, str):
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        steps = [int(value)]
    return sorted(set(max(1, step) for step in steps))


def stats_tensors(loss_kwargs: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(loss_kwargs["correction_mean"], dtype=torch.float32, device=device).view(1, 1, 2)
    std = torch.tensor(loss_kwargs["correction_std"], dtype=torch.float32, device=device).view(1, 1, 2)
    return mean, std.clamp_min(1e-6)


def denormalize(correction: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return correction * std + mean


def predict_projection_step(
    model: torch.nn.Module,
    projection_type: str,
    state: torch.Tensor,
    radius: torch.Tensor,
    solver_steps: int,
    correction_mean: torch.Tensor,
    correction_std: torch.Tensor,
    loss_kwargs: dict[str, Any],
    tau_zero_for_single_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_step = float(loss_kwargs["time_step"])
    proposal = free_position(
        state,
        time_step=time_step,
        gravity_y=float(loss_kwargs["gravity_y"]),
        linear_damping=float(loss_kwargs.get("linear_damping", 0.0)),
    )
    condition = make_projection_condition(state, proposal)

    if projection_type == "delta":
        correction = denormalize(model(proposal, radius, condition), correction_mean, correction_std)
        next_pos = proposal + correction
    else:
        z = proposal
        dtau = 1.0 / float(solver_steps)
        for step in range(solver_steps):
            value = 0.0 if tau_zero_for_single_step and solver_steps == 1 else (step + 0.5) * dtau
            tau = state.new_full((state.size(0), 1), value)
            dz = denormalize(model(z, tau, radius, condition), correction_mean, correction_std)
            z = z + dtau * dz
        next_pos = z

    next_state = finite_difference_state(state, next_pos, time_step)
    return next_state, proposal, next_pos


def add_state_mse(
    totals: dict[str, float],
    counts: dict[str, int],
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    err = (pred - target).pow(2)
    values = {
        f"{prefix}_state_mse": err.mean(dim=(1, 2)),
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
        "rollout_state_mse": err.mean(dim=(2, 3)),
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


def rollout_prediction(
    model: torch.nn.Module,
    projection_type: str,
    initial: torch.Tensor,
    radius: torch.Tensor,
    rollout_steps: int,
    solver_steps: int,
    correction_mean: torch.Tensor,
    correction_std: torch.Tensor,
    loss_kwargs: dict[str, Any],
) -> torch.Tensor:
    state = initial
    trajectory = []
    for _ in range(rollout_steps):
        state, _, _ = predict_projection_step(
            model=model,
            projection_type=projection_type,
            state=state,
            radius=radius,
            solver_steps=solver_steps,
            correction_mean=correction_mean,
            correction_std=correction_std,
            loss_kwargs=loss_kwargs,
        )
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)


def evaluate_rollouts(
    model: torch.nn.Module,
    projection_type: str,
    dataset: NextStepDataset,
    device: torch.device,
    checkpoint: dict[str, Any],
    batch_size: int,
    solver_steps: int,
    requested_steps: int,
) -> dict[str, float]:
    if dataset.rollout_initial is None or dataset.rollout_target is None or dataset.rollout_radius is None:
        return {}

    correction_mean, correction_std = stats_tensors(checkpoint["loss_kwargs"], device)
    saved_steps = int(dataset.rollout_target.size(1))
    steps = min(saved_steps, requested_steps if requested_steps > 0 else saved_steps)
    y_ground = float(checkpoint["loss_kwargs"].get("y_ground", dataset.meta.get("y_ground", 0.0)))
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
            pred = rollout_prediction(
                model=model,
                projection_type=projection_type,
                initial=initial,
                radius=radius,
                rollout_steps=steps,
                solver_steps=solver_steps,
                correction_mean=correction_mean,
                correction_std=correction_std,
                loss_kwargs=checkpoint["loss_kwargs"],
            )
            add_rollout_metrics(totals, counts, pred, target, radius, length, y_ground, slop)

    return {key: totals[key] / counts[key] for key in totals}


def render_predictions(
    model: torch.nn.Module,
    projection_type: str,
    dataset: NextStepDataset,
    device: torch.device,
    checkpoint: dict[str, Any],
    render_dir: str,
    count: int,
    image_size: int,
    solver_steps: int,
) -> None:
    xy_limit = float(dataset.meta.get("xy_limit", 1.0))
    y_ground = float(dataset.meta.get("y_ground", 0.0))
    correction_mean, correction_std = stats_tensors(checkpoint["loss_kwargs"], device)
    out_dir = Path(render_dir)

    with torch.no_grad():
        for idx in range(min(count, len(dataset))):
            item = dataset[idx]
            state = item["source"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            radius = item["radius"].unsqueeze(0).to(device)
            _, proposal, pred_pos = predict_projection_step(
                model=model,
                projection_type=projection_type,
                state=state,
                radius=radius,
                solver_steps=solver_steps,
                correction_mean=correction_mean,
                correction_std=correction_std,
                loss_kwargs=checkpoint["loss_kwargs"],
                tau_zero_for_single_step=True,
            )
            render_transition_panel(
                first=proposal[0].cpu(),
                second=pred_pos[0].cpu(),
                target=target[0, :, :2].cpu(),
                radius=radius[0].cpu(),
                output_path=out_dir / f"{idx:04d}_projection_{projection_type}.png",
                xy_limit=xy_limit,
                y_ground=y_ground,
                image_size=image_size,
                labels=("proposal", "prediction", "target"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate position-projection delta/FM/gradient-FM checkpoint.")
    parser.add_argument("--config", default="configs/eval_projection.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_value = str(get(cfg, "eval", "checkpoint", ""))
    if not checkpoint_value:
        raise ValueError("Set eval.checkpoint in configs/eval_projection.yaml.")
    checkpoint_path = Path(checkpoint_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    projection_type = checkpoint.get("projection_type", str(checkpoint.get("model_type", "projection_fm")).replace("projection_", ""))
    dataset_path = str(get(cfg, "eval", "dataset", "")) or checkpoint["config"]["data"]["dataset"]
    split = str(get(cfg, "eval", "split", "val"))
    device = resolve_device(str(get(cfg, "runtime", "device", "auto")))
    solver_steps_values = parse_steps(get(cfg, "eval", "solver_steps", [1, 4]))
    rollout_solver_steps = int(get(cfg, "eval", "rollout_solver_steps", solver_steps_values[-1]))
    rollout_steps = int(get(cfg, "eval", "rollout_steps", 300))
    batch_size = int(get(cfg, "eval", "batch_size", 512))

    dataset = NextStepDataset(dataset_path, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(get(cfg, "eval", "num_workers", 0)),
    )

    model = build_projection_model(projection_type, **checkpoint["model_kwargs"]).to(device)
    loss_fn = ProjectionLoss(**checkpoint["loss_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss_fn.eval()
    correction_mean, correction_std = stats_tensors(checkpoint["loss_kwargs"], device)

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

            if projection_type == "delta":
                pred_state, _, _ = predict_projection_step(
                    model=model,
                    projection_type=projection_type,
                    state=source,
                    radius=radius,
                    solver_steps=1,
                    correction_mean=correction_mean,
                    correction_std=correction_std,
                    loss_kwargs=checkpoint["loss_kwargs"],
                )
                add_state_mse(totals, counts, "delta", pred_state, target)
                add_state_mse(totals, counts, "delta_dynamic", pred_state, target, dynamic)
                add_state_mse(totals, counts, "delta_resting", pred_state, target, ~dynamic)
            else:
                for solver_steps in solver_steps_values:
                    pred_state, _, _ = predict_projection_step(
                        model=model,
                        projection_type=projection_type,
                        state=source,
                        radius=radius,
                        solver_steps=solver_steps,
                        correction_mean=correction_mean,
                        correction_std=correction_std,
                        loss_kwargs=checkpoint["loss_kwargs"],
                        tau_zero_for_single_step=solver_steps == 1,
                    )
                    prefix = f"k{solver_steps}"
                    add_state_mse(totals, counts, prefix, pred_state, target)
                    add_state_mse(totals, counts, f"{prefix}_dynamic", pred_state, target, dynamic)
                    add_state_mse(totals, counts, f"{prefix}_resting", pred_state, target, ~dynamic)

    metrics = {key: totals[key] / counts[key] for key in totals}
    metrics.update(
        evaluate_rollouts(
            model=model,
            projection_type=projection_type,
            dataset=dataset,
            device=device,
            checkpoint=checkpoint,
            batch_size=batch_size,
            solver_steps=rollout_solver_steps,
            requested_steps=rollout_steps,
        )
    )
    metrics["loss"] = loss_total / max(loss_batches, 1)
    report = {
        "checkpoint": str(checkpoint_path),
        "model_type": checkpoint.get("model_type", f"projection_{projection_type}"),
        "projection_type": projection_type,
        "dataset": dataset_path,
        "split": split,
        "device": str(device),
        "solver_steps": solver_steps_values,
        "rollout_solver_steps": rollout_solver_steps,
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
            model=model,
            projection_type=projection_type,
            dataset=dataset,
            device=device,
            checkpoint=checkpoint,
            render_dir=render_dir,
            count=int(get(cfg, "render", "count", 8)),
            image_size=int(get(cfg, "render", "size", 1024)),
            solver_steps=int(get(cfg, "render", "solver_steps", rollout_solver_steps)),
        )


if __name__ == "__main__":
    main()
