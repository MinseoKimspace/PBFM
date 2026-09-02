from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import ProjectionTransitionDataset
from data.box2d_render import render_transition_panel
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


def sample_flow(
    model: torch.nn.Module,
    source: torch.Tensor,
    radius: torch.Tensor,
    condition: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    z = source
    dtau = 1.0 / float(steps)
    for step in range(steps):
        tau = source.new_full((source.size(0), 1), (step + 0.5) * dtau)
        z = z + dtau * model(z, tau, radius, condition)
    return z


def render_predictions(
    model: torch.nn.Module,
    dataset: ProjectionTransitionDataset,
    device: torch.device,
    render_dir: str,
    count: int,
    image_size: int,
    sample_steps: int,
) -> None:
    xy_limit = float(dataset.meta.get("xy_limit", 1.0))
    y_ground = float(dataset.meta.get("y_ground", 0.0))
    out_dir = Path(render_dir)

    with torch.no_grad():
        for idx in range(min(count, len(dataset))):
            item = dataset[idx]
            source = item["source"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            condition = item["condition"].unsqueeze(0).to(device)
            radius = item["radius"].unsqueeze(0).to(device)
            prediction = sample_flow(model, source, radius, condition, sample_steps)
            render_transition_panel(
                condition=source[0].cpu(),
                source=prediction[0].cpu(),
                target=target[0].cpu(),
                radius=radius[0].cpu(),
                output_path=out_dir / f"{idx:04d}_prediction.png",
                xy_limit=xy_limit,
                y_ground=y_ground,
                image_size=image_size,
                labels=("proposal", "prediction", "target"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate projection flow checkpoint.")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_value = str(get(cfg, "eval", "checkpoint", ""))
    if not checkpoint_value:
        raise ValueError("Set eval.checkpoint in configs/eval.yaml.")
    checkpoint_path = Path(checkpoint_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dataset_path = str(get(cfg, "eval", "dataset", ""))
    if not dataset_path:
        dataset_path = checkpoint["config"]["data"]["dataset"]
    split = str(get(cfg, "eval", "split", "val"))
    device = resolve_device(str(get(cfg, "runtime", "device", "auto")))

    dataset = ProjectionTransitionDataset(dataset_path, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(get(cfg, "eval", "batch_size", 256)),
        shuffle=False,
        num_workers=int(get(cfg, "eval", "num_workers", 0)),
    )

    model = FlowVelocityNet(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = ProjectionFlowLoss(**checkpoint["loss_kwargs"]).to(device)
    model.eval()

    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            condition = batch["condition"].to(device)
            radius = batch["radius"].to(device)
            _, metrics = loss_fn(model, source, target, radius, condition)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

    metrics = {key: value / len(loader) for key, value in totals.items()}
    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_path,
        "split": split,
        "device": str(device),
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
            dataset,
            device,
            render_dir,
            int(get(cfg, "render", "count", 8)),
            int(get(cfg, "render", "size", 1024)),
            int(get(cfg, "render", "sample_steps", checkpoint["loss_kwargs"]["unroll_steps"])),
        )


if __name__ == "__main__":
    main()
