"""Compare PGS and CFM in physical time; no endpoint-pair cache is needed."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.model import load_model
from src.multiplier_flow.rollout import evaluate_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm_large.yaml")
    parser.add_argument("--pgs-only", action="store_true", help="No checkpoint required")
    parser.add_argument("--checkpoint", choices=("best", "best_solver", "last"), default="best_solver")
    parser.add_argument("--device")
    parser.add_argument("--outdir", help="Checkpoint root")
    parser.add_argument("--steps", type=int, help="Physical frames, not CFM integration steps")
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", help="Use a new report directory")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.outdir:
        config["outdir"] = args.outdir
    if args.steps is not None:
        config["rollout"]["steps"] = args.steps
    if args.guarded:
        config["rollout"]["guarded"] = True
    if args.no_render:
        config["rollout"]["render"] = False
    selected_device = device(args.device or config["device"])
    torch.set_num_threads(int(config["cpu_threads"]))
    root = Path(config["outdir"])
    models, metadata = {}, {}
    if not args.pgs_only:
        path = root / "cfm" / f"{args.checkpoint}.pt"
        model, checkpoint = load_model(path, config, selected_device)
        models["cfm"] = model
        metadata["cfm"] = dict(checkpoint=str(path), epoch=checkpoint["epoch"], updates=checkpoint["updates"],
            training_config={key: checkpoint["config"][key] for key in ("seed", "data", "train")},
            selection_metric=checkpoint["selection_metric"])
    label = "pgs" if args.pgs_only else f"{args.checkpoint}_cfm"
    if args.guarded:
        label += "_guarded"
    output = Path(args.output) if args.output else root / "motion" / label
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite motion results: {output}; choose --output")
    evaluate_motion(models, config, selected_device, output / "rollout.json", metadata)


if __name__ == "__main__":
    main()
