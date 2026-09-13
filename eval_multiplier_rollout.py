"""Compare PGS/B/C in physical time. Does not require/generate a segment cache."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.model import MultiplierMap
from src.multiplier_flow.rollout import evaluate_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_flow_large.yaml")
    parser.add_argument("--objectives", nargs="*", choices=("endpoint", "map"), default=["endpoint", "map"])
    parser.add_argument("--checkpoint", choices=("best", "best_solver", "last"), default="best")
    parser.add_argument("--device")
    parser.add_argument("--outdir", help="Checkpoint root, not the report destination")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", help="Use a new directory to preserve previous reports")
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
    for objective in args.objectives:
        path = root / objective / f"{args.checkpoint}.pt"
        checkpoint = torch.load(path, map_location=selected_device, weights_only=True)
        if checkpoint.get("format") != "multiplier_map_v1" or checkpoint["objective"] != objective:
            raise ValueError(f"Invalid {objective} checkpoint: {path}")
        for section in ("model", "physics", "dynamics"):
            if config[section] != checkpoint["config"][section]:
                raise ValueError(f"Checkpoint {section} differs; use its physical settings")
        if config["reference"]["eta_fraction"] != checkpoint["config"]["reference"]["eta_fraction"]:
            raise ValueError("Do not change the learned reference clock at inference")
        model = MultiplierMap(**config["model"]).to(selected_device)
        model.load_state_dict(checkpoint["model"])
        models[objective] = model.eval()
        metadata[objective] = dict(checkpoint=str(path), epoch=checkpoint["epoch"], updates=checkpoint.get("updates"),
            training_config={key: checkpoint["config"][key] for key in ("seed", "data", "train")},
            selection_metric=checkpoint.get("selection_metric"))
    output = Path(args.output) if args.output else root / "motion" / f"{args.checkpoint}_{'-'.join(args.objectives) or 'pgs'}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite motion results: {output}; choose --output")
    evaluate_motion(models, config, selected_device, output / "rollout.json", metadata)


if __name__ == "__main__":
    main()
