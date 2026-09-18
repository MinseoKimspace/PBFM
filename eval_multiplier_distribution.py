"""Evaluate conditional multiplier diversity and KKT validity together."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.distribution import VARIANTS, build_distribution_split, resolve_distribution, validate_distribution
from src.multiplier_flow.distribution_experiment import evaluate_distribution, load_distribution_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm_ablation.yaml")
    parser.add_argument("--variant", choices=VARIANTS, default="D_distribution")
    parser.add_argument("--device")
    parser.add_argument("--outdir", help="Checkpoint experiment root; the variant is appended")
    parser.add_argument("--checkpoint", choices=("best_distribution", "best", "last"), default="best_distribution")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, help="Must match the checkpoint experiment seed")
    parser.add_argument("--calls", type=int, nargs="+")
    parser.add_argument("--samples-per-qp", type=int)
    parser.add_argument("--batch-size", type=int, help="Evaluation chunk size")
    parser.add_argument("--timing-repeats", type=int)
    parser.add_argument("--hybrid", action="store_true", help="Also measure FM + PGS, including changes to its distribution")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", help="New JSON output path; .md/.pt/.png companions are also saved")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    config = resolve_distribution(config, args.variant, args.outdir)
    settings = config["distribution"]["evaluation"]
    for name in ("calls", "samples_per_qp", "batch_size", "timing_repeats"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if args.hybrid:
        settings["hybrid"] = True
    if args.no_render:
        settings["render"] = False
    validate_distribution(config)
    torch.set_num_threads(config["cpu_threads"])
    selected_device = device(args.device or config["device"])
    run = Path(config["outdir"])/"cfm"
    output = Path(args.output) if args.output else run/f"eval_{args.split}_{args.checkpoint}.json"
    if output.suffix.lower() != ".json":
        parser.error("--output must be a .json path")
    if any(output.with_suffix(suffix).exists() for suffix in (".json", ".md", ".pt", ".png")):
        raise FileExistsError(f"Refusing to overwrite {output}; choose --output")
    checkpoint_path = run/f"{args.checkpoint}.pt"
    model, checkpoint = load_distribution_model(checkpoint_path, config, selected_device)
    split = build_distribution_split(config, args.split)
    evaluate_distribution(model, split, config, selected_device, output=output,
        checkpoint_info=dict(path=str(checkpoint_path), updates=checkpoint["updates"],
                             selection_metric=checkpoint["selection_metric"]))
    print(f"Saved {output} and companions")


if __name__ == "__main__":
    main()
