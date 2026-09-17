"""Compare PGS, CFM and FM+PGS in physical time, with side-by-side GIFs."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.model import load_model
from src.multiplier_flow.experiment import VARIANTS, resolve_experiment
from src.multiplier_flow.rollout import evaluate_motion, rollout_budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm_large.yaml")
    parser.add_argument("--pgs-only", action="store_true", help="No checkpoint required")
    parser.add_argument("--checkpoint", choices=("best", "best_solver", "last"), default="best_solver")
    parser.add_argument("--device")
    parser.add_argument("--outdir", help="Checkpoint root")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--steps", type=int, help="Physical frames, not CFM integration steps")
    parser.add_argument("--calls", type=int, nargs="+", help="Standalone FM budgets; hybrid budgets also get matching raw FM")
    hybrid_flags = parser.add_mutually_exclusive_group()
    hybrid_flags.add_argument("--hybrid", action="store_true", help="Include raw FM + PGS to tolerance")
    hybrid_flags.add_argument("--no-hybrid", action="store_true", help="Disable hybrid rollouts")
    parser.add_argument("--hybrid-calls", type=int, nargs="+", help="Enable hybrid rollouts with these FM budgets")
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", help="Use a new report directory")
    args = parser.parse_args()
    if args.no_hybrid and args.hybrid_calls is not None:
        parser.error("--no-hybrid cannot be combined with --hybrid-calls")
    if args.pgs_only and (args.hybrid or args.hybrid_calls is not None):
        parser.error("--pgs-only cannot be combined with explicit hybrid options")
    config = resolve_experiment(load_config(args.config), args.variant, args.outdir)
    if args.calls is not None:
        config["rollout"]["calls"] = args.calls
    if args.hybrid or args.no_hybrid or args.hybrid_calls is not None:
        options = config["rollout"].setdefault("hybrid", {})
        options["enabled"] = not args.no_hybrid
        if args.hybrid_calls is not None:
            options["calls"] = args.hybrid_calls
    try:
        _, hybrid_calls = rollout_budgets(config["rollout"])
    except ValueError as error:
        parser.error(str(error))
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
            checkpoint_format=checkpoint["format"], solver=checkpoint["solver"],
            training_config={key: checkpoint["config"][key] for key in ("seed", "data", "train")},
            selection_metric=checkpoint["selection_metric"])
    label = "pgs" if args.pgs_only else f"{args.checkpoint}_cfm"
    if hybrid_calls and not args.pgs_only:
        label += "_hybrid"
    if args.guarded:
        label += "_guarded"
    output = Path(args.output) if args.output else root / "motion" / label
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite motion results: {output}; choose --output")
    evaluate_motion(models, config, selected_device, output / "rollout.json", metadata)


if __name__ == "__main__":
    main()
