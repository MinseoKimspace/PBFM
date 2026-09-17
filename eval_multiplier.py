"""Validate QP endpoints or evaluate the conditional flow matching solver."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.data import prepare
from src.multiplier_flow.benchmark import validate_budgets
from src.multiplier_flow.evaluation import evaluate, preflight
from src.multiplier_flow.experiment import VARIANTS, pair_cache_path, resolve_experiment
from src.multiplier_flow.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--calls", type=int, nargs="+", help="CFM NFE and matching fixed PGS sweep budgets")
    parser.add_argument("--timing-repeats", type=int, help="Timed repetitions after one warmup per solver")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", help="Evaluation JSON path; also writes a Markdown comparison")
    parser.add_argument("--guarded", action="store_true", help="Also evaluate the explicit Q/nonnegativity guard")
    parser.add_argument("--checkpoint", choices=("best", "best_solver", "last"), default="best_solver")
    args = parser.parse_args()
    config = resolve_experiment(load_config(args.config), args.variant, args.outdir)
    if args.calls is not None:
        config["evaluation"]["calls"] = args.calls
    if args.timing_repeats is not None:
        config["evaluation"]["timing_repeats"] = args.timing_repeats
    if args.no_render:
        config["evaluation"]["render"] = False
    validate_budgets(config["evaluation"])
    if args.output and Path(args.output).suffix.lower() != ".json":
        parser.error("--output must be a .json path; the .md companion is generated automatically")
    if args.guarded:
        config["evaluation"]["guarded"] = True
    torch.set_num_threads(int(config["cpu_threads"]))
    root = Path(config["outdir"])
    if args.preflight:
        result = preflight(config, root / "preflight.json")
        print(f"Preflight {'PASS' if result['passed'] else 'MISS'}: {root / 'preflight.json'}")
        if not result["passed"]:
            raise SystemExit(1)
        return
    selected_device = device(args.device or config["device"])
    path = root / "cfm" / f"{args.checkpoint}.pt"
    model, checkpoint = load_model(path, config, selected_device)
    cache = prepare(config, pair_cache_path(config))
    if cache["spec"] != checkpoint["cache_spec"]:
        raise ValueError("Checkpoint and CFM pair cache differ")
    suffix = "_guarded" if config["evaluation"].get("guarded", False) else ""
    output = Path(args.output) if args.output else root / "cfm" / f"eval_{args.split}_{args.checkpoint}_budget{suffix}.json"
    evaluate(model, cache["splits"][args.split], config, selected_device, output,
             metadata=dict(checkpoint=str(path), objective="cfm", epoch=checkpoint["epoch"],
                           checkpoint_format=checkpoint["format"], solver=checkpoint["solver"],
                           updates=checkpoint["updates"], selection_metric=checkpoint["selection_metric"],
                           split=args.split, cache_format=cache["format"]))
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
