"""Validate QP endpoints or evaluate the conditional flow matching solver."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.data import prepare
from src.multiplier_flow.evaluation import evaluate, preflight
from src.multiplier_flow.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--guarded", action="store_true", help="Also evaluate the explicit Q/nonnegativity guard")
    parser.add_argument("--checkpoint", choices=("best", "best_solver", "last"), default="best_solver")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.outdir:
        config["outdir"] = args.outdir
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
    cache = prepare(config, root / "pairs.pt")
    if cache["spec"] != checkpoint["cache_spec"]:
        raise ValueError("Checkpoint and CFM pair cache differ")
    suffix = "_guarded" if config["evaluation"].get("guarded", False) else ""
    output = root / "cfm" / f"eval_{args.split}_{args.checkpoint}{suffix}.json"
    evaluate(model, cache["splits"][args.split], config, selected_device, output,
             metadata=dict(checkpoint=str(path), objective="cfm", epoch=checkpoint["epoch"],
                           checkpoint_format=checkpoint["format"], solver=checkpoint["solver"],
                           updates=checkpoint["updates"], selection_metric=checkpoint["selection_metric"],
                           split=args.split, cache_format=cache["format"]))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
