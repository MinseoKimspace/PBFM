"""A: numerical preflight. B/C: matched raw and guarded finite-time map tests."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.data import prepare
from src.multiplier_flow.evaluation import evaluate, preflight
from src.multiplier_flow.model import MultiplierMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_flow.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--objective", choices=("endpoint", "map"), default="endpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--checkpoint", choices=("best", "best_solver", "last"), default="best")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.outdir:
        config["outdir"] = args.outdir
    torch.set_num_threads(int(config["cpu_threads"]))
    root = Path(config["outdir"])
    if args.preflight:
        result = preflight(config, root / "preflight.json")
        print(f"Preflight {'PASS' if result['passed'] else 'MISS'}: {root / 'preflight.json'}")
        if not result["passed"]:
            raise SystemExit(1)
        return
    selected_device = device(args.device or config["device"])
    checkpoint_path = root / args.objective / f"{args.checkpoint}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    if checkpoint.get("format") != "multiplier_map_v1" or checkpoint["objective"] != args.objective:
        raise ValueError("Expected a matching multiplier_map_v1 checkpoint")
    for section in ("model", "physics", "dynamics", "reference", "data", "seed"):
        if checkpoint["config"][section] != config[section]:
            raise ValueError(f"Checkpoint {section} differs; do not change the learned problem/clock at inference")
    cache = prepare(config, root / "segments.pt")
    if cache["spec"] != checkpoint["cache_spec"]:
        raise ValueError("Checkpoint and segment cache differ")
    model = MultiplierMap(**config["model"]).to(selected_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    suffix = "" if args.checkpoint == "best" else f"_{args.checkpoint}"
    output = root / args.objective / f"eval_{args.split}{suffix}.json"
    evaluate(model, cache["splits"][args.split], config, selected_device, output,
             metadata=dict(checkpoint=str(checkpoint_path),
                           objective=checkpoint["objective"], epoch=checkpoint["epoch"],
                           split=args.split, cache_format=cache["format"]))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
