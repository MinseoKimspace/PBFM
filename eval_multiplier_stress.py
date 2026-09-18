"""Large-circle comparison: sequential PGS versus D global FM + PGS."""
import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.experiment import resolve_experiment
from src.multiplier_flow.model import load_model
from src.multiplier_flow.stress import evaluate_stress, validate_stress
from src.multiplier_flow.stress_problem import SCENES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm_ablation.yaml")
    parser.add_argument("--device")
    parser.add_argument("--outdir", help="Original training root; D is appended")
    parser.add_argument("--checkpoint", choices=("best_solver", "best", "last"), default="best_solver")
    parser.add_argument("--checkpoint-file", help="Explicit D checkpoint path")
    parser.add_argument("--output", help="New report directory")
    parser.add_argument("--mode", choices=("snapshot", "rollout"))
    parser.add_argument("--pgs-backend", choices=("sparse_cpu", "torch"))
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--scenes", choices=SCENES, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--hybrid-calls", type=int, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--max-sweeps", type=int)
    parser.add_argument("--timing-repeats", type=int)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    config = resolve_experiment(load_config(args.config), "D", args.outdir)
    settings = config["stress"]
    for key in ("mode", "pgs_backend", "sizes", "scenes", "seeds", "hybrid_calls", "steps", "tolerance", "max_sweeps", "timing_repeats"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.no_render:
        settings["render"] = False
    try:
        validate_stress(config)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output) if args.output else Path(config["outdir"])/"stress"/f"{args.checkpoint}_{settings['mode']}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}; choose --output")
    torch.set_num_threads(config["cpu_threads"])
    selected_device = device(args.device or config["device"])
    path = Path(args.checkpoint_file) if args.checkpoint_file else Path(config["outdir"])/"cfm"/f"{args.checkpoint}.pt"
    # Stress settings are separate; training data/physics/model checks remain strict.
    model, checkpoint = load_model(path, config, selected_device)
    evaluate_stress(model, config, selected_device, output,
        metadata=dict(checkpoint=str(path), updates=checkpoint["updates"],
                      selection_metric=checkpoint["selection_metric"], training_sizes=checkpoint["config"]["data"]["train_sizes"]))
    print(f"Saved {output / 'stress.json'} and {output / 'stress.md'}")


if __name__ == "__main__":
    main()
