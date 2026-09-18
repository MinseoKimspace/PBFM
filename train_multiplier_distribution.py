"""Train D on analytic solution distributions (or a single-endpoint control)."""
import argparse

from src.contact_flow.io import device, load_config
from src.multiplier_flow.distribution import VARIANTS, resolve_distribution
from src.multiplier_flow.distribution_experiment import train_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm_ablation.yaml")
    parser.add_argument("--variant", choices=VARIANTS, default="D_distribution")
    parser.add_argument("--device")
    parser.add_argument("--outdir", help="Experiment root; the distribution variant is appended")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    config = resolve_distribution(config, args.variant, args.outdir)
    print(train_distribution(config, device(args.device or config["device"]), max_updates=args.max_updates))


if __name__ == "__main__":
    main()
