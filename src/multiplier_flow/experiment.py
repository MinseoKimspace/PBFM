"""Resolve the controlled CFM ablations without changing the shared dataset."""
from copy import deepcopy
from math import isfinite
from pathlib import Path


VARIANTS = ("A", "B", "C", "D", "D_recovery", "B_direct", "D_direct")


def resolve_experiment(config, variant=None, outdir=None):
    """Apply a variant once, at the CLI boundary, to an unmodified config.

    A/B differ only in communication. C/D additionally train on zero-start
    integration. D_recovery adds perturbed suffix training. B_direct/D_direct
    change only the corresponding B/D output head. Each variant gets its own
    checkpoint directory but reads identical solution pairs.
    """
    resolved = deepcopy(config)
    variant = variant if variant is not None else resolved.get("variant")
    root = Path(outdir if outdir is not None else resolved["outdir"])
    if variant is None:
        resolved["outdir"] = str(root)
        return resolved
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation {variant!r}; expected one of {VARIANTS}")
    model = resolved["model"]
    base_variant = {"B_direct": "B", "D_direct": "D"}.get(variant, variant)
    model["head_type"] = "direct" if variant in ("B_direct", "D_direct") else "analytic"
    model["communication"] = "global" if base_variant in ("B", "D", "D_recovery") else "local"
    model["feature_version"] = "residual"
    inner = resolved["train"].setdefault("inner_rollout", {})
    recovery = resolved["train"].setdefault("recovery", {})
    uses_inner = base_variant in ("C", "D", "D_recovery")
    inner["weight"] = float(inner.get("weight", 1.0)) if uses_inner else 0.0
    recovery["weight"] = float(recovery.get("enabled_weight", 1.0)) if variant == "D_recovery" else 0.0
    if uses_inner and (not isfinite(inner["weight"]) or inner["weight"] <= 0):
        raise ValueError("C/D/D_direct variants require a positive inner_rollout.weight")
    if variant == "D_recovery" and (not isfinite(recovery["weight"]) or recovery["weight"] <= 0):
        raise ValueError("D_recovery requires a positive recovery.enabled_weight")
    resolved["variant"] = variant
    resolved["outdir"] = str(root / variant)
    resolved["cache_dir"] = str(root / "shared_data") if outdir is not None else str(
        resolved.get("cache_dir", root / "shared_data"))
    return resolved


def pair_cache_path(config):
    """Legacy runs keep pairs.pt beside checkpoints; ablations share one cache."""
    return Path(config.get("cache_dir", config["outdir"])) / "pairs.pt"
