"""JSON-native, scene-balanced diagnostics for algorithmic reference paths.

The reference oracle chooses among candidates from the SAME (possibly noisy)
start. Its charged cost is generation of ALL candidates, not just its winner.
It must not be presented as a free or clean-proposal physical solver.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .paths import compute_path_weights


def _number(value: torch.Tensor | float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _distribution(values: torch.Tensor) -> dict[str, float | None]:
    values = values.flatten().to(torch.float64)
    values = values[torch.isfinite(values)]
    if not values.numel():
        return dict(mean=None, min=None, p25=None, median=None, p75=None, max=None)
    quantiles = torch.quantile(values, values.new_tensor([0, .25, .5, .75, 1]))
    return dict(mean=_number(values.mean()),
                **dict(zip(("min", "p25", "median", "p75", "max"),
                           [_number(value) for value in quantiles])))


def path_summary(buffer: dict[str, Any], temperature: float,
                 weighting: str = "score",
                 scene_types: Sequence[str] | None = None) -> dict[str, Any]:
    """Report coverage AND probability mass, retaining initially solved scenes.

    ``clean_*`` groups refer to the clean analytic proposal; ``start_*`` groups
    refer to the actual shared noisy path start. Tolerance is excess overlap
    after physical slop, exactly as used by the reference integrator.
    ``first_tolerance_*`` measures the first visit, not persistence thereafter.
    """
    if buffer.get("format") != "contact_paths_v2":
        raise ValueError("path_summary requires contact_paths_v2; regenerate the old cache")
    scores = buffer["path_scores"].detach().cpu().to(torch.float64)
    weights = compute_path_weights(scores, temperature, weighting)
    batch, candidates = scores.shape
    diag = buffer["diagnostics"]
    tolerance = float(buffer["metadata"]["solver"]["tolerance"])

    def value(name: str, shape: tuple[int, ...], default: float | None = None) -> torch.Tensor:
        source = diag.get(name, default)
        if source is None:
            raise ValueError(f"Missing path diagnostic {name}")
        tensor = torch.as_tensor(source).detach().cpu()
        if tensor.numel() == 1:
            tensor = tensor.expand(*shape)
        return tensor.reshape(shape)

    terminal = value("terminal_violation", (batch, candidates)).double()
    final_energy = value("terminal_energy", (batch, candidates)).double()
    start_violation = value("initial_violation", (batch,)).double()
    start_energy = value("initial_energy", (batch,)).double()
    clean_violation = value("proposal_violation", (batch,)).double()
    completed = value("completed", (batch, candidates), 1).bool()
    failed = value("failed", (batch, candidates), 0).bool()
    success = (terminal <= tolerance) & completed & ~failed & torch.isfinite(terminal)
    nfe = value("nfe", (batch, candidates), 0).double()
    backtracks = value("backtracks", (batch, candidates), 0).double()
    energy_evals = value("energy_evals", (batch, candidates), 0).double()
    first_time = value("first_tolerance_time", (batch, candidates), -1).double()
    first_nfe = value("first_tolerance_nfe", (batch, candidates), -1).double()
    ess = weights.square().sum(dim=1).reciprocal()
    any_success = success.any(dim=1)
    rows = torch.arange(batch)
    score_winner = scores.argmin(dim=1)
    score_miss = any_success & ~success[rows, score_winner]

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        selected = success[mask]
        selected_time = first_time[mask]
        selected_first_nfe = first_nfe[mask]
        positive_start = mask & (start_violation > tolerance)
        energy_start_positive = mask & (start_energy > 0)
        return {
            "scenes": count,
            "candidates_per_scene": candidates,
            "paths": count * candidates,
            "successful_paths": int(selected.sum()),
            "path_success_rate": _number(selected.double().mean()) if count else None,
            "scenes_with_success": int(any_success[mask].sum()),
            "scene_success_rate": _number(any_success[mask].double().mean()) if count else None,
            "weighted_success_mass": _distribution((weights * success)[mask].sum(dim=1)),
            "best_score_misses_success_count": int(score_miss[mask].sum()),
            "ess": _distribution(ess[mask]),
            "score": _distribution(scores[mask]),
            "initial_max_violation": _distribution(start_violation[mask]),
            "final_max_violation": _distribution(terminal[mask]),
            "violation_reduction": _distribution(start_violation[mask, None] - terminal[mask]),
            "violation_ratio_initially_violating": _distribution(
                terminal[positive_start] / start_violation[positive_start, None]),
            "initial_energy": _distribution(start_energy[mask]),
            "final_energy": _distribution(final_energy[mask]),
            "energy_reduction": _distribution(start_energy[mask, None] - final_energy[mask]),
            "energy_ratio_nonzero_initial": _distribution(
                final_energy[energy_start_positive] / start_energy[energy_start_positive, None]),
            "nfe": _distribution(nfe[mask]),
            "backtracks": _distribution(backtracks[mask]),
            "failed_paths": int(failed[mask].sum()),
            "incomplete_paths": int((~completed[mask]).sum()),
            "completed_but_unsolved_paths": int((completed & ~failed & ~success)[mask].sum()),
            "first_tolerance_time": _distribution(selected_time[selected_time >= 0]),
            "first_tolerance_nfe": _distribution(selected_first_nfe[selected_first_nfe >= 0]),
            "all_candidate_cost": {
                "logical_nfe": int(nfe[mask].sum()),
                "logical_backtracks": int(backtracks[mask].sum()),
                "logical_solver_energy_evaluations": int(energy_evals[mask].sum()),
            },
        }

    masks = {
        "all": torch.ones(batch, dtype=torch.bool),
        "clean_violating": clean_violation > tolerance,
        "clean_solved": clean_violation <= tolerance,
        "start_violating": start_violation > tolerance,
        "start_solved": start_violation <= tolerance,
    }
    if scene_types is None:
        scene_types = buffer.get("scene_type")
    if scene_types is not None:
        if len(scene_types) != batch:
            raise ValueError("scene_types must contain one label per physical source")
        labels = [str(label) for label in scene_types]
        for label in sorted(set(labels)):
            masks[f"scene:{label}"] = torch.tensor([entry == label for entry in labels])

    # The chosen candidate is always compared on this buffer's own fixed start.
    # Numerical failure is never promoted to a winner by a small residual.
    oracle_scores = terminal.masked_fill(failed | ~completed, float("inf"))
    oracle_index = oracle_scores.argmin(dim=1)
    oracle_violation = oracle_scores[rows, oracle_index]
    total_cost = {
        "logical_nfe": int(nfe.sum()),
        "logical_backtracks": int(backtracks.sum()),
        "logical_solver_energy_evaluations": int(energy_evals.sum()),
        "generation_seconds": _number(diag.get("generation_seconds", float("nan"))),
        "vectorized_field_calls": int(diag.get("field_calls", 0)),
        "executed_field_rows": int(diag.get("executed_field_rows", 0)),
    }
    return {
        "format": "contact_path_summary_v2",
        "temperature": float(temperature), "weighting": weighting,
        "tolerance_excess_of_slop": tolerance,
        "groups": {name: summarize(mask) for name, mask in masks.items()},
        "candidate_generation_cost": total_cost,
        "oracle": {
            "criterion": "smallest_final_max_violation_among_completed_candidates",
            "start": "buffer.start: same noisy start shared by each scene's candidates",
            "comparable_to_clean_proposal_rollout": False,
            "candidate_index": oracle_index.tolist(),
            "final_max_violation_per_scene": [_number(v) for v in oracle_violation],
            "successful_scenes": int((oracle_violation <= tolerance).sum()),
            "scene_success_rate": _number((oracle_violation <= tolerance).double().mean()),
            "first_tolerance_time_selected_path": [
                _number(v) if v >= 0 else None for v in first_time[rows, oracle_index]],
            "first_tolerance_nfe_selected_path": [
                int(v) if v >= 0 else None for v in first_nfe[rows, oracle_index]],
            "charged_cost": total_cost,
            "cost_note": "ALL candidates are generated and scored; winner-only cost is not a speed result.",
        },
    }
