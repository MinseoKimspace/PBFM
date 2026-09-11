"""Offline, energy-ranked contact-flow paths; no simulator endpoint labels.

The reference is an explicitly integrated energy ODE, not simulation-free FM.
One weight belongs to an entire path. Training records additionally carry the
accepted time interval so adaptive integration does not bias the time measure.
"""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import torch

from .dynamics import free_position, make_projection_condition
from .physics import FlowConfig, PhysicsConfig, contact_velocity, geometry, make_baseline
from .solver import SolverConfig, integrate


def compute_path_weights(scores: torch.Tensor, temperature: float,
                         weighting: str = "score") -> torch.Tensor:
    """Normalize only over candidate paths of the SAME physical problem."""
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("path temperature must be finite and positive")
    if scores.ndim != 2 or not all(scores.shape) or not torch.isfinite(scores).all():
        raise ValueError("path scores must be a finite [sources, candidates] tensor")
    if weighting not in {"score", "uniform"}:
        raise ValueError("path weighting must be 'score' or 'uniform'")
    values = scores.to(torch.float64)
    if weighting == "uniform":
        return torch.full_like(values, 1.0 / scores.shape[1])
    # Subtract before division, including very low-temperature best-of-N limits.
    values = values - values.amin(dim=1, keepdim=True)
    return torch.softmax(-values / temperature, dim=1)


def path_summary(buffer, temperature, weighting="score", scene_types=None):
    """Public diagnostics entry point; kept separate from path-generation logic."""
    from .diagnostics import path_summary as summarize
    return summarize(buffer, temperature, weighting, scene_types)


def build_path_buffer(
    state: torch.Tensor,
    radius: torch.Tensor,
    physics: PhysicsConfig,
    dynamics: dict[str, float],
    path_cfg: dict[str, Any],
    solver_cfg: SolverConfig,
    seed: int,
    flow: FlowConfig | None = None,
) -> dict[str, Any]:
    """Generate fixed-anchor path candidates, retaining completed unsolved paths.

    Random candidates have path-constant PSD controls; contact geometry is
    recomputed at every evaluation. ``candidate_pool='informed'`` replaces one
    candidate with a state-dependent regularized inverse reference, explicitly
    an algorithmic-reference ablation rather than a PBD-free purity claim.
    """
    flow = FlowConfig() if flow is None else flow
    if state.is_cuda:
        torch.cuda.synchronize(state.device)
    generation_started = time.perf_counter()
    if "mobility_bound" in path_cfg:
        raise ValueError("paths.mobility_bound is obsolete; use the shared flow.mobility_bound")
    if state.ndim != 3 or state.shape[-1] != 4 or radius.shape != state.shape[:2]:
        raise ValueError("state must be [B,N,4] and radius [B,N]")
    if state.shape[0] == 0 or state.shape[1] == 0:
        raise ValueError("at least one nonempty source is required")
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError("state must contain finite floating-point values")
    radius = radius.to(device=state.device, dtype=state.dtype)
    if not torch.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError("radii must be finite and positive")
    candidates = int(path_cfg.get("candidates", 8))
    rank = int(path_cfg.get("rank", 4))
    noise = float(path_cfg.get("start_noise_std", 0.01))
    integral_weight = float(path_cfg.get("integral_weight", 1.0))
    proximity_weight = float(path_cfg.get("proximity_weight", 0.0))
    terminal_violation_weight = float(path_cfg.get("terminal_violation_weight", 0.0))
    temperature = float(path_cfg.get("temperature", 0.001))
    weighting = str(path_cfg.get("weighting", "score"))
    pool = str(path_cfg.get("candidate_pool", "random"))
    if candidates < 1 or rank < 0:
        raise ValueError("candidates >= 1 and rank >= 0 required")
    scalars = torch.tensor([noise, integral_weight, proximity_weight, terminal_violation_weight])
    if not torch.isfinite(scalars).all() or scalars.min() < 0:
        raise ValueError("path bounds, noise and score weights must be finite/nonnegative")
    if pool not in {"random", "informed"}:
        raise ValueError("candidate_pool must be 'random' or 'informed'")
    resolved_paths = dict(path_cfg, candidates=candidates, rank=rank,
                          start_noise_std=noise, integral_weight=integral_weight,
                          proximity_weight=proximity_weight, temperature=temperature,
                          terminal_violation_weight=terminal_violation_weight,
                          candidate_pool=pool, weighting=weighting)
    # Validate before spending time on path generation.
    compute_path_weights(torch.zeros(1, candidates), temperature, weighting)
    state = state.detach()
    proposal = free_position(state, **dynamics)
    condition = make_projection_condition(state, proposal)
    batch, nodes = radius.shape
    count = batch * candidates
    rng = torch.Generator(device=state.device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=rng, device=state.device, dtype=state.dtype)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=rng, device=state.device, dtype=state.dtype)

    # Same random starting point for all candidates of a source. It is NOT
    # inserted into the fixed condition, which keeps the original proposal.
    start = proposal + noise * randn(batch, nodes, 2)
    expanded_start = start.repeat_interleave(candidates, dim=0)
    expanded_radius = radius.repeat_interleave(candidates, dim=0)
    initial_geometry = geometry(expanded_start, expanded_radius, physics)
    slots = initial_geometry["J"].shape[1]
    scale = torch.exp(4.0 * rand(count, 1) - 2.0)
    diagonal = torch.exp(2.0 * rand(count, slots) - 1.0) * scale
    low_rank = randn(count, slots, rank) * (scale / max(slots, 1)).sqrt().unsqueeze(-1)
    # Raw controls are fixed per path; the SAME component cap and contact
    # normalization as the learned field are reapplied at each current state.
    # Far slots must never suppress an unrelated active component's mobility.
    inverse = (make_baseline("inverse", radius, physics, flow=flow)
               if pool == "informed" else None)

    def field(z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        velocity = contact_velocity(geometry(z, expanded_radius, physics), diagonal, low_rank, flow)
        if inverse is not None:
            # One clearly identified, state-dependent classical candidate.
            velocity = velocity.clone()
            velocity[::candidates] = inverse(z[::candidates], tau[::candidates])
        return velocity

    with torch.no_grad():
        result = integrate(field, expanded_start, expanded_radius, physics, solver_cfg,
                           record=True, stop_on_tolerance=False)
        incomplete = ~result["completed"] | result["failed"]
        if incomplete.any():
            bad = incomplete.nonzero(as_tuple=False).flatten().tolist()
            fraction = float(result["completed"].float().mean())
            end_time = float(result["time"].min())
            reasons = result.get("failure_reason")
            detail = "" if reasons is None else f" Reasons: {[reasons[i] for i in bad[:12]]}."
            raise RuntimeError(
                f"Reference paths did not complete tau=1 for candidate indices {bad[:12]} "
                f"(completed={fraction:.1%}, min_end_time={end_time:.6g}).{detail} "
                "No failed source was silently removed. Increase solver.max_steps / "
                "max_backtracks, or lower flow.gain/mobility_bound; inspect "
                "the contact geometry and solver diagnostics."
            )
        history = result["history"]
        dt = history["dt"].to(torch.float64)
        if dt.ndim != 2 or dt.shape[1] != count:
            raise ValueError("solver history dt must have shape [accepted_steps, B*candidates]")
        if not torch.isfinite(dt).all() or (dt < 0).any():
            raise RuntimeError("Reference path history has invalid integration intervals")
        duration = dt.sum(dim=0)
        if not torch.allclose(duration, torch.ones_like(duration), atol=2e-5, rtol=2e-5):
            raise RuntimeError("Reference path quadrature must cover the entire tau interval [0,1]")
        energies = history.get("energy")
        if energies is None:
            energies = torch.stack([
                geometry(z, expanded_radius, physics)["energy"] for z in history["z"]
            ])
        energies = energies.to(torch.float64)
        terminal_geometry = geometry(result["final"], expanded_radius, physics)
        terminal = terminal_geometry["energy"].to(torch.float64)
        integral = (energies * dt).sum(dim=0)
        proposal_geometry = geometry(proposal, radius, physics)
        inverse_mass = proposal_geometry["inverse_mass"]
        mass = inverse_mass.reciprocal()
        displacement = result["final"].reshape(batch, candidates, nodes, 2) - proposal[:, None]
        proximity = (displacement.square().sum(dim=-1) * mass[:, None]).sum(dim=-1)
        scores = (terminal + integral_weight * integral).reshape(batch, candidates)
        scores = scores + proximity_weight * proximity.to(torch.float64)
        scores = scores + terminal_violation_weight * terminal_geometry["max_violation"].reshape(batch, candidates).to(torch.float64)
        path_weights = compute_path_weights(scores, temperature, weighting)
        record_weights = dt * path_weights.reshape(1, count)
        valid = dt > 0
        path_index = torch.arange(count, device=state.device).expand(dt.shape[0], count)[valid]

        def cpu(value: torch.Tensor) -> torch.Tensor:
            return value.detach().to("cpu").contiguous()

        def records(name: str) -> torch.Tensor:
            value = history[name][valid]
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Reference path has nonfinite accepted {name} records")
            return cpu(value)

        def per_path(name: str, default: float = 0) -> torch.Tensor:
            value = torch.as_tensor(result.get(name, default), device=state.device)
            if value.numel() == 1:
                value = value.expand(count)
            return cpu(value.reshape(batch, candidates))

        if state.is_cuda:
            torch.cuda.synchronize(state.device)
        generation_seconds = time.perf_counter() - generation_started

        return {
            "format": "contact_paths_v2",
            "z": records("z"),
            "tau": records("tau").reshape(-1, 1),
            "velocity": records("u"),
            "weight": cpu(record_weights[valid]),
            "dt": cpu(dt[valid]),
            "context_index": cpu(path_index // candidates),
            "path_index": cpu(path_index),
            "condition": cpu(condition),
            "radius": cpu(radius),
            "source": cpu(state),
            "proposal": cpu(proposal),
            "start": cpu(start),
            "final_positions": cpu(result["final"].reshape(batch, candidates, nodes, 2)),
            "path_scores": cpu(scores),
            "path_weights": cpu(path_weights),
            "diagnostics": {
                "ess": cpu(path_weights.square().sum(dim=1).reciprocal()),
                "initial_energy": cpu(initial_geometry["energy"][::candidates]),
                "initial_violation": cpu(initial_geometry["max_violation"][::candidates]),
                "proposal_energy": cpu(proposal_geometry["energy"]),
                "proposal_violation": cpu(proposal_geometry["max_violation"]),
                "success_counts": cpu(result["converged"].reshape(batch, candidates).sum(dim=1)),
                "terminal_energy": cpu(terminal.reshape(batch, candidates)),
                "terminal_violation": cpu(terminal_geometry["max_violation"].reshape(batch, candidates)),
                "integral_energy": cpu(integral.reshape(batch, candidates)),
                "nfe": per_path("nfe"),
                "backtracks": per_path("backtracks"),
                "energy_evals": per_path("energy_evals"),
                "completed": per_path("completed"),
                "failed": per_path("failed"),
                "first_tolerance_time": per_path("first_tolerance_time", -1),
                "first_tolerance_nfe": per_path("first_tolerance_nfe", -1),
                "first_tolerance_backtracks": per_path("first_tolerance_backtracks", -1),
                "first_tolerance_energy_evals": per_path("first_tolerance_energy_evals", -1),
                "accepted_roundoff": per_path("accepted_roundoff"),
                "generation_seconds": generation_seconds,
                "field_calls": int(result.get("field_calls", 0)),
                "executed_field_rows": int(result.get("executed_field_rows", 0)),
                "failure_reason": result.get("failure_reason", ["none"] * count),
            },
            "metadata": {"seed": seed, "dynamics": dict(dynamics),
                         "physics": asdict(physics), "solver": asdict(solver_cfg),
                         "flow": asdict(flow),
                         "paths": resolved_paths, "candidate_pool": pool,
                         "reference": "integrated_contact_energy_ode",
                         "quadrature": "accepted_left_endpoint",
                         "roundoff_policy": "within_tolerance_subresolution_nonincrease_only"},
        }
