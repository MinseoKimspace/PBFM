"""Train contact-energy path FM (or the matched energy-only ablation)."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.contact_flow.io import load_config, load_states, device
from src.contact_flow.model import ContactFlowNet
from src.contact_flow.paths import build_path_buffer, compute_path_weights, path_summary
from src.contact_flow.physics import PhysicsConfig, FlowConfig, geometry
from src.contact_flow.solver import SolverConfig


def temperature_weights(buffer: dict, temperature: float, weighting="score") -> tuple[torch.Tensor, float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    path_weights = compute_path_weights(buffer["path_scores"], temperature, weighting)
    weight = path_weights.flatten()[buffer["path_index"]] * buffer["dt"].double()
    ess = (1 / path_weights.square().sum(1)).mean().item()
    return weight, ess


def buffer_batch(buffer, index, target_device):
    context = buffer["context_index"][index]
    return (buffer["z"][index].to(target_device), buffer["tau"][index].to(target_device),
            buffer["radius"][context].to(target_device),
            buffer["condition"][context].to(target_device),
            buffer["velocity"][index].to(target_device))


def energy_objective(model, z, tau, radius, condition, physics, steps=4,
                     max_displacement=.15):
    """Matched architecture ablation: no velocity targets, short differentiable unroll.

    Fixed steps and a scalar displacement cap; inference still uses the common guard.
    This is explicitly NOT the FM objective.
    """
    if steps < 1 or max_displacement <= 0:
        raise ValueError("energy_steps and max_displacement must be positive")
    loss = z.new_zeros(())
    tau = tau.reshape(-1)
    step_size = (1 - tau) / steps
    for _ in range(steps):
        velocity = model(z, tau, radius, condition, physics)
        delta = velocity * step_size[:, None, None]
        scale = (delta.norm(dim=-1).amax(-1) / max_displacement).clamp_min(1)
        z = z + delta / scale[:, None, None]
        tau = tau + step_size / scale
        loss = loss + geometry(z, radius, physics)["energy"].mean() / steps
    return loss


@torch.no_grad()
def validate(model, buffer, physics, batch_size, target_device, scale, temperature, weighting="score"):
    model.eval()
    weights, _ = temperature_weights(buffer, temperature, weighting)
    numerator = 0.0
    for index in torch.arange(len(weights)).split(batch_size):
        z, tau, radius, condition, target = buffer_batch(buffer, index, target_device)
        error = (model(z, tau, radius, condition, physics) - target).double().square().mean((1, 2))
        numerator += (error.cpu() * weights[index]).sum().item()
    return numerator / weights.sum().item() / scale**2


def train(cfg, *, target_device=None, outdir=None, objective=None, epochs=None,
          prepare_only=False, rank=None, weighting=None, candidate_pool=None):
    cfg = deepcopy(cfg)
    if rank is not None:
        cfg.setdefault("model", {})["rank"] = rank
    if weighting is not None:
        cfg.setdefault("paths", {})["weighting"] = weighting
    if candidate_pool is not None:
        cfg.setdefault("paths", {})["candidate_pool"] = candidate_pool
    runtime, data = cfg.get("runtime", {}), cfg.get("data", {})
    settings = cfg.get("train", {})
    seed = int(runtime.get("seed", 42))
    torch.manual_seed(seed)
    target_device = device(target_device or runtime.get("device", "auto"))
    if target_device.type == "cpu":
        torch.set_num_threads(int(runtime.get("cpu_threads", 4)))
    objective = objective or settings.get("objective", "fm")
    if objective not in {"fm", "energy"}:
        raise ValueError("objective must be fm or energy")
    output = Path(outdir or runtime.get("outdir", "runs/contact_fm_v2"))
    if objective == "energy" and outdir is None:
        output = output.with_name(output.name + "_energy")
    if outdir is None and rank is not None:
        output = output.with_name(output.name + f"_rank{rank}")
    if outdir is None and weighting is not None:
        output = output.with_name(output.name + "_" + weighting)
    if outdir is None and candidate_pool is not None:
        output = output.with_name(output.name + "_" + candidate_pool)
    if (output / "best.pt").exists() or (output / "last.pt").exists():
        raise FileExistsError(f"Refusing to overwrite checkpoints in {output}; choose --outdir")
    output.mkdir(parents=True, exist_ok=True)
    physics = PhysicsConfig(**cfg.get("physics", {}))
    flow = FlowConfig(**cfg.get("flow", {}))
    dynamics = {"time_step": 1/60, "gravity_y": -9.8, "linear_damping": .1,
                **cfg.get("dynamics", {})}
    reference_solver = SolverConfig(**cfg.get("reference_solver", {}))
    paths = cfg.get("paths", {})
    if "mobility_bound" in paths or "mobility_bound" in cfg.get("model", {}):
        raise ValueError("v2 uses one shared flow.mobility_bound; remove model/paths.mobility_bound")
    weighting = paths.get("weighting", "score")
    compute_path_weights(torch.zeros(1, 1), float(paths.get("temperature", .001)), weighting)
    if data.get("train_split", "train") == data.get("val_split", "val") and data.get("dataset"):
        raise ValueError("Training and validation must use distinct input splits")
    cache_spec = {"data": data, "physics": asdict(physics), "flow": asdict(flow), "dynamics": dynamics,
                  "paths": paths, "reference_solver": asdict(reference_solver), "seed": seed}
    if data.get("dataset"):
        stamp = Path(data["dataset"]).stat()
        cache_spec["dataset_stamp"] = {"size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns}
    cache_path = output / "paths.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("format") != "contact_path_cache_v2" or cached.get("spec") != cache_spec:
            raise ValueError("v2 path cache/settings differ; regenerate paths in a fresh --outdir")
        buffers = cached["buffers"]
    else:
        buffers = {}
        for split, count, offset in [("train", data.get("train_count", 128), 0),
                                     ("val", data.get("val_count", 32), 1)]:
            source, radius, scene_types = load_states(
                data.get("dataset", ""), data.get(split + "_split", split), int(count), seed + offset,
                int(data.get("num_objects", 5)), physics=physics, return_scene_types=True)
            print(f"Generating {split} energy-reference paths: {len(source)} contexts; no endpoint labels", flush=True)
            buffers[split] = build_path_buffer(source.to(target_device), radius.to(target_device), physics,
                                               dynamics, paths, reference_solver, seed + offset, flow=flow)
            buffers[split]["scene_type"] = scene_types
            print(f"{split}: {len(buffers[split]['z'])} quadrature samples", flush=True)
        torch.save({"format": "contact_path_cache_v2", "spec": cache_spec, "buffers": buffers}, cache_path)
    reference_reports = {}
    quality = {}
    for name, buffer in buffers.items():
        diagnostics = buffer["diagnostics"]
        successes = diagnostics["success_counts"]
        total_paths = buffer["path_scores"].numel()
        print(f"{name} paths: completed={total_paths}/{total_paths}, "
              f"within_tolerance={int(successes.sum())}/{total_paths}, "
              f"contexts_with_solved_candidate={int((successes > 0).sum())}/{len(successes)}, "
              f"ESS={float(diagnostics['ess'].mean()):.2f}, "
              f"roundoff_steps={int(diagnostics['accepted_roundoff'].sum())}", flush=True)
        reference_reports[name] = path_summary(buffer, float(paths.get("temperature", .001)), weighting)
        violating = geometry(buffer["proposal"], buffer["radius"], physics)["max_violation"] > reference_solver.tolerance
        any_success = successes > 0
        coverage = float(any_success[violating].float().mean()) if violating.any() else 1.0
        quality[name] = coverage
        for label in sorted(set(buffer["scene_type"])):
            mask = torch.tensor([value == label for value in buffer["scene_type"]])
            print(f"  {name}/{label}: scenes_with_success={int(any_success[mask].sum())}/{int(mask.sum())}", flush=True)
        print(f"  {name}/clean_violating: scenes_with_success="
              f"{int(any_success[violating].sum())}/{int(violating.sum())}", flush=True)
    (output / "reference_summary.json").write_text(
        json.dumps(reference_reports, indent=2, allow_nan=False), encoding="utf-8")
    if prepare_only:
        print(f"Saved reusable path cache: {cache_path}")
        return {"path_cache": str(cache_path)}
    minimum_coverage = float(settings.get("min_violating_scene_coverage", 0.0))
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("min_violating_scene_coverage must be in [0,1]")
    if any(value < minimum_coverage for value in quality.values()):
        raise ValueError(f"Reference coverage on clean violating scenes is too low: {quality}. "
                         f"Required {minimum_coverage:.1%}; inspect reference_summary.json and tune "
                         "flow/candidates before training. Cache is preserved; no scenes were removed.")

    model_kwargs = cfg.get("model", {})
    model = ContactFlowNet(**model_kwargs, flow=flow).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("lr", 3e-4)),
                                  weight_decay=float(settings.get("weight_decay", 1e-4)))
    train_buffer, val_buffer = buffers["train"], buffers["val"]
    total_epochs = int(epochs if epochs is not None else settings.get("epochs", 100))
    batch_size = int(settings.get("batch_size", 128))
    batches = int(settings.get("steps_per_epoch", 100))
    if min(total_epochs, batch_size, batches) < 1:
        raise ValueError("epochs, batch_size and steps_per_epoch must be positive")
    start_temperature = float(paths.get("temperature", .001))
    end_temperature = float(settings.get("temperature_end", start_temperature))
    weights, _ = temperature_weights(train_buffer, end_temperature, weighting)
    raw_square = train_buffer["velocity"].double().square().mean((1, 2))
    scale = max(float((raw_square * weights).sum().div(weights.sum()).sqrt()), 1e-3)
    from eval_projection import evaluate_projection
    from src.contact_flow.dynamics import make_projection_condition
    validation_solver = SolverConfig(**cfg.get("solver", {}))
    best_score, best_val_fm, history = None, None, []
    for epoch in range(total_epochs):
        fraction = epoch / max(total_epochs - 1, 1)
        temperature = start_temperature * (end_temperature / start_temperature)**fraction
        weights, ess = temperature_weights(train_buffer, temperature, weighting)
        sampler = WeightedRandomSampler(weights, batches * batch_size, replacement=True,
                                        generator=torch.Generator().manual_seed(seed + epoch))
        loader = DataLoader(torch.arange(len(weights)), batch_size=batch_size, sampler=sampler,
                            num_workers=int(settings.get("num_workers", 0)))
        model.train()
        train_loss = 0.0
        for index in loader:
            z, tau, radius, condition, target = buffer_batch(train_buffer, index, target_device)
            if objective == "fm":
                loss = ((model(z, tau, radius, condition, physics) - target) / scale).square().mean()
            else:
                loss = energy_objective(model, z, tau, radius, condition, physics,
                                        int(settings.get("energy_steps", 4)),
                                        float(cfg.get("solver", {}).get("max_displacement", .15)))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective; checkpoint not updated")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 1)),
                                          error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss.item() / batches
        val_fm = validate(model, val_buffer, physics, batch_size, target_device, scale, end_temperature, weighting)
        def field_factory(state, proposal, radius):
            condition = make_projection_condition(state, proposal)
            return lambda z, tau: model(z, tau, radius, condition, physics)
        val_solver, _ = evaluate_projection(val_buffer["source"].to(target_device),
                                            val_buffer["radius"].to(target_device), field_factory,
                                            physics, dynamics, validation_solver,
                                            scene_types=val_buffer["scene_type"])
        score = (val_solver["integration_failed_count"], -val_solver["converged_count"],
                 val_solver["energy_final_mean"])
        path_probabilities = compute_path_weights(train_buffer["path_scores"], temperature, weighting)
        weighted_success = (path_probabilities * (
            train_buffer["diagnostics"]["terminal_violation"] <= reference_solver.tolerance)).sum(1)
        row = {"epoch": epoch + 1, "train_objective": train_loss, "val_fm": val_fm,
               "temperature": temperature, "path_ess": ess, "weighting": weighting,
               "reference_weighted_success": float(weighted_success.mean()), "val_solver": val_solver}
        # Both arms select checkpoints by the SAME physical solver criteria, not teacher MSE.
        history.append(row)
        checkpoint = {"format": "contact_flow_v2", "objective": objective, "epoch": epoch + 1,
                      "model_kwargs": model_kwargs, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "physics": asdict(physics),
                      "dynamics": dynamics, "flow": asdict(flow), "solver": cfg.get("solver", {}), "config": cfg,
                      "reference_cache_spec": cache_spec, "temperature": temperature, "weighting": weighting,
                      "velocity_scale": scale, "selection_metric": "failure_count,negative_success_count,energy",
                      "selection_score": score, "val_fm": val_fm}
        torch.save(checkpoint, output / "last.pt")
        if best_score is None or score < best_score:
            best_score, best_val_fm = score, val_fm
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[{epoch+1:03d}/{total_epochs}] {objective}={train_loss:.6g} val_fm={val_fm:.6g} "
              f"solved={val_solver['converged_count']}/{val_solver['samples']} "
              f"T={temperature:.4g} ESS={ess:.2f}", flush=True)
    return {"best_val_fm": best_val_fm, "checkpoint": str(output / "best.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_projection.yaml")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--objective", choices=["fm", "energy"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rank", type=int, help="Model rank ablation; 0 is learned diagonal only")
    parser.add_argument("--weighting", choices=["score", "uniform"], help="Path-weight ablation")
    parser.add_argument("--candidate-pool", choices=["random", "informed"], help="Reference-generation ablation")
    args = parser.parse_args()
    train(load_config(args.config), target_device=args.device, outdir=args.outdir,
          objective=args.objective, epochs=args.epochs, prepare_only=args.prepare_only,
          rank=args.rank, weighting=args.weighting, candidate_pool=args.candidate_pool)


if __name__ == "__main__":
    main()
