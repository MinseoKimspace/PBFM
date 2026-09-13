"""Train B (endpoint) or C (endpoint + Lagrangian map matching), sharing data."""
from __future__ import annotations

import argparse
import math
from time import perf_counter
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.data import batch, cache_summary, prepare
from src.multiplier_flow.evaluation import solver_validation, timed, write_json
from src.multiplier_flow.model import MultiplierMap, losses


def profile(model, example, selected_device):
    report = {}
    for matching in (False, True):
        def operation():
            model.zero_grad(set_to_none=True)
            endpoint, matching_loss = losses(model, *example, matching=matching)
            (endpoint + matching_loss).backward()
        operation()  # Warm-up, no parameter update and no random sampling.
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
        times = [timed(selected_device, operation)[1] for _ in range(3)]
        report["map" if matching else "endpoint"] = dict(
            forward_backward_seconds=sum(times) / len(times),
            peak_allocated_bytes=(torch.cuda.max_memory_allocated(selected_device)
                                  if selected_device.type == "cuda" else None))
    model.zero_grad(set_to_none=True)
    return report


@torch.no_grad()
def validate(model, split, batch_size, selected_device):
    total = 0.0
    for index in torch.arange(len(split["context"])).split(batch_size):
        loss, _ = losses(model, *batch(split, index, selected_device))
        total += float(loss) * len(index)
    return total / len(split["context"])


def gradient_probe(model, example, weight):
    """One fixed validation batch: loss magnitudes AND weighted gradient sizes."""
    endpoint, matching = losses(model, *example, matching=True)
    parameters = tuple(model.parameters())
    left = torch.autograd.grad(endpoint, parameters, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(weight * matching, parameters, allow_unused=True)
    def norm(values):
        return float(sum(x.detach().square().sum() for x in values if x is not None).sqrt())
    return dict(endpoint=float(endpoint.detach()), matching=float(matching.detach()),
                weighted_matching=float(weight * matching.detach()),
                endpoint_gradient_norm=norm(left), weighted_matching_gradient_norm=norm(right))


def train(config, objective, selected_device, *, prepare_only=False, profile_only=False,
          epochs=None, max_updates=None):
    if objective not in {"endpoint", "map"}:
        raise ValueError("objective must be endpoint or map")
    torch.set_num_threads(int(config["cpu_threads"]))
    root = Path(config["outdir"])
    run = root / objective
    if not prepare_only and not profile_only and any((run / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError(f"Refusing to overwrite {run}; choose a new outdir")
    cache = prepare(config, root / "segments.pt")
    summary = cache_summary(cache, config["evaluation"]["tolerance"])
    write_json(root / "segments_summary.json", summary)
    if prepare_only:
        return dict(cache=str(root / "segments.pt"))
    if cache["format"] != "multiplier_segments_v2":
        raise ValueError("New training requires balanced v2 segments; keep old runs and choose a new outdir")
    if not profile_only and any(summary[s]["horizon_solved"] != summary[s]["scenes"] for s in ("train", "val")):
        raise RuntimeError("Reference horizon misses train/val projection tolerance. Inspect segments_summary.json; "
                           "increase the horizon or improve the reference before full training. Cache is preserved.")
    torch.manual_seed(int(config["seed"]))
    model = MultiplierMap(**config["model"]).to(selected_device)
    settings = config["train"]
    if (epochs is not None and epochs < 1) or settings["epochs"] < 1 or settings["matching_weight"] < 0:
        raise ValueError("Positive epochs and nonnegative matching weight required")
    training, validation = cache["splits"]["train"], cache["splits"]["val"]
    example = batch(training, torch.arange(min(settings["batch_size"], len(training["context"]))), selected_device)
    profiling = profile(model, example, selected_device)
    write_json(root / f"profile_{objective}.json", dict(device=str(selected_device), costs=profiling))
    if profile_only:
        return profiling
    run.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    history, best, best_solver = [], float("inf"), (float("inf"), float("inf"))
    count = len(training["context"])
    batches = math.ceil(count / settings["batch_size"])
    budget = max_updates if max_updates is not None else (
        (epochs * batches) if epochs is not None else settings.get("max_updates", settings["epochs"] * batches))
    if budget < 1:
        raise ValueError("Positive optimizer update budget required")
    updates, last_diagnostic = 0, 0
    began = perf_counter()
    validation_example = batch(validation, torch.arange(min(settings["batch_size"], len(validation["context"]))), selected_device)
    for epoch in range(math.ceil(budget / batches)):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        endpoint_sum = map_sum = 0.0
        seen = 0
        for indices in torch.randperm(count, generator=generator).split(settings["batch_size"]):
            endpoint, matching = losses(model, *batch(training, indices, selected_device), matching=objective == "map")
            loss = endpoint + settings["matching_weight"] * matching
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; checkpoint not updated")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            seen += len(indices)
            endpoint_sum += float(endpoint.detach()) * len(indices)
            map_sum += float(matching.detach()) * len(indices)
            if updates >= budget:
                break
        model.eval()
        val = validate(model, validation, settings["batch_size"], selected_device)
        if not torch.isfinite(torch.tensor(val)):
            raise FloatingPointError("Nonfinite validation loss")
        row = dict(epoch=epoch + 1, updates=updates, train_endpoint=endpoint_sum / seen,
                   train_matching=map_sum / seen, val_endpoint=val,
                   elapsed_seconds=perf_counter()-began)
        interval = int(settings.get("solver_validation_every", 0))
        diagnostic = interval > 0 and (updates-last_diagnostic >= interval or updates == budget or epoch == 0)
        if diagnostic:
            row["solver_validation"] = solver_validation(model, validation, config, selected_device)
            row["gradient_probe"] = gradient_probe(model, validation_example, settings["matching_weight"])
            last_diagnostic = updates
        history.append(row)
        checkpoint = dict(format="multiplier_map_v1", objective=objective, epoch=epoch + 1,
                          updates=updates, config=config, cache_spec=cache["spec"], model=model.state_dict(),
                          optimizer=optimizer.state_dict(), val_endpoint=val,
                          selection_metric="shared validation finite-time endpoint position loss")
        torch.save(checkpoint, run / "last.pt")
        if val < best:
            best = val
            torch.save(checkpoint, run / "best.pt")
        if diagnostic:
            stats = row["solver_validation"]["nontrivial"]
            score = (-stats["success_rate"], stats["mean_projected_gradient"])
            if score < best_solver:
                best_solver = score
                torch.save(dict(checkpoint, selection_metric="nontrivial raw validation success, then PG residual",
                                solver_validation=row["solver_validation"]), run / "best_solver.pt")
        write_json(run / "history.json", history)
        print(f"[{epoch+1:03d} updates={updates}/{budget}] {objective} endpoint={endpoint_sum/seen:.6g} "
              f"matching={map_sum/seen:.6g} val_endpoint={val:.6g}", flush=True)
        if diagnostic:
            print(f"  nontrivial solver validation: {stats}", flush=True)
    return dict(checkpoint=str(run / "best.pt"), best_val_endpoint=best, updates=updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_flow.yaml")
    parser.add_argument("--objective", choices=("endpoint", "map"), default="endpoint")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-updates", type=int, help="Exact update budget, overrides epochs/config")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-pilot", type=int, metavar="SCENES_PER_SIZE",
                        help="Prepare a small timed cache under outdir/prepare_pilot; no training")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.outdir:
        config["outdir"] = args.outdir
    if args.prepare_pilot is not None:
        if args.prepare_pilot < 1:
            parser.error("--prepare-pilot must be positive")
        config["outdir"] = str(Path(config["outdir"]) / "prepare_pilot")
        for split in ("train", "val", "test"):
            config["data"][split + "_per_size"] = min(config["data"][split + "_per_size"], args.prepare_pilot)
        config["data"]["release_count"] = min(config["data"]["release_count"], args.prepare_pilot)
    result = train(config, args.objective, device(args.device or config["device"]),
                   prepare_only=args.prepare_only or args.prepare_pilot is not None,
                   profile_only=args.profile_only, epochs=args.epochs,
                   max_updates=args.max_updates)
    print(result)


if __name__ == "__main__":
    main()
