"""Train B (endpoint) or C (endpoint + Lagrangian map matching), sharing data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.data import batch, cache_summary, prepare
from src.multiplier_flow.evaluation import timed, write_json
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


def train(config, objective, selected_device, *, prepare_only=False, profile_only=False, epochs=None):
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
    history, best = [], float("inf")
    count = len(training["context"])
    for epoch in range(epochs or settings["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        endpoint_sum = map_sum = 0.0
        for indices in torch.randperm(count, generator=generator).split(settings["batch_size"]):
            endpoint, matching = losses(model, *batch(training, indices, selected_device), matching=objective == "map")
            loss = endpoint + settings["matching_weight"] * matching
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; checkpoint not updated")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            endpoint_sum += float(endpoint.detach()) * len(indices)
            map_sum += float(matching.detach()) * len(indices)
        model.eval()
        val = validate(model, validation, settings["batch_size"], selected_device)
        if not torch.isfinite(torch.tensor(val)):
            raise FloatingPointError("Nonfinite validation loss")
        history.append(dict(epoch=epoch + 1, train_endpoint=endpoint_sum / count,
                            train_matching=map_sum / count, val_endpoint=val))
        checkpoint = dict(format="multiplier_map_v1", objective=objective, epoch=epoch + 1,
                          config=config, cache_spec=cache["spec"], model=model.state_dict(),
                          optimizer=optimizer.state_dict(), val_endpoint=val,
                          selection_metric="shared validation finite-time endpoint position loss")
        torch.save(checkpoint, run / "last.pt")
        if val < best:
            best = val
            torch.save(checkpoint, run / "best.pt")
        write_json(run / "history.json", history)
        print(f"[{epoch+1:03d}] {objective} endpoint={endpoint_sum/count:.6g} "
              f"matching={map_sum/count:.6g} val_endpoint={val:.6g}", flush=True)
    return dict(checkpoint=str(run / "best.pt"), best_val_endpoint=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_flow.yaml")
    parser.add_argument("--objective", choices=("endpoint", "map"), default="endpoint")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.outdir:
        config["outdir"] = args.outdir
    result = train(config, args.objective, device(args.device or config["device"]),
                   prepare_only=args.prepare_only, profile_only=args.profile_only, epochs=args.epochs)
    print(result)


if __name__ == "__main__":
    main()
