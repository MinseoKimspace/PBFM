"""Train conditional flow matching to converged frozen-contact QP endpoints."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import torch

from src.contact_flow.io import device, load_config
from src.multiplier_flow.data import batch, cache_summary, prepare
from src.multiplier_flow.evaluation import solver_validation, timed, write_json
from src.multiplier_flow.model import CHECKPOINT_FORMAT, ConditionalField, cfm_loss


def sample_tau(count, rng, zero_fraction):
    tau = torch.rand(count, generator=rng)
    tau[torch.rand(count, generator=rng) < zero_fraction] = 0.
    return tau


def profile(model, example, selected_device):
    tau = example[1].new_full((len(example[1]),), .5)
    def operation():
        model.zero_grad(set_to_none=True)
        loss, _ = cfm_loss(model, *example, tau)
        loss.backward()
    operation()
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    times = [timed(selected_device, operation)[1] for _ in range(3)]
    model.zero_grad(set_to_none=True)
    return dict(forward_backward_seconds=sum(times)/len(times),
                peak_allocated_bytes=(torch.cuda.max_memory_allocated(selected_device)
                                      if selected_device.type == "cuda" else None),
                derivatives="first-order parameter gradients; no JVP or spatial Hessian")


@torch.no_grad()
def validate(model, split, batch_size, selected_device, seed, zero_fraction):
    rng = torch.Generator().manual_seed(seed)
    total, endpoint, count = 0., 0., len(split["context"])
    for index in torch.arange(count).split(batch_size):
        tau = sample_tau(len(index), rng, zero_fraction).to(selected_device)
        loss, mse = cfm_loss(model, *batch(split, index, selected_device), tau)
        total += float(loss) * len(index)
        endpoint += float(mse) * len(index)
    return dict(cfm=total/count, endpoint_estimate_mse=endpoint/count)


def train(config, selected_device, *, prepare_only=False, profile_only=False,
          epochs=None, max_updates=None):
    torch.set_num_threads(int(config["cpu_threads"]))
    root = Path(config["outdir"])
    run = root / "cfm"
    if not prepare_only and not profile_only and any((run / name).exists() for name in ("best.pt", "last.pt", "best_solver.pt")):
        raise FileExistsError(f"Refusing to overwrite {run}; choose a new outdir")
    cache = prepare(config, root / "pairs.pt")
    summary = cache_summary(cache, config["evaluation"]["tolerance"])
    write_json(root / "pairs_summary.json", summary)
    if prepare_only:
        return dict(cache=str(root / "pairs.pt"), summary=summary)
    if any(summary[s]["solved"] != summary[s]["scenes"] for s in ("train", "val")):
        raise RuntimeError("QP labels miss train/val tolerance; inspect pairs_summary.json")
    settings = config["train"]
    if min(settings["epochs"], settings["batch_size"], settings["solver_validation_every"], settings["validation_calls"]) < 1:
        raise ValueError("Positive epochs, batch size and validation budgets required")
    if epochs is not None and epochs < 1:
        raise ValueError("Positive epochs required")
    zero_fraction = float(settings["tau_zero_fraction"])
    if not 0 <= zero_fraction < 1:
        raise ValueError("tau_zero_fraction must be in [0,1); CFM must sample intermediate times")
    torch.manual_seed(int(config["seed"]))
    model = ConditionalField(**config["model"]).to(selected_device)
    training, validation = cache["splits"]["train"], cache["splits"]["val"]
    example = batch(training, torch.arange(min(settings["batch_size"], len(training["context"]))), selected_device)
    profiling = profile(model, example, selected_device)
    write_json(root / "profile_cfm.json", dict(device=str(selected_device), costs=profiling))
    if profile_only:
        return profiling
    run.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    count = len(training["context"])
    batches = math.ceil(count / settings["batch_size"])
    budget = max_updates if max_updates is not None else (
        epochs * batches if epochs is not None else settings.get("max_updates", settings["epochs"] * batches))
    if budget < 1:
        raise ValueError("Positive optimizer update budget required")
    history, best, best_solver = [], float("inf"), (float("inf"), float("inf"))
    updates, last_diagnostic = 0, 0
    began = perf_counter()
    for epoch in range(math.ceil(budget / batches)):
        model.train()
        order_rng = torch.Generator().manual_seed(config["seed"] + epoch)
        tau_rng = torch.Generator().manual_seed(config["seed"] + 100000 + epoch)
        total, endpoint, seen = 0., 0., 0
        for indices in torch.randperm(count, generator=order_rng).split(settings["batch_size"]):
            tau = sample_tau(len(indices), tau_rng, zero_fraction).to(selected_device)
            loss, mse = cfm_loss(model, *batch(training, indices, selected_device), tau)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CFM loss; checkpoint not updated")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            seen += len(indices)
            total += float(loss.detach()) * len(indices)
            endpoint += float(mse) * len(indices)
            if updates >= budget:
                break
        model.eval()
        val = validate(model, validation, settings["batch_size"], selected_device,
                       config["seed"] + 200000, zero_fraction)
        if not all(math.isfinite(x) for x in val.values()):
            raise FloatingPointError("Nonfinite validation metrics")
        row = dict(epoch=epoch+1, updates=updates, train_cfm=total/seen,
                   train_endpoint_estimate_mse=endpoint/seen, val_cfm=val["cfm"],
                   val_endpoint_estimate_mse=val["endpoint_estimate_mse"],
                   elapsed_seconds=perf_counter()-began)
        diagnostic = (updates-last_diagnostic >= settings["solver_validation_every"]
                      or updates == budget or epoch == 0)
        if diagnostic:
            row["solver_validation"] = solver_validation(model, validation, config, selected_device)
            last_diagnostic = updates
        history.append(row)
        checkpoint = dict(format=CHECKPOINT_FORMAT, objective="cfm", epoch=epoch+1, updates=updates,
                          config=config, cache_spec=cache["spec"], model=model.state_dict(),
                          optimizer=optimizer.state_dict(), val_cfm=val["cfm"],
                          selection_metric="fixed-sample validation CFM loss")
        torch.save(checkpoint, run / "last.pt")
        if val["cfm"] < best:
            best = val["cfm"]
            torch.save(checkpoint, run / "best.pt")
        if diagnostic:
            stats = row["solver_validation"]["balanced"]
            score = (-stats["success_rate"], stats["mean_projected_gradient"])
            if score < best_solver:
                best_solver = score
                torch.save(dict(checkpoint,
                    selection_metric="group-balanced raw validation success, then PG residual",
                    solver_validation=row["solver_validation"]), run / "best_solver.pt")
        write_json(run / "history.json", history)
        print(f"[{epoch+1:03d} updates={updates}/{budget}] cfm={total/seen:.6g} "
              f"val_cfm={val['cfm']:.6g} endpoint_estimate_mse={val['endpoint_estimate_mse']:.6g}", flush=True)
        if diagnostic:
            print(f"  balanced solver validation: {stats}", flush=True)
    return dict(checkpoint=str(run / "best_solver.pt"), best_val_cfm=best, updates=updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplier_cfm.yaml")
    parser.add_argument("--device")
    parser.add_argument("--outdir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-updates", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--prepare-pilot", type=int, metavar="SCENES_PER_SIZE")
    modes.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.outdir:
        config["outdir"] = args.outdir
    if args.prepare_pilot is not None:
        if args.prepare_pilot < 1:
            parser.error("--prepare-pilot must be positive")
        config["outdir"] = str(Path(config["outdir"]) / "prepare_pilot")
        for split in ("train", "val", "test"):
            key = split + "_per_size"
            config["data"][key] = min(config["data"][key], args.prepare_pilot)
        for key in ("release_count", "floor_count"):
            config["data"][key] = min(config["data"][key], args.prepare_pilot)
    result = train(config, device(args.device or config["device"]),
                   prepare_only=args.prepare_only or args.prepare_pilot is not None,
                   profile_only=args.profile_only, epochs=args.epochs, max_updates=args.max_updates)
    print(result)


if __name__ == "__main__":
    main()
