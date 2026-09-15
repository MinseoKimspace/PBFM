"""Physical-time diagnostic around the frozen-normal PGS/CFM solve.

Rebuild contacts each physical frame, start lambda=0, update velocity by FD.
No relinearization inside a frame, restitution, friction, CCD resolution,
warm-start transfer, position/velocity clipping, or hidden finishing solver.
CFM DOES contain analytic multiplier projection in its endpoint head. Geometric/swept
checks only measure errors; they NEVER correct the trajectory.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from src.contact_flow.dynamics import free_position, finite_difference_state
from src.contact_flow.physics import PhysicsConfig, geometry, valid_positions
from .evaluation import timed, write_json
from .model import CHECKPOINT_FORMAT, SOLVER_DESCRIPTION, LocalProjection
from .problem import converged, decode, gap, make_problem, pack, position_error, residuals
from .solvers import pgs, run_cfm


def motion_scenes(physics):
    """Deterministic held-out motion templates, not cached training endpoints."""
    def scene(name, positions, velocities):
        p = torch.tensor(positions, dtype=torch.float32)
        p[:, 1] += physics.y_ground
        v = torch.tensor(velocities, dtype=torch.float32)
        return name, torch.cat((p, v), -1), torch.full((len(p),), .45)
    return [
        scene("drop_floor", [[0., 3.]], [[0., 0.]]),
        scene("oblique_collision", [[-1.2, 3.0], [1.2, 3.35]], [[2., 0.], [-2., 0.]]),
        scene("impact_stack", [[0., .452], [0., 1.354], [0., 2.256], [.18, 4.4]],
              [[0., 0.], [0., 0.], [0., 0.], [-.2, -1.]]),
        scene("collapse_stack", [[0., .452], [.24, 1.322], [.48, 2.192], [.72, 3.062]],
              [[0., 0.], [.3, 0.], [.6, 0.], [.9, 0.]])]


def swept_pairs(start, end, radius, slop):
    """Straight physical-frame segment diagnostic, NOT the FM path or a CCD solve."""
    i, j = torch.triu_indices(len(radius), len(radius), 1, device=start.device)
    if not len(i):
        return dict(max_penetration=0., endpoint_missed_count=0)
    relative = start[i] - start[j]
    motion = (end[i] - end[j]) - relative
    t = (-(relative * motion).sum(-1) / motion.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    minimum = (relative + t[:, None] * motion).norm(dim=-1)
    limit = radius[i] + radius[j] - slop
    violation = (limit - minimum).clamp_min(0)
    separated = (relative.norm(dim=-1) >= limit) & ((end[i]-end[j]).norm(dim=-1) >= limit)
    return dict(max_penetration=float(violation.max()),
                endpoint_missed_count=int(((violation > 1e-6) & separated).sum()))


def build_frame_problem(proposal, radius, physics, eta_fraction):
    geo = geometry(proposal[None], radius[None], physics)
    near = geo["near"][0]
    problem = pack([make_problem(proposal, geo["J"][0, near],
        geo["inverse_mass"][0].repeat_interleave(2), geo["gap"][0, near]+physics.slop, eta_fraction)])
    return problem, geo


@torch.no_grad()
def simulate(initial, radius, config, device, model=None, calls=1, guarded=False):
    physics = PhysicsConfig(**config["physics"])
    settings, ref = config["rollout"], config["reference"]
    tolerance = config["evaluation"]["tolerance"]
    dt = float(config["dynamics"]["time_step"])
    steps = int(settings["steps"])
    if steps < 1 or min(settings["max_speed"], settings["max_position"]) <= 0:
        raise ValueError("Positive rollout steps and diagnostic failure bounds required")
    state, radius = initial.to(device), radius.to(device)
    states, rows = [state.cpu().clone()], []
    previous = geometry(state[None, :, :2], radius[None], physics)["gap"][0] <= physics.slop+tolerance
    failure = None
    for frame in range(steps):
        proposal, dynamics_seconds = timed(device, lambda: free_position(state[None], **config["dynamics"])[0])
        if not valid_positions(proposal[None]).all():
            failure = dict(frame=frame+1, reason="degenerate_or_nonfinite_proposal")
            break
        (problem, before), setup_seconds = timed(device, lambda: build_frame_problem(proposal, radius, physics, ref["eta_fraction"]))
        start = torch.zeros_like(problem["c"])
        def solve():
            if model is None:
                return pgs(problem, start, tolerance, ref["max_sweeps"])
            return run_cfm(model, problem, start, calls,
                           guarded, settings.get("max_backtracks", 8))
        result, solver_seconds = timed(device, solve)
        counters = {key: int(result[key].sum()) for key in
                    ("nfe", "neural_evals", "backtracks", "interventions", "sweeps", "contact_evals") if key in result}
        if "completed" in result and not bool(result["completed"].all()):
            failure = dict(frame=frame+1, reason="incomplete_solver_clock", counters=counters,
                           solver_tau=float(result["time"][0]), failure_code=int(result["failure_code"][0]),
                           attempt_seconds=dynamics_seconds+setup_seconds+solver_seconds)
            break  # A partial map is NOT silently promoted to the next physical frame.
        def update():
            position = decode(problem, result["final"]).reshape(1, len(radius), 2)
            return finite_difference_state(state[None], position, dt)[0]
        next_state, update_seconds = timed(device, update)
        if not torch.isfinite(next_state).all() or not valid_positions(next_state[None, :, :2]).all():
            failure = dict(frame=frame+1, reason="nonfinite_or_coincident_endpoint", counters=counters)
            break
        # Diagnostics are deliberately outside the measured simulation timings.
        after = geometry(next_state[None, :, :2], radius[None], physics)
        actual_gap, near = after["gap"][0], before["near"][0]
        contacts = actual_gap <= physics.slop+tolerance
        added, released = contacts & ~previous, previous & ~contacts
        phase = ("onset" if added.any() else "release" if released.any()
                 else "contact" if contacts.any() else "free_flight")
        oracle = pgs(problem, start, ref["qp_tolerance"], ref["max_sweeps"])
        local_stats = residuals(problem, result["final"])
        fixed_gap = gap(problem, result["final"])[0, problem["mask"][0]]
        mass = math.pi * physics.density * radius.square()
        free_velocity = (proposal-state[:, :2])/dt
        kinetic = .5 * (mass * next_state[:, 2:].square().sum(-1)).sum()
        free_kinetic = .5 * (mass * free_velocity.square().sum(-1)).sum()
        pairs = len(radius)*(len(radius)-1)//2
        common = near[:pairs] & after["near"][0, :pairs]
        old_normals, new_normals = before["J"][0, :pairs][common], after["J"][0, :pairs][common]
        cos = (old_normals*new_normals).sum(-1)/2  # Pair rows have squared norm 2.
        rotation = float(torch.acos(cos.clamp(-1, 1)).max()*180/math.pi) if len(cos) else 0.
        swept = swept_pairs(state[:, :2], next_state[:, :2], radius, physics.slop)
        row = dict(frame=frame+1, physical_time=(frame+1)*dt, phase=phase,
            frozen_contacts=int(near.sum()), active_contacts=int(contacts.sum()),
            contacts_added=int(added.sum()), contacts_released=int(released.sum()),
            solver_converged=bool(converged(problem, result["final"], tolerance)[0]),
            oracle_converged=bool(oracle["converged"][0]),
            local_qp_position_mse=(float(position_error(problem, result["final"], oracle["final"])[0])
                                   if bool(oracle["converged"][0]) else None),
            projected_gradient=float(local_stats["projected_gradient"][0]),
            negative_multiplier=float(local_stats["negative_multiplier"][0]),
            linear_penetration=float(local_stats["penetration"][0]),
            geometric_penetration=float(after["max_violation"][0]),
            new_violations_outside_frozen_contacts=int(((actual_gap < -physics.slop-tolerance) & ~near).sum()),
            linearization_gap_error_max=float((actual_gap[near]+physics.slop-fixed_gap).abs().max()) if near.any() else 0.,
            max_normal_rotation_degrees=rotation,
            released_gap_mean=float(actual_gap[released].mean()) if released.any() else 0.,
            kinetic_energy=float(kinetic), projection_kinetic_delta=float(kinetic-free_kinetic),
            max_speed=float(next_state[:, 2:].norm(dim=-1).max()),
            swept_pair_penetration=swept["max_penetration"],
            swept_endpoint_missed_pairs=swept["endpoint_missed_count"],
            dynamics_seconds=dynamics_seconds+update_seconds, setup_seconds=setup_seconds,
            solver_seconds=solver_seconds,
            simulation_seconds=dynamics_seconds+setup_seconds+solver_seconds+update_seconds, **counters)
        rows.append(row)
        state, previous = next_state, contacts
        states.append(state.cpu().clone())
        if row["max_speed"] > settings["max_speed"] or float(state[:, :2].abs().max()) > settings["max_position"]:
            failure = dict(frame=frame+1, reason="diagnostic_bound_exceeded_no_clipping")
            break
    return dict(states=torch.stack(states), frames=rows, requested_steps=steps,
                completed_steps=len(rows), failure=failure, summary=rollout_summary(rows))


def rollout_summary(rows):
    def summarize(items):
        count = len(items)
        return dict(frames=count, unconverged_steps=sum(not x["solver_converged"] for x in items),
            max_geometric_penetration=max((x["geometric_penetration"] for x in items), default=0.),
            new_violations=sum(x["new_violations_outside_frozen_contacts"] for x in items),
            swept_endpoint_missed_pairs=sum(x["swept_endpoint_missed_pairs"] for x in items),
            contacts_added=sum(x["contacts_added"] for x in items),
            contacts_released=sum(x["contacts_released"] for x in items),
            mean_simulation_seconds=sum(x["simulation_seconds"] for x in items)/max(count, 1),
            max_speed=max((x["max_speed"] for x in items), default=0.))
    report = summarize(rows)
    report["phases"] = {phase: summarize([x for x in rows if x["phase"] == phase])
                        for phase in ("free_flight", "onset", "contact", "release")}
    return report


def render_motion(results, radius, physics, settings, path):
    """Common camera/time axis, finite traces only; stopped runs are labelled."""
    from PIL import Image, ImageDraw
    from data.box2d_render import render_state_image
    reference = results["pgs"]["states"]
    position = reference[..., :2]
    lower = (position-radius[None, :, None]).amin(dim=(0, 1))
    upper = (position+radius[None, :, None]).amax(dim=(0, 1))
    bounds = (float(lower[0]-.7), float(upper[0]+.7),
              float(min(lower[1]-.1, physics.y_ground-.1)), float(upper[1]+.7))
    size, stride = int(settings["image_size"]), int(settings["frame_stride"])
    if min(size, stride) < 1:
        raise ValueError("Positive rendering size/stride required")
    height = max(180, min(round(size*(bounds[3]-bounds[2])/(bounds[1]-bounds[0])), round(size*1.25)))
    frames = []
    length = max(len(run["states"]) for run in results.values())
    indices = list(range(0, length, stride))
    if indices[-1] != length-1:
        indices.append(length-1)
    for k in indices:
        canvas = Image.new("RGB", (size*len(results), height+78), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (name, run) in enumerate(results.items()):
            index = min(k, len(run["states"])-1)
            panel = render_state_image(run["states"][index], radius, physics.xy_limit,
                physics.y_ground, size, view_bounds=bounds, canvas_size=(size, height), show_centers=True)
            canvas.paste(panel, (column*size, 78))
            stopped = run["failure"] and k >= run["failure"]["frame"]
            draw.text((column*size+8, 6), name + ("  STOPPED" if stopped else ""), fill="red" if stopped else "black")
            draw.text((column*size+8, 23), f"frame={k}  state_frame={index}", fill="black")
            if index:
                row = run["frames"][index-1]
                draw.text((column*size+8, 40), f"pen={row['geometric_penetration']:.4g}  {row['phase']}", fill="black")
            p = run["states"][index, :, :2]
            outside = int(((p[:, 0] < bounds[0]) | (p[:, 0] > bounds[1]) |
                           (p[:, 1] < bounds[2]) | (p[:, 1] > bounds[3])).sum())
            if outside:
                draw.text((column*size+8, 57), f"OUT OF VIEW: {outside}", fill="red")
        frames.append(canvas)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=max(10, round(1000*settings["time_step"]*stride)))
    frames[-1].save(path.with_suffix(".png"))


def evaluate_motion(models, config, device, output, metadata=None):
    settings = config["rollout"]
    physics = PhysicsConfig(**config["physics"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_formats = {name: getattr(model, "checkpoint_format", CHECKPOINT_FORMAT)
                     for name, model in models.items()}
    formats = set(model_formats.values())
    report = dict(scope="Per-frame frozen contact diagnostic; fresh lambda=0; no finish/restitution/friction/CCD",
        model_format=next(iter(formats)) if len(formats) == 1 else ("mixed" if formats else None),
        model_formats=model_formats, solver=SOLVER_DESCRIPTION,
        raw_definition="Analytic projection in the endpoint head; no EXTRA guard/finish",
        local_definition="No neural coupling; identical endpoint formula and tau schedule",
        timing_scope="Simulation includes free dynamics, geometry/D/eta setup, solve, FD; excludes oracle/metrics/rendering. Not a speed benchmark.",
        swept_scope="Straight previous-to-next position segments only, NOT internal solver trajectories",
        device=str(device), config=config, metadata=metadata or {}, scenes={})
    for name, initial, radius in motion_scenes(physics):
        results = {"pgs": simulate(initial, radius, config, device)}
        compared = dict(local=LocalProjection(), **models) if models else {}
        for objective, model in compared.items():
            for calls in settings["calls"]:
                results[f"{objective}_k{calls}_raw"] = simulate(initial, radius, config, device, model, calls)
                if settings.get("guarded", False):
                    results[f"{objective}_k{calls}_guarded"] = simulate(initial, radius, config, device, model, calls, True)
        for run in results.values():
            count = min(len(run["states"]), len(results["pgs"]["states"]))
            for k in range(1, count):
                difference = run["states"][k]-results["pgs"]["states"][k]
                run["frames"][k-1]["pgs_rollout_position_mse"] = float(difference[:, :2].square().mean())
                run["frames"][k-1]["pgs_rollout_velocity_mse"] = float(difference[:, 2:].square().mean())
            run["pgs_comparison_frames"] = count-1  # Never pad an early-failed baseline.
        traces = {method: run["states"] for method, run in results.items()}
        torch.save(dict(radius=radius, states=traces), output.parent / f"{name}_trajectories.pt")
        report["scenes"][name] = {method: {key: value for key, value in run.items() if key != "states"}
                                 for method, run in results.items()}
        if settings["render"]:
            render_motion(results, radius, physics, dict(settings, time_step=config["dynamics"]["time_step"]),
                          output.parent / f"{name}.gif")
        write_json(output, report)  # Preserve completed scenes if a later scene fails.
        print(f"Motion {name}: " + ", ".join(f"{key}={run['completed_steps']}/{settings['steps']}"
                                             for key, run in results.items()), flush=True)
    return report
