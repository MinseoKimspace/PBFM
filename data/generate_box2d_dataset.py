from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from data.box2d_render import render_split_samples
except ModuleNotFoundError:
    from box2d_render import render_split_samples

SPLITS = ("train", "val", "test")


class SpawnPlacementError(RuntimeError):
    pass


def load_config(path: str) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.")
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Box2D next-step dataset.")
    parser.add_argument("--config", default="configs/dataset_box2d.yaml")
    cli = parser.parse_args()
    cfg = load_config(cli.config)
    render_enabled = bool(get(cfg, "render", "enabled", False))

    return argparse.Namespace(
        config=cli.config,
        output=str(get(cfg, "dataset", "output", "data/box2d_next_step.pt")),
        train_samples=int(get(cfg, "dataset", "train_samples", 8192)),
        val_samples=int(get(cfg, "dataset", "val_samples", 2048)),
        test_samples=int(get(cfg, "dataset", "test_samples", 2048)),
        num_objects=int(get(cfg, "dataset", "num_objects", 8)),
        radius_min=float(get(cfg, "dataset", "radius_min", 0.05)),
        radius_max=float(get(cfg, "dataset", "radius_max", 0.15)),
        transitions_per_world=int(get(cfg, "dataset", "transitions_per_world", 64)),
        dynamic_fraction=float(get(cfg, "dataset", "dynamic_fraction", 0.75)),
        dynamic_score_threshold=float(get(cfg, "dataset", "dynamic_score_threshold", 0.05)),
        contact_margin=float(get(cfg, "dataset", "contact_margin", 0.25)),
        rollout_count=int(get(cfg, "dataset", "rollout_count", 0)),
        rollout_steps=int(get(cfg, "dataset", "rollout_steps", 0)),
        xy_limit=float(get(cfg, "layout", "xy_limit", 1.0)),
        y_ground=float(get(cfg, "layout", "y_ground", 0.0)),
        wall_thickness=float(get(cfg, "layout", "wall_thickness", 0.08)),
        spawn_padding=float(get(cfg, "layout", "spawn_padding", 0.01)),
        spawn_y_min_ratio=float(get(cfg, "layout", "spawn_y_min_ratio", 0.35)),
        spawn_y_max_ratio=float(get(cfg, "layout", "spawn_y_max_ratio", 0.95)),
        max_placement_tries=int(get(cfg, "layout", "max_placement_tries", 128)),
        gravity_y=float(get(cfg, "box2d", "gravity_y", -9.8)),
        density=float(get(cfg, "box2d", "density", 1.0)),
        friction=float(get(cfg, "box2d", "friction", 0.0)),
        restitution=float(get(cfg, "box2d", "restitution", 0.0)),
        linear_damping=float(get(cfg, "box2d", "linear_damping", 0.1)),
        angular_damping=float(get(cfg, "box2d", "angular_damping", 0.0)),
        time_step=float(get(cfg, "box2d", "time_step", 1.0 / 60.0)),
        velocity_iters=int(get(cfg, "box2d", "velocity_iters", 8)),
        position_iters=int(get(cfg, "box2d", "position_iters", 3)),
        max_steps=int(get(cfg, "box2d", "max_steps", 1200)),
        min_steps_before_check=int(get(cfg, "box2d", "min_steps_before_check", 240)),
        sleep_window=int(get(cfg, "box2d", "sleep_window", 90)),
        linear_velocity_eps=float(get(cfg, "box2d", "linear_velocity_eps", 0.03)),
        angular_velocity_eps=float(get(cfg, "box2d", "angular_velocity_eps", 0.05)),
        seed=int(get(cfg, "runtime", "seed", 42)),
        require_settled=bool(get(cfg, "runtime", "require_settled", True)),
        max_resample_attempts=int(get(cfg, "runtime", "max_resample_attempts", 8)),
        render_dir=str(get(cfg, "render", "dir", "")) if render_enabled else "",
        render_split=str(get(cfg, "render", "split", "train")),
        render_count=int(get(cfg, "render", "count", 6)),
        render_size=int(get(cfg, "render", "size", 1024)),
    )


def load_box2d() -> tuple[Any, Any, Any, float]:
    try:
        from Box2D import b2CircleShape, b2PolygonShape, b2World, b2_linearSlop
    except Exception as exc:
        raise RuntimeError("Box2D import failed. Install requirements.txt first.") from exc
    return b2World, b2CircleShape, b2PolygonShape, float(b2_linearSlop)


def make_world(args: argparse.Namespace, b2_world_ctor: Any, b2_polygon_shape_ctor: Any) -> Any:
    world = b2_world_ctor(gravity=(0.0, args.gravity_y), doSleep=False)
    if hasattr(world, "warmStarting"):
        world.warmStarting = False
    half_w = args.xy_limit + args.wall_thickness
    half_t = args.wall_thickness * 0.5
    wall_h = args.xy_limit + args.wall_thickness
    y_mid = args.y_ground + wall_h * 0.5

    world.CreateStaticBody(
        position=(0.0, args.y_ground - half_t),
        shapes=b2_polygon_shape_ctor(box=(half_w, half_t)),
    )
    world.CreateStaticBody(
        position=(-args.xy_limit - half_t, y_mid),
        shapes=b2_polygon_shape_ctor(box=(half_t, wall_h * 0.5)),
    )
    world.CreateStaticBody(
        position=(args.xy_limit + half_t, y_mid),
        shapes=b2_polygon_shape_ctor(box=(half_t, wall_h * 0.5)),
    )
    return world


def sample_radius(args: argparse.Namespace, generator: torch.Generator) -> torch.Tensor:
    return torch.empty(args.num_objects, dtype=torch.float32).uniform_(
        args.radius_min,
        args.radius_max,
        generator=generator,
    )


def sample_spawn_points(args: argparse.Namespace, radius: torch.Tensor, rng: random.Random) -> list[tuple[float, float]]:
    placements: list[tuple[float, float] | None] = [None] * args.num_objects
    placed: list[int] = []
    order = sorted(range(args.num_objects), key=lambda idx: float(radius[idx]), reverse=True)

    for i in order:
        ri = float(radius[i])
        x_min = -args.xy_limit + ri + args.spawn_padding
        x_max = args.xy_limit - ri - args.spawn_padding
        y_min = args.y_ground + max(args.spawn_y_min_ratio * args.xy_limit, ri + args.spawn_padding)
        y_max = args.y_ground + min(args.spawn_y_max_ratio * args.xy_limit, args.xy_limit - ri - args.spawn_padding)
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Spawn area is invalid.")

        for _ in range(args.max_placement_tries):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            ok = True
            for j in placed:
                px, py = placements[j]
                min_dist = ri + float(radius[j]) + args.spawn_padding
                ok = ok and (x - px) ** 2 + (y - py) ** 2 >= min_dist**2
            if ok:
                placements[i] = (x, y)
                placed.append(i)
                break

        if placements[i] is None:
            raise SpawnPlacementError("Failed to place non-overlapping circles.")

    return [point for point in placements if point is not None]


def create_bodies(
    args: argparse.Namespace,
    world: Any,
    b2_circle_shape_ctor: Any,
    radius: torch.Tensor,
    spawn_points: list[tuple[float, float]],
) -> list[Any]:
    if abs(args.friction) > 1e-8:
        raise ValueError("This minimal Markov dataset requires friction=0.0 because angular velocity is not in state.")

    bodies = []
    for i, (x, y) in enumerate(spawn_points):
        body = world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            linearDamping=args.linear_damping,
            angularDamping=args.angular_damping,
            allowSleep=False,
        )
        body.CreateFixture(
            shape=b2_circle_shape_ctor(radius=float(radius[i])),
            density=args.density,
            friction=args.friction,
            restitution=args.restitution,
        )
        bodies.append(body)
    return bodies


def read_state(bodies: list[Any]) -> torch.Tensor:
    state = torch.zeros(len(bodies), 4, dtype=torch.float32)
    for i, body in enumerate(bodies):
        state[i, 0] = float(body.position.x)
        state[i, 1] = float(body.position.y)
        state[i, 2] = float(body.linearVelocity.x)
        state[i, 3] = float(body.linearVelocity.y)
    return state


def simulate_episode(
    args: argparse.Namespace,
    radius: torch.Tensor,
    seed: int,
    b2_world_ctor: Any,
    b2_circle_shape_ctor: Any,
    b2_polygon_shape_ctor: Any,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    rng = random.Random(seed)
    world = make_world(args, b2_world_ctor, b2_polygon_shape_ctor)
    spawn_points = sample_spawn_points(args, radius, rng)
    bodies = create_bodies(args, world, b2_circle_shape_ctor, radius, spawn_points)

    sources = []
    targets = []
    stable_count = 0
    settled = False

    for step in range(args.max_steps):
        source = read_state(bodies)
        world.Step(args.time_step, args.velocity_iters, args.position_iters)
        target = read_state(bodies)

        sources.append(source)
        targets.append(target)

        max_lin = max(float(body.linearVelocity.length) for body in bodies)
        max_ang = max(abs(float(body.angularVelocity)) for body in bodies)
        if step + 1 < args.min_steps_before_check:
            continue
        stable_count = stable_count + 1 if max_lin <= args.linear_velocity_eps and max_ang <= args.angular_velocity_eps else 0
        if stable_count >= args.sleep_window:
            settled = True
            break

    return torch.stack(sources), torch.stack(targets), settled


def transition_scores(source: torch.Tensor, target: torch.Tensor, radius: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    pos = source[..., :2]
    vel = source[..., 2:]
    delta = target - source

    speed = torch.linalg.norm(vel, dim=-1).amax(dim=1)
    step_change = torch.linalg.norm(delta, dim=-1).amax(dim=1)
    ground_gap = pos[..., 1] - radius.unsqueeze(0) - args.y_ground
    ground_near = torch.relu(args.contact_margin - ground_gap).amax(dim=1)

    pair_delta = pos[:, :, None, :] - pos[:, None, :, :]
    pair_dist = torch.linalg.norm(pair_delta, dim=-1)
    pair_radius = radius[None, :, None] + radius[None, None, :]
    separation = pair_dist - pair_radius
    pair_near = torch.relu(args.contact_margin - separation)
    eye = torch.eye(source.size(1), dtype=torch.bool, device=source.device).unsqueeze(0)
    pair_near = pair_near.masked_fill(eye, 0.0).amax(dim=(1, 2))

    return speed + 5.0 * step_change + 2.0 * ground_near + 2.0 * pair_near


def select_transition_indices(
    source: torch.Tensor,
    target: torch.Tensor,
    radius: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_count = args.transitions_per_world
    length = source.size(0)
    if max_count < 1:
        raise ValueError("transitions_per_world must be >= 1.")
    if length <= max_count:
        selected = torch.arange(length)
        scores = transition_scores(source, target, radius, args)
        return selected, scores, scores > args.dynamic_score_threshold

    scores = transition_scores(source, target, radius, args)
    dynamic_count = min(max_count, max(1, int(round(max_count * args.dynamic_fraction))))
    ranked = torch.argsort(scores, descending=True)
    selected = ranked[:dynamic_count]

    rest_count = max_count - selected.numel()
    if rest_count > 0:
        uniform = torch.linspace(0, length - 1, steps=rest_count).round().long()
        selected = torch.cat([selected, uniform]).unique()

    if selected.numel() < max_count:
        keep = torch.ones(length, dtype=torch.bool)
        keep[selected] = False
        fill = ranked[keep[ranked]][: max_count - selected.numel()]
        selected = torch.cat([selected, fill])

    selected = selected[:max_count].sort().values
    selected_scores = scores[selected]
    dynamic = selected_scores > args.dynamic_score_threshold
    return selected, selected_scores, dynamic


def generate_split(
    split: str,
    num_transitions: int,
    split_seed: int,
    args: argparse.Namespace,
    b2_world_ctor: Any,
    b2_circle_shape_ctor: Any,
    b2_polygon_shape_ctor: Any,
) -> dict[str, torch.Tensor]:
    radius_generator = torch.Generator().manual_seed(split_seed)
    source_chunks = []
    target_chunks = []
    radius_chunks = []
    score_chunks = []
    dynamic_chunks = []
    rollout_initial_chunks = []
    rollout_target_chunks = []
    rollout_radius_chunks = []
    rollout_length_chunks = []
    world_idx = 0
    total = 0
    log_next = max(1, num_transitions // 10)

    while total < num_transitions:
        radius = sample_radius(args, radius_generator)
        last_error: Exception | None = None
        episode = None

        for attempt in range(args.max_resample_attempts + 1):
            seed = split_seed + world_idx * 9973 + attempt * 104729
            try:
                source, target, settled = simulate_episode(
                    args,
                    radius,
                    seed,
                    b2_world_ctor,
                    b2_circle_shape_ctor,
                    b2_polygon_shape_ctor,
                )
            except SpawnPlacementError as exc:
                last_error = exc
                continue
            if settled or not args.require_settled:
                episode = (source, target)
                break

        if episode is None:
            raise RuntimeError(f"[{split}] failed to generate a settled episode. Last error: {last_error}")

        source, target = episode
        if args.rollout_steps > 0 and len(rollout_initial_chunks) < args.rollout_count:
            rollout_length = min(args.rollout_steps, target.size(0))
            if rollout_length > 0:
                rollout_target = target[:rollout_length]
                if rollout_length < args.rollout_steps:
                    pad = rollout_target[-1:].expand(args.rollout_steps - rollout_length, -1, -1)
                    rollout_target = torch.cat([rollout_target, pad], dim=0)
                rollout_initial_chunks.append(source[0].clone())
                rollout_target_chunks.append(rollout_target.clone())
                rollout_radius_chunks.append(radius.clone())
                rollout_length_chunks.append(torch.tensor(rollout_length, dtype=torch.long))

        selected, selected_scores, dynamic = select_transition_indices(source, target, radius, args)
        take = min(num_transitions - total, selected.numel())
        selected = selected[:take]
        selected_scores = selected_scores[:take]
        dynamic = dynamic[:take]

        source_chunks.append(source[selected])
        target_chunks.append(target[selected])
        radius_chunks.append(radius.expand(take, -1).clone())
        score_chunks.append(selected_scores)
        dynamic_chunks.append(dynamic)

        total += take
        world_idx += 1
        if total >= log_next:
            print(f"[{split}] generated {total}/{num_transitions} transitions from {world_idx} worlds")
            log_next += max(1, num_transitions // 10)

    split_data = {
        "source": torch.cat(source_chunks).contiguous(),
        "target": torch.cat(target_chunks).contiguous(),
        "radius": torch.cat(radius_chunks).contiguous(),
        "score": torch.cat(score_chunks).float().contiguous(),
        "dynamic": torch.cat(dynamic_chunks).bool().contiguous(),
    }
    if rollout_initial_chunks:
        split_data.update(
            {
                "rollout_initial": torch.stack(rollout_initial_chunks).contiguous(),
                "rollout_target": torch.stack(rollout_target_chunks).contiguous(),
                "rollout_radius": torch.stack(rollout_radius_chunks).contiguous(),
                "rollout_length": torch.stack(rollout_length_chunks).contiguous(),
            }
        )
    return split_data


def delta_stats(split: dict[str, torch.Tensor]) -> tuple[list[float], list[float]]:
    delta = (split["target"] - split["source"]).reshape(-1, 4)
    mean = delta.mean(dim=0)
    std = delta.std(dim=0).clamp_min(1e-6)
    return mean.tolist(), std.tolist()


def build_meta(args: argparse.Namespace, box2d_linear_slop: float, delta_mean: list[float], delta_std: list[float]) -> dict[str, Any]:
    return {
        "generator": "box2d_next_step",
        "state": "x,y,vx,vy",
        "source": "state before Box2D step",
        "target": "state after Box2D step",
        "box2d_linear_slop": box2d_linear_slop,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        **vars(args),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    b2_world_ctor, b2_circle_shape_ctor, b2_polygon_shape_ctor, box2d_linear_slop = load_box2d()
    counts = {"train": args.train_samples, "val": args.val_samples, "test": args.test_samples}
    offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
    splits = {
        split: generate_split(
            split,
            counts[split],
            args.seed + offsets[split],
            args,
            b2_world_ctor,
            b2_circle_shape_ctor,
            b2_polygon_shape_ctor,
        )
        for split in SPLITS
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    delta_mean, delta_std = delta_stats(splits["train"])
    torch.save({"meta": build_meta(args, box2d_linear_slop, delta_mean, delta_std), **splits}, output_path)
    print(f"Saved dataset to {output_path}")

    if args.render_dir:
        render_split_samples(
            splits[args.render_split],
            Path(args.render_dir),
            args.render_split,
            args.render_count,
            args.xy_limit,
            args.y_ground,
            args.render_size,
        )
        print(f"Saved renders to {args.render_dir}")


if __name__ == "__main__":
    main()
