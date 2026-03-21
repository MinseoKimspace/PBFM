from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from data.box2d_render import render_split_samples
except ModuleNotFoundError:
    from box2d_render import render_split_samples

SPLITS = ("train", "val", "test")
META_KEYS = (
    "seed",
    "num_objects",
    "radius_min",
    "radius_max",
    "xy_limit",
    "y_ground",
    "wall_thickness",
    "spawn_padding",
    "spawn_y_min_ratio",
    "spawn_y_max_ratio",
    "max_placement_tries",
    "gravity_y",
    "density",
    "friction",
    "restitution",
    "linear_damping",
    "angular_damping",
    "time_step",
    "velocity_iters",
    "position_iters",
    "max_steps",
    "min_steps_before_check",
    "sleep_window",
    "linear_velocity_eps",
    "angular_velocity_eps",
    "require_settled",
    "max_resample_attempts",
)


class SpawnPlacementError(RuntimeError):
    pass


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)!r}")
    return data


def _get_nested(config: Mapping[str, Any], *keys: str, default: Any) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _cfg_default(cfg: Mapping[str, Any], section: str, key: str, default: Any, cast: Any) -> Any:
    return cast(_get_nested(cfg, section, key, default=default))


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)
    default_render_dir = (
        str(_get_nested(cfg, "render", "dir", default="")) if bool(_get_nested(cfg, "render", "enabled", default=False)) else ""
    )

    parser = argparse.ArgumentParser(
        description="Generate settled circle states using Box2D inside a fixed layout and save dataset splits."
    )
    parser.add_argument("--config", type=str, default=pre_args.config)
    scalar_args = [
        ("--output", str, "dataset", "output", "data/box2d_circles.pt"),
        ("--train-samples", int, "dataset", "train_samples", 8192),
        ("--val-samples", int, "dataset", "val_samples", 2048),
        ("--test-samples", int, "dataset", "test_samples", 2048),
        ("--num-objects", int, "dataset", "num_objects", 8),
        ("--radius-min", float, "dataset", "radius_min", 0.05),
        ("--radius-max", float, "dataset", "radius_max", 0.15),
        ("--xy-limit", float, "layout", "xy_limit", 1.0),
        ("--y-ground", float, "layout", "y_ground", 0.0),
        ("--wall-thickness", float, "layout", "wall_thickness", 0.08),
        ("--spawn-padding", float, "layout", "spawn_padding", 0.01),
        ("--spawn-y-min-ratio", float, "layout", "spawn_y_min_ratio", 0.35),
        ("--spawn-y-max-ratio", float, "layout", "spawn_y_max_ratio", 0.95),
        ("--max-placement-tries", int, "layout", "max_placement_tries", 128),
        ("--gravity-y", float, "box2d", "gravity_y", -9.8),
        ("--density", float, "box2d", "density", 1.0),
        ("--friction", float, "box2d", "friction", 0.4),
        ("--restitution", float, "box2d", "restitution", 0.0),
        ("--linear-damping", float, "box2d", "linear_damping", 0.1),
        ("--angular-damping", float, "box2d", "angular_damping", 0.1),
        ("--time-step", float, "box2d", "time_step", 1.0 / 60.0),
        ("--velocity-iters", int, "box2d", "velocity_iters", 8),
        ("--position-iters", int, "box2d", "position_iters", 3),
        ("--max-steps", int, "box2d", "max_steps", 1200),
        ("--min-steps-before-check", int, "box2d", "min_steps_before_check", 240),
        ("--sleep-window", int, "box2d", "sleep_window", 90),
        ("--linear-velocity-eps", float, "box2d", "linear_velocity_eps", 0.03),
        ("--angular-velocity-eps", float, "box2d", "angular_velocity_eps", 0.05),
        ("--seed", int, "runtime", "seed", 42),
        ("--render-count", int, "render", "count", 6),
        ("--render-size", int, "render", "size", 1024),
        ("--max-resample-attempts", int, "runtime", "max_resample_attempts", 8),
        ("--support-check-eps", float, "runtime", "support_check_eps", 0.02),
    ]
    for flag, cast, section, key, default in scalar_args:
        parser.add_argument(flag, type=cast, default=_cfg_default(cfg, section, key, default, cast))

    parser.add_argument("--render-dir", type=str, default=default_render_dir)
    parser.add_argument(
        "--render-split",
        type=str,
        default=_cfg_default(cfg, "render", "split", "train", str),
        choices=SPLITS,
    )
    parser.add_argument(
        "--require-settled",
        dest="require_settled",
        action="store_true",
        default=bool(_get_nested(cfg, "runtime", "require_settled", default=True)),
    )
    parser.add_argument("--allow-unsettled", dest="require_settled", action="store_false")
    return parser.parse_args(remaining)


def load_box2d() -> tuple[Any, Any, Any, float]:
    try:
        from Box2D import b2CircleShape, b2PolygonShape, b2World, b2_linearSlop
    except Exception as exc:
        raise RuntimeError(
            "Box2D import failed. Install a compatible package first, e.g. `pip install Box2D`."
        ) from exc
    return b2World, b2CircleShape, b2PolygonShape, float(b2_linearSlop)


def make_radius_batch(
    num_samples: int,
    num_objects: int,
    radius_min: float,
    radius_max: float,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    radius = torch.empty(num_samples, num_objects, dtype=torch.float32).uniform_(
        radius_min, radius_max, generator=g
    )
    return radius.contiguous()


def _make_world(
    b2_world_ctor: Any,
    b2_polygon_shape_ctor: Any,
    gravity_y: float,
    xy_limit: float,
    y_ground: float,
    wall_thickness: float,
) -> Any:
    world = b2_world_ctor(gravity=(0.0, gravity_y), doSleep=True)

    half_w = float(xy_limit) + float(wall_thickness)
    half_t = float(wall_thickness) * 0.5
    wall_h = float(xy_limit) + float(wall_thickness)
    half_wall_h = wall_h * 0.5
    y_mid = float(y_ground) + wall_h * 0.5

    world.CreateStaticBody(
        position=(0.0, float(y_ground) - half_t),
        shapes=b2_polygon_shape_ctor(box=(half_w, half_t)),
    )
    world.CreateStaticBody(
        position=(-float(xy_limit) - half_t, y_mid),
        shapes=b2_polygon_shape_ctor(box=(half_t, half_wall_h)),
    )
    world.CreateStaticBody(
        position=(float(xy_limit) + half_t, y_mid),
        shapes=b2_polygon_shape_ctor(box=(half_t, half_wall_h)),
    )
    return world


def _sample_spawn_points(
    rng: random.Random,
    radius: torch.Tensor,
    num_objects: int,
    xy_limit: float,
    y_ground: float,
    spawn_padding: float,
    spawn_y_min_ratio: float,
    spawn_y_max_ratio: float,
    max_placement_tries: int,
) -> list[tuple[float, float]]:
    placements: list[tuple[float, float] | None] = [None] * num_objects
    placed_indices: list[int] = []

    order = sorted(range(num_objects), key=lambda idx: float(radius[idx].item()), reverse=True)
    for i in order:
        ri = float(radius[i].item())
        x_min = -xy_limit + ri + spawn_padding
        x_max = xy_limit - ri - spawn_padding
        y_min = y_ground + max(spawn_y_min_ratio * xy_limit, ri + spawn_padding)
        y_max = y_ground + min(spawn_y_max_ratio * xy_limit, xy_limit - ri - spawn_padding)
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Spawn area is invalid. Adjust layout or radius constraints.")
        placed = False
        for _ in range(max_placement_tries):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            valid = True
            for j in placed_indices:
                placed_point = placements[j]
                if placed_point is None:
                    raise RuntimeError("Internal error: placed index missing spawn point.")
                px, py = placed_point
                rj = float(radius[j].item())
                dx = x - px
                dy = y - py
                if (dx * dx + dy * dy) < (ri + rj + spawn_padding) ** 2:
                    valid = False
                    break
            if valid:
                placements[i] = (x, y)
                placed_indices.append(i)
                placed = True
                break

        if not placed:
            raise SpawnPlacementError(
                "Failed to place non-overlapping initial circles. "
                "Increase max_placement_tries / max_resample_attempts or loosen the layout."
            )

    result: list[tuple[float, float]] = []
    for point in placements:
        if point is None:
            raise RuntimeError("Internal error: incomplete spawn placement.")
        result.append((float(point[0]), float(point[1])))
    return result


def _clamp_state_in_layout_inplace(
    state: torch.Tensor,
    radius: torch.Tensor,
    xy_limit: float,
    y_ground: float,
) -> None:
    x_min = -xy_limit + radius
    x_max = xy_limit - radius
    y_min = y_ground + radius
    y_max = y_ground + xy_limit - radius
    state[:, 0] = torch.maximum(torch.minimum(state[:, 0], x_max), x_min)
    state[:, 1] = torch.maximum(torch.minimum(state[:, 1], y_max), y_min)


def _is_supported_configuration(
    state: torch.Tensor,
    radius: torch.Tensor,
    xy_limit: float,
    y_ground: float,
    eps: float,
) -> bool:
    num_objects = int(state.size(0))
    if num_objects <= 0:
        return True

    seeds: set[int] = set()
    adjacency: list[set[int]] = [set() for _ in range(num_objects)]

    for i in range(num_objects):
        x = float(state[i, 0].item())
        y = float(state[i, 1].item())
        ri = float(radius[i].item())
        on_ground = abs(y - (y_ground + ri)) <= eps
        on_left = abs(x - (-xy_limit + ri)) <= eps
        on_right = abs(x - (xy_limit - ri)) <= eps
        if on_ground or on_left or on_right:
            seeds.add(i)

    if not seeds:
        return False

    for i in range(num_objects - 1):
        for j in range(i + 1, num_objects):
            dist = float(torch.linalg.norm(state[i] - state[j]).item())
            target = float((radius[i] + radius[j]).item())
            if abs(dist - target) <= eps:
                adjacency[i].add(j)
                adjacency[j].add(i)

    frontier = list(seeds)
    seen = set(seeds)
    while frontier:
        cur = frontier.pop()
        for nxt in adjacency[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return len(seen) == num_objects


def simulate_one_sample(
    radius: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    b2_world_ctor: Any,
    b2_circle_shape_ctor: Any,
    b2_polygon_shape_ctor: Any,
    support_check_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    rng = random.Random(seed)
    world = _make_world(
        b2_world_ctor=b2_world_ctor,
        b2_polygon_shape_ctor=b2_polygon_shape_ctor,
        gravity_y=args.gravity_y,
        xy_limit=args.xy_limit,
        y_ground=args.y_ground,
        wall_thickness=args.wall_thickness,
    )
    spawn_points = _sample_spawn_points(
        rng=rng,
        radius=radius,
        num_objects=args.num_objects,
        xy_limit=args.xy_limit,
        y_ground=args.y_ground,
        spawn_padding=args.spawn_padding,
        spawn_y_min_ratio=args.spawn_y_min_ratio,
        spawn_y_max_ratio=args.spawn_y_max_ratio,
        max_placement_tries=args.max_placement_tries,
    )

    bodies: list[Any] = []
    state_init = torch.zeros(args.num_objects, 2, dtype=torch.float32)
    for i in range(args.num_objects):
        ri = float(radius[i].item())
        x, y = spawn_points[i]
        body = world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            linearDamping=args.linear_damping,
            angularDamping=args.angular_damping,
            allowSleep=True,
        )
        body.CreateFixture(
            shape=b2_circle_shape_ctor(radius=ri),
            density=args.density,
            friction=args.friction,
            restitution=args.restitution,
        )
        bodies.append(body)
        state_init[i, 0] = x
        state_init[i, 1] = y

    stable_count = 0
    settle_step = args.max_steps
    for step in range(args.max_steps):
        world.Step(args.time_step, args.velocity_iters, args.position_iters)

        max_lin = 0.0
        max_ang = 0.0
        for body in bodies:
            lin = float(body.linearVelocity.length)
            ang = abs(float(body.angularVelocity))
            if lin > max_lin:
                max_lin = lin
            if ang > max_ang:
                max_ang = ang

        if (step + 1) < args.min_steps_before_check:
            stable_count = 0
            continue

        if max_lin <= args.linear_velocity_eps and max_ang <= args.angular_velocity_eps:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= args.sleep_window:
            settle_step = step + 1
            break

    state_relaxed = torch.zeros(args.num_objects, 2, dtype=torch.float32)
    for i, body in enumerate(bodies):
        state_relaxed[i, 0] = float(body.position.x)
        state_relaxed[i, 1] = float(body.position.y)

    _clamp_state_in_layout_inplace(
        state=state_relaxed,
        radius=radius,
        xy_limit=args.xy_limit,
        y_ground=args.y_ground,
    )

    supported = _is_supported_configuration(
        state=state_relaxed,
        radius=radius,
        xy_limit=args.xy_limit,
        y_ground=args.y_ground,
        eps=support_check_eps,
    )
    settled = 1 if (settle_step < args.max_steps and supported) else 0
    return state_init, state_relaxed, settle_step, settled


def generate_split(
    split_name: str,
    num_samples: int,
    split_seed: int,
    args: argparse.Namespace,
    b2_world_ctor: Any,
    b2_circle_shape_ctor: Any,
    b2_polygon_shape_ctor: Any,
    support_check_eps: float,
) -> dict[str, torch.Tensor]:
    radius = make_radius_batch(
        num_samples=num_samples,
        num_objects=args.num_objects,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        seed=split_seed,
    )

    init_states: list[torch.Tensor] = []
    relaxed_states: list[torch.Tensor] = []
    settle_steps = torch.zeros(num_samples, dtype=torch.int32)
    settled_flags = torch.zeros(num_samples, dtype=torch.int32)
    resample_attempts = torch.zeros(num_samples, dtype=torch.int32)

    log_every = max(1, num_samples // 10)
    for idx in range(num_samples):
        s_init = torch.zeros(args.num_objects, 2, dtype=torch.float32)
        s_relaxed = torch.zeros(args.num_objects, 2, dtype=torch.float32)
        settle_step = args.max_steps
        settled = 0
        used_attempt = 0
        last_error: Exception | None = None

        for attempt in range(args.max_resample_attempts + 1):
            sample_seed = split_seed + idx * 9973 + attempt * 104729
            try:
                s_init, s_relaxed, settle_step, settled = simulate_one_sample(
                    radius=radius[idx],
                    seed=sample_seed,
                    args=args,
                    b2_world_ctor=b2_world_ctor,
                    b2_circle_shape_ctor=b2_circle_shape_ctor,
                    b2_polygon_shape_ctor=b2_polygon_shape_ctor,
                    support_check_eps=support_check_eps,
                )
                last_error = None
            except SpawnPlacementError as exc:
                last_error = exc
                continue
            used_attempt = attempt
            if settled:
                break

        if args.require_settled and not settled:
            extra = f" Last error: {last_error}" if last_error is not None else ""
            raise RuntimeError(
                f"[{split_name}] sample {idx} failed to settle after {args.max_resample_attempts + 1} attempts. "
                "Increase max_steps, loosen stability thresholds, or raise max_resample_attempts."
                f"{extra}"
            )

        init_states.append(s_init)
        relaxed_states.append(s_relaxed)
        settle_steps[idx] = settle_step
        settled_flags[idx] = settled
        resample_attempts[idx] = used_attempt

        if (idx + 1) % log_every == 0 or (idx + 1) == num_samples:
            print(f"[{split_name}] generated {idx + 1}/{num_samples}")

    state_init = torch.stack(init_states, dim=0).contiguous()
    state_relaxed = torch.stack(relaxed_states, dim=0).contiguous()
    settled_ratio = float(settled_flags.float().mean().item())
    mean_steps = float(settle_steps.float().mean().item())
    mean_attempts = float(resample_attempts.float().mean().item())
    print(
        f"[{split_name}] samples={num_samples} settled_ratio={settled_ratio:.3f} "
        f"mean_settle_steps={mean_steps:.1f} mean_resample_attempts={mean_attempts:.2f}"
    )

    return {
        "state_init": state_init,
        "state_relaxed": state_relaxed,
        "radius": radius,
        "settle_steps": settle_steps,
        "settled": settled_flags,
        "resample_attempts": resample_attempts,
    }


def _build_meta(args: argparse.Namespace, support_check_eps: float, box2d_linear_slop: float) -> dict[str, Any]:
    meta = {"generator": "box2d_settle"}
    meta.update({k: getattr(args, k) for k in META_KEYS})
    meta["support_check_eps"] = support_check_eps
    meta["box2d_linear_slop"] = float(box2d_linear_slop)
    return meta


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    b2_world_ctor, b2_circle_shape_ctor, b2_polygon_shape_ctor, box2d_linear_slop = load_box2d()
    support_check_eps = max(float(args.support_check_eps), 2.0 * float(box2d_linear_slop))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_counts = {"train": args.train_samples, "val": args.val_samples, "test": args.test_samples}
    split_offsets = {"train": 0, "val": 1, "test": 2}
    split_map: dict[str, dict[str, torch.Tensor]] = {}
    for split_name in SPLITS:
        split_map[split_name] = generate_split(
            split_name=split_name,
            num_samples=sample_counts[split_name],
            split_seed=args.seed + split_offsets[split_name],
            args=args,
            b2_world_ctor=b2_world_ctor,
            b2_circle_shape_ctor=b2_circle_shape_ctor,
            b2_polygon_shape_ctor=b2_polygon_shape_ctor,
            support_check_eps=support_check_eps,
        )

    payload = {"meta": _build_meta(args, support_check_eps, box2d_linear_slop), **split_map}
    torch.save(payload, output_path)
    print(f"Saved dataset to {output_path}")

    if args.render_dir:
        render_dir = Path(args.render_dir)
        render_split_samples(
            split_data=split_map[args.render_split],
            render_dir=render_dir,
            split_name=args.render_split,
            count=args.render_count,
            xy_limit=args.xy_limit,
            y_ground=args.y_ground,
            image_size=args.render_size,
        )
        print(f"Saved renders to {render_dir}")


if __name__ == "__main__":
    main()
