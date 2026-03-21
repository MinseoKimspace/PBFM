from __future__ import annotations

import math
from pathlib import Path

import torch

COLOR_CYCLE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def _lighten_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    mix = float(min(1.0, max(0.0, factor)))
    return tuple(int(round(255.0 + (channel - 255.0) * mix)) for channel in color)


def _draw_arrow(
    draw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-3:
        return

    draw.line([start, end], fill=color, width=width)
    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux
    head_len = min(12.0, max(6.0, 0.35 * norm))
    head_half_w = 0.5 * head_len
    left = (end[0] - ux * head_len + px * head_half_w, end[1] - uy * head_len + py * head_half_w)
    right = (end[0] - ux * head_len - px * head_half_w, end[1] - uy * head_len - py * head_half_w)
    draw.polygon([end, left, right], fill=color)


def render_state_image(
    state: torch.Tensor,
    radius: torch.Tensor,
    xy_limit: float,
    y_ground: float,
    image_size: int,
    trajectory: torch.Tensor | None = None,
    vector_field: torch.Tensor | None = None,
    vector_stride: int = 1,
    vector_scale: float = 1.0,
    vector_dt: float = 1.0,
):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    world_x_min = -float(xy_limit)
    world_x_max = float(xy_limit)
    world_y_min = float(y_ground)
    world_y_max = float(y_ground) + float(xy_limit)
    world_w = max(1e-6, world_x_max - world_x_min)
    world_h = max(1e-6, world_y_max - world_y_min)

    # Keep world-to-pixel scaling isotropic; otherwise vertical contacts look separated.
    canvas_w = int(max(16, image_size))
    canvas_h = int(max(16, round(canvas_w * (world_h / world_w))))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    pad = 2.0
    usable_x = max(1.0, (canvas_w - 1) - 2.0 * pad)
    usable_y = max(1.0, (canvas_h - 1) - 2.0 * pad)
    scale = min(usable_x / world_w, usable_y / world_h)
    x_offset = pad + 0.5 * (usable_x - world_w * scale)
    y_offset = pad + 0.5 * (usable_y - world_h * scale)

    def to_px(x: float, y: float) -> tuple[float, float]:
        x_px = x_offset + (x - world_x_min) * scale
        y_px = y_offset + (world_y_max - y) * scale
        return x_px, y_px

    x0, ground_line_y = to_px(world_x_min, world_y_min)
    x1, _ = to_px(world_x_max, world_y_min)
    _, y_top = to_px(world_x_min, world_y_max)
    draw.line([(x0, y_top), (x0, ground_line_y)], fill=(0, 0, 0), width=2)
    draw.line([(x1, y_top), (x1, ground_line_y)], fill=(0, 0, 0), width=2)
    draw.line([(x0, ground_line_y), (x1, ground_line_y)], fill=(0, 0, 0), width=2)

    if trajectory is not None and trajectory.dim() == 3 and trajectory.size(1) == state.size(0):
        path_color_factor = 0.55
        stride = max(1, int(vector_stride))
        for i in range(state.size(0)):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            path_color = _lighten_color(color, path_color_factor)
            points = [to_px(float(trajectory[s, i, 0]), float(trajectory[s, i, 1])) for s in range(trajectory.size(0))]
            if len(points) >= 2:
                draw.line(points, fill=path_color, width=2)

            if vector_field is None:
                continue
            max_steps = min(vector_field.size(0), trajectory.size(0))
            for step_idx in range(0, max_steps, stride):
                start_world = trajectory[step_idx, i]
                delta_world = vector_field[step_idx, i] * float(vector_dt * vector_scale)
                end_world = start_world + delta_world
                start_px = to_px(float(start_world[0]), float(start_world[1]))
                end_px = to_px(float(end_world[0]), float(end_world[1]))
                _draw_arrow(draw, start_px, end_px, color=color, width=2)

    for i in range(state.size(0)):
        cx, cy = to_px(float(state[i, 0]), float(state[i, 1]))
        rr = float(radius[i]) * scale
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color, width=2)
    return canvas


def render_one(
    state: torch.Tensor,
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    y_ground: float,
    image_size: int,
) -> None:
    canvas = render_state_image(
        state=state,
        radius=radius,
        xy_limit=xy_limit,
        y_ground=y_ground,
        image_size=image_size,
    )
    if canvas is None:
        print("PIL not available. Skipping render.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_split_samples(
    split_data: dict[str, torch.Tensor],
    render_dir: Path,
    split_name: str,
    count: int,
    xy_limit: float,
    y_ground: float,
    image_size: int,
) -> None:
    num_total = split_data["state_init"].size(0)
    settled = split_data.get("settled")
    if settled is not None:
        settled_idx = torch.nonzero(settled > 0, as_tuple=False).squeeze(-1)
        indices = settled_idx.tolist()
    else:
        indices = list(range(num_total))

    if not indices:
        print(f"[{split_name}] no settled samples available for rendering.")
        return

    selected = indices[: min(count, len(indices))]
    for idx in selected:
        render_one(
            split_data["state_init"][idx],
            split_data["radius"][idx],
            render_dir / f"{split_name}_{idx:04d}_init.png",
            xy_limit=xy_limit,
            y_ground=y_ground,
            image_size=image_size,
        )
        render_one(
            split_data["state_relaxed"][idx],
            split_data["radius"][idx],
            render_dir / f"{split_name}_{idx:04d}_relaxed.png",
            xy_limit=xy_limit,
            y_ground=y_ground,
            image_size=image_size,
        )
