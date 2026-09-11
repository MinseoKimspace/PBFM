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
    view_bounds: tuple[float, float, float, float] | None = None,
    canvas_size: tuple[int, int] | None = None,
    show_centers: bool = False,
):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    if view_bounds is None:
        world_x_min = -float(xy_limit)
        world_x_max = float(xy_limit)
        world_y_min = float(y_ground)
        world_y_max = float(y_ground) + float(xy_limit)
    else:
        world_x_min, world_x_max, world_y_min, world_y_max = map(float, view_bounds)
        if not (world_x_min < world_x_max and world_y_min < world_y_max):
            raise ValueError("view_bounds must be (x_min, x_max, y_min, y_max) with positive extents")
    world_w = max(1e-6, world_x_max - world_x_min)
    world_h = max(1e-6, world_y_max - world_y_min)

    # Keep world-to-pixel scaling isotropic; otherwise vertical contacts look separated.
    if canvas_size is None:
        canvas_w = int(max(16, image_size))
        canvas_h = int(max(16, round(canvas_w * (world_h / world_w))))
    else:
        canvas_w = int(max(16, canvas_size[0]))
        canvas_h = int(max(16, canvas_size[1]))
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

    if world_x_min <= -float(xy_limit) <= world_x_max:
        x_wall, y_bottom = to_px(-float(xy_limit), world_y_min)
        _, y_top = to_px(-float(xy_limit), world_y_max)
        draw.line([(x_wall, y_top), (x_wall, y_bottom)], fill=(0, 0, 0), width=2)
    if world_x_min <= float(xy_limit) <= world_x_max:
        x_wall, y_bottom = to_px(float(xy_limit), world_y_min)
        _, y_top = to_px(float(xy_limit), world_y_max)
        draw.line([(x_wall, y_top), (x_wall, y_bottom)], fill=(0, 0, 0), width=2)
    if world_y_min <= float(y_ground) <= world_y_max:
        x0, ground_line_y = to_px(world_x_min, float(y_ground))
        x1, _ = to_px(world_x_max, float(y_ground))
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
                sx, sy = points[0]
                draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=path_color)

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
        if show_centers:
            draw.line([(cx - 4, cy), (cx + 4, cy)], fill=color, width=2)
            draw.line([(cx, cy - 4), (cx, cy + 4)], fill=color, width=2)
    return canvas


def render_projection_comparison(
    initial: torch.Tensor,
    final: torch.Tensor,
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    y_ground: float,
    image_size: int,
    trajectory: torch.Tensor | None = None,
    title: str = "projection",
    initial_label: str = "initial",
    final_label: str = "final",
) -> bool:
    """Render a zoomed initial/final solver comparison with unambiguous centers.

    A pale dot marks the trajectory start, the pale line is the accepted solver
    path, and the colored cross is the final circle center.  Both panels share
    one world-to-pixel scale so apparent displacement is directly comparable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print("PIL not available. Skipping render.")
        return False
    if initial.shape != final.shape or initial.ndim != 2 or initial.shape[-1] != 2:
        raise ValueError("initial and final must have matching [N,2] shapes")
    if radius.shape != initial.shape[:1]:
        raise ValueError("radius must have shape [N]")

    samples = [initial, final]
    if trajectory is not None and trajectory.ndim == 3 and trajectory.shape[1:] == initial.shape:
        samples.append(trajectory.flatten(0, 1))
    centers = torch.cat(samples, dim=0).detach().cpu().double()
    radii = radius.detach().cpu().double()
    max_radius = max(float(radii.max()), 1e-3)
    particle_radii = radii.repeat(len(centers) // len(radii))
    x_min = float((centers[:, 0] - particle_radii).min())
    x_max = float((centers[:, 0] + particle_radii).max())
    y_min = float((centers[:, 1] - particle_radii).min())
    y_max = float((centers[:, 1] + particle_radii).max())
    margin = max(0.25, 0.6 * max_radius)
    if y_min <= y_ground + max(0.25, max_radius):
        y_min = min(y_min, float(y_ground))
    bounds = (x_min - margin, x_max + margin, y_min - margin, y_max + margin)
    view_w, view_h = bounds[1] - bounds[0], bounds[3] - bounds[2]

    total_width = int(max(480, image_size))
    panel_width = total_width // 2
    panel_height = int(max(300, min(max(480, image_size), round(panel_width * view_h / view_w))))
    panel_size = (panel_width, panel_height)
    first = render_state_image(initial.detach().cpu(), radii, xy_limit, y_ground, panel_width,
                               view_bounds=bounds, canvas_size=panel_size, show_centers=True)
    second = render_state_image(final.detach().cpu(), radii, xy_limit, y_ground, panel_width,
                                trajectory=None if trajectory is None else trajectory.detach().cpu(),
                                view_bounds=bounds, canvas_size=panel_size, show_centers=True)
    if first is None or second is None:
        return False

    header_h, footer_h = 88, 42
    canvas = Image.new("RGB", (panel_width * 2, panel_height + header_h + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 17)
        label_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = label_font = ImageFont.load_default()
    canvas.paste(first, (0, header_h))
    canvas.paste(second, (panel_width, header_h))
    draw.line([(panel_width, header_h), (panel_width, header_h + panel_height)], fill=(190, 190, 190), width=1)
    draw.text((8, 5), title, fill=(0, 0, 0), font=title_font)
    draw.multiline_text((8, 29), initial_label, fill=(0, 0, 0), font=label_font, spacing=1)
    draw.multiline_text((panel_width + 8, 29), final_label, fill=(0, 0, 0), font=label_font, spacing=1)
    footer_y = header_h + panel_height + 4
    draw.text((8, footer_y), "pale dot/line: solver path start/trajectory",
              fill=(70, 70, 70), font=label_font)
    draw.text((8, footer_y + 17), "colored cross: final center",
              fill=(70, 70, 70), font=label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def render_one(
    state: torch.Tensor,
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    y_ground: float,
    image_size: int,
) -> None:
    canvas = render_state_image(
        state=state[..., :2],
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


def render_state_panel(
    states: list[torch.Tensor],
    labels: list[str],
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    y_ground: float,
    image_size: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print("PIL not available. Skipping render.")
        return

    positions = [state[..., :2] for state in states]
    panels = []
    for idx, pos in enumerate(positions):
        trajectory = None
        if idx > 0:
            trajectory = torch.stack([positions[idx - 1], pos], dim=0)
        panels.append(render_state_image(pos, radius, xy_limit, y_ground, image_size, trajectory=trajectory))

    if any(panel is None for panel in panels):
        print("PIL not available. Skipping render.")
        return

    panel_w, panel_h = panels[0].size
    caption_h = 32
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h + caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for col, (panel, label) in enumerate(zip(panels, labels)):
        x_left = col * panel_w
        canvas.paste(panel, (x_left, caption_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x_left + (panel_w - text_w) * 0.5, 7), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_transition_panel(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    y_ground: float,
    image_size: int,
    labels: tuple[str, str, str] = ("source", "prediction", "target"),
) -> None:
    render_state_panel(
        states=[first, second, target],
        labels=list(labels),
        radius=radius,
        output_path=output_path,
        xy_limit=xy_limit,
        y_ground=y_ground,
        image_size=image_size,
    )


def render_split_samples(
    split_data: dict[str, torch.Tensor],
    render_dir: Path,
    split_name: str,
    count: int,
    xy_limit: float,
    y_ground: float,
    image_size: int,
) -> None:
    num_total = split_data["source"].size(0)
    for idx in range(min(count, num_total)):
        render_state_panel(
            states=[split_data["source"][idx], split_data["target"][idx]],
            labels=["source", "target"],
            radius=split_data["radius"][idx],
            output_path=render_dir / f"{split_name}_{idx:04d}_transition.png",
            xy_limit=xy_limit,
            y_ground=y_ground,
            image_size=image_size,
        )
