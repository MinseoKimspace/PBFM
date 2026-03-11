from __future__ import annotations

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


def render_one(
    state: torch.Tensor,
    radius: torch.Tensor,
    output_path: Path,
    xy_limit: float,
    y_ground: float,
    image_size: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("PIL not available. Skipping render.")
        return

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

    for i in range(state.size(0)):
        cx, cy = to_px(float(state[i, 0]), float(state[i, 1]))
        rr = float(radius[i]) * scale
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color, width=2)

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
