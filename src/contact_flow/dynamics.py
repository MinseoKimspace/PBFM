"""Shared free-motion and finite-difference updates, independent of the solver."""
from __future__ import annotations

import math
import torch


def free_position(state: torch.Tensor, time_step: float, gravity_y: float,
                  linear_damping: float = 0.0) -> torch.Tensor:
    if (not all(math.isfinite(v) for v in (time_step, gravity_y, linear_damping))
            or time_step <= 0 or linear_damping < 0):
        raise ValueError("time_step must be positive and damping nonnegative")
    if state.ndim != 3 or state.shape[-1] != 4 or not torch.isfinite(state).all():
        raise ValueError("Expected finite state[B,N,4]")
    gravity = state.new_tensor([0.0, gravity_y])
    velocity = (state[..., 2:] + gravity * time_step) / (1.0 + linear_damping * time_step)
    return state[..., :2] + velocity * time_step


def finite_difference_state(prev_state: torch.Tensor, next_pos: torch.Tensor,
                            time_step: float) -> torch.Tensor:
    if not math.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be positive")
    if prev_state.shape[:-1] != next_pos.shape[:-1] or next_pos.shape[-1] != 2:
        raise ValueError("next_pos must match prev_state's batch and particle dimensions")
    return torch.cat([next_pos, (next_pos - prev_state[..., :2]) / time_step], dim=-1)


def make_projection_condition(state: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    return torch.cat([proposal, state[..., :2], state[..., 2:]], dim=-1)
