from __future__ import annotations

from typing import Tuple

import torch

def linear_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (1-t) * x0 + t * x1


def sample_linear_path(x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype)
    while t.dim() < x1.dim():
        t_state = t.unsqueeze(-1)
    x_t = linear_path(x0, x1, t)
    target_v = x1 - x0
    return x0, x_t, t, target_v
