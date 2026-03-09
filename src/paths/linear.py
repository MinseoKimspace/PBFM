from __future__ import annotations

from typing import Tuple

import torch



def sample_linear_path(x1: torch.Tensor):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.size(0), 1, device=x1.device, dtype=x1.dtype)
    t_state = t.unsqueeze(-1)
    x_t = (1.0 - t_state) * x0 + t_state * x1
    target_v = x1 - x0
    return x0, x_t, t, target_v
