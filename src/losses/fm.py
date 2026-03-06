from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.paths.linear import sample_linear_path

def fm_loss(model: torch.nn.Module, x1: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    _, X_t, T_now, target_v = sample_linear_path(x1)
    t_start = torch.zeros_like(T_now)
    v_hat = model(X_t, t_start, T_now)
    loss = F.mse_loss(v_hat, target_v)
    metrics = {"loss": float(loss.detach().item()), "v_mse": float(loss.detach().item())}
    return loss, metrics
