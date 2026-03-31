from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_energy import PhysicsEnergyLoss
from src.paths.linear import sample_linear_path

class CombinedLoss(nn.Module):
    def __init__(
        self,
        physics_weight: float = 0.1,
        gravity_weight: float = 0.1,
        ground_weight: float = 0.1,
        collision_weight: float = 0.3,
        collision_alpha: float = 0.1,
        collision_epsilon: float = 1e-5,
        collision_constant: float = 0.01,
        y_ground: float = 0.0,
    ) -> None:
        
        super().__init__()
        self.physics_weight = physics_weight
        self.physics = PhysicsEnergyLoss(
            gravity_weight=gravity_weight,
            ground_weight=ground_weight,
            collision_weight=collision_weight,
            alpha=collision_alpha,
            epsilon=collision_epsilon,
            constant=collision_constant,
            y_ground=y_ground,
        )

    def forward(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor, # (B,N,2)
        radius: torch.Tensor, # (B,N)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, x_t, t_now, target_v = sample_linear_path(x0, x1)
        t_start = torch.zeros_like(t_now)
        v_hat = model(x_t, t_start, t_now, radius=radius)
        fm_term = F.mse_loss(v_hat, target_v)
        steps = 4
        x = x_t
        t_cur = t_now
        dt = (1.0 - t_now) / float(steps)
        for _ in range(steps):
            t_mid = t_cur + 0.5 * dt
            t_start = torch.zeros_like(t_mid)
            v_hat = model(x, t_start, t_mid, radius=radius)
            x = x + dt.unsqueeze(-1) * v_hat
            t_cur = t_cur + dt
        x1_hat = x
        physics_per_sample = self.physics(x1_hat, radius)
        t_scale = t_now.squeeze(-1)
        scaled_physics = t_scale * physics_per_sample
        physics_term = scaled_physics.mean()

        metric_eps = 1e-8
        fm_detached = fm_term.detach()
        physics_detached = physics_term.detach()
        residual = self.physics_weight * physics_term
        total = fm_term + residual
        residual_over_fm = residual.detach() / (fm_detached + metric_eps)
        fm_metrics = {
            "loss": float(fm_detached.item()),
            "v_mse": float(fm_detached.item()),
        }

        metrics = {
            "loss": float(total.detach().item()),
            "fm": float(fm_detached.item()),
            "physics": float(physics_detached.item()),
            "physics_residual": float(residual.detach().item()),
            "physics_residual_over_fm": float(residual_over_fm.item()),
        }
        metrics.update({f"fm_{k}": v for k, v in fm_metrics.items()})
        return total, metrics
