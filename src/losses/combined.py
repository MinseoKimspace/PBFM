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
        gravity_weight: float = 0.2,
        ground_weight: float = 0.2,
        collision_weight: float = 0.2,
        y_ground: float = 0.0,
        physics_weight_min: float = 0.05,
        physics_weight_max: float = 2.0,
        physics_weight_eps: float = 1e-8,
    ) -> None:
        
        super().__init__()
        self.physics_weight = physics_weight
        self.physics_weight_min = physics_weight_min
        self.physics_weight_max = physics_weight_max
        self.physics_weight_eps = physics_weight_eps
        self.physics = PhysicsEnergyLoss(
            gravity_weight=gravity_weight,
            ground_weight=ground_weight,
            collision_weight=collision_weight,
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
        x1_hat = x_t + (1.0 - t_now).unsqueeze(-1) * v_hat
        physics_term = self.physics(x1_hat, radius)
        fm_detached = fm_term.detach()
        physics_detached = physics_term.detach()
        fm_over_physics = fm_detached / (physics_detached + self.physics_weight_eps)

        # Dynamic balance: keep physics term from being drowned out by FM scale.
        physics_weight_eff = self.physics_weight * (
            fm_over_physics
        )
        physics_weight_eff = torch.clamp(
            physics_weight_eff,
            min=self.physics_weight_min,
            max=self.physics_weight_max,
        )
        residual = physics_weight_eff * physics_term
        total = fm_term + residual
        residual_over_fm = residual.detach() / (fm_detached + self.physics_weight_eps)
        fm_metrics = {
            "loss": float(fm_detached.item()),
            "v_mse": float(fm_detached.item()),
        }

        metrics = {
            "loss": float(total.detach().item()),
            "fm": float(fm_detached.item()),
            "physics": float(physics_detached.item()),
            "fm_over_physics": float(fm_over_physics.item()),
            "physics_residual": float(residual.detach().item()),
            "physics_residual_over_fm": float(residual_over_fm.item()),
            "physics_weight_eff": float(physics_weight_eff.detach().item()),
        }
        metrics.update({f"fm_{k}": v for k, v in fm_metrics.items()})
        return total, metrics
