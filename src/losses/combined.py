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
        x1: torch.Tensor, # (B,N,2)
        radius: torch.Tensor, # (B,N)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, x_t, t_now, target_v = sample_linear_path(x1)
        t_start = torch.zeros_like(t_now)
        v_hat = model(x_t, t_start, t_now, radius=radius)
        fm_term = F.mse_loss(v_hat, target_v)
        x1_hat = x_t + (1.0 - t_now).unsqueeze(-1) * v_hat
        physics_term = self.physics(x1_hat, radius)

        # Dynamic balance: keep physics term from being drowned out by FM scale.
        physics_weight_eff = self.physics_weight * (
            fm_term.detach() / (physics_term.detach() + self.physics_weight_eps)
        )
        physics_weight_eff = torch.clamp(
            physics_weight_eff,
            min=self.physics_weight_min,
            max=self.physics_weight_max,
        )
        residual = physics_weight_eff * physics_term
        total = fm_term + residual
        fm_metrics = {
            "loss": float(fm_term.detach().item()),
            "v_mse": float(fm_term.detach().item()),
        }

        metrics = {
            "loss": float(total.detach().item()),
            "fm": float(fm_term.detach().item()),
            "physics": float(physics_term.detach().item()),
            "physics_residual": float(residual.detach().item()),
            "physics_weight_eff": float(physics_weight_eff.detach().item()),
        }
        metrics.update({f"fm_{k}": v for k, v in fm_metrics.items()})
        return total, metrics
