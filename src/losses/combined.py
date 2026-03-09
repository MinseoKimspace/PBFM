from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.losses.fm import fm_loss
from src.losses.physics_energy import PhysicsEnergyLoss

class CombinedLoss(nn.Module):
    def __init__(
        self,
        physics_weight: float = 0.1,
        gravity_weight: float = 0.2,
        ground_weight: float = 0.2,
        collision_weight: float = 0.2,
        y_ground: float = 0.0,
    ) -> None:
        
        super().__init__()
        self.physics_weight = physics_weight
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
        
        fm_term, fm_metrics = fm_loss(model, x1)
        physics_term = self.physics(x1, radius)
        residual = self.physics_weight * physics_term
        total = fm_term + residual

        metrics = {
            "loss": float(total.detach().item()),
            "fm": float(fm_term.detach().item()),
            "physics": float(physics_term.detach().item()),
            "physics_residual": float(residual.detach().item()),
        }
        metrics.update({f"fm_{k}": v for k, v in fm_metrics.items()})
        return total, metrics
