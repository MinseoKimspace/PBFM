from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_energy import PhysicsEnergyLoss
from src.paths.linear import sample_linear_path


class NextStepFlowLoss(nn.Module):
    def __init__(
        self,
        physics_weight: float = 0.1,
        ground_weight: float = 0.1,
        collision_weight: float = 0.3,
        y_ground: float = 0.0,
        unroll_steps: int = 4,
    ) -> None:
        super().__init__()
        self.physics_weight = physics_weight
        self.unroll_steps = unroll_steps
        self.physics = PhysicsEnergyLoss(
            ground_weight=ground_weight,
            collision_weight=collision_weight,
            y_ground=y_ground,
        )

    def forward(
        self,
        model: torch.nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
        radius: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, z_tau, tau, target_v = sample_linear_path(source, target)
        v_hat = model(z_tau, tau, radius)
        fm_loss = F.mse_loss(v_hat, target_v)

        z = source
        physics_loss = source.new_tensor(0.0)
        dtau = 1.0 / float(self.unroll_steps)
        for step in range(self.unroll_steps):
            tau_mid = source.new_full((source.size(0), 1), (step + 0.5) * dtau)
            z = z + dtau * model(z, tau_mid, radius)
            physics_loss = physics_loss + self.physics(z[..., :2], radius).mean()
        physics_loss = physics_loss / float(self.unroll_steps)

        total = fm_loss + self.physics_weight * physics_loss
        metrics = {
            "loss": float(total.detach().item()),
            "fm": float(fm_loss.detach().item()),
            "physics": float(physics_loss.detach().item()),
            "unroll_steps": float(self.unroll_steps),
        }
        return total, metrics
