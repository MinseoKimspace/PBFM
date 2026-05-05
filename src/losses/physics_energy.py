from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsEnergyLoss(nn.Module):
    def __init__(
        self,
        gravity_weight=0.1,
        ground_weight=0.1,
        collision_weight=0.3,
        y_ground=0.0,
        alpha=None,
        epsilon=None,
        constant=None,
    ):
        super().__init__()
        self.gravity_weight = gravity_weight
        self.ground_weight = ground_weight
        self.collision_weight = collision_weight
        self.y_ground = y_ground

    def forward(self, pos: torch.Tensor, radius: torch.Tensor): # pos = (B, N, 2), radius = (B, N)
        y = pos[:, :, 1]
        gravity_loss = torch.mean(y, dim=1)
        ground_loss = torch.mean(F.relu(radius - y + self.y_ground), dim=1)
        
        pairwise_distances = pos[:, :, None, :] - pos[:, None, :, :]
        pairwise_radius = radius[:, :, None] + radius[:, None, :]
        center_distance = torch.linalg.norm(pairwise_distances, dim=-1)
        overlap = pairwise_radius - center_distance
        pair_mask = torch.triu(
            torch.ones(pos.size(1), pos.size(1), device=pos.device, dtype=torch.bool),
            diagonal=1,
        )
        pair_mask_f = pair_mask.unsqueeze(0).to(overlap.dtype)
        collision_penalty = F.relu(overlap).pow(2)
        collision_count = pair_mask.sum().clamp_min(1).to(collision_penalty.dtype)
        collision_loss = (collision_penalty * pair_mask_f).sum(dim=(1, 2)) / collision_count

        total_loss = (self.gravity_weight * gravity_loss + self.ground_weight * ground_loss + self.collision_weight * collision_loss)
        return total_loss
