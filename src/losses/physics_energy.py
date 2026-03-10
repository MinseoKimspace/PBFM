from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg

class PhysicsEnergyLoss(nn.Module):
    def __init__(self, gravity_weight=0.2, ground_weight=0.2, collision_weight=0.2, y_ground=0.0):
        super().__init__()
        self.gravity_weight = gravity_weight
        self.ground_weight = ground_weight
        self.collision_weight = collision_weight
        self.y_ground = y_ground
        
    def forward(self, pos: torch.Tensor, radius: torch.Tensor): # pos = (B, N, 2), radius = (B, N)
        gravity_loss = torch.mean(pos[:, :, 1])
        ground_loss = torch.mean(F.relu(radius[:,:] - pos[:, :, 1] + self.y_ground))
        pairwise_distances = pos[:, :, None, :] - pos[:, None, :, :]
        pairwise_radius = radius[:, :,None] + radius[:, None, :]
        overlap = pairwise_radius - torch.linalg.norm(pairwise_distances, dim=-1)
        collision_matrix = F.relu(overlap).pow(2)
        num_objects = overlap.size(1)
        upper_mask = torch.triu(
            torch.ones(num_objects, num_objects, device=overlap.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).expand_as(collision_matrix)
        collision_values = collision_matrix.masked_select(upper_mask)
        collision_loss = collision_values.mean() if collision_values.numel() > 0 else collision_matrix.new_tensor(0.0)
        total_loss = (self.gravity_weight * gravity_loss + self.ground_weight * ground_loss + self.collision_weight * collision_loss)
        return total_loss