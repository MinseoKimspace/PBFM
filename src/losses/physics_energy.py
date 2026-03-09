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
        collision_loss = F.softplus((pairwise_radius - torch.linalg.norm(pairwise_distances, dim=-1)).mean())
        total_loss = (self.gravity_weight * gravity_loss + self.ground_weight * ground_loss + self.collision_weight * collision_loss)
        return total_loss