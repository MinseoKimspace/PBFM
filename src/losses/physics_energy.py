from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg

class PhysicsEnergyLoss(nn.Module):
    def __init__(self, gravity_weight=0.1, ground_weight=0.1, collision_weight=0.3, y_ground=0.0, alpha=0.1, epsilon=1e-5, constant=0.01):
        super().__init__()
        self.gravity_weight = gravity_weight
        self.ground_weight = ground_weight
        self.collision_weight = collision_weight
        self.y_ground = y_ground
        self.alpha = alpha
        self.epsilon = epsilon
        self.constant = constant

    def forward(self, pos: torch.Tensor, radius: torch.Tensor): # pos = (B, N, 2), radius = (B, N)
        gravity_loss = torch.mean(pos[:, :, 1])

        ground_loss = torch.mean(F.relu(radius[:,:] - pos[:, :, 1] + self.y_ground))
        
        pairwise_distances = pos[:, :, None, :] - pos[:, None, :, :]
        pairwise_radius = radius[:, :, None] + radius[:, None, :]
        distance = torch.linalg.norm(pairwise_distances, dim=-1) - pairwise_radius
        pair_mask = torch.triu(torch.ones(pos.size(1), pos.size(1), device=pos.device, dtype=torch.bool), diagonal=1,).unsqueeze(0)
        d_hat = self.alpha * pairwise_radius
        
        barrier = torch.zeros_like(distance)
        barrier = torch.where(((distance < d_hat) & (distance >= 0)), -(distance - d_hat).pow(2)*torch.log(distance.clamp_min(self.epsilon) / d_hat.clamp_min(self.epsilon)), barrier)
        barrier = torch.where(distance < 0, (-distance + self.epsilon).pow(2) + self.constant, barrier)
        collision_loss = barrier.masked_select(pair_mask).mean()

        total_loss = (self.gravity_weight * gravity_loss + self.ground_weight * ground_loss + self.collision_weight * collision_loss)
        return total_loss