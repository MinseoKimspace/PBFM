from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsEnergyLoss(nn.Module):
    def __init__(self, gravity_weight=0.2, ground_weight=0.2, collision_weight=0.2, y_ground=0.0):
        super().__init__()
        self.gravity_weight = gravity_weight
        self.ground_weight = ground_weight
        self.collision_weight = collision_weight
        self.y_ground = y_ground
        
    def forward(self, pos, radius):
        