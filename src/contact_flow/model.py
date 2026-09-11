"""FM velocity in contact space; the network predicts a bounded PSD mobility."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.time_embedding import TimeEmbeddingMLP
from .physics import PhysicsConfig, FlowConfig, geometry, contact_velocity


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ContactFlowNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, time_dim: int = 32,
                 message_steps: int = 3, rank: int = 4,
                 flow: FlowConfig | None = None):
        super().__init__()
        if min(hidden_dim, message_steps) < 1 or time_dim < 2 or rank < 0:
            raise ValueError("Invalid contact network dimensions; rank must be nonnegative")
        self.rank, self.message_steps = rank, message_steps
        self.flow = flow or FlowConfig()
        self.time_embedding = TimeEmbeddingMLP(time_dim)
        self.node = mlp(12 + time_dim, hidden_dim, hidden_dim)
        self.message = mlp(2 * hidden_dim + 6, hidden_dim, hidden_dim)
        self.update = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.head = mlp(2 * hidden_dim + 6, hidden_dim, 1 + rank)

    def mobility(self, z, tau, radius, condition, geo, physics):
        batch, nodes, _ = z.shape
        tau = tau.reshape(batch, 1)
        if condition.shape != (batch, nodes, 6):
            raise ValueError("condition must be fixed physical context [B,N,6]")
        time = self.time_embedding(tau).to(z.dtype)[:, None].expand(-1, nodes, -1)
        features = torch.cat([z, condition, radius[..., None],
                              geo["inverse_mass"][..., None],
                              geo["gradient"].reshape_as(z), time], -1)
        hidden = self.node(features)
        i, j = geo["edge_i"], geo["edge_j"]
        other = j.clamp_min(0)
        internal = (j >= 0)[None, :, None]
        mask = (geo["gap"] <= physics.contact_margin)[..., None]
        jac = geo["J"].reshape(batch, len(i), nodes, 2)
        normal = jac[:, torch.arange(len(i), device=z.device), i]
        diagonal_d = (jac.square().sum(-1) * geo["inverse_mass"][:, None]).sum(-1)
        static = torch.cat([normal, geo["gap"][..., None], geo["residual"][..., None],
                            diagonal_d[..., None], (~internal).to(z.dtype).expand(batch, -1, -1)], -1)
        reverse_static = torch.cat([-normal, static[..., 2:]], -1)

        def edges(h, reverse=False):
            # A message is expressed from its recipient's point of view. Pair
            # ordering i<j is storage only and must not change particle physics.
            if reverse:
                return torch.cat([h[:, other], h[:, i] * internal, reverse_static], -1)
            return torch.cat([h[:, i], h[:, other] * internal, static], -1)

        for _ in range(self.message_steps):
            message = self.message(edges(hidden)) * mask
            reverse_message = self.message(edges(hidden, reverse=True)) * mask * internal
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(1, i, message)
            aggregate.index_add_(1, other, reverse_message)
            hidden = hidden + self.update(torch.cat([hidden, aggregate], -1))
        output = self.head(edges(hidden))
        reverse_output = self.head(edges(hidden, reverse=True))
        output = torch.where(internal, 0.5 * (output + reverse_output), output)
        diagonal = (F.softplus(output[..., 0]) + 1e-4) * mask.squeeze(-1)
        low_rank = output[..., 1:] * mask
        # These are dimensionless C controls. Both candidates and this model
        # use contact_velocity for the SAME component cap, P scaling and gain.
        return diagonal, low_rank

    def forward(self, z, tau, radius, condition, physics: PhysicsConfig):
        geo = geometry(z, radius, physics)
        diagonal, low_rank = self.mobility(z, tau, radius, condition, geo, physics)
        valid = torch.isfinite(diagonal).all(1) & torch.isfinite(low_rank).all((1, 2))
        velocity = contact_velocity(geo, torch.where(valid[:, None], diagonal, 0.0),
                                    torch.where(valid[:, None, None], low_rank, 0.0), self.flow)
        # Let the integrator report per-world neural overflow, not crash the batch
        # inside the otherwise strict PSD-controls validator.
        return torch.where(valid[:, None, None], velocity, float("nan"))
