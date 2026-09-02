from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import TimeEmbeddingMLP

STATE_DIM = 4


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(depth):
        layers += [nn.Linear(dim, hidden_dim), nn.SiLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class FlowVelocityNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbeddingMLP(time_dim)
        self.pair_mlp = make_mlp(2 * 3 * STATE_DIM + 3 + time_dim, hidden_dim, hidden_dim, num_layers)
        self.node_mlp = make_mlp(2 * STATE_DIM + 1 + time_dim + hidden_dim, hidden_dim, STATE_DIM, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != STATE_DIM:
            raise ValueError(f"Expected x shape (B, N, {STATE_DIM}), got {tuple(x.shape)}")
        if condition.shape != x.shape:
            raise ValueError(f"Expected condition shape {tuple(x.shape)}, got {tuple(condition.shape)}")
        if radius.shape != x.shape[:2]:
            raise ValueError(f"Expected radius shape {tuple(x.shape[:2])}, got {tuple(radius.shape)}")
        if tau.shape != (x.size(0), 1):
            raise ValueError(f"Expected tau shape {(x.size(0), 1)}, got {tuple(tau.shape)}")

        batch_size, num_objects, _ = x.shape
        tau_node = self.time_embed(tau).unsqueeze(1).expand(batch_size, num_objects, -1)
        tau_pair = tau_node.unsqueeze(2).expand(batch_size, num_objects, num_objects, -1)

        x_i = x.unsqueeze(2).expand(batch_size, num_objects, num_objects, STATE_DIM)
        x_j = x.unsqueeze(1).expand(batch_size, num_objects, num_objects, STATE_DIM)
        c_i = condition.unsqueeze(2).expand(batch_size, num_objects, num_objects, STATE_DIM)
        c_j = condition.unsqueeze(1).expand(batch_size, num_objects, num_objects, STATE_DIM)
        r_i = radius.unsqueeze(2).expand(batch_size, num_objects, num_objects).unsqueeze(-1)
        r_j = radius.unsqueeze(1).expand(batch_size, num_objects, num_objects).unsqueeze(-1)

        pair_features = torch.cat(
            [x_i, x_j, x_j - x_i, c_i, c_j, c_j - c_i, r_i, r_j, r_i + r_j, tau_pair],
            dim=-1,
        )
        messages = self.pair_mlp(pair_features)
        eye = torch.eye(num_objects, device=x.device, dtype=torch.bool).view(1, num_objects, num_objects, 1)
        messages = messages.masked_fill(eye, 0.0)
        messages = messages.sum(dim=2) / float(max(num_objects - 1, 1))

        node_features = torch.cat([x, condition, radius.unsqueeze(-1), tau_node, messages], dim=-1)
        return self.node_mlp(node_features)
