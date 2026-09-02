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


class MessagePassingCore(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_layers: int,
        message_steps: int,
        contact_margin: float,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        self.message_steps = int(message_steps)
        self.contact_margin = float(contact_margin)
        self.time_dim = int(time_dim)

        node_dim = 2 * STATE_DIM + 1 + self.time_dim
        pair_dim = 2 * hidden_dim + 6 * STATE_DIM + 6 + self.time_dim
        self.node_proj = make_mlp(node_dim, hidden_dim, hidden_dim, 1)
        self.pair_mlp = make_mlp(pair_dim, hidden_dim, hidden_dim, mlp_layers)
        self.update_mlp = make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(
        self,
        x: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != STATE_DIM:
            raise ValueError(f"Expected x shape (B, N, {STATE_DIM}), got {tuple(x.shape)}")
        if condition.shape != x.shape:
            raise ValueError(f"Expected condition shape {tuple(x.shape)}, got {tuple(condition.shape)}")
        if radius.shape != x.shape[:2]:
            raise ValueError(f"Expected radius shape {tuple(x.shape[:2])}, got {tuple(radius.shape)}")

        batch_size, num_objects, _ = x.shape
        if self.time_dim > 0:
            if time_features is None or time_features.shape != (batch_size, num_objects, self.time_dim):
                raise ValueError("time_features has an invalid shape")
            node_features = [x, condition, radius.unsqueeze(-1), time_features]
            time_pair = time_features.unsqueeze(2).expand(batch_size, num_objects, num_objects, -1)
        else:
            node_features = [x, condition, radius.unsqueeze(-1)]
            time_pair = x.new_zeros(batch_size, num_objects, num_objects, 0)

        h = self.node_proj(torch.cat(node_features, dim=-1))

        x_i = x.unsqueeze(2).expand(batch_size, num_objects, num_objects, STATE_DIM)
        x_j = x.unsqueeze(1).expand(batch_size, num_objects, num_objects, STATE_DIM)
        c_i = condition.unsqueeze(2).expand(batch_size, num_objects, num_objects, STATE_DIM)
        c_j = condition.unsqueeze(1).expand(batch_size, num_objects, num_objects, STATE_DIM)
        r_i = radius.unsqueeze(2).expand(batch_size, num_objects, num_objects).unsqueeze(-1)
        r_j = radius.unsqueeze(1).expand(batch_size, num_objects, num_objects).unsqueeze(-1)

        pos_delta = x_j[..., :2] - x_i[..., :2]
        distance = torch.linalg.norm(pos_delta, dim=-1, keepdim=True).clamp_min(1e-8)
        radius_sum = r_i + r_j
        separation = distance - radius_sum
        penetration = torch.relu(-separation)
        proximity = torch.relu(self.contact_margin - separation)
        gate = torch.exp(-torch.relu(separation).pow(2) / max(self.contact_margin**2, 1e-8))
        eye = torch.eye(num_objects, device=x.device, dtype=torch.bool).view(1, num_objects, num_objects, 1)
        gate = gate.masked_fill(eye, 0.0)

        static_pair = torch.cat(
            [
                x_i,
                x_j,
                x_j - x_i,
                c_i,
                c_j,
                c_j - c_i,
                r_i,
                r_j,
                radius_sum,
                distance,
                penetration,
                proximity,
                time_pair,
            ],
            dim=-1,
        )

        for _ in range(self.message_steps):
            h_i = h.unsqueeze(2).expand(batch_size, num_objects, num_objects, -1)
            h_j = h.unsqueeze(1).expand(batch_size, num_objects, num_objects, -1)
            messages = self.pair_mlp(torch.cat([h_i, h_j, static_pair], dim=-1)) * gate
            dh = self.update_mlp(torch.cat([h, messages.sum(dim=2)], dim=-1))
            h = h + dh

        return h


class FlowVelocityNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
        message_steps: int = 3,
        contact_margin: float = 0.25,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbeddingMLP(time_dim)
        self.core = MessagePassingCore(
            hidden_dim=hidden_dim,
            mlp_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            time_dim=time_dim,
        )
        self.head = make_mlp(hidden_dim, hidden_dim, STATE_DIM, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if tau.shape != (x.size(0), 1):
            raise ValueError(f"Expected tau shape {(x.size(0), 1)}, got {tuple(tau.shape)}")
        time_features = self.time_embed(tau).unsqueeze(1).expand(x.size(0), x.size(1), -1)
        return self.head(self.core(x, radius, condition, time_features=time_features))


class DeltaVelocityNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        message_steps: int = 3,
        contact_margin: float = 0.25,
    ) -> None:
        super().__init__()
        self.core = MessagePassingCore(
            hidden_dim=hidden_dim,
            mlp_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
        )
        self.head = make_mlp(hidden_dim, hidden_dim, STATE_DIM, num_layers)

    def forward(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(x, radius, condition=x))
