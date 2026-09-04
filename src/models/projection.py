from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import TimeEmbeddingMLP

POS_DIM = 2
COND_DIM = 6


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(depth):
        layers += [nn.Linear(dim, hidden_dim), nn.SiLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ProjectionMessagePassingCore(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_layers: int,
        message_steps: int,
        contact_margin: float,
        condition_dim: int = COND_DIM,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        self.message_steps = int(message_steps)
        self.contact_margin = float(contact_margin)
        self.condition_dim = int(condition_dim)
        self.time_dim = int(time_dim)

        node_dim = POS_DIM + self.condition_dim + 1 + self.time_dim
        pair_dim = 2 * hidden_dim + 3 * POS_DIM + 3 * self.condition_dim + 6 + self.time_dim
        self.node_proj = make_mlp(node_dim, hidden_dim, hidden_dim, 1)
        self.pair_mlp = make_mlp(pair_dim, hidden_dim, hidden_dim, mlp_layers)
        self.update_mlp = make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(
        self,
        z: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.dim() != 3 or z.size(-1) != POS_DIM:
            raise ValueError(f"Expected z shape (B, N, {POS_DIM}), got {tuple(z.shape)}")
        if condition.shape[:2] != z.shape[:2] or condition.size(-1) != self.condition_dim:
            raise ValueError(
                f"Expected condition shape (B, N, {self.condition_dim}), got {tuple(condition.shape)}"
            )
        if radius.shape != z.shape[:2]:
            raise ValueError(f"Expected radius shape {tuple(z.shape[:2])}, got {tuple(radius.shape)}")

        batch_size, num_objects, _ = z.shape
        if self.time_dim > 0:
            if time_features is None or time_features.shape != (batch_size, num_objects, self.time_dim):
                raise ValueError("time_features has an invalid shape")
            node_features = [z, condition, radius.unsqueeze(-1), time_features]
            time_pair = time_features.unsqueeze(2).expand(batch_size, num_objects, num_objects, -1)
        else:
            node_features = [z, condition, radius.unsqueeze(-1)]
            time_pair = z.new_zeros(batch_size, num_objects, num_objects, 0)

        h = self.node_proj(torch.cat(node_features, dim=-1))

        z_i = z.unsqueeze(2).expand(batch_size, num_objects, num_objects, POS_DIM)
        z_j = z.unsqueeze(1).expand(batch_size, num_objects, num_objects, POS_DIM)
        c_i = condition.unsqueeze(2).expand(batch_size, num_objects, num_objects, self.condition_dim)
        c_j = condition.unsqueeze(1).expand(batch_size, num_objects, num_objects, self.condition_dim)
        r_i = radius.unsqueeze(2).expand(batch_size, num_objects, num_objects).unsqueeze(-1)
        r_j = radius.unsqueeze(1).expand(batch_size, num_objects, num_objects).unsqueeze(-1)

        pos_delta = z_j - z_i
        distance = torch.linalg.norm(pos_delta, dim=-1, keepdim=True).clamp_min(1e-8)
        radius_sum = r_i + r_j
        separation = distance - radius_sum
        penetration = torch.relu(-separation)
        proximity = torch.relu(self.contact_margin - separation)
        gate = torch.exp(-torch.relu(separation).pow(2) / max(self.contact_margin**2, 1e-8))
        eye = torch.eye(num_objects, device=z.device, dtype=torch.bool).view(1, num_objects, num_objects, 1)
        gate = gate.masked_fill(eye, 0.0)

        static_pair = torch.cat(
            [
                z_i,
                z_j,
                z_j - z_i,
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


class ProjectionDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        message_steps: int = 4,
        contact_margin: float = 0.25,
        condition_dim: int = COND_DIM,
    ) -> None:
        super().__init__()
        self.core = ProjectionMessagePassingCore(
            hidden_dim=hidden_dim,
            mlp_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            condition_dim=condition_dim,
        )
        self.head = make_mlp(hidden_dim, hidden_dim, POS_DIM, num_layers)

    def forward(self, proposal: torch.Tensor, radius: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(proposal, radius, condition))


class ProjectionFlowNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 3,
        message_steps: int = 4,
        contact_margin: float = 0.25,
        condition_dim: int = COND_DIM,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbeddingMLP(time_dim)
        self.core = ProjectionMessagePassingCore(
            hidden_dim=hidden_dim,
            mlp_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            condition_dim=condition_dim,
            time_dim=time_dim,
        )
        self.head = make_mlp(hidden_dim, hidden_dim, POS_DIM, num_layers)

    def forward(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if tau.shape != (z.size(0), 1):
            raise ValueError(f"Expected tau shape {(z.size(0), 1)}, got {tuple(tau.shape)}")
        time_features = self.time_embed(tau).unsqueeze(1).expand(z.size(0), z.size(1), -1)
        return self.head(self.core(z, radius, condition, time_features=time_features))


class ProjectionEnergyNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 3,
        message_steps: int = 4,
        contact_margin: float = 0.25,
        condition_dim: int = COND_DIM,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbeddingMLP(time_dim)
        self.core = ProjectionMessagePassingCore(
            hidden_dim=hidden_dim,
            mlp_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            condition_dim=condition_dim,
            time_dim=time_dim,
        )
        self.energy_head = make_mlp(hidden_dim, hidden_dim, 1, num_layers)

    def energy(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if tau.shape != (z.size(0), 1):
            raise ValueError(f"Expected tau shape {(z.size(0), 1)}, got {tuple(tau.shape)}")
        time_features = self.time_embed(tau).unsqueeze(1).expand(z.size(0), z.size(1), -1)
        return self.energy_head(self.core(z, radius, condition, time_features=time_features)).squeeze(-1)

    def forward(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        radius: torch.Tensor,
        condition: torch.Tensor,
        create_graph: bool | None = None,
    ) -> torch.Tensor:
        create_graph = self.training if create_graph is None else create_graph
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            energy = self.energy(z_req, tau, radius, condition).sum()
            grad_z = torch.autograd.grad(
                energy,
                z_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
        return -grad_z if create_graph else -grad_z.detach()


def build_projection_model(
    model_type: str,
    hidden_dim: int = 256,
    time_dim: int = 64,
    num_layers: int = 3,
    message_steps: int = 4,
    contact_margin: float = 0.25,
    condition_dim: int = COND_DIM,
) -> nn.Module:
    if model_type == "delta":
        return ProjectionDeltaNet(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            condition_dim=condition_dim,
        )
    if model_type == "fm":
        return ProjectionFlowNet(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            condition_dim=condition_dim,
        )
    if model_type == "grad_fm":
        return ProjectionEnergyNet(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            message_steps=message_steps,
            contact_margin=contact_margin,
            condition_dim=condition_dim,
        )
    raise ValueError(f"Unknown projection model_type: {model_type}")
