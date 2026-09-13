"""Conditional flow matching of multiplier paths ending at converged QP solutions."""
from __future__ import annotations

import torch
from torch import nn

from .problem import gap, matvec, position_error


CHECKPOINT_FORMAT = "multiplier_cfm_v1"


def load_model(path, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("objective") != "cfm":
        raise ValueError("Expected a CFM checkpoint; old B/C flow-map weights cannot be reused")
    for section in ("model", "physics", "dynamics", "reference", "data", "seed"):
        if checkpoint["config"][section] != config[section]:
            raise ValueError(f"Checkpoint {section} differs; use its training configuration")
    model = ConditionalField(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ConditionalField(nn.Module):
    """Instantaneous d(lambda)/d(tau), not an endpoint or finite-time map.

    The condition is the original frozen contact problem. Neither the solution
    nor a separate clean source channel is provided. Signed rates allow release.
    """

    def __init__(self, hidden_dim=64, message_steps=3, length_scale=0.1):
        super().__init__()
        if hidden_dim < 1 or message_steps < 1 or length_scale <= 0:
            raise ValueError("Positive model dimensions/scale required")
        self.length_scale = length_scale
        self.encoder = mlp(5, hidden_dim, hidden_dim)
        self.updates = nn.ModuleList(mlp(2 * hidden_dim, hidden_dim, hidden_dim)
                                     for _ in range(message_steps))
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, lam, tau, problem):
        tau = tau.reshape(-1, 1).expand_as(lam)
        mask = problem["mask"]
        diagonal = problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        scale = self.length_scale / diagonal
        features = torch.stack((lam / scale, problem["c"] / self.length_scale,
                                gap(problem, lam) / self.length_scale,
                                torch.log1p(diagonal), tau), -1)
        embedding = self.encoder(features) * mask[..., None]
        coupling = problem["D"] / (diagonal[:, :, None] * diagonal[:, None, :]).sqrt()
        coupling = coupling / coupling.abs().sum(-1, keepdim=True).clamp_min(1)
        for update in self.updates:
            embedding = (embedding + update(torch.cat((embedding, coupling @ embedding), -1))) * mask[..., None]
        rate = scale * self.head(embedding).squeeze(-1)
        # Exact identity only for disconnected, initially feasible components
        # with zero multiplier. No projected-gradient update is added to u.
        activity = lam.abs() + problem["c"].clamp_max(0).abs()
        live = matvec(problem["reach"].to(lam.dtype), activity) > 0
        return rate * (mask & live)


def cfm_loss(model, problem, source, target, tau):
    """Straight conditional path, first-order parameter gradients only.

    lambda_tau=(1-tau)*source+tau*target, u_target=target-source.
    target is a converged QP label, never a finite-time reference ODE label.
    Endpoint MSE is a diagnostic, not an additional training objective.
    """
    tau = tau.reshape(-1, 1)
    state = torch.lerp(source, target, tau)
    rate = model(state, tau[:, 0], problem)
    scale = model.length_scale / problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
    error = (rate - (target - source)) / scale
    loss = ((error.square() * problem["mask"]).sum(-1)
            / problem["mask"].sum(-1).clamp_min(1)).mean()
    endpoint_mse = position_error(problem, state + (1 - tau) * rate, target).mean()
    return loss, endpoint_mse.detach()
