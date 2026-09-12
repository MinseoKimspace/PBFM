"""One shared contact-graph map for the endpoint and Lagrangian-matching arms."""
from __future__ import annotations

import torch
from torch import nn
from torch.func import jvp

from .problem import field, gap, matvec, position_error, projected_step


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class MultiplierMap(nn.Module):
    """Contact nodes communicate through normalized Delassus coupling.

    F = exp(-h)*lambda + (1-exp(-h))*nonnegative_candidate.
    Thus F(lambda,0)=lambda and F>=0, while F-lambda can be negative.
    The exact b=T-lambda flow has this same variation-of-constants form.
    A zero-initialized residual head starts at a frozen-T exponential step;
    BOTH experimental arms receive this identical analytic inductive bias.
    """

    def __init__(self, hidden_dim=64, message_steps=3, length_scale=0.1):
        super().__init__()
        if hidden_dim < 1 or message_steps < 1 or length_scale <= 0:
            raise ValueError("Positive model dimensions/scale required")
        self.length_scale = length_scale
        self.encoder = mlp(6, hidden_dim, hidden_dim)
        self.updates = nn.ModuleList(mlp(2 * hidden_dim, hidden_dim, hidden_dim)
                                     for _ in range(message_steps))
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, lam, h, problem):
        h = h.reshape(-1, 1)
        mask = problem["mask"]
        diagonal = problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        multiplier_scale = self.length_scale / diagonal
        g = gap(problem, lam)
        target = projected_step(problem, lam)
        features = torch.stack((lam / multiplier_scale,
                                problem["c"] / self.length_scale,
                                g / self.length_scale,
                                (target - lam) / multiplier_scale,
                                (problem["eta"] * diagonal),
                                torch.log1p(h).expand_as(lam)), -1)
        embedding = self.encoder(features) * mask[..., None]
        norm = (diagonal[:, :, None] * diagonal[:, None, :]).sqrt()
        coupling = problem["D"] / norm
        coupling = coupling / coupling.abs().sum(-1, keepdim=True).clamp_min(1)
        for update in self.updates:
            embedding = (embedding + update(torch.cat((embedding, coupling @ embedding), -1))) * mask[..., None]
        candidate = (target + multiplier_scale * self.head(embedding).squeeze(-1)).clamp_min(0)
        weight = -torch.expm1(-h)
        output = (1 - weight) * lam + weight * candidate
        # Preserve exact fixed components, including nearby but feasible contacts.
        live = matvec(problem["reach"].to(lam.dtype), (target - lam).abs()) > 0
        return torch.where(live & mask, output, lam) * mask


def losses(model, problem, start, duration, target, *, matching=False):
    """C differentiates THROUGH both d_h F and b(F); no target detach.

    Reverse-over-forward AD computes a mixed parameter/time derivative. It is
    not a dense spatial Hessian, but is more expensive than endpoint training.
    """
    if matching:
        prediction, time_derivative = jvp(lambda h: model(start, h, problem),
                                           (duration,), (torch.ones_like(duration),))
        error = time_derivative - field(problem, prediction)
        diagonal = problem["D"].diagonal(dim1=1, dim2=2)
        normalized = error * diagonal / model.length_scale
        map_loss = ((normalized.square() * problem["mask"]).sum(-1)
                    / problem["mask"].sum(-1).clamp_min(1)).mean()
    else:
        prediction = model(start, duration, problem)
        map_loss = prediction.new_zeros(())
    endpoint_loss = position_error(problem, prediction, target).mean() / model.length_scale**2
    return endpoint_loss, map_loss
