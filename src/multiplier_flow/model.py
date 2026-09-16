"""Shared contact CFM backbone with analytic or direct velocity outputs."""
from __future__ import annotations

import torch
from torch import nn

from .problem import contact_endpoint, gap, matvec, position_error


CHECKPOINT_FORMAT = "multiplier_contact_cfm_v4"
LEGACY_CHECKPOINT_FORMAT = "multiplier_contact_cfm_v2"
ANALYTIC_CHECKPOINT_FORMAT = "multiplier_contact_cfm_v3"

SOLVER_DESCRIPTION = (
    "Contact-structured CFM: learned cross-contact rates, analytic projected "
    "coordinate endpoint, convex-combination Euler; not sequential PGS"
)
DIRECT_SOLVER_DESCRIPTION = (
    "Direct-velocity contact CFM: signed learned field, projected Euler updates; "
    "raw-field velocity matching, no analytic output endpoint or PGS finish"
)


def load_model(path, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("format") not in (
            CHECKPOINT_FORMAT, ANALYTIC_CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT)
            or checkpoint.get("objective") != "cfm"):
        raise ValueError("Expected contact CFM v2/v3/v4; raw CFM v1 and old B/C weights cannot be reused")
    for section in ("model", "physics", "dynamics", "reference", "data", "seed"):
        saved, requested = checkpoint["config"][section], config[section]
        if section == "model":
            # v2/v3 predate the head selector and always mean analytic. Preserve
            # their parameter names and accept an explicit analytic default.
            saved = {"head_type": "analytic", **saved}
            requested = {"head_type": "analytic", **requested}
            if checkpoint["format"] != CHECKPOINT_FORMAT and saved["head_type"] != "analytic":
                raise ValueError("Legacy v2/v3 checkpoints require an analytic head")
        if saved != requested:
            raise ValueError(f"Checkpoint {section} differs; use its training configuration")
    if checkpoint["format"] == LEGACY_CHECKPOINT_FORMAT:
        settings = config["model"]
        if (settings.get("communication", "local") != "local"
                or settings.get("feature_version", "legacy") != "legacy"):
            raise ValueError("CFM v2 weights require local communication and legacy features")
    model = ConditionalField(**config["model"]).to(device)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint weights do not match their declared model configuration") from error
    model.checkpoint_format = checkpoint["format"]
    return model.eval(), checkpoint


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ComponentAttention(nn.Module):
    """Full attention within each dynamic contact component, without pooling.

    A signed, mass-normalized D entry biases each head's attention score. Zero
    D entries are still visible when reach says the contacts share a component.
    LayerNorm acts only on one token's channels, so disconnected problems and
    padding cannot change another component's feature normalization.
    """

    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.relation_weight = nn.Parameter(torch.ones(heads))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.update = mlp(hidden_dim, 2 * hidden_dim, hidden_dim)

    def forward(self, embedding, relation, reach, mask):
        batch, contacts, hidden = embedding.shape
        qkv = self.qkv(self.attention_norm(embedding))
        qkv = qkv.reshape(batch, contacts, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-1, -2)) / self.head_dim ** 0.5
        scores = scores + self.relation_weight[None, :, None, None] * relation[:, None]
        allowed = reach & mask[:, :, None] & mask[:, None, :]
        # A padded query has no keys. A finite sentinel plus explicit masking
        # produces zero attention (and finite gradients) for those empty rows.
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * allowed[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, contacts, hidden)
        embedding = (embedding + self.output(mixed)) * mask[..., None]
        return (embedding + self.update(self.update_norm(embedding))) * mask[..., None]


class LocalProjection:
    """No-network ablation: the same endpoint formula/clock, no predicted coupling.

    This is NOT native Jacobi iteration: run_cfm interpolates to this endpoint
    with its remaining-time schedule. PGS remains the native solver baseline.
    """

    neural_evaluations = 0
    head_type = "analytic"

    def endpoint(self, lam, tau, problem):
        return contact_endpoint(problem, lam)


class ConditionalField(nn.Module):
    """Identical features/backbone; only the final velocity parameterization varies.

    Analytic: q=lambda+(1-tau)*r; u=(contact_endpoint(q)-lambda)/(1-tau).
    Direct: u=r, with no clipping or endpoint transform inside the field.
    Both rates are signed. Direct states are projected by the integrator, so
    CFM still trains the raw velocity, including negative release directions.
    The fixed original problem is the condition; no solution is an input.
    """

    neural_evaluations = 1

    def __init__(self, hidden_dim=64, message_steps=3, length_scale=0.1,
                 communication="local", attention_heads=4, attention_layers=2,
                 feature_version="legacy", head_type="analytic"):
        super().__init__()
        if hidden_dim < 1 or message_steps < 1 or length_scale <= 0:
            raise ValueError("Positive model dimensions/scale required")
        if communication not in ("local", "global"):
            raise ValueError("communication must be 'local' or 'global'")
        if feature_version not in ("legacy", "residual"):
            raise ValueError("feature_version must be 'legacy' or 'residual'")
        if head_type not in ("analytic", "direct"):
            raise ValueError("head_type must be 'analytic' or 'direct'")
        if communication == "global" and (attention_heads < 1 or attention_layers < 1
                                           or hidden_dim % attention_heads != 0):
            raise ValueError("Global attention requires positive heads/layers and hidden_dim divisible by heads")
        self.length_scale = length_scale
        self.communication = communication
        self.feature_version = feature_version
        self.head_type = head_type
        # Legacy defaults keep the original v2 parameter names and dimensions.
        self.encoder = mlp(5 if feature_version == "legacy" else 6, hidden_dim, hidden_dim)
        self.updates = nn.ModuleList(mlp(2 * hidden_dim, hidden_dim, hidden_dim)
                                     for _ in range(message_steps))
        self.attention = nn.ModuleList(
            ComponentAttention(hidden_dim, attention_heads) for _ in range(
                attention_layers if communication == "global" else 0))
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def solver_description(self):
        return DIRECT_SOLVER_DESCRIPTION if self.head_type == "direct" else SOLVER_DESCRIPTION

    def coupling_rate(self, lam, tau, problem):
        """Shared signed network output, in multiplier units per FM time.

        The legacy method name is retained for analytic checkpoints and exact-
        rate diagnostics. With a direct head this IS the final FM velocity.
        Analytic contact endpoints may still appear in input residual features;
        neither head changes those features or their normalization.
        """
        tau = tau.reshape(-1, 1).expand_as(lam)
        mask = problem["mask"]
        diagonal = problem["D"].diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        scale = self.length_scale / diagonal
        features = [lam / scale, problem["c"] / self.length_scale,
                    gap(problem, lam) / self.length_scale,
                    torch.log1p(diagonal), tau]
        if self.feature_version == "residual":
            # Coordinate projected residual avoids the scene-wide spectral eta:
            # adding an independent stiff component cannot change this feature.
            # Only fixed diagonal scaling is used: a component RMS would leak
            # a global dynamic summary into the local communication ablation.
            features.append((lam - contact_endpoint(problem, lam)) / scale)
        features = torch.stack(features, -1)
        embedding = self.encoder(features) * mask[..., None]
        relation = problem["D"] / (diagonal[:, :, None] * diagonal[:, None, :]).sqrt()
        coupling = relation / relation.abs().sum(-1, keepdim=True).clamp_min(1)
        for update in self.updates:
            embedding = (embedding + update(torch.cat((embedding, coupling @ embedding), -1))) * mask[..., None]
        for attention in self.attention:
            embedding = attention(embedding, relation, problem["reach"], mask)
        rate = scale * self.head(embedding).squeeze(-1)
        # Do not invent coupling for a disconnected, initially feasible component.
        activity = lam.abs() + problem["c"].clamp_max(0).abs()
        live = matvec(problem["reach"].to(lam.dtype), activity) > 0
        return rate * (mask & live)

    def endpoint(self, lam, tau, problem):
        if self.head_type != "analytic":
            raise ValueError("Direct velocity heads have no analytic endpoint; use the field integrator")
        # Predict a RATE so the learned remaining correction vanishes as tau->1.
        # This keeps the straight-path training target representable: if
        # r_theta=target-source and target is KKT, this returns target exactly.
        remaining = 1 - tau.reshape(-1, 1)
        predicted = lam + remaining * self.coupling_rate(lam, tau, problem)
        return contact_endpoint(problem, predicted)

    def forward(self, lam, tau, problem):
        remaining = 1 - tau.reshape(-1, 1)
        if (remaining <= 0).any() or (remaining > 1).any():
            raise ValueError("CFM field requires 0 <= tau < 1; never evaluate at tau=1")
        return self.velocity(lam, tau, problem)

    def velocity(self, lam, tau, problem):
        """Field implementation shared by CFM and the solver-owned valid clock.

        Public forward validates external times. Integrators already construct
        times in [0,1), so they avoid a GPU synchronization for that same check.
        """
        if self.head_type == "direct":
            return self.coupling_rate(lam, tau, problem)
        return (self.endpoint(lam, tau, problem) - lam) / (1 - tau.reshape(-1, 1))


def cfm_loss(model, problem, source, target, tau):
    """Straight conditional path, first-order parameter gradients only.

    lambda_tau=(1-tau)*source+tau*target, u_target=target-source.
    target is a converged QP label, never a finite-time reference ODE label.
    Match the final field returned by forward: analytic endpoint-derived
    velocity or raw direct velocity. The direct integrator's state projection
    is deliberately absent here. Endpoint MSE is diagnostic, not an objective.
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
