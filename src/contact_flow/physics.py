"""Analytic circle contacts. E is overlap energy, not an inertial potential."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PhysicsConfig:
    y_ground: float = 0.0
    xy_limit: float = 10.0
    slop: float = 0.005
    contact_margin: float = 0.25
    density: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Physics parameters must be finite")
        if min(self.xy_limit, self.density) <= 0 or min(self.slop, self.contact_margin) < 0:
            raise ValueError("Positive extent/density and nonnegative slop/margin required")


@dataclass(frozen=True)
class FlowConfig:
    """C is dimensionless when normalization='diagonal'; gain sets tau speed.

    The spectral bound applies to C, BEFORE B=P C P. It does not bound the
    dimensional B again. epsilon is relative to each positive contact diagonal,
    so changing all masses by a common factor cancels analytically for fixed C.
    """
    gain: float = 1.0
    normalization: str = "diagonal"
    mobility_bound: float = 16.0
    normalization_eps: float = 1e-4

    def __post_init__(self):
        if self.normalization not in {"diagonal", "none"}:
            raise ValueError("normalization must be diagonal or none")
        values = (self.gain, self.mobility_bound, self.normalization_eps)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Flow parameters must be finite")
        if min(self.gain, self.mobility_bound) <= 0 or self.normalization_eps < 0:
            raise ValueError("Positive gain/bound and nonnegative relative epsilon required")


def valid_positions(z):
    """Per-world mask; coincident centers have no defined circle normal."""
    i, j = torch.triu_indices(z.shape[1], z.shape[1], 1, device=z.device)
    distance = torch.linalg.vector_norm(z[:, i] - z[:, j], dim=-1)
    return torch.isfinite(z).all(dim=(1, 2)) & (distance > 1e-10).all(dim=1)


def validate(z, radius):
    if z.ndim != 3 or z.shape[-1] != 2 or not z.shape[0] or not z.shape[1] or radius.shape != z.shape[:2]:
        raise ValueError("Expected nonempty z[B,N,2] and radius[B,N]")
    if not z.is_floating_point() or z.dtype != radius.dtype or z.device != radius.device:
        raise ValueError("Positions and radii must share a floating dtype and device")
    if not torch.isfinite(radius).all() or not (radius > 0).all():
        raise ValueError("Radii must be finite and positive")
    if not valid_positions(z).all():
        raise ValueError("Nonfinite positions or coincident/degenerate circle centers")


def _gaps(z, radius, cfg):
    i, j = torch.triu_indices(z.shape[1], z.shape[1], 1, device=z.device)
    difference = z[:, i] - z[:, j]
    distance = torch.linalg.vector_norm(difference, dim=-1)
    gap = torch.cat((distance - radius[:, i] - radius[:, j],
                     z[..., 1] - radius - cfg.y_ground,
                     z[..., 0] - radius + cfg.xy_limit,
                     cfg.xy_limit - z[..., 0] - radius), dim=1)
    return gap, difference, distance, i, j


def energy(z, radius, cfg):
    """Cheap candidate energy; callers validate positions before differentiating."""
    gap = _gaps(z, radius, cfg)[0]
    return 0.5 * torch.minimum(gap + cfg.slop, torch.zeros_like(gap)).square().sum(dim=1)


def energy_and_violation(z, radius, cfg):
    """One gap evaluation supplies both line-search energy and stopping residual."""
    residual = (_gaps(z, radius, cfg)[0] + cfg.slop).clamp_max(0.0)
    return 0.5 * residual.square().sum(dim=1), (-residual).amax(dim=1)


def geometry(z, radius, cfg):
    """Fixed slots: upper-triangle pairs, then ground/left/right for every body.

    J contains geometric normals; G=W J^T additionally masks far contacts.
    Components connect bodies through near *pairs*, never through a shared wall.
    """
    validate(z, radius)
    batch, nodes, _ = z.shape
    gap, difference, distance, i, j = _gaps(z, radius, cfg)
    pairs, contacts = len(i), gap.shape[1]
    mass = math.pi * cfg.density * radius.square()
    if not torch.isfinite(mass).all() or not (mass > 0).all():
        raise ValueError("Radii/density produce invalid floating-point masses")
    inverse_mass = mass.reciprocal()
    if not torch.isfinite(inverse_mass).all():
        raise ValueError("Radii produce nonfinite inverse masses")
    residual = (gap + cfg.slop).clamp_max(0.0)
    near = gap <= cfg.contact_margin
    jacobian = z.new_zeros(batch, contacts, nodes, 2)
    slot = torch.arange(pairs, device=z.device)
    normal = difference / distance.unsqueeze(-1)
    jacobian[:, slot, i] = normal
    jacobian[:, slot, j] = -normal
    body = torch.arange(nodes, device=z.device)
    jacobian[:, pairs + body, body, 1] = 1.0
    jacobian[:, pairs + nodes + body, body, 0] = 1.0
    jacobian[:, pairs + 2 * nodes + body, body, 0] = -1.0
    jacobian = jacobian.flatten(2)
    gradient = torch.bmm(jacobian.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
    mobility_basis = (jacobian.transpose(1, 2) * inverse_mass.repeat_interleave(2, dim=1).unsqueeze(-1)
                      * near.unsqueeze(1))
    adjacency = torch.eye(nodes, dtype=torch.bool, device=z.device).expand(batch, nodes, nodes).clone()
    adjacency[:, i, j] = near[:, :pairs]
    adjacency[:, j, i] = near[:, :pairs]
    labels = body.expand(batch, nodes)
    for _ in range(nodes - 1):
        labels = torch.where(adjacency, labels.unsqueeze(1), nodes).amin(dim=2)
    edge_i = torch.cat((i, body.repeat(3)))
    edge_j = torch.cat((j, torch.full((3 * nodes,), -1, dtype=torch.long, device=z.device)))
    component = labels[:, edge_i].masked_fill(~near, -1)
    return dict(gap=gap, residual=residual, energy=0.5 * residual.square().sum(dim=1),
                gradient=gradient, J=jacobian, G=mobility_basis, inverse_mass=inverse_mass,
                max_violation=(-residual).amax(dim=1), penetration=(-gap).clamp_min(0.0),
                edge_i=edge_i, edge_j=edge_j, component=component, near=near)


def apply_mobility(geo, diagonal, low_rank):
    """u=-G B G^T grad E; B=diag(d)+LL^T independently in each component."""
    near, basis = geo["near"], geo["G"]
    if diagonal.shape != near.shape or low_rank.ndim != 3 or low_rank.shape[:2] != near.shape:
        raise ValueError("Expected diagonal[B,M] and low_rank[B,M,R]")
    if not torch.isfinite(diagonal).all() or not (diagonal >= 0).all() or not torch.isfinite(low_rank).all():
        raise ValueError("Mobility requires finite nonnegative diagonal and finite factors")
    h = torch.bmm(basis.transpose(1, 2), geo["gradient"].unsqueeze(-1)).squeeze(-1)
    weighted = diagonal * near * h
    if low_rank.shape[-1]:
        factor = low_rank.to(dtype=h.dtype) * near.unsqueeze(-1)
        index = geo["component"].clamp_min(0).unsqueeze(-1).expand_as(factor)
        pooled = factor.new_zeros(factor.shape[0], basis.shape[1] // 2, factor.shape[-1])
        pooled = pooled.scatter_add(1, index, factor * h.unsqueeze(-1))
        weighted = weighted + (factor * pooled.gather(1, index)).sum(dim=-1)
    return -torch.bmm(basis, weighted.unsqueeze(-1)).reshape(basis.shape[0], -1, 2)


def cap_mobility(geo, diagonal, low_rank, bound):
    """Shared candidate/model cap of diag(d)+LL^T per near-pair component.

    max(d)+||L||_F^2 bounds the largest eigenvalue. Far slots and completely
    separate components do not consume another component's mobility budget.
    """
    near = geo["near"]
    if diagonal.shape != near.shape or low_rank.ndim != 3 or low_rank.shape[:2] != near.shape:
        raise ValueError("Expected diagonal[B,M] and low_rank[B,M,R]")
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("mobility bound must be positive and finite")
    if not torch.isfinite(diagonal).all() or not (diagonal >= 0).all() or not torch.isfinite(low_rank).all():
        raise ValueError("Mobility requires finite nonnegative diagonal and finite factors")
    diagonal = diagonal * near
    low_rank = low_rank * near.unsqueeze(-1)
    index = geo["component"].clamp_min(0)
    nodes = geo["inverse_mass"].shape[1]
    maximum = diagonal.new_zeros(diagonal.shape[0], nodes).scatter_reduce(
        1, index, diagonal, reduce="amax", include_self=True)
    squared = diagonal.new_zeros(diagonal.shape[0], nodes).scatter_add(
        1, index, low_rank.square().sum(dim=-1))
    scale = ((maximum + squared).gather(1, index) / bound).clamp_min(1.0)
    return diagonal / scale, low_rank / scale.sqrt().unsqueeze(-1)


def mobility_diagonal_scale(geo, flow):
    """P_cc=1/((1+relative_epsilon) A_cc), or identity on near slots.

    A=J W J^T. No absolute epsilon/floor is used: a common rescaling of mass
    must not change the normalized field at fixed controls. Unrepresentable
    scales fail explicitly instead of silently changing that physical scaling.
    """
    near = geo["near"]
    if flow.normalization == "none":
        return near.to(geo["G"].dtype)
    diagonal = (geo["J"] * geo["G"].transpose(1, 2)).sum(dim=-1)
    if not torch.isfinite(diagonal).all() or not (diagonal[near] > 0).all():
        raise ValueError("Near-contact mass diagonal must be positive and finite")
    safe = torch.where(near, diagonal, torch.ones_like(diagonal))
    scale = safe.reciprocal() / (1.0 + flow.normalization_eps)
    scale = scale * near
    if not torch.isfinite(scale).all() or not (scale[near] > 0).all():
        raise ValueError("Unrepresentable contact normalization scale")
    return scale


def contact_velocity(geo, diagonal, low_rank, flow):
    """u=-gain G P C P G^T grad(E), with the SAME cap for data and model."""
    diagonal, low_rank = cap_mobility(geo, diagonal, low_rank, flow.mobility_bound)
    scale = mobility_diagonal_scale(geo, flow)
    return flow.gain * apply_mobility(
        geo, diagonal * scale.square(), low_rank * scale.unsqueeze(-1))


def make_baseline(method, radius, cfg, mobility_bound=None, flow=None):
    """Numerical fields, with identical physical energy and current contact graph.

    'inverse' is regularized inverse-square mobility, NOT exact Newton/PBD.
    'diagonal' is diagonal inverse-square mobility, NOT Jacobi.
    With flow, isotropic/inverse use its normalized C coordinates and cap.
    Classical gradient/diagonal/jacobi retain their named formulas and receive
    only its scalar gain; normalizing them a second time changes the baseline.
    flow=None preserves the original raw-B legacy behavior.
    """
    if method not in {"gradient", "isotropic", "diagonal", "inverse", "jacobi"}:
        raise ValueError(f"Unknown contact baseline: {method}")
    if mobility_bound is not None and (not math.isfinite(mobility_bound) or mobility_bound <= 0):
        raise ValueError("mobility_bound must be positive and finite")

    def field(z, tau):
        del tau
        geo = geometry(z, radius, cfg)
        gradient, basis = geo["gradient"], geo["G"]
        gain = 1.0 if flow is None else flow.gain
        if method == "gradient":
            return -gain * (gradient * geo["inverse_mass"].repeat_interleave(2, dim=1)).reshape_as(z)
        if method == "isotropic":
            diagonal = torch.ones_like(geo["gap"])
            factor = z.new_zeros(*diagonal.shape, 0)
            if flow is not None:
                return contact_velocity(geo, diagonal, factor, flow)
            if mobility_bound is not None:
                diagonal = diagonal.clamp_max(mobility_bound)
            return apply_mobility(geo, diagonal, factor)
        gram = torch.bmm(geo["J"], basis) * geo["near"].unsqueeze(-1)
        gram = 0.5 * (gram + gram.transpose(1, 2))
        diagonal = gram.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        if method == "jacobi":
            return -gain * torch.bmm(basis, (geo["residual"] / diagonal).unsqueeze(-1)).reshape_as(z)
        if method == "diagonal":
            weights = diagonal.reciprocal().square()
            if mobility_bound is not None:
                weights = weights.clamp_max(mobility_bound)
            return gain * apply_mobility(geo, weights * geo["near"],
                                         z.new_zeros(z.shape[0], diagonal.shape[1], 0))
        if flow is not None:
            # C*=(P A^2 P)^-1=P^-1 A^-2 P^-1 in the nonsingular,
            # zero-regularization case. Spectral clipping is in C coordinates,
            # not in the dimensional B. Masked contacts are identity dummies.
            p = mobility_diagonal_scale(geo, flow)
            normalized = gram * p.unsqueeze(1)
            normal = normalized.transpose(1, 2) @ normalized
            normal = 0.5 * (normal + normal.transpose(1, 2))
            index = geo["component"].clamp_min(0)
            component_scale = z.new_zeros(z.shape[0], z.shape[1]).scatter_reduce(
                1, index, normal.diagonal(dim1=1, dim2=2), reduce="amax", include_self=True)
            ridge = torch.where(geo["near"], 1e-8 * component_scale.gather(1, index), 1.0)
            values, vectors = torch.linalg.eigh(normal + torch.diag_embed(ridge))
            # Saturate before reciprocal, avoiding overflow in singular modes.
            weights = values.clamp_min(1.0 / flow.mobility_bound).reciprocal()
            h = torch.bmm(basis.transpose(1, 2), gradient.unsqueeze(-1)) * p.unsqueeze(-1)
            coefficients = vectors @ (weights.unsqueeze(-1) * (vectors.transpose(1, 2) @ h))
            return -gain * torch.bmm(basis, p.unsqueeze(-1) * coefficients).reshape_as(z)
        # Masked rows are identity dummies. No inverse is explicitly formed.
        scale = gram.diagonal(dim1=1, dim2=2).amax(dim=1).clamp_min(1e-8)
        ridge = torch.where(geo["near"], (1e-4 * scale).unsqueeze(-1), 1.0)
        regularized = gram + torch.diag_embed(ridge)
        h = torch.bmm(basis.transpose(1, 2), gradient.unsqueeze(-1))
        if mobility_bound is None:
            coefficients = torch.linalg.solve(regularized, torch.linalg.solve(regularized, h))
        else:
            # Clipping preserves useful range-space modes; global scaling would
            # be dominated by regularized nullspace modes that G annihilates.
            values, vectors = torch.linalg.eigh(regularized)
            weights = values.clamp_min(torch.finfo(z.dtype).tiny).reciprocal().square().clamp_max(mobility_bound)
            coefficients = vectors @ (weights.unsqueeze(-1) * (vectors.transpose(1, 2) @ h))
        return -torch.bmm(basis, coefficients).reshape_as(z)

    return field
