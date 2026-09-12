"""The convex, fixed-normal contact problem. All solver tensors are batched.

min_d 1/2 d^T M d,  subject to c + J d >= 0
min_lambda>=0 Q = 1/2 lambda^T D lambda + c^T lambda
D = J W J^T, z = p + W J^T lambda, W = M^-1.
"""
from __future__ import annotations

import torch

from src.contact_flow.physics import PhysicsConfig, geometry


def make_problem(p, jacobian, inverse_mass, gap, eta_fraction=0.9):
    """Construct one problem; positive masses and nonzero contact rows required."""
    p, inverse_mass, gap = p.flatten(), inverse_mass.flatten(), gap.flatten()
    if (jacobian.shape != (len(gap), len(p)) or inverse_mass.shape != p.shape
            or not 0 < eta_fraction <= 1):
        raise ValueError("Invalid problem shapes or eta_fraction (expected (0,1])")
    tensors = (p, jacobian, inverse_mass, gap)
    if any(t.dtype != p.dtype or t.device != p.device or not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Problem tensors must be finite with one floating dtype/device")
    if not p.is_floating_point() or not (inverse_mass > 0).all():
        raise ValueError("Positive inverse masses and floating tensors required")
    d = (jacobian * inverse_mass) @ jacobian.T
    if len(gap) and not (d.diagonal() > 0).all():
        raise ValueError("Contact rows must have nonzero mass-weighted normals")
    spectral = float(torch.linalg.eigvalsh(d).amax()) if len(gap) else 1.0
    return dict(p=p, J=jacobian, w=inverse_mass, c=gap,
                eta=p.new_tensor(eta_fraction / spectral))


def circle_problem(proposal, radius, physics=None, eta_fraction=0.9):
    """Freeze *near*, not just violated, contacts at the original proposal.

    A separated contact remains in this problem so its multiplier can decrease.
    Slop is part of c. Missing/new contacts and CCD are outside this local solve.
    """
    physics = physics or PhysicsConfig()
    geo = geometry(proposal[None], radius[None], physics)
    mask = geo["near"][0]
    return make_problem(proposal, geo["J"][0, mask],
                        geo["inverse_mass"][0].repeat_interleave(2),
                        geo["gap"][0, mask] + physics.slop, eta_fraction)


def pack(problems):
    """Pad different contact/particle counts; padding never participates in Q."""
    if not problems:
        raise ValueError("At least one problem required")
    count = len(problems)
    contacts = max(1, max(len(p["c"]) for p in problems))
    coords = max(len(p["p"]) for p in problems)
    base = problems[0]["p"]
    result = dict(p=base.new_zeros(count, coords), w=base.new_zeros(count, coords),
                  J=base.new_zeros(count, contacts, coords), c=base.new_zeros(count, contacts),
                  mask=torch.zeros(count, contacts, dtype=torch.bool, device=base.device),
                  eta=base.new_zeros(count, 1))
    for i, problem in enumerate(problems):
        m, n = problem["J"].shape
        result["p"][i, :n], result["w"][i, :n] = problem["p"], problem["w"]
        result["J"][i, :m, :n], result["c"][i, :m] = problem["J"], problem["c"]
        result["mask"][i, :m], result["eta"][i] = True, problem["eta"]
    result["D"] = (result["J"] * result["w"][:, None]) @ result["J"].transpose(1, 2)
    # Exact disconnected components of this *linear* coupling matrix. A solved
    # component should stay fixed even when another component needs correction.
    reach = result["D"].abs() > 0
    for k in range(contacts):
        reach = reach | (reach[:, :, k:k+1] & reach[:, k:k+1, :])
    result["reach"] = reach
    return result


def select(problem, index):
    return {name: value[index] for name, value in problem.items()}


def move(problem, device, dtype=None):
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None)
            for k, v in problem.items()}


def matvec(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def gap(problem, lam):
    return problem["c"] + matvec(problem["D"], lam)


def value(problem, lam):
    return (0.5 * lam * matvec(problem["D"], lam) + problem["c"] * lam).sum(-1)


def projected_step(problem, lam):
    return (lam - problem["eta"] * gap(problem, lam)).clamp_min(0) * problem["mask"]


def field(problem, lam):
    """b=T_eta(lambda)-lambda, NOT a discrete projection step of size one."""
    return projected_step(problem, lam) - lam


def decode(problem, lam):
    return problem["p"] + problem["w"] * matvec(problem["J"].transpose(1, 2), lam)


def position_error(problem, prediction, target):
    difference = decode(problem, prediction) - decode(problem, target)
    mass = torch.where(problem["w"] > 0, problem["w"].clamp_min(1e-30).reciprocal(), 0)
    return (mass * difference.square()).sum(-1) / mass.sum(-1).clamp_min(1e-30)


def residuals(problem, lam):
    g = gap(problem, lam)
    return dict(penetration=(-g).clamp_min(0).amax(-1),
                projected_gradient=(field(problem, lam) / problem["eta"]).abs().amax(-1),
                complementarity=(lam * g).abs().amax(-1),
                negative_multiplier=(-lam).clamp_min(0).amax(-1), Q=value(problem, lam))


def converged(problem, lam, tolerance):
    r = residuals(problem, lam)
    return ((r["projected_gradient"] <= tolerance) & (r["penetration"] <= tolerance)
            & (r["negative_multiplier"] <= tolerance))


def relinearize(anchor, position, jacobian, inverse_mass, gaps_at_position,
                old_keys, old_lambda, new_keys, eta_fraction=0.9):
    """Bookkeeping prototype, NOT a nonlinear solver or a consistent dual map.

    Keep primal position explicitly. Warm-start retained multipliers by contact
    ID; expose the stationarity mismatch instead of silently decoding a jump.
    The next actual primal-dual solve must resolve this mismatch. Original p is
    never reset. A removed contact's displacement remains in primal history.
    """
    if len(set(old_keys)) != len(old_keys) or len(set(new_keys)) != len(new_keys):
        raise ValueError("Contact IDs must be unique")
    if len(old_keys) != len(old_lambda) or len(new_keys) != len(gaps_at_position):
        raise ValueError("Contact IDs and multiplier/gap lengths differ")
    c = gaps_at_position + jacobian @ (anchor.flatten() - position.flatten())
    problem = make_problem(anchor, jacobian, inverse_mass, c, eta_fraction)
    old = dict(zip(old_keys, old_lambda))
    guess = anchor.new_zeros(len(new_keys))
    for i, key in enumerate(new_keys):
        if key in old:
            guess[i] = old[key]
    mismatch = position.flatten() - anchor.flatten() - inverse_mass.flatten() * (jacobian.T @ guess)
    return dict(problem=problem, position=position.clone(), multiplier_guess=guess,
                stationarity_mismatch=mismatch)
