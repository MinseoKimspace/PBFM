"""Large circle scenes and near-contact-only QPs for the unchanged D solver.

All pair distances are checked, but only near contacts get Jacobian rows.
The resulting QP is identical to the small-scene builder. A safe spectral
upper bound replaces the eigenvalue solve when choosing the residual step eta.
"""
from __future__ import annotations

import math

import torch


SCENES = ("pile_drop", "pile_impact", "pile_shear")


def circle_gaps(position, radius, physics):
    """Original slot order, O(N^2) distances/storage, no all-pair Jacobian."""
    if (position.ndim != 2 or position.shape[1] != 2 or not len(position)
            or radius.shape != (len(position),) or position.dtype != radius.dtype
            or position.device != radius.device or not position.is_floating_point()):
        raise ValueError("Expected matching floating position[N,2] and radius[N]")
    if not torch.isfinite(position).all() or not torch.isfinite(radius).all() or not (radius > 0).all():
        raise ValueError("Finite positions and positive radii required")
    first, second = torch.triu_indices(len(radius), len(radius), 1, device=position.device)
    difference = position[first] - position[second]
    distance = difference.norm(dim=-1)
    if not (distance > 1e-10).all():
        raise ValueError("Coincident circle centers")
    gaps = torch.cat((distance-radius[first]-radius[second],
                      position[:, 1]-radius-physics.y_ground,
                      position[:, 0]-radius+physics.xy_limit,
                      physics.xy_limit-position[:, 0]-radius))
    return dict(gap=gaps, first=first, second=second,
                difference=difference, distance=distance, near=gaps <= physics.contact_margin)


def coupling_components(matrix):
    """Exact connected components of nonzero D, without cubic transitive closure.

    Sparse nonzero edge indices go to a CPU union-find. Both transfer and work
    are included in QP setup timing. Dense reach is still needed by D attention.
    """
    count = len(matrix)
    parents = list(range(count))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = torch.triu(matrix != 0, diagonal=1).nonzero().cpu().tolist()
    for i, j in edges:
        a, b = root(i), root(j)
        if a != b:
            parents[b] = a
    labels = torch.tensor([root(i) for i in range(count)], device=matrix.device)
    reach = labels[:, None] == labels[None, :]
    return reach, labels


def compact_problem(position, radius, physics, eta_fraction):
    if not 0 < eta_fraction <= 1:
        raise ValueError("eta_fraction must be in (0,1]")
    geo = circle_gaps(position, radius, physics)
    near = geo["near"]
    slots = near.nonzero().flatten()
    count, bodies = len(slots), len(radius)
    # Preserve a masked padding row for a completely contact-free world.
    jacobian = position.new_zeros(max(count, 1), bodies, 2)
    pair_count = len(geo["first"])
    pair_rows = (slots < pair_count).nonzero().flatten()
    pair_slots = slots[pair_rows]
    normal = geo["difference"][pair_slots] / geo["distance"][pair_slots, None]
    jacobian[pair_rows, geo["first"][pair_slots]] = normal
    jacobian[pair_rows, geo["second"][pair_slots]] = -normal
    wall_rows = (slots >= pair_count).nonzero().flatten()
    wall_slots = slots[wall_rows]-pair_count
    wall_kind, wall_body = wall_slots//bodies, wall_slots % bodies
    wall_normal = position.new_tensor([[0., 1.], [1., 0.], [-1., 0.]])
    jacobian[wall_rows, wall_body] = wall_normal[wall_kind]
    jacobian = jacobian.flatten(1)
    inverse_mass = (math.pi*physics.density*radius.square()).reciprocal().repeat_interleave(2)
    if not torch.isfinite(inverse_mass).all() or not (inverse_mass > 0).all():
        raise ValueError("Invalid inverse masses")
    matrix = (jacobian*inverse_mass) @ jacobian.T
    reach, labels = coupling_components(matrix)
    mask = torch.arange(max(count, 1), device=position.device) < count
    reach &= mask[:, None] & mask[None, :]
    # For symmetric PSD D, lambda_max <= max_i sum_j |D_ij|.
    # The analytic D field does not use eta; PGS coordinate updates also do not.
    bound = matrix.abs().sum(-1).max().clamp_min(1e-30) if count else position.new_tensor(1.)
    gap = position.new_zeros(max(count, 1))
    gap[:count] = geo["gap"][slots]+physics.slop
    problem = dict(p=position.flatten()[None], J=jacobian[None], w=inverse_mass[None],
                   c=gap[None], D=matrix[None], mask=mask[None], reach=reach[None],
                   eta=(eta_fraction/bound).reshape(1, 1))
    counts = torch.unique(labels[:count], return_counts=True)[1]
    meta = dict(circles=bodies, contacts=count, components=len(counts),
                largest_component_contacts=int(counts.max()) if len(counts) else 0,
                spectral_upper_bound=float(bound), eta=float(problem["eta"][0, 0]),
                qp_tensor_bytes=sum(t.numel()*t.element_size() for t in problem.values()))
    return problem, geo, meta


def large_scene(kind, count, seed, physics, *, radius=.45, spacing_gap=.002):
    """Connected staggered piles in the original box; exact requested body count.

    The width saturates the existing domain while height grows with N. Equal
    radii keep the checkpoint's physical scale. Random velocities distinguish
    seeds without introducing initial overlaps or changing the box/masses.
    """
    if kind not in SCENES or type(count) is not int or count < 4:
        raise ValueError("Known stress scene and at least four circles required")
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(spacing_gap) or spacing_gap < 0:
        raise ValueError("Positive radius and nonnegative spacing_gap required")
    pitch = 2*radius+spacing_gap
    # Reserve half a pitch for staggered odd rows, plus a small wall clearance.
    available = 2*physics.xy_limit-2*radius-.02
    columns = min(math.ceil(math.sqrt(count)), math.floor(available/pitch+.5))
    if columns < 2:
        raise ValueError("Domain too narrow for a staggered pile")
    index = torch.arange(count)
    row, column = index//columns, index % columns
    x = (column.double() + .5*(row % 2))*pitch
    x -= .5*(float(x.min())+float(x.max()))
    y = radius+physics.y_ground+spacing_gap + row.double()*pitch*math.sqrt(3)/2
    rng = torch.Generator().manual_seed(seed)
    velocity = .1*(torch.rand(count, 2, generator=rng, dtype=torch.float64)-.5)
    if kind == "pile_drop":
        y += .005
        # Upper layers arrive faster: compress the entire pile on frame one,
        # rather than benchmarking a large but almost unconstrained free fall.
        velocity[:, 1] -= 2. + .75*row.double()
    elif kind == "pile_impact":
        first_impact_row = max(1, int(row.max())*3//4)
        falling = row >= first_impact_row
        y[falling] += .03
        velocity[:, 1] -= 1.5 + .65*row.double()
        velocity[falling, 1] -= 3. + .5*(row[falling]-first_impact_row).double()
    else:
        velocity[:, 1] -= 2. + .75*row.double()
        velocity[:, 0] += .8*(2*(row % 2).double()-1)
    state = torch.cat((torch.stack((x, y), -1), velocity), -1).float()
    return state, torch.full((count,), radius)
