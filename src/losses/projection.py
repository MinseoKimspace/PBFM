from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_energy import PhysicsEnergyLoss


def _stats_tensor(values: Sequence[float] | None, default: list[float]) -> torch.Tensor:
    return torch.tensor(default if values is None else list(values), dtype=torch.float32).view(1, 1, -1)


def free_position(
    state: torch.Tensor,
    time_step: float,
    gravity_y: float,
    linear_damping: float = 0.0,
) -> torch.Tensor:
    pos = state[..., :2]
    vel = state[..., 2:]
    gravity = state.new_tensor([0.0, float(gravity_y)]).view(1, 1, 2)
    vel_free = vel + gravity * float(time_step)
    if abs(float(linear_damping)) > 1e-12:
        vel_free = vel_free / (1.0 + float(linear_damping) * float(time_step))
    return pos + vel_free * float(time_step)


def finite_difference_state(prev_state: torch.Tensor, next_pos: torch.Tensor, time_step: float) -> torch.Tensor:
    prev_pos = prev_state[..., :2]
    next_vel = (next_pos - prev_pos) / float(time_step)
    return torch.cat([next_pos, next_vel], dim=-1)


def make_projection_condition(state: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    return torch.cat([proposal, state[..., :2], state[..., 2:]], dim=-1)


def near_contact_mask(
    pos: torch.Tensor,
    radius: torch.Tensor,
    y_ground: float,
    contact_margin: float,
) -> torch.Tensor:
    ground_near = pos[..., 1] - radius - float(y_ground) <= float(contact_margin)
    pair_delta = pos.unsqueeze(2) - pos.unsqueeze(1)
    pair_dist = torch.linalg.norm(pair_delta, dim=-1)
    pair_radius = radius.unsqueeze(2) + radius.unsqueeze(1)
    pair_near = pair_dist - pair_radius <= float(contact_margin)
    eye = torch.eye(pos.size(1), dtype=torch.bool, device=pos.device).view(1, pos.size(1), pos.size(1))
    pair_near = pair_near.masked_fill(eye, False).any(dim=2)
    return ground_near | pair_near


def projection_node_weights(
    proposal: torch.Tensor,
    target_pos: torch.Tensor,
    radius: torch.Tensor,
    y_ground: float,
    contact_margin: float,
    contact_loss_weight: float,
    correction_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    proposal_contact = near_contact_mask(proposal, radius, y_ground, contact_margin)
    target_contact = near_contact_mask(target_pos, radius, y_ground, contact_margin)
    correction_active = torch.linalg.norm(target_pos - proposal, dim=-1) > float(correction_threshold)
    mask = proposal_contact | target_contact | correction_active
    weight = 1.0 + float(contact_loss_weight) * mask.to(proposal.dtype)
    return weight, mask


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, node_weight: torch.Tensor) -> torch.Tensor:
    per_node = (pred - target).pow(2).mean(dim=-1)
    return (per_node * node_weight).sum() / node_weight.sum().clamp_min(1.0)


class ProjectionNormalizerMixin:
    def _init_correction_stats(
        self,
        correction_mean: Sequence[float] | None,
        correction_std: Sequence[float] | None,
    ) -> None:
        self.register_buffer("correction_mean", _stats_tensor(correction_mean, [0.0, 0.0]))
        self.register_buffer("correction_std", _stats_tensor(correction_std, [1.0, 1.0]).clamp_min(1e-6))

    def normalize_correction(self, correction: torch.Tensor) -> torch.Tensor:
        return (correction - self.correction_mean) / self.correction_std

    def denormalize_correction(self, correction: torch.Tensor) -> torch.Tensor:
        return correction * self.correction_std + self.correction_mean


class ProjectionLoss(nn.Module, ProjectionNormalizerMixin):
    def __init__(
        self,
        model_type: str = "fm",
        time_step: float = 1.0 / 60.0,
        gravity_y: float = -9.8,
        linear_damping: float = 0.0,
        physics_weight: float = 0.1,
        ground_weight: float = 1.0,
        collision_weight: float = 1.0,
        y_ground: float = 0.0,
        slop: float = 0.005,
        contact_margin: float = 0.25,
        contact_loss_weight: float = 4.0,
        correction_threshold: float = 1e-4,
        position_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
        correction_mean: Sequence[float] | None = None,
        correction_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if model_type not in {"delta", "fm", "grad_fm"}:
            raise ValueError(f"Unknown projection model_type: {model_type}")
        self.model_type = model_type
        self.time_step = float(time_step)
        self.gravity_y = float(gravity_y)
        self.linear_damping = float(linear_damping)
        self.physics_weight = float(physics_weight)
        self.y_ground = float(y_ground)
        self.contact_margin = float(contact_margin)
        self.contact_loss_weight = float(contact_loss_weight)
        self.correction_threshold = float(correction_threshold)
        self.position_noise_std = float(position_noise_std)
        self.velocity_noise_std = float(velocity_noise_std)
        self._init_correction_stats(correction_mean, correction_std)
        self.physics = PhysicsEnergyLoss(
            ground_weight=ground_weight,
            collision_weight=collision_weight,
            y_ground=y_ground,
            slop=slop,
        )

    def noisy_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.training or (self.position_noise_std <= 0.0 and self.velocity_noise_std <= 0.0):
            return state
        scale = state.new_tensor(
            [self.position_noise_std, self.position_noise_std, self.velocity_noise_std, self.velocity_noise_std]
        ).view(1, 1, 4)
        return state + torch.randn_like(state) * scale

    def projection_inputs(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_noisy = self.noisy_state(source)
        proposal = free_position(source_noisy, self.time_step, self.gravity_y, self.linear_damping)
        condition = make_projection_condition(source_noisy, proposal)
        target_pos = target[..., :2]
        node_weight, contact_mask = projection_node_weights(
            proposal=proposal,
            target_pos=target_pos,
            radius=radius,
            y_ground=self.y_ground,
            contact_margin=self.contact_margin,
            contact_loss_weight=self.contact_loss_weight,
            correction_threshold=self.correction_threshold,
        )
        return proposal, condition, target_pos, node_weight, contact_mask

    def forward(
        self,
        model: torch.nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
        radius: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        proposal, condition, target_pos, node_weight, contact_mask = self.projection_inputs(source, target, radius)
        target_correction = target_pos - proposal
        target_norm = self.normalize_correction(target_correction)

        if self.model_type == "delta":
            pred_norm = model(proposal, radius, condition)
            main_loss = weighted_mse(pred_norm, target_norm, node_weight)
            endpoint = proposal + self.denormalize_correction(pred_norm)
            metric_name = "delta"
        else:
            tau = torch.rand(source.size(0), 1, device=source.device, dtype=source.dtype)
            tau_state = tau.unsqueeze(-1)
            z_tau = (1.0 - tau_state) * proposal + tau_state * target_pos
            pred_norm = model(z_tau, tau, radius, condition)
            main_loss = weighted_mse(pred_norm, target_norm, node_weight)
            endpoint = z_tau + (1.0 - tau_state) * self.denormalize_correction(pred_norm)
            metric_name = "fm"

        physics_loss = self.physics(endpoint, radius).mean()
        total = main_loss + self.physics_weight * physics_loss
        metrics = {
            "loss": float(total.detach().item()),
            metric_name: float(main_loss.detach().item()),
            "physics": float(physics_loss.detach().item()),
            "contact_fraction": float(contact_mask.float().mean().detach().item()),
            "correction_l2": float(torch.linalg.norm(target_correction, dim=-1).mean().detach().item()),
        }
        return total, metrics
