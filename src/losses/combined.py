from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_energy import PhysicsEnergyLoss
from src.paths.linear import sample_linear_path


def _stats_tensor(values: Sequence[float] | None, default: list[float]) -> torch.Tensor:
    return torch.tensor(default if values is None else list(values), dtype=torch.float32).view(1, 1, -1)


class NormalizedDeltaMixin:
    def _init_delta_stats(
        self,
        delta_mean: Sequence[float] | None,
        delta_std: Sequence[float] | None,
    ) -> None:
        self.register_buffer("delta_mean", _stats_tensor(delta_mean, [0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("delta_std", _stats_tensor(delta_std, [1.0, 1.0, 1.0, 1.0]).clamp_min(1e-6))

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self.delta_mean) / self.delta_std

    def denormalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return delta * self.delta_std + self.delta_mean


class NextStepFlowLoss(nn.Module, NormalizedDeltaMixin):
    def __init__(
        self,
        physics_weight: float = 0.1,
        ground_weight: float = 0.1,
        collision_weight: float = 0.3,
        y_ground: float = 0.0,
        slop: float = 0.005,
        unroll_steps: int = 4,
        position_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
        delta_mean: Sequence[float] | None = None,
        delta_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.physics_weight = physics_weight
        self.unroll_steps = unroll_steps
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        self._init_delta_stats(delta_mean, delta_std)
        self.physics = PhysicsEnergyLoss(
            ground_weight=ground_weight,
            collision_weight=collision_weight,
            y_ground=y_ground,
            slop=slop,
        )

    def noisy_source(self, source: torch.Tensor) -> torch.Tensor:
        if not self.training or (self.position_noise_std <= 0.0 and self.velocity_noise_std <= 0.0):
            return source
        scale = source.new_tensor(
            [self.position_noise_std, self.position_noise_std, self.velocity_noise_std, self.velocity_noise_std]
        ).view(1, 1, 4)
        return source + torch.randn_like(source) * scale

    def forward(
        self,
        model: torch.nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
        radius: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        source_noisy = self.noisy_source(source)
        condition = source_noisy
        _, z_tau, tau, target_v = sample_linear_path(source_noisy, target)
        v_hat = model(z_tau, tau, radius, condition)
        fm_loss = F.mse_loss(v_hat, self.normalize_delta(target_v))

        z = source_noisy
        physics_loss = source.new_tensor(0.0)
        dtau = 1.0 / float(self.unroll_steps)
        for step in range(self.unroll_steps):
            tau_mid = source.new_full((source.size(0), 1), (step + 0.5) * dtau)
            v_step = self.denormalize_delta(model(z, tau_mid, radius, condition))
            z = z + dtau * v_step
            physics_loss = physics_loss + self.physics(z[..., :2], radius).mean()
        physics_loss = physics_loss / float(self.unroll_steps)

        total = fm_loss + self.physics_weight * physics_loss
        metrics = {
            "loss": float(total.detach().item()),
            "fm": float(fm_loss.detach().item()),
            "physics": float(physics_loss.detach().item()),
            "unroll_steps": float(self.unroll_steps),
        }
        return total, metrics


class DeltaLoss(nn.Module, NormalizedDeltaMixin):
    def __init__(
        self,
        delta_mean: Sequence[float] | None = None,
        delta_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self._init_delta_stats(delta_mean, delta_std)

    def forward(
        self,
        model: torch.nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
        radius: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        target_delta = self.normalize_delta(target - source)
        pred_delta = model(source, radius)
        loss = F.mse_loss(pred_delta, target_delta)
        return loss, {"loss": float(loss.detach().item()), "delta": float(loss.detach().item())}
