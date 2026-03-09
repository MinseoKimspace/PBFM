from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class TimeEmbeddingMLP(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)
