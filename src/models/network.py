from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import TimeEmbeddingMLP


class FlowVelocityNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        if ffn_mult < 1:
            raise ValueError(f"ffn_mult must be >= 1, got {ffn_mult}")

        self.time_embed = TimeEmbeddingMLP(time_dim)
        in_dim = 2 + 2 + 3 * time_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != 2:
            raise ValueError(f"Expected x shape (B, N, 2), got {tuple(x.shape)}")
        if t_start.dim() != 2 or t_start.size(1) != 1:
            raise ValueError(f"Expected t_start shape (B, 1), got {tuple(t_start.shape)}")
        if t_end.dim() != 2 or t_end.size(1) != 1:
            raise ValueError(f"Expected t_end shape (B, 1), got {tuple(t_end.shape)}")
        if t_start.size(0) != x.size(0) or t_end.size(0) != x.size(0):
            raise ValueError(
                f"Batch dimension mismatch between x/t_start/t_end: {tuple(x.shape)}, {tuple(t_start.shape)}, {tuple(t_end.shape)}"
            )

        batch_size, num_objects, _ = x.shape
        global_mean = x.mean(dim=1, keepdim=True).expand(batch_size, num_objects, 2)

        t_delta = (t_end - t_start).clamp_min(0.0)
        t0_emb = self.time_embed(t_start).unsqueeze(1).expand(batch_size, num_objects, -1)
        t1_emb = self.time_embed(t_end).unsqueeze(1).expand(batch_size, num_objects, -1)
        td_emb = self.time_embed(t_delta).unsqueeze(1).expand(batch_size, num_objects, -1)

        features = torch.cat([x, global_mean, t0_emb, t1_emb, td_emb], dim=-1)
        tokens = self.input_proj(features)
        encoded = self.encoder(tokens)
        return self.head(encoded)
