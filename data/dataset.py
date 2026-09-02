from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class NextStepDataset(Dataset):
    def __init__(self, dataset_path: str | Path, split: str = "train") -> None:
        payload = torch.load(Path(dataset_path), map_location="cpu", weights_only=True)
        data = payload[split]
        self.source = data["source"].float().contiguous()
        self.target = data["target"].float().contiguous()
        self.radius = data["radius"].float().contiguous()
        self.score = data.get("score", torch.zeros(self.source.size(0))).float().contiguous()
        self.dynamic = data.get("dynamic", torch.ones(self.source.size(0), dtype=torch.bool)).bool().contiguous()
        self.rollout_initial = self._optional_float(data, "rollout_initial")
        self.rollout_target = self._optional_float(data, "rollout_target")
        self.rollout_radius = self._optional_float(data, "rollout_radius")
        self.rollout_length = data.get("rollout_length")
        if self.rollout_length is not None:
            self.rollout_length = self.rollout_length.long().contiguous()
        self.meta = payload.get("meta", {})
        delta = (self.target - self.source).reshape(-1, self.source.size(-1))
        self.delta_mean = torch.as_tensor(self.meta.get("delta_mean", delta.mean(dim=0).tolist()), dtype=torch.float32)
        self.delta_std = torch.as_tensor(self.meta.get("delta_std", delta.std(dim=0).clamp_min(1e-6).tolist()), dtype=torch.float32)

    @staticmethod
    def _optional_float(data: dict, key: str) -> torch.Tensor | None:
        value = data.get(key)
        if value is None:
            return None
        return value.float().contiguous()

    def __len__(self) -> int:
        return self.source.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "source": self.source[idx],
            "target": self.target[idx],
            "radius": self.radius[idx],
            "score": self.score[idx],
            "dynamic": self.dynamic[idx],
        }
