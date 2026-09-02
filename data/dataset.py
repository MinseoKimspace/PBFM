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
        self.meta = payload.get("meta", {})

    def __len__(self) -> int:
        return self.source.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "source": self.source[idx],
            "target": self.target[idx],
            "radius": self.radius[idx],
        }
