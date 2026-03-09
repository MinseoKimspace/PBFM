from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset


class RelaxedCirclesDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        use_relaxed_state: bool = True,
        return_init_state: bool = False,
    ) -> None:
        super().__init__()
        dataset_path = Path(dataset_path)
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)

        if split not in payload:
            raise KeyError(f"Split '{split}' not found in dataset. Available keys: {list(payload.keys())}")

        split_data = payload[split]
        state_key = "state_relaxed" if use_relaxed_state else "state_init"
        if state_key not in split_data:
            raise KeyError(f"Key '{state_key}' not found in split '{split}'.")
        if "radius" not in split_data:
            raise KeyError(f"Key 'radius' not found in split '{split}'.")

        self.state = split_data[state_key].float().contiguous()
        self.radius = split_data["radius"].float().contiguous()
        self.state_init = split_data.get("state_init")
        self.return_init_state = return_init_state
        self.meta = payload.get("meta", {})
        self.split = split

        if self.state.dim() != 3 or self.state.size(-1) != 2:
            raise ValueError(f"Expected state shape (S, N, 2), got {tuple(self.state.shape)}")
        if self.radius.dim() != 2:
            raise ValueError(f"Expected radius shape (S, N), got {tuple(self.radius.shape)}")
        if self.state.shape[:2] != self.radius.shape:
            raise ValueError(
                f"State/radius leading shape mismatch: {tuple(self.state.shape)} vs {tuple(self.radius.shape)}"
            )

    def __len__(self) -> int:
        return self.state.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "state": self.state[idx],
            "radius": self.radius[idx],
        }
        if self.return_init_state and self.state_init is not None:
            item["state_init"] = self.state_init[idx]
        return item
