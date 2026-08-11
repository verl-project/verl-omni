# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic BAGEL SFT dataset used only by the special e2e smoke test."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class BagelSFTSmokeDataset(Dataset):
    """Return tiny text+image-supervision batches for actor-only SFT."""

    def __init__(
        self,
        data_files=None,
        tokenizer=None,
        processor=None,
        config=None,
        is_train: bool = True,
        max_samples: int = -1,
        **kwargs,
    ) -> None:
        del data_files, tokenizer, processor, config, is_train, kwargs
        self.num_samples = 4 if max_samples is None or max_samples < 0 else max(1, max_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        input_ids = torch.tensor([1, 2, 1, 2, 2, 0], dtype=torch.long)
        labels = torch.tensor([-100, -100, 1, 2, 2, -100], dtype=torch.long)
        attention_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.long)

        latent_len = 4
        patch_dim = 8
        image_hidden_states = torch.linspace(-1.0, 1.0, latent_len * patch_dim, dtype=torch.float32).reshape(
            latent_len, patch_dim
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_hidden_states": image_hidden_states,
            "timesteps": torch.tensor(0.25, dtype=torch.float32),
            "latent_pos_ids": torch.arange(latent_len, dtype=torch.long),
            "image_velocity_target": torch.zeros(latent_len, patch_dim, dtype=torch.float32),
            "image_loss_mask": torch.ones(latent_len, dtype=torch.bool),
        }


def bagel_sft_smoke_collate_fn(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}
