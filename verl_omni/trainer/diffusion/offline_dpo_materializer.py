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

"""Validate and attach precomputed offline DPO diffusion training tensors."""

from typing import Any

import numpy as np
import torch
from verl import DataProto


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class OfflineDPOMaterializer:
    """Materializer facade for offline DPO batches with precomputed SD3 tensors."""

    REQUIRED_KEYS = (
        "image_latents",
        "prompt_embeds",
        "prompt_embeds_mask",
        "pooled_prompt_embeds",
    )

    OPTIONAL_KEYS = (
        "negative_prompt_embeds",
        "negative_prompt_embeds_mask",
        "negative_pooled_prompt_embeds",
    )

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _stack_values(value: Any, dtype: torch.dtype) -> torch.Tensor:
        values = _to_list(value)
        tensors = []
        for item in values:
            if isinstance(item, torch.Tensor):
                tensors.append(item.to(dtype=dtype))
            else:
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                tensors.append(torch.tensor(item, dtype=dtype))
        return torch.stack(tensors, dim=0)

    def _collect_precomputed_tensors(self, batch: DataProto) -> dict[str, torch.Tensor]:
        tensor_dict: dict[str, torch.Tensor] = {}
        for key in (*self.REQUIRED_KEYS, *self.OPTIONAL_KEYS):
            if key in batch.batch:
                continue
            if key not in batch.non_tensor_batch:
                if key in self.REQUIRED_KEYS:
                    raise KeyError(
                        f"Offline DPO parquet must provide precomputed `{key}`. "
                        "Regenerate it with examples/dpo_trainer/data_process/prepare_offline_dpo.py."
                    )
                continue
            dtype = torch.int32 if key.endswith("_mask") else torch.float32
            tensor_dict[key] = self._stack_values(batch.non_tensor_batch[key], dtype)
        return tensor_dict

    def materialize(self, batch: DataProto) -> DataProto:
        missing = [key for key in self.REQUIRED_KEYS if key not in batch.batch and key not in batch.non_tensor_batch]
        if missing:
            raise KeyError(
                f"Offline DPO batch is missing precomputed tensors: {missing}. "
                "Regenerate the parquet with examples/dpo_trainer/data_process/prepare_offline_dpo.py."
            )

        tensor_dict = self._collect_precomputed_tensors(batch)
        if not tensor_dict:
            return batch
        return batch.union(DataProto.from_single_dict(tensor_dict))
