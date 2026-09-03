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
"""Padding utilities for diffusion model training."""

import torch
from tensordict import TensorDict


def embeds_padding_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert padded diffusion sequence fields to jagged nested tensors.

    Masks are expected to be left-aligned (``[1111000...]``). Prompt embeddings
    are always considered; row tensors are discovered through ``*_rows_mask``.

    Args:
        data: TensorDict containing padded sequence tensors and their masks.

    Returns:
        TensorDict with padding stripped from prompt embeddings and masked row
        tensors. Missing prompt masks keep the full embedding sequence intact.
    """

    def _to_nested(values: torch.Tensor, mask: torch.Tensor | None, key: str):
        """Strip sequence padding from a dense tensor and return jagged tensors."""
        if values.ndim != 3:
            raise ValueError(f"{key} must have shape [batch, sequence, width], got {tuple(values.shape)}.")
        if mask is None:
            return (
                torch.nested.as_nested_tensor([values[i] for i in range(values.shape[0])], layout=torch.jagged),
                None,
            )
        if mask.ndim != 2 or mask.shape != values.shape[:2]:
            raise ValueError(
                f"{key}_mask shape {tuple(mask.shape)} does not match {key} batch/sequence shape "
                f"{tuple(values.shape[:2])}."
            )

        values_list, mask_list = [], []
        for i in range(mask.shape[0]):
            curr_mask = mask[i].bool()
            values_list.append(values[i, curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(values_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    padded_keys = {"prompt_embeds", "negative_prompt_embeds"}
    padded_keys.update(
        key.removesuffix("_mask") for key in data.keys() if isinstance(key, str) and key.endswith("_rows_mask")
    )
    for key in padded_keys:
        values = data.get(key, None)
        if not isinstance(values, torch.Tensor) or values.is_nested:
            continue
        mask_key = f"{key}_mask"
        mask = data.get(mask_key, None)
        data[key], data[mask_key] = _to_nested(values, mask, key)

    return data
