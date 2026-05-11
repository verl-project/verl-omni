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
    Convert TensorDict from prompt embeds with padding to no-padding format.
    For diffusion model training only.

    Currently we expect the prompt embedding mask to be [1111000...] format,
    which means the valid tokens are continuous and start from the left.

    Args:
        data: TensorDict with ``prompt_embeds``, ``prompt_embeds_mask``,
            ``negative_prompt_embeds``, ``negative_prompt_embeds_mask``.

    Returns:
        TensorDict where ``prompt_embeds``, ``prompt_embeds_mask``,
        ``negative_prompt_embeds``, and ``negative_prompt_embeds_mask`` have been
        replaced with jagged ``torch.nested`` tensors with padding stripped.
    """

    def _to_nested(embeds: torch.Tensor, mask: torch.Tensor):
        """Strip padding from (bs, seq_len, dim) embeds using the boolean mask and return nested tensors."""
        embeds_list, mask_list = [], []
        for i in range(mask.shape[0]):
            curr_mask = mask[i].bool()
            embeds_list.append(embeds[i, curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    data["prompt_embeds"], data["prompt_embeds_mask"] = _to_nested(data["prompt_embeds"], data["prompt_embeds_mask"])

    if isinstance(data.get("negative_prompt_embeds", None), torch.Tensor):
        data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"] = _to_nested(
            data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"]
        )

    return data
