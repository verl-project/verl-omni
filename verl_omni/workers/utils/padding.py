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

    def _coerce(item):
        if isinstance(item, torch.Tensor):
            return item
        return torch.as_tensor(item)

    def _to_nested(embeds, mask):
        """Strip padding from (bs, seq_len, dim) embeds using the boolean mask and return nested tensors.

        Both ``embeds`` and ``mask`` may arrive as torch tensors (uniform padding)
        or as list-like containers of per-sample arrays/tensors (variable
        padding, e.g. when stored in DataProto.non_tensor_batch and surfaced as
        a ``tensordict.utils.LinkedList`` after ``to_tensordict()``).
        """
        bs = mask.shape[0] if isinstance(mask, torch.Tensor) else len(mask)
        embeds_list, mask_list = [], []
        for i in range(bs):
            curr_mask = _coerce(mask[i]).bool()
            curr_embeds = _coerce(embeds[i])
            embeds_list.append(curr_embeds[curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    data["prompt_embeds"], data["prompt_embeds_mask"] = _to_nested(data["prompt_embeds"], data["prompt_embeds_mask"])

    neg_embeds = data.get("negative_prompt_embeds", None)
    if neg_embeds is not None:
        data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"] = _to_nested(
            neg_embeds, data["negative_prompt_embeds_mask"]
        )

    return data
