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
"""Padding utilities for model training."""

import logging

import torch
from tensordict import TensorDict
from verl.trainer.ppo import padding_utils as _padding_utils
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template as ori_padding_template

logger = logging.getLogger(__name__)


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


# TODO (wsc): temporary monkey-patch. Remove once verl's padding_utils pads teacher-side
# fields (teacher_ids / teacher_logprobs) natively in construct_minimal_padding_template.
def patched_padding_template(source_td, source_tag, eos_token_id):
    """Wrap verl's minimal padding template to also pad teacher-side fields.

    The stock template only knows about student-token fields; teacher_ids /
    teacher_logprobs ride along on the batch and must be extended to the padded
    sequence length with eos / zeros respectively.
    """
    sample, tag = ori_padding_template(source_td, source_tag, eos_token_id)
    if not getattr(patched_padding_template, "_warned", False):
        logger.warning(
            "Using patched_padding_template to pad teacher_ids / teacher_logprobs. "
            "This is a temporary workaround; remove once verl's padding_utils pads "
            "teacher-side fields natively."
        )
        patched_padding_template._warned = True
    sequence_length = sample["input_ids"].size(0)

    teacher_ids = sample.get("teacher_ids")
    if isinstance(teacher_ids, torch.Tensor):
        sample["teacher_ids"] = teacher_ids.new_full((sequence_length, *teacher_ids.shape[1:]), eos_token_id)

    teacher_logprobs = sample.get("teacher_logprobs")
    if isinstance(teacher_logprobs, torch.Tensor):
        sample["teacher_logprobs"] = teacher_logprobs.new_zeros((sequence_length, *teacher_logprobs.shape[1:]))

    return sample, tag


# Patch the module attribute so upsample_batch_to_divisible_size resolves the
# patched template through verl's padding_utils namespace.
_padding_utils.construct_minimal_padding_template = patched_padding_template
