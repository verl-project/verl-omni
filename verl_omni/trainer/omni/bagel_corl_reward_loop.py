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
"""Colocated reward loop for ``bagel_corl_sync``: enablement rules and the TQ-backed episode scoring."""

from __future__ import annotations

from typing import Any

#: Row-scoped dataset fields ``VisualRewardManager`` dispatches on: ``data_source`` picks the
#: scorer, ``reward_model`` carries its ``ground_truth``.
REWARD_ROW_FIELDS = ("data_source", "reward_model", "extra_info")


def named_reward_models_enabled(config) -> bool:
    """Whether the run hosts the named-model table (``reward.models``)."""
    reward_cfg = getattr(config, "reward", None)
    try:
        return bool(reward_cfg is not None and (reward_cfg.get("models", {}) or {}))
    except (AttributeError, TypeError):
        return False


def bagel_rm_enabled(config) -> bool:
    """Whether the run deploys a colocated reward model at all (legacy switch or named-model table)."""
    reward_cfg = getattr(config, "reward", None)
    if reward_cfg is None:
        return False
    try:
        if bool(reward_cfg.get("reward_model", {}).get("enable", False)):
            return True
        return bool(reward_cfg.get("models", {}) or {})
    except (AttributeError, TypeError):
        return False


def compute_reward_colocate(batch, *, reward_loop_manager, tokenizer, lengths_to_mask):
    """Score a TQ batch with the colocated RM, handing it the unwrapped row fields the scorer needs."""
    import numpy as np
    import torch
    import transfer_queue as tq
    from tensordict import TensorDict
    from verl.protocol import DataProto
    from verl.utils import tensordict_utils as tu

    assert reward_loop_manager is not None, "RewardLoopManager is None"
    fields = ["prompts", "responses", "raw_prompt", *REWARD_ROW_FIELDS]
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
    prompt_lengths = data["prompts"].offsets().diff()
    response_lengths = data["responses"].offsets().diff()
    prompts = data["prompts"].to_padded_tensor(padding=tokenizer.pad_token_id)
    responses = data["responses"].to_padded_tensor(padding=tokenizer.pad_token_id)
    attention_mask = torch.cat(
        [lengths_to_mask(prompt_lengths, prompts.size(1)), lengths_to_mask(response_lengths, responses.size(1))], dim=1
    )

    def _object_array(items: list) -> np.ndarray:
        arr = np.empty(len(items), dtype=object)
        arr[:] = items
        return arr

    non_tensor_batch: dict[str, Any] = {"raw_prompt": _object_array(list(data["raw_prompt"]))}
    for key in REWARD_ROW_FIELDS:
        values = data.pop(key, None)
        if values is not None:
            non_tensor_batch[key] = _object_array([tu.unwrap_non_tensor_data(value) for value in list(values)])
    rm_input = DataProto(
        batch=TensorDict(
            {"prompts": prompts, "responses": responses, "attention_mask": attention_mask}, batch_size=len(batch)
        ),
        non_tensor_batch=non_tensor_batch,
    )
    rm_output = reward_loop_manager.compute_rm_score(rm_input)
    padded = rm_output.batch["rm_scores"]
    write_back = {
        "rm_scores": torch.nested.as_nested_tensor(
            [padded[i, : response_lengths[i]] for i in range(len(batch))], layout=torch.jagged
        )
    }
    for key in rm_output.meta_info.get("reward_extra_keys", []):
        write_back[key] = rm_output.non_tensor_batch[key]
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=tu.get_tensordict(write_back))
    return batch
