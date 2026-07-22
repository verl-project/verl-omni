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
"""TransferQueue <-> diffusion DataProto conversion helpers.

These helpers bridge the TransferQueue row format produced by
``DiffusionAgentLoopWorkerTQ`` and the diffusion ``DataProto`` layout consumed
by the existing policy-gradient compute path (reward, old/ref log-prob,
advantage, actor update). They centralize object-array handling for non-tensor
fields and key sorting for validation/rollout dumping.
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# Fields written by the diffusion TQ worker that are tensors with a fixed
# (per-batch) shape and therefore stackable directly from TransferQueue.
_DIFFUSION_TENSOR_FIELDS = [
    "prompts",
    "responses",
    "rollout_log_probs",
    "rm_scores",
    "attention_mask",
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "pooled_prompt_embeds",
    "negative_pooled_prompt_embeds",
    "all_latents",
    "all_timesteps",
]

# Non-tensor fields carried through TransferQueue as object arrays.
_DIFFUSION_NON_TENSOR_FIELDS = [
    "uid",
    "reward_model",
    "data_source",
    "extra_info",
    "raw_prompt",
    "__num_turns__",
    "extra_fields",
]


def _stack_field(value: Any, padding: float = 0.0) -> torch.Tensor | None:
    """Normalize a TransferQueue-returned field into a stacked tensor.

    Diffusion worker outputs are pre-padded to fixed shapes (prompt_length for
    token ids, fixed C/H/W for images, max_prompt_embed_length for embeds), so
    most fields come back already stacked. Variable-length fallbacks are padded
    so the downstream diffusion compute path receives uniform batch tensors.
    """
    if value is None:
        return None
    # Nested/jagged tensors returned by some TransferQueue backends must be
    # converted to dense padded tensors BEFORE the generic torch.Tensor check
    # below, since nested tensors are also instances of torch.Tensor. Leaving
    # them nested would expose symbolic ``NestedIntNode`` shapes downstream
    # (e.g. ``range(tensor.shape[1])`` raises ``AttributeError``).
    if isinstance(value, torch.Tensor) and value.is_nested:
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "to_padded_tensor"):
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, (list, tuple)):
        tensors = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in value]
        if not tensors:
            return None
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding)
    return torch.as_tensor(value)


def diffusion_tq_batch_to_dataproto(
    batch_meta: KVBatchMeta,
    pad_token_id: int = 0,
) -> DataProto:
    """Read selected TQ rows and assemble a diffusion ``DataProto``.

    Args:
        batch_meta: ``KVBatchMeta`` returned by ``ReplayBuffer.sample``.
        pad_token_id: Padding token id for variable-length prompt token tensors.

    Returns:
        ``DataProto`` whose ``batch`` carries diffusion tensors (prompts,
        responses, rollout_log_probs, rm_scores, embeds, ...) and whose
        ``non_tensor_batch`` carries uid/reward_model/data_source/extra_fields.
    """
    keys = list(batch_meta.keys)
    partition_id = batch_meta.partition_id

    available = set(_DIFFUSION_TENSOR_FIELDS) | set(_DIFFUSION_NON_TENSOR_FIELDS)
    data = tq.kv_batch_get(
        keys=keys,
        partition_id=partition_id,
        select_fields=list(available),
    )

    batch_dict: dict[str, torch.Tensor] = {}
    for field in _DIFFUSION_TENSOR_FIELDS:
        if field not in data:
            continue
        padding = float(pad_token_id) if field == "prompts" else 0.0
        stacked = _stack_field(data[field], padding=padding)
        if stacked is not None:
            batch_dict[field] = stacked

    non_tensor_batch: dict[str, Any] = {}
    for field in _DIFFUSION_NON_TENSOR_FIELDS:
        if field not in data:
            continue
        value = data[field]
        # Normalize LinkedList / NonTensorStack / numpy object arrays to a plain
        # list, then wrap as an object ndarray for DataProto compatibility.
        if isinstance(value, np.ndarray):
            non_tensor_batch[field] = value
        else:
            items = list(value)
            arr = np.empty(len(items), dtype=object)
            arr[:] = items
            non_tensor_batch[field] = arr

    # Unpack extra_fields dict rows into top-level non_tensor_batch keys so the
    # diffusion compute path can read min/max_global_steps and reward_extra_info.
    extra_fields_arr = non_tensor_batch.pop("extra_fields", None)
    if extra_fields_arr is not None:
        for i, extra in enumerate(extra_fields_arr.tolist()):
            if not isinstance(extra, dict):
                continue
            for k, v in extra.items():
                if k not in non_tensor_batch:
                    non_tensor_batch[k] = np.empty(len(extra_fields_arr), dtype=object)
                non_tensor_batch[k][i] = v

    batch = TensorDict(batch_dict, batch_size=len(keys))
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def put_dataproto_fields_to_tq(
    batch_meta: KVBatchMeta,
    data: DataProto,
    fields: list[str],
) -> None:
    """Write selected diffusion ``DataProto`` fields back to TransferQueue.

    Used after the trainer computes old_log_probs / ref_log_prob / advantages /
    returns on the driver and needs to persist them so the actor update worker
    can read them by key.
    """
    output: dict[str, Any] = {}
    for field in fields:
        if field not in data.batch:
            continue
        output[field] = data.batch[field]
    if not output:
        return
    tq.kv_batch_put(
        keys=list(batch_meta.keys),
        partition_id=batch_meta.partition_id,
        fields=tu.get_tensordict(output),
    )


def sort_diffusion_tq_keys(keys: list[str]) -> list[int]:
    """Return sort indices that order keys by ``(uid, session_id, index)``.

    Keys have the format ``{uid}_{session_id}_{index}``. Sorting by this tuple
    keeps generations from the same rollout group together for dumping.
    """
    sort_keys = []
    for key in keys:
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
        else:
            sort_keys.append((key, 0, 0))
    return sorted(range(len(keys)), key=lambda i: sort_keys[i])
