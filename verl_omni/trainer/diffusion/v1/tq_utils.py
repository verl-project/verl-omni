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

import logging
import os
from typing import Any, Literal

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def diffusion_persisted_tq_fields(
    algorithm: Literal["policy_gradient", "direct_preference"],
) -> list[str]:
    """Return trainer-computed fields persisted after a diffusion update."""
    if algorithm == "policy_gradient":
        return ["old_log_probs", "advantages", "returns", "sample_level_scores", "sample_level_rewards"]
    if algorithm == "direct_preference":
        return ["sample_level_scores", "sample_level_rewards"]
    raise ValueError(f"Unsupported diffusion trainer algorithm: {algorithm}")


def diffusion_metric_tq_fields(
    algorithm: Literal["policy_gradient", "direct_preference"],
) -> list[str]:
    """Return fields persisted and consumed by diffusion metric computation."""
    if algorithm == "policy_gradient":
        return ["sample_level_rewards", "sample_level_scores", "advantages", "returns", "uid", "extra_fields"]
    if algorithm == "direct_preference":
        return ["sample_level_rewards", "sample_level_scores", "uid", "extra_fields"]
    raise ValueError(f"Unsupported diffusion trainer algorithm: {algorithm}")


def _unwrap_non_tensor_item(item: Any) -> Any:
    """Unwrap common non-tensor wrappers (e.g. NonTensorData) to raw values."""
    value = item
    # Some TransferQueue paths return wrapper objects with a ``data`` attribute.
    # Unwrap a few levels defensively until reaching a plain python value.
    for _ in range(4):
        if isinstance(value, str | bytes | bytearray | dict | list | tuple | np.ndarray | torch.Tensor):
            break
        data_attr = getattr(value, "data", None)
        if data_attr is None or data_attr is value:
            break
        value = data_attr
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_object_array(value: Any) -> np.ndarray:
    """Normalize non-tensor TransferQueue values to object ndarray."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        items = value.tolist()
    elif isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, str | bytes | dict):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    items = [_unwrap_non_tensor_item(item) for item in items]
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def _stack_field(value: Any, padding: float = 0.0) -> torch.Tensor | None:
    """Normalize a TransferQueue-returned field into a stacked tensor.

    Diffusion worker outputs are pre-padded to fixed shapes (prompt_length for
    token ids, fixed C/H/W for images, max_prompt_embed_length for embeds), so
    most fields come back already stacked. Variable-length fallbacks are padded
    so the downstream diffusion compute path receives uniform batch tensors.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.is_nested:
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "to_padded_tensor"):
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, list | tuple):
        tensors = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in value]
        if not tensors:
            return None
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding)
    return torch.as_tensor(value)


def diffusion_tq_batch_to_dataproto(
    batch_meta: KVBatchMeta,
    pad_token_id: int = 0,
    select_fields: list[str] | None = None,
) -> DataProto:
    """Read TQ rows and assemble a diffusion ``DataProto``.

    Args:
        batch_meta: ``KVBatchMeta`` returned by ``ReplayBuffer.sample``.
        pad_token_id: Padding token id for variable-length prompt token tensors.
        select_fields: Optional TQ fields to retrieve. ``None`` preserves the
            full-payload behavior required by training and validation.

    Returns:
        ``DataProto`` whose ``batch`` carries diffusion tensors (prompts,
        responses, rollout_log_probs, rm_scores, embeds, ...) and whose
        ``non_tensor_batch`` carries uid/reward_model/data_source/extra_fields.
    """
    keys = list(batch_meta.keys)
    partition_id = batch_meta.partition_id

    data = tq.kv_batch_get(
        keys=keys,
        partition_id=partition_id,
        select_fields=select_fields,
    )

    batch_dict: dict[str, torch.Tensor] = {}
    non_tensor_batch: dict[str, Any] = {}
    for field, value in data.items():
        padding = float(pad_token_id) if field == "prompts" else 0.0
        stacked = None
        try:
            stacked = _stack_field(value, padding=padding)
        except (TypeError, ValueError, RuntimeError):
            # Non-numeric/object fields are forwarded via non_tensor_batch.
            stacked = None

        if stacked is not None and isinstance(stacked, torch.Tensor):
            batch_dict[field] = stacked
            continue
        non_tensor_batch[field] = _to_object_array(value)

    # Unpack extra_fields dict rows into top-level non_tensor_batch keys so the
    # diffusion compute path can read min/max_global_steps and reward_extra_info.
    extra_fields_arr = non_tensor_batch.pop("extra_fields", None)
    if extra_fields_arr is not None:
        for i, extra in enumerate(extra_fields_arr.tolist()):
            if not isinstance(extra, dict):
                continue
            for k, v in extra.items():
                if k not in non_tensor_batch:
                    non_tensor_batch[k] = np.full(len(extra_fields_arr), None, dtype=object)
                non_tensor_batch[k][i] = v

    batch = TensorDict(batch_dict, batch_size=len(keys))
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def put_dataproto_fields_to_tq(
    batch_meta: KVBatchMeta,
    data: DataProto,
    fields: list[str],
) -> None:
    """Write selected ``DataProto`` batch fields back to TransferQueue.

    Args:
        batch_meta: ``KVBatchMeta`` whose ``keys`` and ``partition_id`` target
            the rows to update.
        data: ``DataProto`` containing the computed tensor fields.
        fields: Batch field names to persist; missing fields are skipped.
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
    """Return indices that sort TransferQueue keys in rollout order.

    Args:
        keys: TransferQueue row keys from ``KVBatchMeta.keys``.

    Returns:
        Permutation indices that reorder ``keys`` by ``(uid, rollout, output)``.
    """
    return sorted(range(len(keys)), key=lambda i: _parse_tq_key(keys[i]))


def _parse_tq_key(key: str) -> tuple[str, int, int]:
    """Split a ``{uid}_{session}_{output}`` TransferQueue key."""
    parts = key.rsplit("_", 2)
    if len(parts) == 3:
        try:
            return parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            return key, 0, 0
    return key, 0, 0


def canonicalize_diffusion_tq_meta(batch_meta: KVBatchMeta) -> KVBatchMeta:
    """Return a copy of ``batch_meta`` with rows in v0 rollout order.

    Upstream ``_materialize_batch`` emits trajectory keys in TransferQueue
    ``kv_list`` iteration order, which is storage-arbitrary and differs per
    run, while the v0 trainer produced rows prompt-major in dataset order
    (``prompt_index * rollout.n + session``). Rows whose tag lacks
    ``prompt_index`` (written before this ordering existed) fall back to uid
    order, which is deterministic but not dataset order.
    """
    if len(batch_meta.keys) < 2:
        return batch_meta

    def sort_key(i: int) -> tuple:
        uid, session, output = _parse_tq_key(batch_meta.keys[i])
        tag = batch_meta.tags[i] if i < len(batch_meta.tags) else {}
        prompt_index = tag.get("prompt_index") if isinstance(tag, dict) else None
        if prompt_index is None:
            return (1, 0, uid, session, output)
        return (0, int(prompt_index), uid, session, output)

    perm = sorted(range(len(batch_meta.keys)), key=sort_key)
    return KVBatchMeta(
        partition_id=batch_meta.partition_id,
        keys=[batch_meta.keys[i] for i in perm],
        tags=[batch_meta.tags[i] for i in perm],
    )
