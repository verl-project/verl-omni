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
"""Worker-side materialization of diffusion trajectory fields."""

from dataclasses import dataclass
from typing import TypedDict

import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import kv_batch_meta2batch_meta

from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

TRAJECTORY_KEYS = "_diffusion_tq_keys"
TRAJECTORY_META = "_diffusion_tq_trajectory"


@dataclass(frozen=True)
class TQTrainingFields:
    """Fields an adapter needs beyond the V1 trainer's common driver inputs.

    Declare S+1 state tensors in `transition_fields` and variable-length
    worker sequences as (value, mask) pairs in `sequence_masks`. A dense
    sequence may omit its mask if the mask field is optional.
    """

    worker_required: frozenset[str]
    worker_optional: frozenset[str] = frozenset()
    driver_required: frozenset[str] = frozenset()
    driver_optional: frozenset[str] = frozenset()
    transition_fields: frozenset[str] = frozenset()
    sequence_masks: tuple[tuple[str, str], ...] = ()

    @property
    def worker_fields(self) -> frozenset[str]:
        """Return the fields eligible for the worker read."""
        return self.worker_required | self.worker_optional


class TrajectoryDescriptor(TypedDict):
    """Batch-wide worker projection metadata carried beside the per-row keys."""

    partition_id: str
    fields: list[str]
    required: list[str]
    num_timesteps: int
    transition_fields: list[str]
    sequence_masks: dict[str, str]


def validate_transition_shape(shape: tuple[int, ...], steps: int, field: str, context: str) -> None:
    """Require one more latent state than timesteps in the per-sample tensor."""
    if len(shape) < 2 or shape[0] != steps + 1:
        raise ValueError(f"Expected S+1 latents for S timesteps in {field!r} ({context})")


def _validate_nested_embed_mask(
    field: str, values: torch.Tensor, fetched: TensorDict, mask_field: str, context: str
) -> None:
    mask = fetched.get(mask_field, None)
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"Nested trajectory field {field!r} requires {mask_field!r} ({context})")

    lengths = [row.shape[0] for row in values.unbind()]
    if mask.is_nested:
        mask_rows = mask.unbind()
        valid = len(mask_rows) == len(lengths) and all(
            mask_row.shape == (length,) for mask_row, length in zip(mask_rows, lengths, strict=True)
        )
    else:
        max_length = max(lengths)
        columns = torch.arange(max_length, device=mask.device)
        expected = columns < mask.new_tensor(lengths, dtype=torch.long)[:, None]
        valid = torch.equal(mask.bool(), expected)
    if not valid:
        raise ValueError(f"Nested trajectory field {field!r} has an incompatible {mask_field!r} ({context})")


def _read_trajectory_handle(data: TensorDict) -> tuple[TrajectoryDescriptor, list[str]] | None:
    descriptor = tu.get_non_tensor_data(data, TRAJECTORY_META, None)
    row_keys = data.get(TRAJECTORY_KEYS, None)
    if descriptor is None and row_keys is None:
        return None
    if descriptor is None or row_keys is None:
        raise ValueError("Incomplete diffusion TQ trajectory handle")
    keys = row_keys.tolist()
    if len(keys) != len(data) or not all(isinstance(key, str) for key in keys):
        raise ValueError("Diffusion TQ keys must match the worker batch dimension")
    if not keys:
        raise ValueError("Cannot materialize an empty diffusion TQ shard")
    timesteps = data.get("all_timesteps", None)
    num_timesteps = descriptor["num_timesteps"]
    if not isinstance(timesteps, torch.Tensor) or timesteps.shape != (len(keys), num_timesteps):
        raise ValueError("Deferred training requires driver-owned timesteps matching the trajectory handle")
    fields = set(descriptor["fields"])
    if not set(descriptor["required"]).issubset(fields):
        raise ValueError("Diffusion TQ projection omits required trajectory fields")
    if fields & set(data.keys()):
        raise ValueError("Diffusion TQ trajectory conflicts with driver-owned fields")
    return descriptor, keys


def _fetch_trajectory_rows(descriptor: TrajectoryDescriptor, keys: list[str], context: str) -> TensorDict:
    fields = descriptor["fields"]
    metadata = kv_batch_meta2batch_meta(
        KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id=descriptor["partition_id"], fields=fields)
    )
    missing = set(fields) - set(metadata.field_names)
    if missing or metadata.size != len(keys) or not metadata.is_ready:
        raise ValueError(f"Invalid diffusion trajectory metadata ({context}); missing fields={sorted(missing)}")
    fetched = tq.get_client().get_data(metadata)
    missing = set(fields) - set(fetched.keys())
    if missing:
        raise ValueError(f"Missing diffusion trajectory data ({context}): {sorted(missing)}")
    return fetched


def _restore_trajectory_fields(
    fetched: TensorDict, descriptor: TrajectoryDescriptor, indices: torch.Tensor, unique_count: int, context: str
) -> dict[str, torch.Tensor]:
    restored = {}
    sequence_masks = descriptor["sequence_masks"]
    transition_fields = set(descriptor["transition_fields"])
    for field in descriptor["fields"]:
        value = fetched[field]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor trajectory field {field!r} ({context})")
        if value.is_nested:
            if field in sequence_masks:
                _validate_nested_embed_mask(field, value, fetched, sequence_masks[field], context)
            value = value.to_padded_tensor(padding=0.0)
        if not value.ndim or value.shape[0] != unique_count:
            raise ValueError(f"Invalid batch dimension for trajectory field {field!r} ({context})")
        if field in transition_fields:
            validate_transition_shape(value.shape[1:], descriptor["num_timesteps"], field, context)
        restored[field] = value[indices.to(value.device)]
    return restored


def materialize_diffusion_trajectory(data: TensorDict) -> TensorDict:
    """Load this worker shard's declared inputs, preserving row order and overlays."""
    handle = _read_trajectory_handle(data)
    if handle is None:
        return data

    descriptor, keys = handle
    positions = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    unique_keys = list(positions)
    context = f"partition={descriptor['partition_id']!r}, keys={unique_keys!r}"
    fetched = _fetch_trajectory_rows(descriptor, unique_keys, context)
    indices = torch.tensor([positions[key] for key in keys], dtype=torch.long)
    fields = _restore_trajectory_fields(fetched, descriptor, indices, len(unique_keys), context)
    result = data.exclude(TRAJECTORY_KEYS, TRAJECTORY_META).update(fields)
    return embeds_padding_2_no_padding(result, sequence_masks=descriptor["sequence_masks"])
