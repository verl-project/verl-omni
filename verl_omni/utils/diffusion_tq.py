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

import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import kv_batch_meta2batch_meta

TRAJECTORY_KEYS = "_diffusion_tq_keys"
TRAJECTORY_META = "_diffusion_tq_trajectory"


def _validate_nested_embed_mask(field: str, values: torch.Tensor, mask: torch.Tensor | None, context: str) -> None:
    mask_field = f"{field}_mask"
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"Nested trajectory field {field!r} requires {mask_field!r} ({context})")

    value_rows = values.unbind()
    lengths = [row.shape[0] for row in value_rows]
    if mask.is_nested:
        mask_rows = mask.unbind()
        valid = len(mask_rows) == len(value_rows) and all(
            mask_row.shape == (length,) for mask_row, length in zip(mask_rows, lengths, strict=True)
        )
    else:
        valid = mask.ndim == 2 and mask.shape[0] == len(value_rows) and mask.sum(dim=1).tolist() == lengths
    if not valid:
        raise ValueError(f"Nested trajectory field {field!r} has an incompatible {mask_field!r} ({context})")


def materialize_diffusion_trajectory(data: TensorDict) -> TensorDict:
    """Fetch this worker shard's trajectory, preserving overlays and duplicated rows.

    Args:
        data: TensorDict whose row keys follow its batch dimension through DP
            dispatch and padding. Batches without a TQ handle pass through.

    Returns:
        A worker-owned TensorDict with padded trajectory tensors and jagged embeds.

    Raises:
        ValueError: A handle, required field, or trajectory shape is inconsistent.
    """
    descriptor = tu.get_non_tensor_data(data, TRAJECTORY_META, None)
    row_keys = data.get(TRAJECTORY_KEYS, None)
    if descriptor is None and row_keys is None:
        return data
    if descriptor is None or row_keys is None:
        raise ValueError("Incomplete diffusion TQ trajectory handle")
    keys = row_keys.tolist()
    if len(keys) != len(data) or not all(isinstance(key, str) for key in keys):
        raise ValueError("Diffusion TQ keys must match the worker batch dimension")
    timesteps = data.get("all_timesteps", None)
    num_timesteps = descriptor["num_timesteps"]
    if not isinstance(timesteps, torch.Tensor) or timesteps.shape != (len(keys), num_timesteps):
        raise ValueError("Deferred training requires driver-owned timesteps matching the trajectory handle")
    fields = descriptor["fields"]
    required = set(descriptor["required"])
    if not required.issubset(fields):
        raise ValueError("Diffusion TQ projection omits required trajectory fields")
    if set(fields) & set(data.keys()):
        raise ValueError("Diffusion TQ trajectory conflicts with driver-owned fields")
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        raise ValueError("Cannot materialize an empty diffusion TQ shard")
    context = f"partition={descriptor['partition_id']!r}, keys={unique_keys!r}"
    metadata = kv_batch_meta2batch_meta(
        KVBatchMeta(
            keys=unique_keys, tags=[{} for _ in unique_keys], partition_id=descriptor["partition_id"], fields=fields
        )
    )
    missing = set(fields) - set(metadata.field_names)
    if missing or metadata.size != len(unique_keys) or not metadata.is_ready:
        raise ValueError(f"Invalid diffusion trajectory metadata ({context}); missing fields={sorted(missing)}")
    fetched = tq.get_client().get_data(metadata)
    missing = set(fields) - set(fetched.keys())
    if missing:
        raise ValueError(f"Missing diffusion trajectory data ({context}): {sorted(missing)}")
    positions = {key: index for index, key in enumerate(unique_keys)}
    indices = torch.tensor([positions[key] for key in keys], dtype=torch.long)
    result = data.clone(recurse=False)
    result.pop(TRAJECTORY_KEYS)
    result.pop(TRAJECTORY_META)
    for field in fields:
        value = fetched[field]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor trajectory field {field!r} ({context})")
        if value.is_nested and field in {"prompt_embeds", "negative_prompt_embeds"}:
            _validate_nested_embed_mask(field, value, fetched.get(f"{field}_mask", None), context)
        if value.is_nested:
            value = value.to_padded_tensor(padding=0.0)
        if value.shape[0] != len(unique_keys):
            raise ValueError(f"Invalid batch dimension for trajectory field {field!r} ({context})")
        result[field] = value[indices.to(value.device)]
    latents = result["all_latents"]
    if latents.ndim < 3 or latents.shape[1] != num_timesteps + 1:
        raise ValueError(f"Expected S+1 trajectory latents for S timesteps ({context})")

    from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

    return embeds_padding_2_no_padding(result)
