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
"""Read deferred diffusion training fields on actor and reference workers."""

import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl_omni.workers.config.diffusion.model import DIFFUSION_TQ_KEYS, DiffusionTrajectoryFieldsConfig
from verl_omni.workers.utils.padding import diffusion_sequence_fields, embeds_padding_2_no_padding


def _dense_trajectory_fields(fetched: TensorDict) -> TensorDict:
    sequence_fields = diffusion_sequence_fields(fetched)
    dense_fields = {}
    trimmed_masks = {}
    for field, value in fetched.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor in trajectory field {field!r}")
        if value.is_nested and field in sequence_fields:
            mask_field = f"{field}_mask"
            mask = fetched.get(mask_field, None)
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"Nested trajectory field {field!r} requires {mask_field!r}")
            lengths = [row.shape[0] for row in value.unbind()]
            if mask.is_nested:
                valid = all(row.shape == (length,) for row, length in zip(mask.unbind(), lengths, strict=True))
            else:
                width = max(lengths)
                valid = mask.ndim == 2 and mask.shape[0] == len(lengths) and mask.shape[1] >= width
                if valid:
                    columns = torch.arange(mask.shape[1], device=mask.device)
                    outside_row = columns >= mask.new_tensor(lengths, dtype=torch.long)[:, None]
                    valid = not (mask.bool() & outside_row).any()
                    # A shard can have a shorter maximum sequence than the stored dense mask.
                    trimmed_masks[mask_field] = mask[:, :width]
            if not valid:
                raise ValueError(f"Nested trajectory field {field!r} has an incompatible {mask_field!r}")
        dense_fields[field] = value.to_padded_tensor(padding=0.0) if value.is_nested else value
    dense_fields.update(trimmed_masks)
    return TensorDict(dense_fields, batch_size=fetched.batch_size)


def materialize_diffusion_trajectory(data: TensorDict, fields: DiffusionTrajectoryFieldsConfig) -> TensorDict:
    """Read the shard's configured training fields and keep driver-owned values."""
    keys = data.get(DIFFUSION_TQ_KEYS).tolist()
    unique_keys = list(dict.fromkeys(keys))
    fetched = tq.kv_batch_get(keys=unique_keys, partition_id="train", select_fields=fields.required + fields.optional)
    if missing := set(fields.required).difference(fetched.keys()):
        raise ValueError(f"Missing required diffusion trajectory fields: {sorted(missing)}")
    # TQ drops missing keys when other requested rows still exist.
    if len(fetched) != len(unique_keys):
        raise ValueError(f"Diffusion trajectory row count mismatch: expected {len(unique_keys)}, got {len(fetched)}")
    trajectory = _dense_trajectory_fields(fetched.exclude(*data.keys()))
    # TQ returns rows in request order. Only padded shards with duplicate keys need a gather.
    if len(keys) != len(unique_keys):
        positions = {key: index for index, key in enumerate(unique_keys)}
        trajectory = trajectory[[positions[key] for key in keys]]
    return embeds_padding_2_no_padding(data.exclude(DIFFUSION_TQ_KEYS).update(trajectory))
