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
                columns = torch.arange(max(lengths), device=mask.device)
                valid = torch.equal(mask.bool(), columns < mask.new_tensor(lengths, dtype=torch.long)[:, None])
            if not valid:
                raise ValueError(f"Nested trajectory field {field!r} has an incompatible {mask_field!r}")
        dense_fields[field] = value.to_padded_tensor(padding=0.0) if value.is_nested else value
    return TensorDict(dense_fields, batch_size=fetched.batch_size)


def materialize_diffusion_trajectory(data: TensorDict, fields: DiffusionTrajectoryFieldsConfig) -> TensorDict:
    """Read the shard's configured training fields and keep driver-owned values."""
    keys = data.get(DIFFUSION_TQ_KEYS).tolist()
    positions = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    fetched = tq.kv_batch_get(
        keys=list(positions), partition_id="train", select_fields=fields.required + fields.optional
    )
    if missing := set(fields.required).difference(fetched.keys()):
        raise ValueError(f"Missing required diffusion trajectory fields: {sorted(missing)}")
    # TQ drops missing keys when other requested rows still exist.
    if len(fetched) != len(positions):
        raise ValueError(f"Diffusion trajectory row count mismatch: expected {len(positions)}, got {len(fetched)}")
    indices = [positions[key] for key in keys]
    trajectory = _dense_trajectory_fields(fetched.exclude(*data.keys()))[indices]
    return embeds_padding_2_no_padding(data.exclude(DIFFUSION_TQ_KEYS).update(trajectory))
