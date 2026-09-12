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
"""Shared generated-media projection for visual and audio reward managers."""

from collections.abc import Mapping

import numpy as np
import torch


def _metadata_mapping(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Reward media metadata must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _reward_extra_info(data_item) -> dict:
    """Copy metadata and project generated media, rejecting conflicting sources."""
    extra_info = _metadata_mapping(data_item.non_tensor_batch.get("extra_info"))
    tool_extra_fields = _metadata_mapping(data_item.non_tensor_batch.get("tool_extra_fields"))
    extra_info.update(tool_extra_fields)
    tensor_fields = data_item.batch if data_item.batch is not None else {}
    for key in ("audio", "audio_sample_rate", "media_kind"):
        value = tensor_fields.get(key)
        if value is None:
            value = data_item.non_tensor_batch.get(key)
        if value is None:
            continue
        tool_value = tool_extra_fields.get(key)
        if tool_value is not None and tool_value is not value:
            if isinstance(value, torch.Tensor) and isinstance(tool_value, torch.Tensor):
                matches = value.device == tool_value.device and torch.equal(value, tool_value)
            else:
                matches = np.array_equal(value, tool_value)
            if not matches:
                raise ValueError(f"Conflicting rollout media field {key!r} in batch and tool_extra_fields")
        extra_info[key] = value
    return extra_info
