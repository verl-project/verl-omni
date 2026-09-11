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
"""CPU tests for ragged-collation tolerance in the MiniCPM media normalizers."""

from __future__ import annotations

import numpy as np
import torch

from verl_omni.pipelines.minicpm.media_inputs import (
    _unwrap_collated,
    batch_audio_feature_lens,
    normalize_audio_features,
    sample_pixel_slices,
    sample_tgt_sizes,
)


def _collated(*rows):
    """Mimic DataProto's ragged collation: object-dtype ndarray, None padding.

    Assigned element-wise — np.array(..., dtype=object) would try to broadcast
    same-leading-dim arrays during construction.
    """
    packed = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        packed[index] = row
    return packed


def test_unwrap_collated_drops_none_and_object_arrays():
    packed = _collated([np.zeros((3, 2, 2)), None], [np.zeros((3, 2, 2))])
    unwrapped = _unwrap_collated(packed)
    assert isinstance(unwrapped, list)
    assert len(unwrapped) == 2 and len(unwrapped[0]) == 1 and len(unwrapped[1]) == 1
    assert _unwrap_collated(None) is None
    assert _unwrap_collated(np.zeros((2, 2))) is not None  # numeric arrays pass through


def test_sample_pixel_slices_accepts_collated_pack_with_none_padding():
    packed = _collated([np.zeros((3, 4, 4)), np.zeros((3, 4, 4)), None], [np.zeros((3, 4, 4))])
    slices = sample_pixel_slices(packed)
    assert len(slices) == 3
    assert all(isinstance(item, torch.Tensor) and item.shape == (3, 4, 4) for item in slices)


def test_sample_tgt_sizes_accepts_collated_pack_with_none_padding():
    packed = _collated([np.array([2, 3], dtype=np.int64), None])
    sizes = sample_tgt_sizes(packed, n_slices=0, device=torch.device("cpu"))
    assert sizes.shape == (1, 2)
    assert sizes.tolist() == [[2, 3]]


def test_normalize_audio_features_accepts_collated_clips():
    clips = [np.zeros((80, 10), dtype=np.float32), np.zeros((80, 6), dtype=np.float32)]
    packed = _collated(clips)
    features = normalize_audio_features(packed)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (2, 80, 10)  # shorter clip zero-padded to the max frame count


def test_normalize_audio_features_collated_empties_stay_empty():
    assert normalize_audio_features(_collated([], None)) == []


def test_batch_audio_feature_lens_accepts_collated_lens():
    packed = _collated(np.array([5], dtype=np.int64), None)
    lens = batch_audio_feature_lens(packed, torch.device("cpu"))
    assert [item.tolist() for item in lens] == [[5]]
