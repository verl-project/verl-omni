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

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from dill import dumps, loads

pytest.importorskip("cachetools")
pytest.importorskip("vllm")

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.import_utils import load_extern_object

from verl_omni.utils.dataset.omni_rl_datasets import QwenOmniRLHFDataset


def test_package_loaded_dataset_preserves_base_class_after_serialization():
    dataset_cls = load_extern_object(
        "pkg://verl_omni.utils.dataset.omni_rl_datasets",
        "QwenOmniRLHFDataset",
    )
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.serialize_dataset = False

    restored = loads(dumps(dataset))

    assert isinstance(restored, RLHFDataset)
    assert hasattr(restored, "_build_messages")


def test_process_multi_modal_info_uses_qwen_omni_utils_and_reorders_outputs(monkeypatch):
    calls = []
    L = 8001
    padded_len = L + (-L % 160)
    audios = [np.zeros(L, dtype=np.float32)]
    images = [object()]
    videos = [object()]

    def fake_process_mm_info(messages, use_audio_in_video):
        calls.append((messages, use_audio_in_video))
        return audios, images, videos

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=fake_process_mm_info))
    messages = [{"role": "user", "content": [{"type": "audio", "audio": "/data/sample.wav"}]}]

    result = QwenOmniRLHFDataset._process_multi_modal_info(messages, image_patch_size=14, config={})

    assert result[0] is images
    assert result[1] is videos
    assert len(result[2]) == 1
    # Audio is padded to a hop multiple (160) so the actor recompute and the
    # vllm-omni rollout expand it to the same audio token count.
    assert result[2][0].shape == (padded_len,)
    np.testing.assert_array_equal(result[2][0][:L], audios[0])
    np.testing.assert_array_equal(result[2][0][L:], 0.0)
    assert calls == [(messages, False)]
