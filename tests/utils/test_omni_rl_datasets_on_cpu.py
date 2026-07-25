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

import numpy as np
import pytest
from datasets.utils._dill import dumps
from dill import loads

pytest.importorskip("cachetools")
pytest.importorskip("vllm")

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.import_utils import load_extern_object

from verl_omni.utils.dataset.omni_rl_datasets import OmniRLHFDataset


def test_package_loaded_dataset_preserves_base_class_after_serialization():
    dataset_cls = load_extern_object(
        "pkg://verl_omni.utils.dataset.omni_rl_datasets",
        "OmniRLHFDataset",
    )
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.serialize_dataset = False

    restored = loads(dumps(dataset))

    assert isinstance(restored, RLHFDataset)
    assert hasattr(restored, "_build_messages")


def test_extract_audio_info_loads_path_at_configured_sampling_rate(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    waveform = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    calls = []

    def fake_load_audio(path, sampling_rate):
        calls.append((path, sampling_rate))
        return waveform

    monkeypatch.setattr("transformers.audio_utils.load_audio", fake_load_audio)
    messages = [{"role": "user", "content": [{"type": "audio", "audio": str(audio_path)}]}]

    result = OmniRLHFDataset._extract_audio_info(messages, sampling_rate=16000)

    assert result is not None
    np.testing.assert_array_equal(result[0], waveform)
    assert calls == [(str(audio_path), 16000)]


def test_load_audio_reports_unmounted_media_path(tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError, match="mounted at the same path"):
        OmniRLHFDataset._load_audio_for_processor(str(missing), sampling_rate=16000)
