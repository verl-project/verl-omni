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
"""CPU tests for the generic CLAP reward scorer."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_scorer_module():
    module_path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/clap.py"
    spec = importlib.util.spec_from_file_location("clap_reward_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clap = _load_scorer_module()


def test_get_audio_normalizes_batch_and_channels():
    audio, sample_rate = clap._get_audio(
        {
            "audio": torch.ones(1, 2, 16),
            "audio_sample_rate": torch.tensor(48_000),
        }
    )

    assert audio.shape == (16,)
    assert sample_rate == 48_000


def test_compute_score_uses_generic_reward_interface(monkeypatch):
    functional = ModuleType("torchaudio.functional")
    functional.resample = lambda waveform, **kwargs: waveform
    torchaudio = ModuleType("torchaudio")
    torchaudio.functional = functional
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", functional)

    class FakeProcessor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == ["forest ambience"]
            return {"input_values": torch.ones(1, 4)}

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                audio_embeds=torch.tensor([[3.0, 4.0]]),
                text_embeds=torch.tensor([[0.0, 5.0]]),
            )

    monkeypatch.setattr(clap, "_load_clap", lambda model_name_or_path, device: (FakeModel(), FakeProcessor()))
    result = clap.compute_score(
        data_source="test",
        solution_image=None,
        ground_truth="forest ambience",
        extra_info={"audio": torch.ones(16), "audio_sample_rate": 48_000},
        device="cpu",
    )

    assert result == {"score": pytest.approx(0.8), "source_sample_rate": 48_000}
