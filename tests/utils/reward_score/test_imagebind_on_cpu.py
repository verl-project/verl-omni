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
"""CPU tests for the generic ImageBind reward scorer."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_scorer_module():
    module_path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/imagebind.py"
    spec = importlib.util.spec_from_file_location("imagebind_reward_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


imagebind = _load_scorer_module()


class _ModalityType:
    AUDIO = "audio"
    TEXT = "text"
    VISION = "vision"


def _install_imagebind_modules(monkeypatch, imagebind_model=None):
    root = ModuleType("imagebind")
    models = ModuleType("imagebind.models")
    model_module = imagebind_model or ModuleType("imagebind.models.imagebind_model")
    model_module.ModalityType = _ModalityType
    models.imagebind_model = model_module
    root.models = models
    monkeypatch.setitem(sys.modules, "imagebind", root)
    monkeypatch.setitem(sys.modules, "imagebind.models", models)
    monkeypatch.setitem(sys.modules, "imagebind.models.imagebind_model", model_module)


def test_to_tchw_accepts_channels_last_video():  # trufflehog:ignore
    video = torch.full((3, 8, 10, 3), 255, dtype=torch.uint8)

    converted = imagebind._to_tchw(video)

    assert converted.shape == (3, 3, 8, 10)
    assert converted.max() == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_to_tchw_rejects_non_uint8_video(dtype):
    with pytest.raises(ValueError, match=rf"Expected uint8 video input, got {dtype}\."):
        imagebind._to_tchw(torch.zeros(3, 8, 10, 3, dtype=dtype))


@pytest.mark.parametrize(
    ("mode", "expected_modalities", "expected_score"),
    [
        ("audio_video", {_ModalityType.AUDIO, _ModalityType.VISION}, 0.8),
        ("text_audio", {_ModalityType.TEXT, _ModalityType.AUDIO}, 0.6),
        ("text_video", {_ModalityType.TEXT, _ModalityType.VISION}, 0.0),
        ("all", {_ModalityType.TEXT, _ModalityType.AUDIO, _ModalityType.VISION}, 0.55),
    ],
)
def test_compute_score_supports_flowfactory_modes(
    monkeypatch,
    mode,
    expected_modalities,
    expected_score,
):
    _install_imagebind_modules(monkeypatch)
    observed = {}

    class FakeModel:
        def __call__(self, inputs):
            assert set(inputs) == expected_modalities
            embeddings = {
                _ModalityType.AUDIO: torch.tensor([[3.0, 4.0]]),
                _ModalityType.VISION: torch.tensor([[0.0, 5.0]]),
                _ModalityType.TEXT: torch.tensor([[5.0, 0.0]]),
            }
            return {key: embeddings[key] for key in inputs}

    def fake_preprocess_audio(audio, source_rate, device):
        observed["sample_rate"] = source_rate
        return torch.ones(1)

    monkeypatch.setattr(imagebind, "_load_imagebind", lambda device, model_path: FakeModel())
    monkeypatch.setattr(imagebind, "_preprocess_audio", fake_preprocess_audio)
    monkeypatch.setattr(imagebind, "_preprocess_video", lambda video, device: torch.ones(1))
    monkeypatch.setattr(imagebind, "_preprocess_text", lambda text, device: torch.ones(1))

    result = imagebind.compute_score(
        data_source="test",
        solution_image=torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        ground_truth="forest ambience",
        extra_info={} if mode == "text_video" else {"audio": torch.ones(16)},
        device="cpu",
        mode=mode,
    )

    if _ModalityType.AUDIO in expected_modalities:
        assert observed["sample_rate"] == 16_000
    assert result["score"] == pytest.approx(expected_score)
    if mode == "all":
        assert result["audio_video_similarity"] == pytest.approx(0.8)
        assert result["text_audio_similarity"] == pytest.approx(0.6)
        assert result["text_video_similarity"] == pytest.approx(0.0)


def test_compute_score_rejects_unknown_mode(monkeypatch):
    _install_imagebind_modules(monkeypatch)

    with pytest.raises(ValueError, match="Unknown ImageBind mode"):
        imagebind.compute_score(
            data_source="test",
            solution_image=None,
            ground_truth="",
            extra_info={},
            device="cpu",
            mode="unknown",
        )


def test_model_cache_is_scoped_by_model_path_and_device(monkeypatch):
    created_models = []
    model_module = ModuleType("imagebind.models.imagebind_model")

    class FakeModel:
        def load_state_dict(self, state_dict):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    def imagebind_huge(pretrained):
        model = FakeModel()
        created_models.append(model)
        return model

    model_module.imagebind_huge = imagebind_huge
    _install_imagebind_modules(monkeypatch, model_module)
    monkeypatch.setattr(imagebind.os.path, "exists", lambda path: True)
    monkeypatch.setattr(imagebind.torch, "load", lambda path, weights_only: {})
    imagebind._MODEL_CACHE.clear()

    first = imagebind._load_imagebind("cpu", "first.pth")
    assert imagebind._load_imagebind("cpu", "first.pth") is first
    assert imagebind._load_imagebind("cpu", "second.pth") is not first
    assert len(created_models) == 2
