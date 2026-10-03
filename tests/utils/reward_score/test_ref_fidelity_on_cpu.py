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
"""CPU tests for the Ref2VA reference-fidelity reward."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image


def _load_file(dotted_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stub_verl_omni_reward_utils(repo_root: Path) -> None:
    """Register ``verl_omni.utils.reward_score.reward_utils`` without importing
    ``verl_omni/__init__.py``, whose heavy pipeline registrations are unrelated
    to this scorer and needn't be installed to test it."""
    package_names = ("verl_omni", "verl_omni.utils", "verl_omni.utils.reward_score")
    for name in package_names:
        sys.modules.setdefault(name, ModuleType(name))
    reward_utils_path = repo_root / "verl_omni/utils/reward_score/reward_utils.py"
    sys.modules["verl_omni.utils.reward_score.reward_utils"] = _load_file(
        "verl_omni.utils.reward_score.reward_utils", reward_utils_path
    )


def _load_module():
    repo_root = Path(__file__).parents[3]
    _stub_verl_omni_reward_utils(repo_root)
    return _load_file("ref_fidelity", repo_root / "verl_omni/utils/reward_score/ref_fidelity.py")


ref_fidelity = _load_module()


_COLOR_TO_FEATURE = {
    (255, 0, 0): [1.0, 0.0, 0.0],
    (0, 255, 0): [0.0, 1.0, 0.0],
    (0, 0, 255): [0.0, 0.0, 1.0],
}


class _FakeBatch(dict):
    def __init__(self, **fields):
        super().__init__(**fields)

    def to(self, device):
        return self


class _FakeClipProcessor:
    """Maps each fixed-color PIL image to a distinct one-hot feature vector."""

    def __call__(self, *, images, return_tensors="pt"):
        features = [_COLOR_TO_FEATURE[image.getpixel((0, 0))] for image in images]
        return _FakeBatch(pixel_values=torch.tensor(features))


class _FakeClipModel:
    def get_image_features(self, pixel_values):
        return pixel_values


class _FakeClapProcessor:
    """Maps a 1-element waveform (its own value) to a one-hot-ish feature vector."""

    def __call__(self, *, audio, sampling_rate, return_tensors="pt"):
        features = [[float(waveform[0]), float(waveform[0] < 0)] for waveform in audio]
        return _FakeBatch(input_features=torch.tensor(features))


class _FakeClapModel:
    def get_audio_features(self, input_features):
        return input_features


def _solid_image(color: tuple[int, int, int], size: int = 4) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _solid_video(color: tuple[int, int, int], num_frames: int = 3, size: int = 4) -> torch.Tensor:
    frame = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1).expand(3, size, size)
    return frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def _patch_clip(monkeypatch):
    monkeypatch.setattr(
        ref_fidelity, "_load_clip", lambda model_name_or_path, device: (_FakeClipModel(), _FakeClipProcessor())
    )


def _patch_clap(monkeypatch):
    monkeypatch.setattr(
        ref_fidelity, "_load_clap", lambda model_name_or_path, device: (_FakeClapModel(), _FakeClapProcessor())
    )


# --- helpers -----------------------------------------------------------------


def test_reference_visual_frames_requires_image_or_video():
    with pytest.raises(KeyError, match="source_images"):
        ref_fidelity._reference_visual_frames({}, frames_per_video=2)


def test_reference_visual_frames_accepts_single_image_path(tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((255, 0, 0)).save(image_path)

    frames = ref_fidelity._reference_visual_frames({"source_images": str(image_path)}, frames_per_video=2)

    assert len(frames) == 1
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)


def test_reference_visual_frames_decodes_reference_videos(monkeypatch, tmp_path):
    video_path = tmp_path / "ref.mp4"
    video_path.write_bytes(b"not a real video, decoding is mocked")
    raw_frames = [np.full((4, 4, 3), 255, dtype=np.uint8) for _ in range(5)]

    fake_iio = ModuleType("imageio.v3")
    fake_iio.imiter = lambda path, plugin: iter(raw_frames)
    monkeypatch.setitem(sys.modules, "imageio", ModuleType("imageio"))
    monkeypatch.setitem(sys.modules, "imageio.v3", fake_iio)

    frames = ref_fidelity._reference_visual_frames({"source_videos": [str(video_path)]}, frames_per_video=2)

    assert len(frames) == 2


def test_sample_generated_frames_rejects_missing_solution():
    with pytest.raises(ValueError, match="requires generated video/image"):
        ref_fidelity._sample_generated_frames(None, num_frames=4)


def test_sample_generated_frames_caps_at_available_frame_count():
    video = _solid_video((0, 255, 0), num_frames=2)

    frames = ref_fidelity._sample_generated_frames(video, num_frames=8)

    assert len(frames) == 2


# --- compute_score_ref_fidelity: visual only ----------------------------------


def test_compute_score_visual_perfect_match(monkeypatch, tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((0, 255, 0)).save(image_path)
    _patch_clip(monkeypatch)

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((0, 255, 0)),
        ground_truth="a green scene",
        extra_info={"source_images": [str(image_path)]},
        device="cpu",
    )

    assert result["score"] == pytest.approx(1.0)
    assert result["ref_fidelity_visual_similarity"] == pytest.approx(1.0)
    assert "ref_fidelity_audio_similarity" not in result
    assert result["ref_fidelity_num_references"] == 1
    assert result["ref_fidelity_num_frames"] == 3


def test_compute_score_visual_orthogonal_mismatch(monkeypatch, tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((255, 0, 0)).save(image_path)
    _patch_clip(monkeypatch)

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((0, 0, 255)),
        ground_truth="a blue scene",
        extra_info={"source_images": [str(image_path)]},
        device="cpu",
    )

    assert result["score"] == pytest.approx(0.0, abs=1e-6)


def test_compute_score_averages_multiple_image_references(monkeypatch, tmp_path):
    red_path = tmp_path / "red.png"
    green_path = tmp_path / "green.png"
    _solid_image((255, 0, 0)).save(red_path)
    _solid_image((0, 255, 0)).save(green_path)
    _patch_clip(monkeypatch)

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((255, 0, 0)),
        ground_truth="a red scene",
        extra_info={"source_images": [str(red_path), str(green_path)]},
        device="cpu",
    )

    assert result["ref_fidelity_num_references"] == 2
    assert 0.0 < result["score"] < 1.0


# --- compute_score_ref_fidelity: with an audio reference ----------------------


def test_compute_score_blends_audio_similarity_when_present(monkeypatch, tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((0, 255, 0)).save(image_path)
    audio_path = tmp_path / "ref.wav"
    audio_path.write_bytes(b"not real audio, decoding is mocked")
    _patch_clip(monkeypatch)
    _patch_clap(monkeypatch)
    monkeypatch.setattr(ref_fidelity, "_load_reference_waveform", lambda path, sr: np.array([1.0], dtype=np.float32))

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((0, 255, 0)),
        ground_truth="a green scene",
        extra_info={
            "source_images": [str(image_path)],
            "source_audios": [str(audio_path)],
            "audio": torch.tensor([1.0]),
            "audio_sample_rate": ref_fidelity._CLAP_SAMPLE_RATE,
        },
        device="cpu",
        audio_weight=0.5,
    )

    assert result["ref_fidelity_visual_similarity"] == pytest.approx(1.0)
    assert result["ref_fidelity_audio_similarity"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.0)


def test_compute_score_skips_audio_term_without_generated_audio(monkeypatch, tmp_path):
    image_path = tmp_path / "ref.png"
    _solid_image((0, 255, 0)).save(image_path)
    audio_path = tmp_path / "ref.wav"
    audio_path.write_bytes(b"not real audio, decoding is mocked")
    _patch_clip(monkeypatch)
    _patch_clap(monkeypatch)

    result = ref_fidelity.compute_score_ref_fidelity(
        data_source="minimax_h3_ref2va",
        solution_image=_solid_video((0, 255, 0)),
        ground_truth="a green scene",
        extra_info={"source_images": [str(image_path)], "source_audios": [str(audio_path)]},
        device="cpu",
    )

    assert "ref_fidelity_audio_similarity" not in result
    assert result["score"] == result["ref_fidelity_visual_similarity"]
