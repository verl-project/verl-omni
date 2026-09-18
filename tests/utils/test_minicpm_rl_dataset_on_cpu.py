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
"""CPU tests for the MiniCPM-o RL dataset media loader."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("soundfile")
pytest.importorskip("PIL")

from verl_omni.utils.dataset.omni_rl_datasets import (  # noqa: E402
    MiniCPMORLHFDataset,
    QwenOmniRLHFDataset,
    _load_minicpm_audio,
    pad_audio_to_hop_multiple,
)


def _media_files(tmp_path):
    from PIL import Image

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)
    audio_path = tmp_path / "clip.wav"
    import soundfile as sf

    sf.write(audio_path, np.zeros(161, dtype=np.float32), 16000, subtype="FLOAT")
    return str(image_path), str(audio_path)


def _avqa_messages(image_path: str, audio_path: str) -> list[dict]:
    # What verl's RLHFDataset._build_messages produces from the AVQA parquet
    # ("<image><audio>{problem}" + images/audios columns).
    return [
        {"role": "system", "content": "Answer with <answer>X</answer>."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Which song is playing?"},
            ],
        },
    ]


def test_minicpm_loader_reads_image_and_hop_pads_audio(tmp_path):
    image_path, audio_path = _media_files(tmp_path)
    images, videos, audios = MiniCPMORLHFDataset._process_multi_modal_info(
        _avqa_messages(image_path, audio_path),
        image_patch_size=14,
        config=OmegaConf.create({"mm_processor_kwargs": {"sampling_rate": 16000}}),
    )
    assert videos is None
    assert len(images) == 1 and images[0].size == (8, 8)
    assert len(audios) == 1
    # 161 samples zero-pad to the next 160-sample hop multiple.
    assert audios[0].shape == (320,)
    assert audios[0].dtype == np.float32


def test_minicpm_loader_rejects_video_blocks():
    messages = [{"role": "user", "content": [{"type": "video", "video": "/tmp/clip.mp4"}]}]
    with pytest.raises(ValueError, match="does not support video rows"):
        MiniCPMORLHFDataset._process_multi_modal_info(messages, image_patch_size=14, config=None)


def test_minicpm_loader_rejects_media_blocks_without_a_path():
    messages = [{"role": "user", "content": [{"type": "audio", "audio": None}]}]
    with pytest.raises(ValueError, match="audio block has no path"):
        MiniCPMORLHFDataset._process_multi_modal_info(messages, image_patch_size=14, config=None)


def test_minicpm_loader_resamples_to_the_configured_rate(tmp_path):
    image_path, audio_path = _media_files(tmp_path)
    _, _, audios = MiniCPMORLHFDataset._process_multi_modal_info(
        _avqa_messages(image_path, audio_path),
        image_patch_size=14,
        config=OmegaConf.create({"mm_processor_kwargs": {"sampling_rate": 8000}}),
    )
    # The config rate reaches the decoder: the 16kHz clip comes back as the
    # 8kHz-resampled, hop-padded waveform. Decoding at the wrong rate would
    # desync the waveform from the mel frames the processor expects.
    expected = pad_audio_to_hop_multiple(_load_minicpm_audio(audio_path, 8000))
    assert np.array_equal(audios[0], expected)


def test_qwen_dataset_keeps_qwen_resolution_order():
    # QwenOmniRLHFDataset still delegates to qwen_omni_utils (skipped when the
    # extra is absent); its patch lives in that method, not in a shared base.
    try:
        import qwen_omni_utils  # noqa: F401
    except ImportError:
        pytest.skip("qwen-omni-utils not installed")
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    images, videos, audios = QwenOmniRLHFDataset._process_multi_modal_info(messages, image_patch_size=14, config=None)
    # Newer qwen_omni_utils returns None instead of [] for absent modalities.
    assert images in ([], None) and videos in ([], None) and audios in ([], None)
