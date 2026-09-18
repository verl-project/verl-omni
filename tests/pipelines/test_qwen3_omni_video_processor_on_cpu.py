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

"""Video timing regression tests using real HF processor execution on CPU."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def processor_module():
    path = Path(__file__).resolve().parents[2] / "verl_omni/pipelines/qwen3_omni/video_processor.py"
    spec = importlib.util.spec_from_file_location("nextqa_processor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_processor_passes_actual_sampled_fps_without_resampling(processor_module, monkeypatch):
    calls = []
    monkeypatch.setattr(processor_module.Qwen3OmniMoeProcessor, "__call__", lambda self, **kw: calls.append(kw))
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    meta = [{"fps": 30, "total_num_frames": 1800, "frames_indices": list(range(32))}]
    original = {"fps": 1, "do_sample_frames": True}
    processor(videos=[object()], video_metadata=meta, videos_kwargs=original, fps=99)
    assert calls[0]["videos_kwargs"] == {"fps": 32 / 60, "do_sample_frames": False}
    assert "fps" not in calls[0]
    assert original == {"fps": 1, "do_sample_frames": True}


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_processor_rejects_invalid_video_clock(processor_module, duration):
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    with pytest.raises(ValueError, match="duration"):
        processor(videos=[object()], video_metadata=[{"duration": duration, "frames_indices": [0, 1]}])


def test_processor_rejects_mixed_clocks(processor_module):
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    with pytest.raises(ValueError, match="one sampled FPS"):
        processor(
            videos=[object(), object()],
            video_metadata=[{"duration": 2, "frames_indices": [0, 1]}, {"duration": 4, "frames_indices": [0, 1]}],
        )


def test_real_hf_processor_returns_sampled_video_time(processor_module):
    import torch

    class Tokenizer:
        init_kwargs = {}

        def __call__(self, text, **kwargs):
            return {"input_ids": [[1, 2]]}

    class VideoProcessor:
        temporal_patch_size = 2

        def __call__(self, videos, **kwargs):
            assert kwargs["do_sample_frames"] is False
            return {"video_grid_thw": [[16, 2, 2]]}

    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    processor.tokenizer = Tokenizer()
    processor.video_processor = VideoProcessor()
    processor.replace_multimodal_special_tokens = lambda text, *args, **kwargs: text
    result = processor(
        text="video",
        videos=[torch.zeros(32, 3, 28, 28)],
        video_metadata=[{"fps": 30, "total_num_frames": 1800, "frames_indices": list(range(32))}],
        return_tensors="pt",
    )
    torch.testing.assert_close(result["video_second_per_grid"], torch.tensor([3.75]))


@pytest.mark.parametrize("metadata", [[], [{"duration": 1}], [{"duration": 1, "frames_indices": []}]])
def test_processor_rejects_missing_sample_indices(processor_module, metadata):
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    with pytest.raises(ValueError):
        processor(videos=[object()], video_metadata=metadata)


def test_processor_preserves_nonvideo_inputs(processor_module, monkeypatch):
    calls = []
    monkeypatch.setattr(processor_module.Qwen3OmniMoeProcessor, "__call__", lambda self, **kw: calls.append(kw))
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    images, audio = [object()], [object()]
    processor(text="question", images=images, audio=audio, return_tensors="pt")
    assert calls == [{"text": "question", "images": images, "videos": None, "audio": audio, "return_tensors": "pt"}]


def test_explicit_duration_and_nested_metadata_control_sample_clock(processor_module, monkeypatch):
    calls = []
    monkeypatch.setattr(processor_module.Qwen3OmniMoeProcessor, "__call__", lambda self, **kw: calls.append(kw))
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    metadata = [{"duration": 4, "fps": 30, "total_num_frames": 300, "frames_indices": [0, 30]}]
    kwargs = {"video_metadata": metadata, "fps": 99, "do_sample_frames": True}
    processor(videos=[object()], videos_kwargs=kwargs)
    forwarded = calls[0]["videos_kwargs"]
    assert forwarded["fps"] == pytest.approx(0.5)
    assert forwarded["do_sample_frames"] is False
    assert forwarded["video_metadata"] == metadata
    assert kwargs["fps"] == 99
    assert kwargs["do_sample_frames"] is True
