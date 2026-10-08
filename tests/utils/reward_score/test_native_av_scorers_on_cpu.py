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
"""Native audio/video scorer contracts with CPU tensors and injected models."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

# Load only scorers, avoiding top-level engine/rollout auto-registration on CPU.
_PACKAGE = "native_av_scorers_under_test"
package = ModuleType(_PACKAGE)
package.__path__ = [str(Path(__file__).parents[3] / "verl_omni/utils/reward_score")]
sys.modules[_PACKAGE] = package
audiobox, clap, desync, hpsv3_reward, videoalign, reward_utils = (
    importlib.import_module(f"{_PACKAGE}.{name}")
    for name in ("audiobox", "clap", "desync", "hpsv3_reward", "videoalign", "reward_utils")
)


def _batch(*, audio=None, sample_rate=48000, video=None, fps=24.0):
    tensors = {}
    if audio is not None:
        tensors["audio"] = audio
    if video is not None:
        tensors["responses"] = video
    return [
        SimpleNamespace(
            batch=tensors,
            non_tensor_batch={
                "audio_sample_rate": sample_rate,
                "fps": fps,
                "reward_inputs": {"text": {"audio": "audio prompt", "video": "video prompt"}},
            },
        )
    ]


def _video(values):
    return torch.stack([torch.full((3, 2, 2), value, dtype=torch.uint8) for value in values])


def test_audio_batch_overrides_extra_info_for_all_scorers():
    info = reward_utils.audio_info_from_batch(
        {"audio": torch.zeros(4), "audio_sample_rate": 8000},
        _batch(audio=torch.ones(4), sample_rate=16000),
        scorer="Audio",
    )
    waveform, rate = reward_utils.get_audio(info)
    assert waveform.tolist() == [1.0] * 4
    assert rate == 16000
    with pytest.raises(ValueError, match="exactly one sample"):
        reward_utils.audio_info_from_batch({}, _batch() * 2, scorer="Audio")


def test_hpsv3_and_videoalign_reject_out_of_range_float_pixels():
    frame = torch.full((3, 2, 2), 1.1)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        hpsv3_reward._frame_to_pil(frame)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        videoalign._sample_video(frame.unsqueeze(0).expand(2, -1, -1, -1), 24.0)


def test_desync_head_mask_compat_is_instance_scoped(monkeypatch):
    class ASTBase(torch.nn.Module):
        pass

    class Synchformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio = ASTBase()

    monkeypatch.setitem(
        sys.modules,
        "flow_grpo.audio_video_align.synchformer.hf_src.modeling_ast",
        SimpleNamespace(ASTPreTrainedModel=ASTBase),
    )
    model = Synchformer()
    untouched = ASTBase()
    desync._install_ast_head_mask_compat(model)
    assert callable(model.audio.get_head_mask)
    assert not hasattr(ASTBase, "get_head_mask")
    assert not hasattr(untouched, "get_head_mask")


@pytest.mark.asyncio
async def test_clap_native_audio_prompt_and_calibration():
    model = SimpleNamespace(
        infer=AsyncMock(
            return_value={
                "audio_embeddings": torch.tensor([[3.0, 4.0]]),
                "text_embeddings": torch.tensor([[0.0, 2.0]]),
            }
        )
    )
    result = await clap.compute_score(
        "test",
        None,
        "fallback",
        {},
        reward_model=model,
        batch=_batch(audio=torch.ones(2, 480)),
        prompt_key="audio",
        score_scale=2.0,
        score_offset=-0.5,
        score_max=1.0,
    )
    assert result == {"score": pytest.approx(1.0), "source_sample_rate": 48000}
    waveforms, prompts = model.infer.call_args.args
    assert prompts == ["audio prompt"]
    assert waveforms[0].shape == (480,)
    assert waveforms[0].tolist() == pytest.approx([1.0] * 480)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_batch_size, batch_sizes", [(1, [1, 1, 1]), (2, [2, 1]), (4, [3])])
async def test_hpsv3_frame_inference_preserves_order_and_batch_limit(max_batch_size, batch_sizes):
    async def infer(images, prompts):
        assert prompts == ["video prompt"] * len(images)
        return torch.tensor([[float(image.getpixel((0, 0))[0]), -1.0] for image in images])

    model = SimpleNamespace(infer=AsyncMock(side_effect=infer))
    images = [hpsv3_reward._frame_to_pil(frame) for frame in _video([5, 1, 3])]

    scores = await hpsv3_reward._infer_frame_scores(model, images, "video prompt", max_batch_size)

    assert scores == pytest.approx([5.0, 1.0, 3.0])
    assert [len(call.args[0]) for call in model.infer.await_args_list] == batch_sizes


@pytest.mark.asyncio
async def test_hpsv3_frame_inference_rejects_empty_input():
    model = SimpleNamespace(infer=AsyncMock())
    with pytest.raises(ValueError, match="at least one image frame"):
        await hpsv3_reward._infer_frame_scores(model, [], "video prompt", 2)
    model.infer.assert_not_awaited()


@pytest.mark.asyncio
async def test_hpsv3_native_frame_sampling_microbatches_and_top_scores():
    async def infer(images, prompts):
        assert prompts == ["video prompt"] * len(images)
        return torch.tensor([[float(image.getpixel((0, 0))[0]), -1.0] for image in images])

    model = SimpleNamespace(infer=AsyncMock(side_effect=infer))
    result = await hpsv3_reward.compute_score_hpsv3(
        "test",
        _video([1, 2, 3, 4, 5]),
        "fallback",
        {},
        reward_model=model,
        batch=_batch(),
        prompt_key="video",
        num_frames=3,
        max_batch_size=2,
        top_fraction=0.5,
        score_cap=4.0,
    )
    assert model.infer.await_count == 2
    assert result == {"score": pytest.approx(0.35), "hpsv3_raw": pytest.approx(3.5)}


@pytest.mark.parametrize("fraction, cap, expected", [(1.0, None, 3.0), (0.5, None, 4.0), (0.5, 4.0, 3.5)])
def test_hpsv3_frame_aggregation(fraction, cap, expected):
    assert hpsv3_reward._aggregate_frame_scores([1.0, 3.0, 5.0], fraction, cap) == pytest.approx(expected)


def test_hpsv3_default_and_uniform_frame_selection():
    video = _video(range(9))
    assert [im.getpixel((0, 0))[0] for im in hpsv3_reward._select_reward_frames(video, {}, None)] == [0, 4, 8]
    assert [im.getpixel((0, 0))[0] for im in hpsv3_reward._select_reward_frames(video, {}, 3)] == [0, 4, 8]


@pytest.mark.asyncio
async def test_audiobox_native_duration_weighted_windows():
    async def infer(windows, masks):
        assert windows.shape == (2, 1, 160000)
        assert masks.sum(dim=(1, 2)).tolist() == [160000, 80000]
        return {
            "predictions": {axis: torch.tensor([1.0, 3.0]) for axis in audiobox._AXES},
            "target_transform": {axis: (1.0, 2.0) for axis in audiobox._AXES},
        }

    result = await audiobox.compute_score(
        reward_model=SimpleNamespace(infer=AsyncMock(side_effect=infer)),
        batch=_batch(audio=torch.ones(1, 240000), sample_rate=16000),
    )
    # De-normalized windows score 3 and 7; CE+CU-PC+PQ gives 2x each.
    assert result["score"] == pytest.approx(((3.0 * 2 + 7.0) / 3) * 2 * 0.025)


def test_audiobox_windows_preserve_tail_and_mask_padding():
    windows, masks, weights = audiobox._make_windows([torch.ones(160003), torch.ones(2) * 2])
    assert windows.shape == (3, 1, 160000)
    assert weights == pytest.approx([1.0, 3 / 160000, 2 / 160000])
    assert masks.sum(dim=(1, 2)).tolist() == [160000, 3, 2]
    torch.testing.assert_close(windows[1, 0, :3], torch.ones(3))
    assert torch.count_nonzero(windows[1, 0, 3:]) == 0


@pytest.mark.asyncio
async def test_videoalign_native_sampling_prompt_and_calibration():
    model = SimpleNamespace(infer=AsyncMock(return_value=torch.tensor([[3.0, 5.0, 7.0]])))
    result = await videoalign.compute_score(
        reward_model=model,
        batch=_batch(video=_video(range(8)), fps=48),
        prompt_key="video",
        score_weights=[0.5, 0.0, 0.5],
        score_means=[1.0, 0.0, 3.0],
        score_stds=[2.0, 1.0, 2.0],
    )
    assert result["score"] == pytest.approx(1.5)
    videos, prompts = model.infer.call_args.args
    assert prompts == ["video prompt"]
    assert videos[0][:, 0, 0, 0].tolist() == [0, 2, 5, 7]
    assert videos[0].dtype == torch.uint8


@pytest.mark.parametrize("count, fps, expected", [(8, 48, [0, 2, 5, 7]), (2, 24, [0, 1])])
def test_videoalign_temporal_sampling(count, fps, expected):
    sampled, indices = videoalign._sample_video(_video(range(count)), fps)
    assert indices == expected
    assert sampled[:, 0, 0, 0].tolist() == expected


def test_desync_temporal_sampling_uses_frame_centers():
    sampled = desync._temporal_resample_video(_video(range(8)), 50.0)
    assert sampled[:, 0, 0, 0].tolist() == [0, 2, 4, 6]


@pytest.mark.asyncio
async def test_desync_native_preparation_and_offset_score():
    async def infer(video, audio):
        assert video.shape == (1, 200, 3, 224, 224)
        assert audio.shape == (1, 128000)
        assert torch.all(video[:, 2:] == -1)
        assert torch.count_nonzero(audio[:, 1600:]) == 0
        logits = torch.zeros(2, 1, 21)
        logits[0, 0, 10] = 1  # zero offset
        logits[1, 0, 15] = 1  # one-second offset
        return logits

    result = await desync.compute_score(
        reward_model=SimpleNamespace(infer=AsyncMock(side_effect=infer)),
        batch=_batch(video=_video([0, 255]), audio=torch.ones(1, 1600), sample_rate=16000, fps=25),
    )
    assert result["score"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", [videoalign, desync])
@pytest.mark.parametrize("fps", [0, float("nan")])
async def test_av_scorers_reject_invalid_fps_before_inference(scorer, fps):
    model = SimpleNamespace(infer=AsyncMock())
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        await scorer.compute_score(reward_model=model, solution_image=_video([0, 1]), fps=fps)
    model.infer.assert_not_awaited()


@pytest.mark.asyncio
async def test_videoalign_rejects_nonfinite_model_scores():
    model = SimpleNamespace(infer=AsyncMock(return_value=torch.tensor([[float("nan"), 0.0, 0.0]])))
    with pytest.raises(ValueError, match="score must be finite"):
        await videoalign.compute_score(reward_model=model, solution_image=_video([0, 1]), fps=24)


@pytest.mark.asyncio
async def test_audiobox_rejects_invalid_model_output():
    model = SimpleNamespace(infer=AsyncMock(return_value={"predictions": {}, "target_transform": {}}))
    with pytest.raises(ValueError, match="AudioBox CE output"):
        await audiobox.compute_score(reward_model=model, batch=_batch(audio=torch.ones(1, 8), sample_rate=16000))


@pytest.mark.asyncio
async def test_clap_native_resamples_to_model_rate_and_preserves_source_rate():
    model = SimpleNamespace(
        infer=AsyncMock(
            return_value={
                "audio_embeddings": torch.tensor([[1.0, 0.0]]),
                "text_embeddings": torch.tensor([[1.0, 0.0]]),
            }
        )
    )
    result = await clap.compute_score(
        "test",
        None,
        "prompt",
        {
            "audio": torch.ones(1, 160),
            "audio_sample_rate": 16000,
        },
        reward_model=model,
    )
    assert result == {"score": pytest.approx(1.0), "source_sample_rate": 16000}
    waveforms, prompts = model.infer.call_args.args
    assert waveforms[0].shape == (480,)
    assert prompts == ["prompt"]
