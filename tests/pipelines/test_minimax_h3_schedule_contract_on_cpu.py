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
"""Contract tests that pin MiniMax H3 FlowGRPO's step count to vLLM-Omni's.

Whether a sigma schedule holds ``num_steps`` or ``num_steps + 1`` boundaries is an upstream
convention that changed between vLLM-Omni releases. A rollout that derives its step count from
``num_steps`` then stops short of sigma 0 and returns partly noisy video. These tests compare our
rollout loop against upstream's own ``minimax_h3_denoise_loop`` so such a drift fails on CPU.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    H3_AUDIO_SHIFT,
    H3_AUDIO_WIDTH,
    H3_VIDEO_SHIFT,
    H3_VIDEO_WIDTH,
    h3_sigma_schedules,
    h3_transition_count,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import MiniMaxH3PipelineWithLogProb

STEP_COUNTS = [2, 5, 10, 40]
VIDEO_ROWS = 2
AUDIO_ROWS = 3


class _FakeBranch:
    """Minimal denoise branch shared by our rollout and upstream's loop."""

    def __init__(self, **kwargs):
        del kwargs
        self.img_pos = torch.arange(VIDEO_ROWS)
        self.audio_pos = torch.arange(VIDEO_ROWS, VIDEO_ROWS + AUDIO_ROWS)
        self.update_mask = torch.ones(VIDEO_ROWS, dtype=torch.bool)
        self.update_mask_dev = self.update_mask
        self.audio_update_mask = torch.ones(AUDIO_ROWS, dtype=torch.bool)
        self.audio_update_mask_dev = self.audio_update_mask
        self.locked_audio_rows = None

    def forward_kwargs(self, *, video_rows, audio_rows, **kwargs):
        del kwargs
        return {"hidden_states": video_rows, "audio_hidden_states": audio_rows}


class _CountingModel:
    """Zero-velocity DiT stand-in that counts evaluations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, **model_inputs):
        self.calls += 1
        return torch.zeros_like(model_inputs["hidden_states"]), torch.zeros_like(model_inputs["audio_hidden_states"])


def _upstream_schedule(num_steps: int) -> tuple[list[float], list[float]]:
    return (
        minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=H3_VIDEO_SHIFT),
        minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=H3_AUDIO_SHIFT),
    )


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
def test_schedule_runs_from_one_to_zero_and_matches_upstream(num_steps: int) -> None:
    video, audio = h3_sigma_schedules(num_steps)
    upstream_video, upstream_audio = _upstream_schedule(num_steps)

    assert video == upstream_video
    assert audio == upstream_audio
    assert video[0] == pytest.approx(1.0)
    assert video[-1] == pytest.approx(0.0)
    assert h3_transition_count(video, audio) == len(video) - 1


@pytest.mark.parametrize(
    ("video", "audio", "message"),
    [
        ([1.0, 0.5, 0.0], [1.0, 0.5], "differ in length"),
        ([1.0], [1.0], "at least two"),
        ([1.0, 0.5, 0.2], [1.0, 0.5, 0.0], "1.0 to 0.0"),
        ([0.9, 0.5, 0.0], [1.0, 0.5, 0.0], "1.0 to 0.0"),
        ([1.0, 0.5, 0.5, 0.0], [1.0, 0.5, 0.2, 0.0], "strictly decreasing"),
    ],
)
def test_transition_count_rejects_malformed_schedules(video, audio, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        h3_transition_count(video, audio)


def _rollout_pipeline() -> MiniMaxH3PipelineWithLogProb:
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_sde_contiguous = True
    pipeline._flow_grpo_seed = 123
    pipeline._h3_max_text_len = 2
    pipeline._initial_noise = MagicMock(
        return_value=(torch.zeros(VIDEO_ROWS, H3_VIDEO_WIDTH), torch.zeros(AUDIO_ROWS, H3_AUDIO_WIDTH))
    )
    pipeline.record_denoise_step = MagicMock()
    pipeline.progress_bar = lambda total: nullcontext(SimpleNamespace(update=MagicMock()))
    pipeline._resident_dit_layers_on_device = lambda enabled: nullcontext()
    pipeline._layout_outputs = MagicMock(return_value={})
    pipeline.transformer = _CountingModel()
    pipeline._transformer_for_task = MagicMock(return_value=pipeline.transformer)
    return pipeline


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
def test_rollout_matches_upstream_loop_and_reaches_sigma_zero(monkeypatch, num_steps: int) -> None:
    module = "verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter"
    packed = {
        "token_tags": torch.zeros(VIDEO_ROWS + AUDIO_ROWS + 2, dtype=torch.long),
        "text_pos": torch.tensor([5, 6]),
    }
    boundaries: dict[str, list[tuple[float, float]]] = {"video": [], "audio": []}

    def recording_transition(scheduler, sample, _velocity, step, **kwargs):
        modality = "video" if sample.shape[-1] == H3_VIDEO_WIDTH else "audio"
        boundaries[modality].append((float(scheduler.sigmas[step]), float(scheduler.sigmas[step + 1])))
        log_prob = torch.zeros(1) if kwargs["return_log_prob"] else None
        return sample, log_prob, sample, torch.tensor(0.0), torch.tensor(0.0)

    monkeypatch.setattr(f"{module}.minimax_h3_packed_sequence", lambda **kwargs: packed)
    monkeypatch.setattr(f"{module}.MiniMaxH3DenoiseBranch", _FakeBranch)
    monkeypatch.setattr(f"{module}.sample_h3_transition", recording_transition)
    monkeypatch.setattr(f"{module}.minimax_h3_unpatchify_video_tokens", lambda *args, **kwargs: torch.zeros(1))
    monkeypatch.setattr(f"{module}.minimax_h3_unpack_audio_tokens", lambda *args, **kwargs: torch.zeros(1))

    pipeline = _rollout_pipeline()
    pipeline.diffuse(
        task="t2va",
        text_embeddings=torch.randn(2, 8),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=7,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        num_frames=1,
        num_steps=num_steps,
        video_shift=H3_VIDEO_SHIFT,
        audio_shift=H3_AUDIO_SHIFT,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
    )

    upstream_model = _CountingModel()
    video_sigmas, audio_sigmas = _upstream_schedule(num_steps)
    minimax_h3_denoise_loop(
        model=upstream_model,
        positive=_FakeBranch(),
        initial_video_rows=torch.zeros(VIDEO_ROWS, H3_VIDEO_WIDTH),
        initial_audio_rows=torch.zeros(AUDIO_ROWS, H3_AUDIO_WIDTH),
        keyframe_cond_rows=None,
        sigmas_video=video_sigmas,
        sigmas_audio=audio_sigmas,
        device=torch.device("cpu"),
    )

    # Upstream's own loop defines how many denoiser evaluations a schedule means.
    assert upstream_model.calls == len(video_sigmas) - 1
    assert pipeline.transformer.calls == upstream_model.calls
    for modality in ("video", "audio"):
        assert len(boundaries[modality]) == upstream_model.calls
        # The walk ends exactly on sigma 0; stopping one boundary early is the failure this guards against.
        assert boundaries[modality][-1][1] == pytest.approx(0.0)
        assert boundaries[modality][0][0] == pytest.approx(1.0)
