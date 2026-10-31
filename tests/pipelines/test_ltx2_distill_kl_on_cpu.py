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
"""CPU tests for distilling LTX-2.3 joint audio-video transitions with distill_kl."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.ltx2_flow_grpo.common import set_ltx23_timesteps
from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.trainer.diffusion.diffusion_algos import DistillKLLoss

SCHEDULER_CONFIG = {
    "base_image_seq_len": 1024,
    "max_image_seq_len": 4096,
    "base_shift": 0.95,
    "max_shift": 2.05,
    "shift_terminal": None,
    "num_train_timesteps": 1000,
}

BATCH = 2
VIDEO_ROWS = 6
AUDIO_ROWS = 2
WIDTH = 4


class Transformer(torch.nn.Module):
    """Stand-in DiT whose velocity scale stands for a checkpoint's weights."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states, audio_hidden_states, **_kwargs):
        return hidden_states * self.scale, audio_hidden_states * self.scale


def build_scheduler():
    scheduler = FlowMatchSDEDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    set_ltx23_timesteps(scheduler, 4, torch.device("cpu"))
    return scheduler


def replay_transition(scale, video, audio, next_sample, timestep):
    """Run one LTX-2.3 transition the way the actor and the frozen teacher both do."""
    model_inputs = {
        "hidden_states": video,
        "audio_hidden_states": audio,
        "timestep": timestep[:, None].expand(-1, video.shape[1]),
        "audio_timestep": timestep[:, None].expand(-1, audio.shape[1]),
        "sigma": timestep,
        "audio_sigma": timestep,
        "num_frames": VIDEO_ROWS,
        "height": 1,
        "width": 1,
    }
    config = SimpleNamespace(
        pipeline=SimpleNamespace(guidance_scale=1.0),
        algo=SimpleNamespace(noise_level=0.8, sde_type="cps"),
    )
    return LTX23FlowGRPO.forward_and_sample_previous_step(
        Transformer(scale),
        build_scheduler(),
        config,
        model_inputs,
        None,
        {"all_next_latents": next_sample.unsqueeze(1), "all_timesteps": timestep.unsqueeze(1)},
        0,
    )


@pytest.fixture
def transition():
    torch.manual_seed(0)
    video = torch.randn(BATCH, VIDEO_ROWS, WIDTH)
    audio = torch.randn(BATCH, AUDIO_ROWS, WIDTH)
    timestep = build_scheduler().timesteps[:1].repeat(BATCH)
    next_sample = torch.randn(BATCH, VIDEO_ROWS + AUDIO_ROWS, WIDTH)
    return video, audio, next_sample, timestep


def test_student_transition_mean_spans_video_and_audio_rows(transition):
    video, audio, next_sample, timestep = transition

    _, prev_sample_mean, std_dev_t, _ = replay_transition(0.1, video, audio, next_sample, timestep)

    # the teacher scores this tensor as-is, so its layout is the distillation contract
    assert prev_sample_mean.shape == (BATCH, VIDEO_ROWS + AUDIO_ROWS, WIDTH)
    assert std_dev_t.shape == (BATCH, 1, 1)


def test_identical_checkpoints_give_zero_kl(transition):
    video, audio, next_sample, timestep = transition
    _, student_mean, std_dev_t, _ = replay_transition(0.1, video, audio, next_sample, timestep)

    loss, metrics = DistillKLLoss.compute_loss(
        prev_sample_mean=student_mean,
        teacher_prev_sample_mean=student_mean.clone(),
        std_dev_t=std_dev_t,
    )

    assert loss.item() == 0.0
    assert metrics["actor/distill_kl_loss"] == 0.0


def test_teacher_mean_is_independent_of_the_sampled_next_step(transition):
    """The teacher scores the student's state, not the step the student happened to sample."""
    video, audio, next_sample, timestep = transition

    _, mean, _, _ = replay_transition(0.4, video, audio, next_sample, timestep)
    _, other_mean, _, _ = replay_transition(0.4, video, audio, torch.randn_like(next_sample), timestep)

    torch.testing.assert_close(mean, other_mean)


def test_differing_checkpoints_give_positive_kl(transition):
    video, audio, next_sample, timestep = transition
    _, student_mean, std_dev_t, _ = replay_transition(0.1, video, audio, next_sample, timestep)
    _, teacher_mean, _, _ = replay_transition(0.4, video, audio, next_sample, timestep)

    loss, metrics = DistillKLLoss.compute_loss(
        prev_sample_mean=student_mean,
        teacher_prev_sample_mean=teacher_mean,
        std_dev_t=std_dev_t,
    )

    assert loss.item() > 0.0
    assert metrics["actor/distill_kl_loss"] == pytest.approx(loss.item())


def test_kl_is_monotone_in_the_teacher_gap(transition):
    video, audio, next_sample, timestep = transition
    _, student_mean, std_dev_t, _ = replay_transition(0.1, video, audio, next_sample, timestep)

    losses = []
    for scale in (0.2, 0.4, 0.8):
        _, teacher_mean, _, _ = replay_transition(scale, video, audio, next_sample, timestep)
        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=student_mean,
            teacher_prev_sample_mean=teacher_mean,
            std_dev_t=std_dev_t,
        )
        losses.append(loss.item())

    assert losses == sorted(losses)


def test_audio_rows_enter_the_loss(transition):
    """The KL reduces over the joint trajectory, so an audio-only gap is not silently dropped."""
    video, audio, next_sample, timestep = transition
    _, student_mean, std_dev_t, _ = replay_transition(0.1, video, audio, next_sample, timestep)

    teacher_mean = student_mean.clone()
    teacher_mean[:, VIDEO_ROWS:] += 1.0

    loss, _ = DistillKLLoss.compute_loss(
        prev_sample_mean=student_mean,
        teacher_prev_sample_mean=teacher_mean,
        std_dev_t=std_dev_t,
    )

    # video and audio are weighted by row count; there is no per-modality weighting
    expected = (torch.ones(BATCH, 1, 1) * AUDIO_ROWS / (VIDEO_ROWS + AUDIO_ROWS) / (2 * std_dev_t**2)).mean()
    assert loss.item() > 0.0
    torch.testing.assert_close(loss, expected)
