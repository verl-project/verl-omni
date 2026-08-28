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

"""CPU coverage for the Boogu-Image rollout adapter.

Guards the numerical conventions the rollout shares with training: the velocity
negation and text CFG that reach ``scheduler.step``, the ``t = 1 - sigma``
timestep mapping, the float32 sample the SDE step is entitled to, and the
SDE-window collection contract. A break in any of these keeps generating
images — it only makes the rollout log-probs disagree with the training ones —
so an e2e run cannot catch it. Training-side conventions live in
``test_boogu_image_adapters_on_cpu.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

NUM_TRAIN_TIMESTEPS = 1000


class _RecordingScheduler:
    """Stand-in for FlowMatchSDEDiscreteScheduler that records every step call."""

    def __init__(self, step_dtype=torch.float32, noise_coupled=False) -> None:
        self.config = SimpleNamespace(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
        self.begin_index = None
        self.step_dtype = step_dtype
        # Fold noise_level into the step so a packed row's trajectory actually
        # depends on the window it was given, not just on the step index.
        self.noise_coupled = noise_coupled
        self.calls: list[SimpleNamespace] = []

    def set_begin_index(self, index) -> None:
        self.begin_index = index

    def step(self, model_output, timestep, sample, **kwargs):
        self.calls.append(
            SimpleNamespace(
                model_output=model_output.clone(),
                timestep=timestep,
                sample=sample.clone(),
                noise_level=kwargs["noise_level"],
                sde_type=kwargs["sde_type"],
            )
        )
        log_prob = torch.full((sample.shape[0],), float(len(self.calls))) if kwargs["return_logprobs"] else None
        stepped = sample + 1.0
        if self.noise_coupled:
            stepped = stepped + kwargs["noise_level"]
        return stepped.to(self.step_dtype), log_prob, None, None


def _make_pipeline(predictions, step_dtype=torch.float32, noise_coupled=False):
    """Build an uninitialised pipeline whose ``predict`` replays ``predictions``."""
    pipeline = object.__new__(BooguImagePipelineWithLogProb)
    pipeline.scheduler = _RecordingScheduler(step_dtype, noise_coupled)
    pipeline.predict_calls = []
    values = list(predictions)

    def _predict(boogu_t, x, embeds, freqs_cis, mask, ref_image_hidden_states):
        pipeline.predict_calls.append(
            SimpleNamespace(boogu_t=boogu_t, x=x, embeds=embeds, ref_image_hidden_states=ref_image_hidden_states)
        )
        return values[(len(pipeline.predict_calls) - 1) % len(values)]

    pipeline.predict = _predict
    return pipeline


def _levels(noise_level, batch=2):
    """Per-row noise levels, whether the loop emitted a shared float or a tensor."""
    if isinstance(noise_level, torch.Tensor):
        return noise_level.flatten().tolist()
    return [noise_level] * batch


def _diffuse(
    pipeline, *, timesteps, sde_window, guidance_scale=1.0, negative=False, noise_level=1.2, dtype=None, batch=2
):
    embeds = torch.zeros(batch, 5, 8, dtype=dtype or torch.float32)
    mask = torch.ones(batch, 5, dtype=torch.long)
    return BooguImagePipelineWithLogProb.diffuse(
        pipeline,
        prompt_embeds=embeds,
        prompt_embeds_mask=mask,
        negative_prompt_embeds=embeds if negative else None,
        negative_prompt_embeds_mask=mask if negative else None,
        latents=torch.zeros(batch, 3, 2, 2),
        freqs_cis="freqs",
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        noise_level=noise_level,
        sde_window=sde_window,
        sde_type="sde",
        generator=None,
        logprobs=True,
    )


# ------------------------------------------------------------------ registration


def test_rollout_registration_resolves_the_boogu_architecture():
    assert VllmOmniPipelineBase.get_class("BooguImagePipeline", "flow_grpo") is BooguImagePipelineWithLogProb


# ------------------------------------------------------- velocity / CFG numerics


def test_rollout_negates_the_predicted_velocity_for_the_scheduler():
    """Boogu predicts x0 - noise; the diffusers scheduler wants noise - x0."""
    pipeline = _make_pipeline([torch.full((2, 3, 2, 2), 3.0)])

    _diffuse(pipeline, timesteps=torch.tensor([900.0, 500.0]), sde_window=(0, 2))

    for call in pipeline.scheduler.calls:
        torch.testing.assert_close(call.model_output, torch.full((2, 3, 2, 2), -3.0))
        assert call.model_output.dtype is torch.float32
        assert call.sample.dtype is torch.float32


def test_rollout_applies_text_cfg_before_negating():
    """CFG combines the raw predictions; the negation happens once, afterwards."""
    positive = torch.full((2, 3, 2, 2), 2.0)
    negative = torch.full((2, 3, 2, 2), 1.0)
    pipeline = _make_pipeline([positive, negative])

    _diffuse(
        pipeline,
        timesteps=torch.tensor([900.0]),
        sde_window=(0, 1),
        guidance_scale=4.0,
        negative=True,
    )

    # 2 + (4 - 1) * (2 - 1) = 5, negated once.
    assert len(pipeline.predict_calls) == 2
    torch.testing.assert_close(pipeline.scheduler.calls[0].model_output, torch.full((2, 3, 2, 2), -5.0))


def test_rollout_skips_the_negative_forward_without_cfg():
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])

    _diffuse(pipeline, timesteps=torch.tensor([900.0]), sde_window=(0, 1), guidance_scale=1.0, negative=True)

    assert len(pipeline.predict_calls) == 1


# --------------------------------------------------------------- time mapping


def test_rollout_feeds_boogu_time_to_the_transformer():
    """The transformer consumes t = 1 - sigma, not the scheduler's raw timestep."""
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])

    _diffuse(pipeline, timesteps=torch.tensor([750.0, 250.0]), sde_window=(0, 2))

    assert [call.boogu_t.item() for call in pipeline.predict_calls] == pytest.approx([0.25, 0.75])
    # The scheduler still receives the untouched native timestep.
    assert [call.timestep.item() for call in pipeline.scheduler.calls] == pytest.approx([750.0, 250.0])


# ------------------------------------------------------------- float32 discipline


def test_rollout_steps_in_float32_under_a_bf16_model():
    """Model dtype stays inside the transformer forward; the SDE step gets float32.

    Stacking promotes the collected trajectory back to float32 on its own, so
    the contract worth pinning is the step input, not the per-entry cast.
    """
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2, dtype=torch.bfloat16)], step_dtype=torch.bfloat16)

    _, all_latents, _, _ = _diffuse(
        pipeline, timesteps=torch.tensor([900.0, 500.0]), sde_window=(0, 2), dtype=torch.bfloat16
    )

    # Only the transformer forward sees the model dtype.
    assert pipeline.predict_calls[0].x.dtype is torch.bfloat16
    assert pipeline.scheduler.calls[0].sample.dtype is torch.float32
    assert pipeline.scheduler.calls[1].sample.dtype is torch.float32
    assert all_latents.dtype is torch.float32


# ------------------------------------------------------------ SDE window contract


def test_rollout_collects_one_more_latent_than_log_probs():
    """all_latents holds the window's entry state plus one post-step latent each."""
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])
    timesteps = torch.tensor([900.0, 800.0, 700.0, 600.0, 500.0, 400.0])

    _, all_latents, all_log_probs, all_timesteps = _diffuse(pipeline, timesteps=timesteps, sde_window=(1, 4))

    assert all_latents.shape == (2, 4, 3, 2, 2)
    assert all_log_probs.shape == (2, 3)
    assert all_timesteps.shape == (2, 3)
    # The recorded latents are contiguous across the window boundary: the step
    # that opens the window contributes its input, every step its output.
    assert [all_latents[0, k].flatten()[0].item() for k in range(4)] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(all_timesteps[0], torch.tensor([800.0, 700.0, 600.0]))


def test_rollout_applies_noise_only_inside_the_window():
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])
    timesteps = torch.tensor([900.0, 800.0, 700.0, 600.0, 500.0, 400.0])

    _diffuse(pipeline, timesteps=timesteps, sde_window=(1, 4), noise_level=1.2)

    assert [call.noise_level for call in pipeline.scheduler.calls] == [0.0, 1.2, 1.2, 1.2, 0.0, 0.0]


def test_rollout_degenerate_window_reports_no_trajectory():
    """The engine warm-up denoises outside any window; callers get None fields."""
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])

    latents, all_latents, all_log_probs, all_timesteps = _diffuse(
        pipeline, timesteps=torch.tensor([900.0]), sde_window=(0, 0)
    )

    assert (all_latents, all_log_probs, all_timesteps) == (None, None, None)
    assert latents.shape == (2, 3, 2, 2)


# ------------------------------------------------------- packed request batching


def test_rollout_declares_request_batching():
    """Without the flag the engine clamps max_num_seqs to 1 and serialises rollout."""
    assert BooguImagePipelineWithLogProb.supports_request_batch is True


def test_rollout_noises_each_packed_row_over_its_own_window():
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])
    timesteps = torch.tensor([900.0, 800.0, 700.0, 600.0, 500.0])

    _diffuse(pipeline, timesteps=timesteps, sde_window=[(0, 2), (2, 4)], noise_level=1.2)

    levels = [call.noise_level for call in pipeline.scheduler.calls]

    # A step where the rows disagree carries a per-row tensor; where they agree
    # the loop keeps the cheaper shared float.
    assert [isinstance(level, torch.Tensor) for level in levels] == [True, True, True, True, False]
    assert [_levels(level) for level in levels] == [
        pytest.approx([1.2, 0.0]),
        pytest.approx([1.2, 0.0]),
        pytest.approx([0.0, 1.2]),
        pytest.approx([0.0, 1.2]),
        pytest.approx([0.0, 0.0]),
    ]


def test_rollout_collects_each_packed_row_over_its_own_window():
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])
    timesteps = torch.tensor([900.0, 800.0, 700.0, 600.0, 500.0])

    _, all_latents, all_log_probs, all_timesteps = _diffuse(pipeline, timesteps=timesteps, sde_window=[(0, 2), (2, 4)])

    assert all_latents.shape == (2, 3, 3, 2, 2)
    torch.testing.assert_close(all_timesteps[0], torch.tensor([900.0, 800.0]))
    torch.testing.assert_close(all_timesteps[1], torch.tensor([700.0, 600.0]))
    # Each row records the latent it entered its own window with, then one per step.
    assert [all_latents[0, k].flatten()[0].item() for k in range(3)] == pytest.approx([0.0, 1.0, 2.0])
    assert [all_latents[1, k].flatten()[0].item() for k in range(3)] == pytest.approx([2.0, 3.0, 4.0])
    # Log-probs are indexed per row, so each keeps the steps inside its window.
    torch.testing.assert_close(all_log_probs, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_packed_rows_match_the_same_windows_run_alone():
    """Packing must not move a row's trajectory: the training-side recompute
    replays these latents, so a shift here silently breaks the log-prob ratio."""
    timesteps = torch.tensor([900.0, 800.0, 700.0, 600.0, 500.0])
    windows = [(0, 2), (2, 4)]

    packed = _make_pipeline([torch.zeros(2, 3, 2, 2)], noise_coupled=True)
    _, packed_latents, packed_log_probs, packed_timesteps = _diffuse(packed, timesteps=timesteps, sde_window=windows)

    for row, window in enumerate(windows):
        alone = _make_pipeline([torch.zeros(1, 3, 2, 2)], noise_coupled=True)
        _, latents, log_probs, row_timesteps = _diffuse(alone, timesteps=timesteps, sde_window=window, batch=1)
        torch.testing.assert_close(packed_latents[row : row + 1], latents)
        torch.testing.assert_close(packed_log_probs[row : row + 1], log_probs)
        torch.testing.assert_close(packed_timesteps[row : row + 1], row_timesteps)


def test_rollout_rejects_one_window_per_row_mismatch():
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])

    with pytest.raises(ValueError, match="SDE windows"):
        _diffuse(pipeline, timesteps=torch.tensor([900.0, 800.0]), sde_window=[(0, 1)])


def test_rollout_rejects_packed_windows_of_different_sizes():
    """Ragged windows would leave the rows with trajectories that cannot stack."""
    pipeline = _make_pipeline([torch.zeros(2, 3, 2, 2)])

    with pytest.raises(ValueError, match="same size"):
        _diffuse(pipeline, timesteps=torch.tensor([900.0, 800.0, 700.0]), sde_window=[(0, 1), (1, 3)])


# --------------------------------------------------------------- prompt plumbing


def test_extract_prompt_ids_reads_the_pretokenised_payload():
    pipeline = object.__new__(BooguImagePipelineWithLogProb)
    prompts = [
        {
            "prompt_token_ids": [1, 2, 3],
            "prompt_mask": [1, 1, 1],
            "negative_prompt_ids": [4, 5],
            "negative_prompt_mask": [1, 1],
        }
    ]

    assert pipeline._extract_prompt_ids(prompts) == ([1, 2, 3], [1, 1, 1], [4, 5], [1, 1])


@pytest.mark.parametrize("prompts", [[{"prompt": "a cat"}], ["a cat"]])
def test_extract_prompt_ids_falls_back_to_tokenising_raw_text(prompts):
    pipeline = object.__new__(BooguImagePipelineWithLogProb)
    pipeline._tokenize_text_prompt = lambda text: (f"ids:{text}", f"mask:{text}")

    prompt_ids, prompt_mask, _, _ = pipeline._extract_prompt_ids(prompts)

    assert (prompt_ids, prompt_mask) == ("ids:a cat", "mask:a cat")


def test_extract_prompt_ids_returns_nothing_without_prompts():
    pipeline = object.__new__(BooguImagePipelineWithLogProb)

    assert pipeline._extract_prompt_ids([]) == (None, None, None, None)


def test_collate_prompt_batch_pads_every_request_into_one_grid():
    pipeline = object.__new__(BooguImagePipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    prompts = [
        {
            "prompt_token_ids": [1, 2, 3],
            "prompt_mask": [1, 1, 1],
            "negative_prompt_ids": [7, 8],
            "negative_prompt_mask": [1, 1],
        },
        {
            "prompt_token_ids": [4, 5],
            "prompt_mask": [1, 1],
            "negative_prompt_ids": [7, 8],
            "negative_prompt_mask": [1, 1],
        },
    ]

    prompt_ids, prompt_mask, negative_ids, negative_mask = pipeline._collate_prompt_batch(prompts)

    torch.testing.assert_close(prompt_ids, torch.tensor([[1, 2, 3], [4, 5, 0]]))
    torch.testing.assert_close(prompt_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
    torch.testing.assert_close(negative_ids, torch.tensor([[7, 8], [7, 8]]))
    torch.testing.assert_close(negative_mask, torch.tensor([[1, 1], [1, 1]]))
    # The Qwen3VL encoder and the Boogu transformer read a long mask, not a bool.
    assert prompt_mask.dtype == torch.long


def test_collate_prompt_batch_falls_back_to_raw_text():
    """The engine warm-up sends an untokenised prompt, and never sends it packed."""
    pipeline = object.__new__(BooguImagePipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._tokenize_text_prompt = lambda text: (f"ids:{text}", f"mask:{text}")

    prompt_ids, prompt_mask, _, _ = pipeline._collate_prompt_batch([{"prompt": "a cat"}])

    assert (prompt_ids, prompt_mask) == ("ids:a cat", "mask:a cat")
