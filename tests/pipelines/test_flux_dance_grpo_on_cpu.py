# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from verl_omni.pipelines.flux_dance_grpo.common import select_dance_grpo_transitions
from verl_omni.pipelines.flux_dance_grpo.diffusers_training_adapter import FluxDanceGRPO
from verl_omni.pipelines.flux_dance_grpo.vllm_omni_rollout_adapter import (
    FluxDanceGRPOPipelineWithLogProb,
    _extract_extra_prompt_ids,
    _normalize_window_range,
    _pad_token_ids,
    _sample_sde_windows,
)


class _PipelineConfig(dict):
    """Minimal Hydra-like mapping with attribute access for adapter tests."""

    __getattr__ = dict.__getitem__


def test_select_transitions_drops_last_and_keeps_gathered_fields_aligned() -> None:
    current = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    next_latents = current + 100
    timesteps = current.squeeze(-1) + 200
    log_probs = current.squeeze(-1) + 300
    generators = [torch.Generator().manual_seed(7), torch.Generator().manual_seed(11)]

    selected_current, selected_next, selected_timesteps, selected_log_probs = select_dance_grpo_transitions(
        current,
        next_latents,
        timesteps,
        log_probs,
        strategy="random_subset",
        fraction=0.5,
        drop_last=True,
        generator=generators,
    )

    assert selected_current.shape == (2, 2, 1)
    torch.testing.assert_close(selected_next, selected_current + 100)
    torch.testing.assert_close(selected_timesteps, selected_current.squeeze(-1) + 200)
    torch.testing.assert_close(selected_log_probs, selected_current.squeeze(-1) + 300)
    assert 4 not in selected_current[0, :, 0].tolist()
    assert 9 not in selected_current[1, :, 0].tolist()


def test_select_transitions_validates_fraction_and_generator_count() -> None:
    latents = torch.zeros(2, 3, 1)
    timesteps = torch.zeros(2, 3)
    log_probs = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="fraction must be in"):
        select_dance_grpo_transitions(latents, latents, timesteps, log_probs, fraction=0.0)
    with pytest.raises(ValueError, match="expected 2 transition generators"):
        select_dance_grpo_transitions(
            latents,
            latents,
            timesteps,
            log_probs,
            strategy="random_subset",
            generator=[torch.Generator()],
        )


def test_normalize_and_sample_sde_windows() -> None:
    assert _normalize_window_range(None, 16) == (0, 16)
    assert _normalize_window_range("[2, 12]", 16) == (2, 12)
    assert _sample_sde_windows(
        window_size=None,
        window_range=None,
        num_steps=16,
        batch_size=2,
        generators=None,
        device=torch.device("cpu"),
    ) == [(0, 16), (0, 16)]

    generators = [torch.Generator().manual_seed(3), torch.Generator().manual_seed(5)]
    windows = _sample_sde_windows(
        window_size=4,
        window_range=(2, 12),
        num_steps=16,
        batch_size=2,
        generators=generators,
        device=torch.device("cpu"),
    )
    assert all(2 <= start < end <= 12 and end - start == 4 for start, end in windows)

    with pytest.raises(ValueError, match="invalid sde_window_size"):
        _sample_sde_windows(
            window_size=11,
            window_range=(2, 12),
            num_steps=16,
            batch_size=1,
            generators=None,
            device=torch.device("cpu"),
        )


def test_pad_token_ids_truncates_and_pads() -> None:
    padded = _pad_token_ids(
        [[1, 2], [3, 4, 5, 6]],
        max_length=3,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert padded.dtype == torch.long
    assert padded.tolist() == [[1, 2, 0], [3, 4, 5]]


def test_extra_prompt_ids_require_both_configured_tokenizers() -> None:
    with pytest.raises(ValueError, match=r"extra_tokenizers\.clip and \.t5"):
        _extract_extra_prompt_ids([{"extra_prompt_ids": {"clip": [1, 2]}}])
    with pytest.raises(TypeError, match=r"extra_prompt_ids\['t5'\]"):
        _extract_extra_prompt_ids([{"extra_prompt_ids": {"clip": [1, 2], "t5": "invalid"}}])


def test_rollout_and_actor_use_identical_non_dyadic_sigma_schedule() -> None:
    rollout = object.__new__(FluxDanceGRPOPipelineWithLogProb)
    rollout.device = torch.device("cpu")
    rollout.scheduler = SimpleNamespace(set_timesteps=MagicMock(), timesteps=torch.empty(0))
    FluxDanceGRPOPipelineWithLogProb._set_timesteps(rollout, num_steps=20, shift=3.0)
    rollout_sigmas = rollout.scheduler.set_timesteps.call_args.kwargs["sigmas"]

    actor_scheduler = SimpleNamespace(set_timesteps=MagicMock())
    model_config = SimpleNamespace(pipeline=_PipelineConfig(num_inference_steps=20, shift=3.0))
    FluxDanceGRPO.set_timesteps(actor_scheduler, model_config, "cpu")
    actor_sigmas = actor_scheduler.set_timesteps.call_args.kwargs["sigmas"]

    np.testing.assert_array_equal(rollout_sigmas, actor_sigmas)


class _GuidanceModule:
    config = SimpleNamespace(guidance_embeds=True)

    @staticmethod
    def parameters():
        return iter(())


def test_actor_missing_guidance_scale_defaults_to_one() -> None:
    latents = torch.zeros(2, 1, 3, 4)
    timesteps = torch.full((2, 1), 500.0)
    prompt_embeds = torch.zeros(2, 5, 4)
    micro_batch = {
        "text_ids": torch.zeros(5, 3),
        "image_ids": torch.zeros(3, 3),
        "pooled_prompt_embeds": torch.zeros(2, 4),
    }

    model_inputs, negative_inputs = FluxDanceGRPO.prepare_model_inputs(
        _GuidanceModule(),
        SimpleNamespace(pipeline={}),
        latents,
        timesteps,
        prompt_embeds,
        torch.ones(2, 5),
        torch.empty(0),
        torch.empty(0),
        micro_batch,
        0,
    )

    assert negative_inputs is None
    torch.testing.assert_close(model_inputs["guidance"], torch.ones(2))
