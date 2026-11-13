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

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

import verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter as wan22_adapter
from verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter import (
    Wan22DanceGRPOPipelineWithLogProb,
    _unwrap_single_request,
    _validate_wan_rollout_inputs,
)


def test_wan22_dance_grpo_unwraps_vllm_omni_request_batch() -> None:
    request = SimpleNamespace(prompt={}, sampling_params=SimpleNamespace())
    request_batch = DiffusionRequestBatch(requests=[request])

    assert _unwrap_single_request(request_batch) is request
    assert _unwrap_single_request(request) is request


def test_wan22_dance_grpo_rejects_packed_request_batch() -> None:
    request_batch = DiffusionRequestBatch(requests=[SimpleNamespace(), SimpleNamespace()])

    with pytest.raises(ValueError, match="expects one request, got 2"):
        _unwrap_single_request(request_batch)


def test_wan22_dance_grpo_validates_token_prompt_contract() -> None:
    _validate_wan_rollout_inputs(
        prompt_ids=torch.tensor([1, 2, 3]),
        negative_prompt_ids=None,
        height=704,
        width=1280,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )

    with pytest.raises(ValueError, match="both `prompt_ids` and `prompt_embeds`"):
        _validate_wan_rollout_inputs(
            prompt_ids=torch.tensor([1, 2, 3]),
            negative_prompt_ids=None,
            height=704,
            width=1280,
            prompt_embeds=torch.randn(1, 3, 8),
            negative_prompt_embeds=None,
        )


def test_wan22_dance_grpo_forward_accepts_vllm_omni_request_batch(monkeypatch) -> None:
    sampling_params = OmniDiffusionSamplingParams(
        height=64,
        width=96,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        output_type="latent",
        seed=123,
        max_sequence_length=4,
        extra_args={"global_steps": 7},
    )
    request = OmniDiffusionRequest(
        request_id="wan22-request",
        prompt={"prompt_token_ids": [11, 12, 13]},
        sampling_params=sampling_params,
    )
    request_batch = DiffusionRequestBatch(requests=[request])

    pipeline = object.__new__(Wan22DanceGRPOPipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2), in_channels=4)
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae_scale_factor_temporal = 4
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.transformer_2 = None
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline.boundary_ratio = None
    pipeline.expand_timesteps = False
    pipeline.scheduler = SimpleNamespace(
        set_timesteps=MagicMock(),
        timesteps=torch.tensor([900.0, 500.0]),
        config=SimpleNamespace(num_train_timesteps=1000),
    )

    prompt_embeds = torch.zeros(1, 4, 8)
    prompt_mask = torch.ones(1, 4, dtype=torch.long)
    pipeline.encode_prompt = MagicMock(return_value=(prompt_embeds, prompt_mask))
    latents = torch.zeros(1, 4, 2, 4, 6)
    pipeline.prepare_latents = MagicMock(return_value=latents)
    trajectory_latents = torch.zeros(1, 2, 4, 2, 4, 6)
    trajectory_log_probs = torch.zeros(1, 2)
    trajectory_timesteps = torch.tensor([[900.0, 500.0]])
    pipeline.diffuse = MagicMock(return_value=(latents, trajectory_latents, trajectory_log_probs, trajectory_timesteps))

    expected = DiffusionOutput(output=latents)
    rollout_output = MagicMock(return_value=expected)
    monkeypatch.setattr(wan22_adapter, "rollout_output", rollout_output)
    monkeypatch.setattr(wan22_adapter, "seed_from_prompt_ids", lambda **_: 321)
    monkeypatch.setattr(
        wan22_adapter,
        "current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )

    output = pipeline.forward(
        request_batch,
        height=128,
        width=128,
        frame_num=9,
        output_type="np",
    )

    assert output.output["metadata"]["media_artifacts"]["primary"] == "video_latent"
    torch.testing.assert_close(output.output["payload"]["video"][0]["video_latent"], latents[0])
    encoded_prompt_ids = pipeline.encode_prompt.call_args.kwargs["prompt_ids"]
    assert encoded_prompt_ids.tolist() == [11, 12, 13]
    prepare_kwargs = pipeline.prepare_latents.call_args.kwargs
    assert prepare_kwargs["height"] == 64
    assert prepare_kwargs["width"] == 96
    assert prepare_kwargs["num_frames"] == 5
    assert prepare_kwargs["generator"].initial_seed() == 321
    pipeline.diffuse.assert_called_once()
    rollout_output.assert_called_once_with(
        media=None,
        media_key="video",
        trajectory_latents=trajectory_latents,
        trajectory_log_probs=trajectory_log_probs,
        trajectory_timesteps=trajectory_timesteps,
        prompt_embeddings={
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_mask,
            "negative_prompt_embeds": None,
            "negative_prompt_embeds_mask": None,
        },
        to_cpu=True,
    )


def test_wan22_sde_window_uses_the_training_step_not_the_sample_seed(monkeypatch) -> None:
    recorded: list[int | None] = []
    real_randint = torch.randint

    def _spy(*args, **kwargs):
        generator = kwargs.get("generator")
        recorded.append(None if generator is None else int(generator.initial_seed()))
        return real_randint(*args, **kwargs)

    monkeypatch.setattr(wan22_adapter.torch, "randint", _spy)
    monkeypatch.setattr(wan22_adapter, "seed_from_prompt_ids", lambda **_: 321)
    monkeypatch.setattr(wan22_adapter, "current_omni_platform", SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        wan22_adapter, "rollout_output", MagicMock(side_effect=lambda **_: DiffusionOutput(output=None))
    )

    def _run(sample_seed: int, step: int) -> None:
        sampling_params = OmniDiffusionSamplingParams(
            height=64,
            width=96,
            num_frames=5,
            num_inference_steps=2,
            guidance_scale=1.0,
            output_type="latent",
            seed=sample_seed,
            max_sequence_length=4,
            extra_args={
                "sde_window_seed": 0,
                "global_steps": step,
                "sde_window_size": 2,
                "sde_window_range": [0, 5],
            },
        )
        request = OmniDiffusionRequest(
            request_id="wan22-window",
            prompt={"prompt_token_ids": [11, 12, 13]},
            sampling_params=sampling_params,
        )
        pipeline = object.__new__(Wan22DanceGRPOPipelineWithLogProb)
        pipeline.device = torch.device("cpu")
        pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2), in_channels=4)
        pipeline.vae_scale_factor_spatial = 8
        pipeline.vae_scale_factor_temporal = 4
        pipeline.transformer = SimpleNamespace(dtype=torch.float32)
        pipeline.transformer_2 = None
        pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
        pipeline.boundary_ratio = None
        pipeline.expand_timesteps = False
        pipeline.scheduler = SimpleNamespace(
            set_timesteps=MagicMock(),
            timesteps=torch.tensor([900.0, 500.0]),
            config=SimpleNamespace(num_train_timesteps=1000),
        )
        latents = torch.zeros(1, 4, 2, 4, 6)
        pipeline.encode_prompt = MagicMock(return_value=(torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.long)))
        pipeline.prepare_latents = MagicMock(return_value=latents)
        pipeline.diffuse = MagicMock(
            return_value=(latents, latents.unsqueeze(1), torch.zeros(1, 1), torch.tensor([[900.0]]))
        )
        pipeline.forward(DiffusionRequestBatch(requests=[request]))

    _run(123, 7)
    _run(999, 7)
    _run(5, 1)
    _run(6, 4)
    assert recorded == [7, 7, 1, 4]
