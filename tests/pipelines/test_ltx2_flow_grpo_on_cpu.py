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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXPromptContext
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_flow_grpo.agent_loop import _messages_to_text
from verl_omni.pipelines.ltx2_flow_grpo.common import (
    LTX2_LORA_TARGET_MODULES,
    apply_x0_cfg,
    calculate_shift,
)
from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase


def test_ltx2_reference_lora_targets_are_complete() -> None:
    assert len(LTX2_LORA_TARGET_MODULES) == 28
    assert len(set(LTX2_LORA_TARGET_MODULES)) == 28
    assert "audio_to_video_attn.to_q" in LTX2_LORA_TARGET_MODULES
    assert "video_to_audio_attn.to_q" in LTX2_LORA_TARGET_MODULES


def test_ltx2_checkpoint_architecture_registers_both_adapters() -> None:
    assert DiffusionModelBase.get_class_by_name("LTX2Pipeline", "flow_grpo") is LTX23FlowGRPO
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "flow_grpo") is LTX23PipelineWithLogProb


def test_ltx2_x0_cfg_and_resolution_dependent_shift() -> None:
    sample = torch.tensor([[[4.0]]])
    positive = torch.tensor([[[2.0]]])
    negative = torch.tensor([[[1.0]]])
    sigma = torch.tensor([[[0.5]]])
    assert torch.equal(apply_x0_cfg(sample, positive, negative, sigma, 4.0), torch.tensor([[[5.0]]]))
    assert calculate_shift(6144, 1024, 4096, 0.95, 2.05) > 2.05


def test_ltx2_raw_prompt_normalization() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "  jungle ambience  "}]}]
    assert _messages_to_text(messages) == "jungle ambience"


def test_ltx2_training_adapter_splits_joint_latents() -> None:
    batch_size = 2
    latents = torch.randn(batch_size, 3, 12, 128)
    timesteps = torch.tensor([[900.0, 700.0, 500.0]]).expand(batch_size, -1)
    prompt_embeds = torch.randn(batch_size, 4, 32)
    prompt_mask = torch.ones(batch_size, 4, dtype=torch.long)
    micro_batch = TensorDict(
        {
            "audio_prompt_embeds": torch.randn(batch_size, 4, 32),
            "video_seq_len": torch.full((batch_size,), 5),
            "all_next_latents": torch.randn_like(latents),
        },
        batch_size=[batch_size],
    )
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            num_frames=121,
            height=512,
            width=768,
            frame_rate=24.0,
            guidance_scale=1.0,
        )
    )

    positive, negative = LTX23FlowGRPO.prepare_model_inputs(
        module=None,
        model_config=config,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=1,
    )
    assert positive["hidden_states"].shape == (batch_size, 5, 128)
    assert positive["audio_hidden_states"].shape == (batch_size, 7, 128)
    assert positive["timestep"].tolist() == [700.0, 700.0]
    assert negative is None


def test_ltx2_non_contiguous_sde_step_selection_is_seeded() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline._flow_grpo_window_size = 3
    pipeline._flow_grpo_window_range = [0, 10]
    pipeline._flow_grpo_sde_contiguous = False
    pipeline._flow_grpo_seed = 42

    first = pipeline._select_sde_steps(24, torch.device("cpu"))
    second = pipeline._select_sde_steps(24, torch.device("cpu"))
    assert first == second
    assert len(first) == 3
    assert first == sorted(first)
    assert set(first).issubset(set(range(10)))
    assert first != list(range(first[0], first[0] + len(first)))


def test_ltx2_rollout_adapter_configure_flow_grpo() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            output_type="image",
            extra_args={
                "noise_level": 0.5,
                "sde_type": "cps",
                "sde_window_size": 4,
                "sde_window_range": [2, 12],
                "sde_contiguous": True,
                "logprobs": True,
                "sde_window_seed": 100,
                "global_steps": 5,
            },
        )
    )
    pipeline._configure_flow_grpo(req)
    assert req.sampling_params.output_type == "pt"
    assert pipeline._flow_grpo_noise_level == 0.5
    assert pipeline._flow_grpo_sde_type == "cps"
    assert pipeline._flow_grpo_window_size == 4
    assert pipeline._flow_grpo_window_range == [2, 12]
    assert pipeline._flow_grpo_sde_contiguous is True
    assert pipeline._flow_grpo_logprobs is True
    assert pipeline._flow_grpo_seed == 104


def test_ltx2_rollout_adapter_inject_precomputed_prompt_embeds() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline.tokenizer_max_length = 128
    pipeline.device = torch.device("cpu")

    def mock_encode_token_ids(token_ids, mask, max_seq_len):
        embeds = torch.full((1, max_seq_len, 64), float(len(token_ids)))
        attn_mask = torch.ones((1, max_seq_len), dtype=torch.long)
        return embeds, attn_mask

    pipeline._encode_token_ids = mock_encode_token_ids

    req = SimpleNamespace(
        prompt={
            "prompt_token_ids": [10, 20, 30],
            "prompt_mask": [1, 1, 1],
            "negative_prompt_ids": [40, 50],
            "negative_prompt_mask": [1, 1],
        },
        sampling_params=SimpleNamespace(max_sequence_length=128),
    )
    pipeline._inject_precomputed_prompt_embeds(req)
    assert isinstance(req.prompt, dict)
    assert "prompt_embeds" in req.prompt
    assert "prompt_attention_mask" in req.prompt
    assert "negative_prompt_embeds" in req.prompt
    assert "negative_prompt_attention_mask" in req.prompt
    assert req.prompt["prompt_embeds"].shape == (128, 64)
    assert req.prompt["negative_prompt_embeds"].shape == (128, 64)


def test_ltx2_rollout_denoise_step_collects_sde_trajectory() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_logprobs = True
    pipeline._selected_sde_steps = {0}
    pipeline._current_latents = []
    pipeline._next_latents = []
    pipeline._selected_timesteps = []
    pipeline._log_probs = []

    # Mock scheduler
    mock_scheduler = MagicMock()
    mock_scheduler.step.return_value = (
        torch.ones(1, 12, 32),  # stepped sample (5 video + 7 audio)
        torch.tensor([0.42]),  # logprob
        None,
        None,
    )
    pipeline.scheduler = mock_scheduler

    # Mock noise prediction and synchronization
    pipeline._predict_noise_for_step = MagicMock(return_value=(torch.zeros(1, 5, 32), torch.zeros(1, 7, 32)))
    pipeline._synchronize_guidance_parallel_step_output = MagicMock(side_effect=lambda latents, **kwargs: latents)

    state = LTXAVState(video=torch.randn(1, 5, 32), audio=torch.randn(1, 7, 32))
    forward_ctx = SimpleNamespace(
        request_inputs=SimpleNamespace(generator=None),
        guidance_parallel_ready=False,
    )
    denoise_ctx = SimpleNamespace(latents=None, audio_latents=None)

    next_state = pipeline._denoise_step(
        index=0,
        timestep=torch.tensor([900.0]),
        state=state,
        forward_ctx=forward_ctx,
        denoise_ctx=denoise_ctx,
    )

    assert next_state.video.shape == (1, 5, 32)
    assert next_state.audio.shape == (1, 7, 32)
    assert len(pipeline._current_latents) == 1
    assert len(pipeline._next_latents) == 1
    assert len(pipeline._selected_timesteps) == 1
    assert len(pipeline._log_probs) == 1
    assert pipeline._log_probs[0].item() == pytest.approx(0.42, rel=1e-5)


def test_ltx2_rollout_forward_attaches_trajectory_and_metadata() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline._configure_flow_grpo = MagicMock()
    pipeline._inject_precomputed_prompt_embeds = MagicMock()
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))

    prompt_context = LTXPromptContext(
        batch_size=1,
        connector_prompt_embeds=torch.randn(1, 10, 32),
        connector_audio_prompt_embeds=torch.randn(1, 10, 32),
        connector_attention_mask=torch.ones(1, 10),
        positive_connector_prompt_embeds=torch.randn(1, 10, 32),
        positive_connector_audio_prompt_embeds=torch.randn(1, 10, 32),
        positive_connector_attention_mask=torch.ones(1, 10),
        negative_connector_prompt_embeds=torch.randn(1, 10, 32),
        negative_connector_audio_prompt_embeds=torch.randn(1, 10, 32),
        negative_connector_attention_mask=torch.ones(1, 10),
    )
    pipeline._flow_grpo_prompt_context = prompt_context
    pipeline._flow_grpo_trajectory = {
        "all_latents": torch.randn(1, 3, 12, 32),
        "all_next_latents": torch.randn(1, 3, 12, 32),
        "all_timesteps": torch.tensor([[900.0, 700.0, 500.0]]),
        "all_log_probs": torch.tensor([[0.1, 0.2, 0.3]]),
        "video_seq_len": torch.tensor([5]),
    }

    req = MagicMock(spec=DiffusionRequestBatch)
    req.num_reqs = 1
    req.requests = [MagicMock()]

    # Mock super().forward
    video_tensor = torch.randn(1, 3, 16, 64, 64)
    audio_tensor = torch.randn(1, 2, 24000)
    mock_base_output = DiffusionOutput(output=(video_tensor, audio_tensor))

    # Test forward with monkeypatched super().forward
    with torch.no_grad():
        with unittest_mock_super_forward(pipeline, mock_base_output):
            output = pipeline.forward(req)

    assert isinstance(output, DiffusionOutput)
    assert output.trajectory_latents is not None
    assert output.trajectory_log_probs is not None
    assert output.trajectory_timesteps is not None

    envelope = output.output
    assert isinstance(envelope, dict)
    assert "payload" in envelope
    assert "metadata" in envelope
    metadata = envelope["metadata"]
    assert "prompt_embeddings" in metadata
    assert "rl" in metadata
    assert metadata["rl"]["video_seq_len"].item() == 5
    assert metadata["rl"]["audio_sample_rate"] == 24000
    assert "audio_prompt_embeds" in metadata["prompt_embeddings"]


def unittest_mock_super_forward(target, return_value):
    from unittest.mock import patch

    return patch.object(LTX23PipelineWithLogProb.__bases__[0], "forward", return_value=return_value)
