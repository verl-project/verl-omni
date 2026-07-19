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

import torch
from tensordict import TensorDict

from verl_omni.pipelines.ltx2_flow_grpo.agent_loop import _messages_to_text
from verl_omni.pipelines.ltx2_flow_grpo.common import (
    LTX2_LORA_TARGET_MODULES,
    apply_x0_cfg,
    calculate_shift,
)
from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.utils.reward_score.ltx2_clap import _get_audio
from verl_omni.utils.reward_score.ltx2_imagebind import _to_tchw
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


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


def test_ltx2_raw_prompt_and_reward_media_normalization() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "  jungle ambience  "}]}]
    assert _messages_to_text(messages) == "jungle ambience"

    audio, sample_rate = _get_audio({"audio": torch.ones(1, 2, 16), "audio_sample_rate": torch.tensor(48_000)})
    assert audio.shape == (16,)
    assert sample_rate == 48_000

    thwc = torch.zeros(3, 8, 10, 3)
    assert _to_tchw(thwc).shape == (3, 3, 8, 10)


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
    pipeline._flow_grpo_sde_steps = list(range(10))
    pipeline._flow_grpo_num_sde_steps = 3
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_seed = 42

    first = pipeline._select_sde_steps(24, torch.device("cpu"))
    second = pipeline._select_sde_steps(24, torch.device("cpu"))
    assert first == second
    assert len(first) == 3
    assert first == sorted(first)
    assert set(first).issubset(set(range(10)))


def test_vllm_omni_server_forwards_audio_for_rewards() -> None:
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = False
    server.global_steps = 7
    final_result = SimpleNamespace(
        images=[torch.zeros(3, 3, 8, 8)],
        custom_output={"all_latents": torch.ones(1, 2, 4, 8)},
        multimodal_output={
            "audio": torch.ones(1, 1, 32),
            "audio_sample_rate": 48_000,
        },
        request_output=None,
    )

    output = server._process_output(final_result, params=None, sampling_params={"logprobs": False})
    assert output.extra_fields["audio"].shape == (1, 32)
    assert output.extra_fields["audio_sample_rate"] == 48_000
    assert output.extra_fields["global_steps"] == 7
