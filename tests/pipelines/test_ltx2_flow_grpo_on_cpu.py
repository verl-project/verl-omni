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

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from tensordict import TensorDict
from verl.utils.dataset.rl_dataset import get_dataset_class
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXPromptContext
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXPhaseResult, _official_ltx_sigmas
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTXPhaseRecipe
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_flow_grpo.agent_loop import LTX2DiffusionSingleTurnAgentLoop, _messages_to_text
from verl_omni.pipelines.ltx2_flow_grpo.common import (
    LTX2_LORA_TARGET_MODULES,
    apply_x0_cfg,
    calculate_shift,
    set_ltx23_timesteps,
)
from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb
from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.utils.dataset.rl_dataset import RLHFDataset, create_rl_dataset


def test_ltx2_reference_lora_targets_are_complete() -> None:
    assert len(LTX2_LORA_TARGET_MODULES) == 28
    assert len(set(LTX2_LORA_TARGET_MODULES)) == 28
    assert "audio_to_video_attn.to_q" in LTX2_LORA_TARGET_MODULES
    assert "video_to_audio_attn.to_q" in LTX2_LORA_TARGET_MODULES


def test_ltx2_checkpoint_architecture_registers_both_adapters() -> None:
    assert DiffusionModelBase.get_class_by_name("LTX2Pipeline", "flow_grpo") is LTX23FlowGRPO
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "flow_grpo") is LTX23PipelineWithLogProb
    assert issubclass(LTX23FlowGRPO, DiffusionI2IModelBase)


def test_ltx2_processor_files_use_text_tokenizer_path(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    result = LTX23FlowGRPO.prepare_processor_files(str(tmp_path))

    assert result == str(tokenizer_dir)


def test_ltx2_ti2va_default_dataset_forwards_image_without_hf_processor(tmp_path) -> None:
    image = Image.new("RGB", (56, 56), "red")
    image_path = tmp_path / "frame.png"
    image.save(image_path)
    data_path = tmp_path / "data.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "prompt": [{"role": "user", "content": "<image>Animate this frame."}],
                    "negative_prompt": [{"role": "user", "content": ""}],
                    "images": [str(image_path)],
                }
            ]
        )
    )
    data_config = OmegaConf.create({"filter_overlong_prompts": False})
    tokenizer = MagicMock(return_value={"input_ids": [1, 2, 3]})
    dataset = create_rl_dataset(str(data_path), data_config, tokenizer, processor=None)
    assert type(dataset) is RLHFDataset
    assert dataset.processor is None
    row = dataset[0]
    assert row["raw_negative_prompt"] == [{"role": "user", "content": ""}]
    server = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                diffusion_output=torch.zeros(1), log_probs=None, num_preempted=None, extra_fields={}
            )
        )
    )

    async def run():
        rollout = SimpleNamespace(prompt_length=128, enable_prompt_embed_cache=False)
        agent = LTX2DiffusionSingleTurnAgentLoop(
            SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout))),
            server_manager=server,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=get_dataset_class(data_config),
            data_config=SimpleNamespace(config=data_config),
        )
        await agent.run({}, **row)
        assert agent.processor is None

    asyncio.run(run())

    call = server.generate.await_args.kwargs
    assert len(call["image_data"]) == 1
    assert call["image_data"][0].tobytes() == image.tobytes()
    assert call["video_data"] is None
    assert call["audio_data"] is None
    assert call["prompt_ids"] == [1, 2, 3]
    assert call["negative_prompt_ids"] == [1, 2, 3]
    assert [call.args[0] for call in tokenizer.call_args_list] == ["Animate this frame.", ""]


def test_ltx2_x0_cfg_and_resolution_dependent_shift() -> None:
    sample = torch.tensor([[[4.0]]])
    positive = torch.tensor([[[2.0]]])
    negative = torch.tensor([[[1.0]]])
    sigma = torch.tensor([[[0.5]]])
    assert torch.equal(apply_x0_cfg(sample, positive, negative, sigma, 4.0), torch.tensor([[[5.0]]]))
    assert calculate_shift(6144, 1024, 4096, 0.95, 2.05) > 2.05


def test_ltx2_agent_loop_extracts_images_without_hf_processor() -> None:
    image = object()

    class Dataset:
        @classmethod
        async def process_multi_modal_info(cls, messages, image_patch_size, config):
            del cls, messages, image_patch_size, config
            return [image], None, None

    rollout = SimpleNamespace(prompt_length=128)
    agent = LTX2DiffusionSingleTurnAgentLoop(
        SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout))),
        server_manager=object(),
        tokenizer=MagicMock(),
        processor=None,
        dataset_cls=Dataset,
        data_config=SimpleNamespace(config={}),
    )

    media = asyncio.run(agent.process_multi_modal_info([{"role": "user", "content": []}]))

    assert agent.processor is None
    assert media == {"images": [image]}


def test_ltx2_agent_loop_keeps_image_out_of_text_tokens() -> None:
    calls = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return {"input_ids": [1, 2, 3]}

    async def run():
        agent = object.__new__(LTX2DiffusionSingleTurnAgentLoop)
        agent.tokenizer = Tokenizer()
        agent.rollout_config = SimpleNamespace(prompt_length=128)
        agent.loop = asyncio.get_running_loop()
        return await agent.ct_build_initial_tokens(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": object()},
                        {"type": "text", "text": "Animate this frame."},
                    ],
                }
            ],
            images=[object()],
        )

    assert asyncio.run(run()) == [1, 2, 3]
    assert calls[0][0] == "Animate this frame."


def test_ltx23_actor_schedule_matches_current_vllm_omni() -> None:
    config = {
        "base_image_seq_len": 1024,
        "max_image_seq_len": 4096,
        "base_shift": 0.95,
        "max_shift": 2.05,
        "shift_terminal": None,
        "num_train_timesteps": 1000,
    }
    scheduler = SimpleNamespace(config=config, sigmas=None, timesteps=None)

    set_ltx23_timesteps(scheduler, 4, torch.device("cpu"))

    expected = _official_ltx_sigmas(SimpleNamespace(config=config), 4, torch.device("cpu"))
    torch.testing.assert_close(scheduler.sigmas, expected)
    torch.testing.assert_close(scheduler.timesteps, expected[:-1] * 1000)


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
    assert positive["timestep"].tolist() == [[700.0] * 5] * batch_size
    assert positive["audio_timestep"].tolist() == [[700.0] * 7] * batch_size
    assert negative is None


def test_ltx2_training_adapter_reconstructs_first_frame_condition() -> None:
    batch_size = 2
    target_video = torch.randn(batch_size, 3, 4)
    audio = torch.randn(batch_size, 2, 4)
    condition = torch.randn(batch_size, 1, 4)
    latents = torch.cat([target_video, audio], dim=1).unsqueeze(1)
    timesteps = torch.full((batch_size, 1), 700.0)
    micro_batch = TensorDict(
        {
            "audio_prompt_embeds": torch.randn(batch_size, 4, 8),
            "video_seq_len": torch.full((batch_size,), 3),
            "all_next_latents": torch.randn_like(latents),
            "condition_image_latents": condition,
        },
        batch_size=[batch_size],
    )
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            num_frames=25,
            height=32,
            width=32,
            frame_rate=24.0,
            guidance_scale=1.0,
        )
    )

    positive, negative = LTX23FlowGRPO.prepare_model_inputs(
        module=None,
        model_config=config,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=torch.randn(batch_size, 4, 8),
        prompt_embeds_mask=torch.ones(batch_size, 4),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )
    prepared = LTX23FlowGRPO.prepare_condition(micro_batch, latents, 0)
    positive, negative = LTX23FlowGRPO.inject_condition(positive, negative, prepared)

    torch.testing.assert_close(positive["hidden_states"][:, :1], condition)
    torch.testing.assert_close(positive["hidden_states"][:, 1:], target_video)
    assert positive["timestep"].tolist() == [[0.0, 700.0, 700.0, 700.0]] * batch_size
    assert positive["_condition_video_seq_len"] == 1
    assert negative is None


def test_ltx2_ti2va_actor_rejects_empty_condition() -> None:
    target = torch.randn(1, 3, 4)
    model_inputs = {
        "hidden_states": target,
        "audio_hidden_states": torch.randn(1, 2, 4),
        "timestep": torch.ones(1, 3),
        "height": 1,
        "width": 1,
        "num_frames": 3,
        "_require_image_condition": True,
    }

    with pytest.raises(ValueError, match="requires condition_image_latents"):
        LTX23FlowGRPO.inject_condition(
            model_inputs,
            None,
            {"image_latents": torch.empty(1, 0, 4)},
        )


def test_ltx2_training_prediction_excludes_fixed_first_frame() -> None:
    class Transformer(torch.nn.Module):
        def forward(self, hidden_states, audio_hidden_states, **_kwargs):
            return hidden_states + 1, audio_hidden_states + 2

    video, audio = LTX23FlowGRPO._predict(
        Transformer(),
        {
            "hidden_states": torch.zeros(1, 4, 3),
            "audio_hidden_states": torch.zeros(1, 2, 3),
            "_condition_video_seq_len": 1,
        },
    )

    assert video.shape == (1, 3, 3)
    torch.testing.assert_close(video, torch.ones_like(video))
    torch.testing.assert_close(audio, torch.full_like(audio, 2))


def test_ltx2_i2av_actor_replays_target_only_log_prob() -> None:
    scheduler_config = {
        "base_image_seq_len": 1024,
        "max_image_seq_len": 4096,
        "base_shift": 0.95,
        "max_shift": 2.05,
        "shift_terminal": None,
        "num_train_timesteps": 1000,
    }
    rollout_scheduler = FlowMatchSDEDiscreteScheduler.from_config(scheduler_config)
    actor_scheduler = FlowMatchSDEDiscreteScheduler.from_config(scheduler_config)
    set_ltx23_timesteps(rollout_scheduler, 4, torch.device("cpu"))
    set_ltx23_timesteps(actor_scheduler, 4, torch.device("cpu"))

    condition = torch.randn(1, 1, 4)
    video = torch.randn(1, 3, 4)
    audio = torch.randn(1, 2, 4)
    sample = torch.cat([video, audio], dim=1)
    video_velocity = video * 0.1
    audio_velocity = audio * 0.1
    model_output = torch.cat([video_velocity, audio_velocity], dim=1)
    timestep = rollout_scheduler.timesteps[:1]
    next_sample, rollout_log_prob, _, _ = rollout_scheduler.step(
        model_output,
        timestep[0],
        sample,
        generator=torch.Generator().manual_seed(42),
        noise_level=0.8,
        sde_type="cps",
        return_logprobs=True,
        return_dict=False,
    )

    class Transformer(torch.nn.Module):
        def forward(self, hidden_states, audio_hidden_states, **_kwargs):
            return hidden_states * 0.1, audio_hidden_states * 0.1

    model_inputs = {
        "hidden_states": video,
        "audio_hidden_states": audio,
        "timestep": timestep[:, None].expand(-1, video.shape[1]),
        "audio_timestep": timestep[:, None].expand(-1, audio.shape[1]),
        "sigma": timestep,
        "audio_sigma": timestep,
        "num_frames": 4,
        "height": 1,
        "width": 1,
    }
    model_inputs, _ = LTX23FlowGRPO.inject_condition(
        model_inputs,
        None,
        {"image_latents": condition},
    )
    config = SimpleNamespace(
        pipeline=SimpleNamespace(guidance_scale=1.0),
        algo=SimpleNamespace(noise_level=0.8, sde_type="cps"),
    )
    actor_log_prob, _, _, _ = LTX23FlowGRPO.forward_and_sample_previous_step(
        Transformer(),
        actor_scheduler,
        config,
        model_inputs,
        None,
        {
            "all_next_latents": next_sample.unsqueeze(1),
            "all_timesteps": timestep.unsqueeze(1),
        },
        0,
    )

    torch.testing.assert_close(actor_log_prob, rollout_log_prob)


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
            guidance_scale=4.0,
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
    assert req.sampling_params.extra_args["video_cfg_scale"] == 4.0
    assert req.sampling_params.extra_args["audio_cfg_scale"] == 4.0
    assert req.sampling_params.extra_args["video_stg_scale"] == 0.0
    assert req.sampling_params.extra_args["audio_modality_scale"] == 1.0


def test_ltx2_rollout_rejects_guidance_the_actor_cannot_replay() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            output_type="pt",
            guidance_scale=4.0,
            extra_args={"audio_cfg_scale": 7.0},
        )
    )

    with pytest.raises(NotImplementedError, match="audio_cfg_scale"):
        pipeline._configure_flow_grpo(req)


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
    pipeline._flow_grpo_video_seq_len = 0
    pipeline._flow_grpo_condition_image_latents = None

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
        original_audio_num_frames=7,
        latent_num_frames=1,
    )
    denoise_ctx = SimpleNamespace(latents=None, audio_latents=None, conditioning_mask=None)

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
    assert pipeline._flow_grpo_condition_image_latents.shape == (1, 0, 32)


def test_ltx2_i2av_rollout_steps_only_generated_video_and_audio() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_logprobs = True
    pipeline._selected_sde_steps = {0}
    pipeline._current_latents = []
    pipeline._next_latents = []
    pipeline._selected_timesteps = []
    pipeline._log_probs = []
    pipeline._flow_grpo_video_seq_len = 0
    pipeline._flow_grpo_condition_image_latents = None

    video = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    state = LTXAVState(video=video, audio=audio)
    conditioning_mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    denoise_ctx = SimpleNamespace(
        latents=None,
        audio_latents=None,
        conditioning_mask=conditioning_mask,
    )
    forward_ctx = SimpleNamespace(
        request_inputs=SimpleNamespace(generator=None),
        guidance_parallel_ready=False,
        original_audio_num_frames=2,
        latent_num_frames=3,
    )
    pipeline._predict_noise_for_step = MagicMock(return_value=(torch.zeros_like(video), torch.zeros_like(audio)))
    pipeline._synchronize_guidance_parallel_step_output = MagicMock(side_effect=lambda latents, **kwargs: latents)

    def step(_prediction, _timestep, sample, **_kwargs):
        assert sample.shape == (1, 6, 4)
        torch.testing.assert_close(sample[:, :4], video[:, 2:])
        torch.testing.assert_close(sample[:, 4:], audio[:, :2])
        return sample + 10, torch.tensor([0.25]), None, None

    pipeline.scheduler = SimpleNamespace(step=step)

    next_state = pipeline._denoise_step(
        index=0,
        timestep=torch.tensor(900.0),
        state=state,
        forward_ctx=forward_ctx,
        denoise_ctx=denoise_ctx,
    )

    torch.testing.assert_close(next_state.video[:, :2], video[:, :2])
    torch.testing.assert_close(next_state.video[:, 2:], video[:, 2:] + 10)
    torch.testing.assert_close(next_state.audio[:, :2], audio[:, :2] + 10)
    torch.testing.assert_close(next_state.audio[:, 2:], torch.zeros_like(audio[:, 2:]))
    torch.testing.assert_close(pipeline._flow_grpo_condition_image_latents, video[:, :2])
    assert pipeline._flow_grpo_video_seq_len == 4
    assert pipeline._current_latents[0].shape == (1, 6, 4)
    assert pipeline._next_latents[0].shape == (1, 6, 4)


def test_ltx2_rollout_run_phase_accepts_first_frame_condition(monkeypatch) -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_sde_contiguous = True
    condition = torch.randn(1, 2, 4)
    expected = LTXPhaseResult(
        forward_context=MagicMock(),
        video=torch.randn(1, 4, 2, 2, 2),
        audio=torch.randn(1, 2, 2, 2),
    )

    def fake_run_phase(self, *_args, **kwargs):
        assert kwargs["image"] is not None
        self._current_latents = [torch.randn(1, 5, 4)]
        self._next_latents = [torch.randn(1, 5, 4)]
        self._selected_timesteps = [torch.tensor(500.0)]
        self._log_probs = [torch.tensor([0.1])]
        self._flow_grpo_video_seq_len = 3
        self._flow_grpo_condition_image_latents = condition
        return expected

    monkeypatch.setattr(LTX23PipelineWithLogProb.__bases__[0], "run_phase", fake_run_phase)
    result = pipeline.run_phase(
        MagicMock(),
        SimpleNamespace(num_inference_steps=1),
        noise_scale=0.0,
        sigmas=None,
        timesteps=None,
        attention_kwargs=None,
        phase_recipe=LTXPhaseRecipe(name="generate", guidance=MagicMock()),
        image=object(),
    )

    assert result is expected
    torch.testing.assert_close(pipeline._flow_grpo_trajectory["condition_image_latents"], condition)


def test_ltx2_ti2va_rollout_rejects_missing_image() -> None:
    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._flow_grpo_task = "ti2va"

    with pytest.raises(ValueError, match="requires exactly one first-frame image"):
        pipeline.run_phase(
            MagicMock(),
            SimpleNamespace(num_inference_steps=1),
            noise_scale=0.0,
            sigmas=None,
            timesteps=None,
            attention_kwargs=None,
            phase_recipe=LTXPhaseRecipe(name="generate", guidance=MagicMock()),
            image=None,
        )


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
        "condition_image_latents": torch.randn(1, 2, 32),
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
    assert metadata["rl"]["condition_image_latents"].shape == (1, 2, 32)
    assert metadata["rl"]["audio_sample_rate"] == 24000
    assert "audio_prompt_embeds" in metadata["prompt_embeddings"]


def unittest_mock_super_forward(target, return_value):
    from unittest.mock import patch

    return patch.object(LTX23PipelineWithLogProb.__bases__[0], "forward", return_value=return_value)
