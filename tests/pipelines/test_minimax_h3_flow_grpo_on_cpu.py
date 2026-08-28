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
"""CPU contract tests for MiniMax H3 FlowGRPO."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict
from vllm_omni.diffusion.data import DiffusionOutput as VllmDiffusionOutput
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

from verl_omni.pipelines.minimax_h3_flow_grpo.agent_loop import MiniMaxH3DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    H3_AUDIO_WIDTH,
    H3_VIDEO_WIDTH,
    combine_log_probs,
    flatten_joint_latents,
    split_joint_latents,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import MiniMaxH3PipelineWithLogProb
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


def _trajectory() -> dict[str, torch.Tensor]:
    video = torch.randn(1, 2, H3_VIDEO_WIDTH)
    audio = torch.randn(1, 3, H3_AUDIO_WIDTH)
    next_video = video + 0.1
    next_audio = audio - 0.1
    text_len = 2
    seq_len = video.shape[1] + audio.shape[1] + text_len
    return {
        "all_latents": flatten_joint_latents(video, audio).unsqueeze(1),
        "all_next_latents": flatten_joint_latents(next_video, next_audio).unsqueeze(1),
        "all_timesteps": torch.tensor([[0.25]]),
        "all_log_probs": torch.tensor([[0.4]]),
        "h3_step_indices": torch.tensor([[0]]),
        "h3_audio_timesteps": torch.tensor([[0.5]]),
        "prompt_embeds": torch.randn(1, text_len, 8),
        "prompt_embeds_mask": torch.ones(1, text_len, dtype=torch.long),
        "h3_seq_len": torch.tensor([seq_len]),
        "h3_video_rows": torch.tensor([video.shape[1]]),
        "h3_audio_rows": torch.tensor([audio.shape[1]]),
        "h3_position_ids": torch.zeros(1, seq_len, 3, dtype=torch.long),
        "h3_token_tags": torch.zeros(1, seq_len, dtype=torch.long),
        "h3_video_indices": torch.tensor([[0, 1]]),
        "h3_audio_indices": torch.tensor([[2, 3, 4]]),
        "h3_text_indices": torch.tensor([[5, 6]]),
    }


def test_minimax_h3_flow_grpo_registers_both_adapters_and_joint_helpers() -> None:
    assert DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "flow_grpo") is MiniMaxH3FlowGRPO
    assert VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", "flow_grpo") is MiniMaxH3PipelineWithLogProb

    video = torch.randn(2, 3, H3_VIDEO_WIDTH)
    audio = torch.randn(2, 4, H3_AUDIO_WIDTH)
    joint = flatten_joint_latents(video, audio)
    actual_video, actual_audio = split_joint_latents(joint, 3, 4)
    torch.testing.assert_close(actual_video, video)
    torch.testing.assert_close(actual_audio, audio)

    combined = combine_log_probs(
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        video_weight=0.25,
        audio_weight=0.75,
    )
    torch.testing.assert_close(combined, torch.tensor([2.5]))


def test_h3_agent_loop_uses_parent_init_and_raw_text_tokens() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.calls = []

        def encode(self, text, add_special_tokens=False):
            assert text == "\n"
            assert add_special_tokens is False
            return [198]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 151645

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("H3 prompt encoding must not apply a chat template")

        def __call__(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return {"input_ids": [11, 12, 13]}

    tokenizer = Tokenizer()
    rollout = SimpleNamespace(prompt_length=128)
    trainer_config = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)))
    processor = SimpleNamespace(image_processor=object())

    async def initialize_and_build() -> tuple[MiniMaxH3DiffusionSingleTurnAgentLoop, list[int]]:
        agent_loop = MiniMaxH3DiffusionSingleTurnAgentLoop(
            trainer_config,
            server_manager=object(),
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=None,
            data_config=SimpleNamespace(config={}),
            hf_model_type="qwen3_vl",
        )
        prompt_ids = await agent_loop.ct_build_initial_tokens([{"role": "user", "content": "Raw H3 prompt"}])
        return agent_loop, prompt_ids

    agent_loop, prompt_ids = asyncio.run(initialize_and_build())
    assert prompt_ids == [11, 12, 13]
    assert agent_loop.continuous_token_builder is not None
    assert agent_loop.extra_tokenizer_map == {}
    assert tokenizer.calls == [
        (
            "Raw H3 prompt",
            {
                "padding": False,
                "truncation": True,
                "max_length": 128,
                "add_special_tokens": False,
            },
        )
    ]


def test_rollout_output_reaches_actor_and_replays_joint_transition(monkeypatch) -> None:
    trajectory = _trajectory()
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.decode.return_value = "a bounded piano performance"
    pipeline._flow_grpo_trajectory = trajectory
    request = SimpleNamespace(
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params=SimpleNamespace(
            extra_args={},
            max_sequence_length=2,
            num_outputs_per_prompt=1,
        ),
    )
    request_batch = SimpleNamespace(requests=[request])
    video_pixels = torch.zeros(1, 3, 2, 8, 8, dtype=torch.uint8)
    audio_waveform = torch.zeros(1, 32000)

    with patch.object(
        MiniMaxH3Pipeline,
        "forward",
        return_value=VllmDiffusionOutput(output=(video_pixels, audio_waveform)),
    ):
        rollout_output = pipeline.forward(request_batch)

    pipeline.tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=False)
    assert request.prompt["prompt"] == "a bounded piano performance"
    torch.testing.assert_close(rollout_output.trajectory_latents, trajectory["all_latents"])
    metadata = rollout_output.output["metadata"]
    assert set(metadata["prompt_embeddings"]) == {"prompt_embeds", "prompt_embeds_mask"}
    assert "all_next_latents" in metadata["rl"]

    final_res = SimpleNamespace(
        images=[(video_pixels, audio_waveform)],
        trajectory_latents=rollout_output.trajectory_latents,
        trajectory_timesteps=rollout_output.trajectory_timesteps,
        trajectory_log_probs=rollout_output.trajectory_log_probs,
        multimodal_output=rollout_output.output,
        request_output=None,
    )
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 1
    processed = DiffusionStrategy(server).process_output(final_res, None, {"output_type": "pt", "logprobs": True})

    for key in ("all_latents", "all_next_latents", "h3_step_indices", "h3_audio_timesteps"):
        assert key in processed.extra_fields
    assert processed.extra_fields["audio_sample_rate"] == 32000

    actor_fields = {
        key: value.unsqueeze(0)
        for key, value in processed.extra_fields.items()
        if isinstance(value, torch.Tensor) and key != "audio"
    }
    micro_batch = TensorDict(actor_fields, batch_size=[1])
    model_inputs, negative_inputs = MiniMaxH3FlowGRPO.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=micro_batch["all_latents"],
        timesteps=micro_batch["all_timesteps"],
        prompt_embeds=micro_batch["prompt_embeds"],
        prompt_embeds_mask=micro_batch["prompt_embeds_mask"],
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )
    assert negative_inputs is None
    assert model_inputs["hidden_states"].shape == (1, 2, H3_VIDEO_WIDTH)
    assert model_inputs["audio_hidden_states"].shape == (1, 3, H3_AUDIO_WIDTH)

    module = MagicMock(
        return_value=(
            torch.zeros_like(model_inputs["hidden_states"]),
            torch.zeros_like(model_inputs["audio_hidden_states"]),
        )
    )
    log_probs = iter((0.2, 0.6))

    def fake_transition(_scheduler, sample, _velocity, _step, **kwargs):
        assert kwargs["prev_sample"] is not None
        return sample, torch.tensor([next(log_probs)]), sample + 1, torch.tensor(0.3), torch.tensor(0.1)

    monkeypatch.setattr(
        "verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter.sample_h3_transition",
        fake_transition,
    )
    model_config = SimpleNamespace(algo=SimpleNamespace(noise_level=0.8, sde_type="cps"))
    log_prob, mean, std, sqrt_dt = MiniMaxH3FlowGRPO.forward_and_sample_previous_step(
        module=module,
        scheduler=(MagicMock(), MagicMock()),
        model_config=model_config,
        model_inputs=model_inputs,
        negative_model_inputs=None,
        scheduler_inputs=micro_batch,
        step=0,
    )
    torch.testing.assert_close(log_prob, torch.tensor([0.4]))
    assert mean.shape == micro_batch["all_latents"][:, 0].shape
    assert std.shape == (1, 1, 1)
    assert sqrt_dt.shape == (1,)


class _RecordingBase:
    def load_weights(self, weights):
        self.forwarded = list(weights)
        return {name for name, _ in self.forwarded}


class _Component:
    def __init__(self, params):
        self._params = params
        self.arch = SimpleNamespace(ffn_hidden_size=4)

    def named_parameters(self):
        return self._params.items()


class _SyncPipeline(MiniMaxH3WeightSyncMixin, _RecordingBase):
    def __init__(self):
        self.qkv = SimpleNamespace(weight_loader=MagicMock())
        self.fc1 = SimpleNamespace(weight_loader=MagicMock())
        self.transformer = _Component(
            {
                "blocks.0.attn.qkv_proj.weight": self.qkv,
                "blocks.0.mlp.fc1.weight": self.fc1,
            }
        )


def test_full_weight_sync_uses_fused_loaders_and_renames_plain_weights() -> None:
    pipeline = _SyncPipeline()
    q = torch.randn(8, 8)
    geglu = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    pipeline.load_weights(
        [
            ("transformer.transformer_blocks.0.attn.to_q.weight", q),
            ("transformer.transformer_blocks.0.ff.net.0.proj.weight", geglu),
            ("transformer.audio_proj_in.weight", torch.ones(2, 2)),
        ]
    )

    pipeline.qkv.weight_loader.assert_called_once_with(pipeline.qkv, q, "q")
    assert pipeline.fc1.weight_loader.call_count == 2
    torch.testing.assert_close(pipeline.fc1.weight_loader.call_args_list[0].args[1], geglu[4:])
    torch.testing.assert_close(pipeline.fc1.weight_loader.call_args_list[1].args[1], geglu[:4])
    assert pipeline.forwarded[0][0] == "transformer.audio_patch_proj.weight"


def test_lora_weight_sync_splits_geglu_and_rewrites_targets() -> None:
    pipeline = _SyncPipeline()
    tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    name = "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_B.weight"
    config = {"r": 4, "target_modules": ["to_q", "ff.net.0.proj"]}

    mapped, mapped_config = pipeline.map_lora_update_to_engine({name: tensor}, config)

    torch.testing.assert_close(mapped["transformer.blocks.0.mlp.fc1_0.lora_B.weight"], tensor[4:])
    torch.testing.assert_close(mapped["transformer.blocks.0.mlp.fc1_1.lora_B.weight"], tensor[:4])
    assert mapped_config["target_modules"] == ["fc1_0", "fc1_1", "to_q"]


class _FakeH3Branch:
    """Small stand-in that keeps the denoise test focused on trajectory semantics."""

    def __init__(self, *, packed, text_embeddings, token_tags, device):
        del packed, text_embeddings, token_tags, device

    def forward_kwargs(self, *, video_rows, audio_rows, **kwargs):
        del kwargs
        return {
            "hidden_states": video_rows,
            "audio_hidden_states": audio_rows,
        }


def _layout_metadata() -> dict[str, torch.Tensor]:
    """Return a valid fixed H3 packed layout for two video, three audio and two text rows."""
    return {
        "prompt_embeds": torch.randn(1, 2, 8),
        "prompt_embeds_mask": torch.ones(1, 2, dtype=torch.long),
        "h3_seq_len": torch.tensor([7]),
        "h3_video_rows": torch.tensor([2]),
        "h3_audio_rows": torch.tensor([3]),
        "h3_position_ids": torch.zeros(1, 7, 3, dtype=torch.long),
        "h3_token_tags": torch.tensor([[0, 0, 2, 2, 2, 1, 1]]),
        "h3_video_indices": torch.tensor([[0, 1]]),
        "h3_audio_indices": torch.tensor([[2, 3, 4]]),
        "h3_text_indices": torch.tensor([[5, 6]]),
    }


def test_rollout_denoise_window_records_replayable_aligned_trajectory(monkeypatch) -> None:
    """Exercise the H3 denoise loop and verify selected CPS transitions stay aligned.

    The configured range has exactly one legal contiguous window, steps 1 and 2.
    This lets the test prove that unselected steps are still executed
    deterministically, while only selected transitions enter the Actor payload.
    """
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_window_size = 2
    pipeline._flow_grpo_window_range = [1, 3]
    pipeline._flow_grpo_sde_contiguous = True
    pipeline._flow_grpo_seed = 123
    pipeline._h3_max_text_len = 2
    pipeline._initial_noise = MagicMock(
        return_value=(
            torch.zeros(2, H3_VIDEO_WIDTH),
            torch.zeros(3, H3_AUDIO_WIDTH),
        )
    )
    pipeline.record_denoise_step = MagicMock()
    pipeline.progress_bar = lambda total: nullcontext(SimpleNamespace(update=MagicMock()))
    pipeline._resident_dit_layers_on_device = lambda enabled: nullcontext()
    pipeline._layout_outputs = MagicMock(return_value=_layout_metadata())

    def transformer(**model_inputs):
        return (
            torch.zeros_like(model_inputs["hidden_states"]),
            torch.zeros_like(model_inputs["audio_hidden_states"]),
        )

    pipeline.transformer = transformer
    pipeline._transformer_for_task = MagicMock(return_value=transformer)

    video_sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    audio_sigmas = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
    transition_calls = []

    def fake_transition(_scheduler, sample, _velocity, step, **kwargs):
        is_video = sample.shape[-1] == H3_VIDEO_WIDTH
        transition_calls.append(
            {
                "step": step,
                "modality": "video" if is_video else "audio",
                "noise_level": kwargs["noise_level"],
                "return_log_prob": kwargs["return_log_prob"],
            }
        )
        log_prob = torch.tensor([0.2 if is_video else 0.6]) if kwargs["return_log_prob"] else None
        next_sample = sample + float(step + 1)
        return next_sample, log_prob, sample, torch.tensor(0.1), torch.tensor(0.2)

    packed = {
        "token_tags": torch.zeros(7, dtype=torch.long),
        "text_pos": torch.tensor([5, 6]),
    }
    module = "verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter"
    monkeypatch.setattr(f"{module}.minimax_h3_packed_sequence", lambda **kwargs: packed)
    monkeypatch.setattr(f"{module}.MiniMaxH3DenoiseBranch", _FakeH3Branch)
    monkeypatch.setattr(f"{module}.h3_sigma_schedules", lambda *args: (video_sigmas, audio_sigmas))
    monkeypatch.setattr(f"{module}.configure_flow_scheduler", lambda *args: None)
    monkeypatch.setattr(f"{module}.sample_h3_transition", fake_transition)
    monkeypatch.setattr(f"{module}.minimax_h3_unpatchify_video_tokens", lambda *args, **kwargs: torch.tensor([11.0]))
    monkeypatch.setattr(f"{module}.minimax_h3_unpack_audio_tokens", lambda *args, **kwargs: torch.tensor([22.0]))

    video_output, audio_output = pipeline.diffuse(
        task="t2va",
        text_embeddings=torch.randn(2, 8),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=7,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        num_frames=1,
        num_steps=6,
        video_shift=12.0,
        audio_shift=3.0,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
    )

    torch.testing.assert_close(video_output, torch.tensor([11.0]))
    torch.testing.assert_close(audio_output, torch.tensor([22.0]))
    assert len(transition_calls) == 10
    for call in transition_calls:
        selected = call["step"] in {1, 2}
        assert call["noise_level"] == (0.8 if selected else 0.0)
        assert call["return_log_prob"] is selected

    trajectory = pipeline._flow_grpo_trajectory
    assert set(_layout_metadata()).issubset(trajectory)
    assert trajectory["all_latents"].shape == (1, 2, 1, 2 * H3_VIDEO_WIDTH + 3 * H3_AUDIO_WIDTH)
    assert trajectory["all_next_latents"].shape == trajectory["all_latents"].shape
    assert trajectory["all_log_probs"].shape == (1, 2)
    assert trajectory["h3_step_indices"].tolist() == [[1, 2]]
    torch.testing.assert_close(trajectory["all_timesteps"], torch.tensor([[0.2, 0.4]]))
    torch.testing.assert_close(trajectory["h3_audio_timesteps"], torch.tensor([[0.3, 0.5]]))
    torch.testing.assert_close(trajectory["all_log_probs"], torch.tensor([[0.4, 0.4]]))

    # At selected step 1 the transition adds 2, and at step 2 it adds 3.
    # This checks each recorded next latent is paired with its own current latent.
    first_delta = trajectory["all_next_latents"][:, 0] - trajectory["all_latents"][:, 0]
    second_delta = trajectory["all_next_latents"][:, 1] - trajectory["all_latents"][:, 1]
    torch.testing.assert_close(first_delta, torch.full_like(first_delta, 2.0))
    torch.testing.assert_close(second_delta, torch.full_like(second_delta, 3.0))
    pipeline.record_denoise_step.assert_any_call(1, normalized_timestep=0.8)
    pipeline.record_denoise_step.assert_any_call(2, normalized_timestep=0.6)
    pipeline.record_denoise_step.assert_called_with(None)


def _batched_actor_payload(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Repeat a valid single-sample rollout payload into one Actor micro-batch."""
    payload = _trajectory()
    return {key: value.repeat(batch_size, *([1] * (value.ndim - 1))) for key, value in payload.items()}


def _prepare_actor_payload(payload: dict[str, torch.Tensor], module=None):
    module = module or MagicMock()
    result = MiniMaxH3FlowGRPO.prepare_model_inputs(
        module=module,
        model_config=MagicMock(),
        latents=payload["all_latents"],
        timesteps=payload["all_timesteps"],
        prompt_embeds=payload["prompt_embeds"],
        prompt_embeds_mask=payload["prompt_embeds_mask"],
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=payload,
        step=0,
    )
    return module, result


def test_actor_accepts_a_shared_replicated_h3_layout() -> None:
    """A batch larger than one is valid when every packed-sequence layout matches."""
    payload = _batched_actor_payload()
    module, (model_inputs, negative_inputs) = _prepare_actor_payload(payload)

    assert negative_inputs is None
    assert model_inputs["hidden_states"].shape == (2, 2, H3_VIDEO_WIDTH)
    assert model_inputs["audio_hidden_states"].shape == (2, 3, H3_AUDIO_WIDTH)
    assert model_inputs["encoder_hidden_states"].shape == (2, 2, 8)
    assert model_inputs["timestep"].tolist() == [0.25, 0.5]
    assert model_inputs["_h3_scheduler_step"] == 0
    module.assert_not_called()


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (
            lambda payload: payload.pop("h3_audio_indices"),
            KeyError,
            "missing fields: \\['h3_audio_indices'\\]",
        ),
        (
            lambda payload: payload["h3_video_rows"].__setitem__(1, 3),
            ValueError,
            "one shared video row count",
        ),
        (
            lambda payload: payload["h3_position_ids"].__setitem__((1, 0, 0), 9),
            ValueError,
            "different position_ids layouts",
        ),
        (
            lambda payload: payload["h3_audio_timesteps"].__setitem__((1, 0), 0.75),
            ValueError,
            "shared video/audio timesteps",
        ),
        (
            lambda payload: payload.__setitem__("all_latents", payload["all_latents"][..., :-1]),
            ValueError,
            "joint width .* does not match video/audio metadata",
        ),
    ],
    ids=[
        "missing-trajectory-field",
        "mixed-video-row-count",
        "mixed-position-layout",
        "mixed-modality-timestep",
        "malformed-joint-latent",
    ],
)
def test_actor_rejects_non_replayable_rollout_contract_before_transformer(
    mutate,
    error_type,
    message,
) -> None:
    """Contract violations must fail before an invalid packed batch reaches H3."""
    payload = _batched_actor_payload()
    mutate(payload)
    module = MagicMock()

    with pytest.raises(error_type, match=message):
        _prepare_actor_payload(payload, module)

    # prepare_model_inputs only validates and reshapes data; a bad rollout must
    # never invoke the model or defer the error until the expensive forward.
    module.assert_not_called()
