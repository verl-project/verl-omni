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

"""
This test script validates single-turn return of `CompositeAgentLoopWorker`,
especially returns from LLM part, e.g.,
`rollout_llm_log_probs`,
`llm_response_ids`,
`text_encoder_responses`,
`llm_rm_scores` (optional)
"""

import gc
import os
import shutil
import tempfile

import numpy as np
import pytest
import ray
import torch
from omegaconf import DictConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.workers.rollout.llm_server import LLMServerManager

from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopWorker

from ..utils.gpu_test_topology import resolve_diffusion_agent_loop_gpu_topology


class _FakeRemoteComputeScore:
    def __init__(self):
        self.received_data: DataProto | None = None

    async def remote(self, data: DataProto) -> dict:
        self.received_data = data
        return {"reward_score": 1.0, "reward_extra_info": {"dit_msg": "dummy_dit_reward_info"}}


class _FakeARRemoteComputeScore:
    def __init__(self):
        self.received_data: DataProto | None = None

    async def remote(self, data: DataProto) -> dict:
        self.received_data = data
        return {"reward_score": 1.0, "reward_extra_info": {"ar_msg": "dummy_ar_reward_info"}}


class _FakeRewardLoopWorkerHandle:
    def __init__(self):
        self.compute_score = _FakeRemoteComputeScore()


class _FakeARRewardLoopWorkerHandle:
    def __init__(self):
        self.compute_score = _FakeARRemoteComputeScore()


def _assert_non_empty_tensor(value, field_name: str) -> None:
    assert value is not None, f"{field_name} should not be None"
    assert isinstance(value, torch.Tensor), f"{field_name} should be a torch.Tensor, got {type(value).__name__}"
    assert value.numel() > 0, f"{field_name} should not be empty"


def _assert_text_encoder_outputs(result: DataProto, *, batch_size: int, max_token_len: int) -> None:
    """Validate Qwen-Image text-encoder returns by rollout."""
    llm_response_ids = result.batch["llm_response_ids"]
    llm_all_log_probs = result.batch.get("rollout_llm_log_probs")
    text_encoder_responses = result.non_tensor_batch["text_encoder_responses"]  # list[str]
    _assert_non_empty_tensor(llm_response_ids, "llm_response_ids")
    _assert_non_empty_tensor(llm_all_log_probs, "llm_all_log_probs")

    assert llm_response_ids.shape == (batch_size, max_token_len)
    if llm_all_log_probs is not None:
        assert llm_all_log_probs.shape[1] <= max_token_len
        assert llm_all_log_probs.shape == (batch_size, llm_all_log_probs.shape[1], llm_all_log_probs.shape[-1])
    assert len(text_encoder_responses) == batch_size


def _assert_qwen_image_outputs(result: DataProto, *, batch_size: int, height: int, width: int) -> None:
    """Validate Qwen-Image diffusion output is a batched RGB image tensor."""
    responses = result.batch["responses"]
    _assert_non_empty_tensor(responses, "responses")
    assert responses.shape == (batch_size, 3, height, width), (
        f"Expected responses shape {(batch_size, 3, height, width)}, got {tuple(responses.shape)}"
    )
    assert torch.isfinite(responses).all(), "Generated image tensor contains non-finite values"
    assert responses.min() >= 0.0 and responses.max() <= 1.0, (
        f"Generated image pixels should be in [0, 1], got [{responses.min():.4f}, {responses.max():.4f}]"
    )


def _create_tp_compatible_model(parent_dir, src_model_path, num_attention_heads=2):
    """Copy base model and recreate transformer on-the-fly with TP-compatible head count.

    The tiny-random Qwen-Image model has num_attention_heads=1 in its transformer config,
    which is not divisible by tensor_model_parallel_size=2. This helper copies the full
    model directory (vae, text_encoder, tokenizer, scheduler) and overwrites only the
    transformer component with a freshly-initialized one that has the desired head count.
    """
    from diffusers import QwenImageTransformer2DModel

    dst = os.path.join(parent_dir, "Qwen-Image")
    shutil.copytree(src_model_path, dst)
    transformer = QwenImageTransformer2DModel(
        num_attention_heads=num_attention_heads,
        attention_head_dim=32,
        num_layers=2,
        in_channels=64,
        out_channels=16,
        patch_size=2,
        joint_attention_dim=32,
        axes_dims_rope=(8, 12, 12),
        guidance_embeds=False,
    )
    transformer.save_pretrained(os.path.join(dst, "transformer"))

    return dst


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config")):
        config = compose(config_name="diffusion_trainer")

    requested_gpus, tp_size, attention_heads = resolve_diffusion_agent_loop_gpu_topology()
    base_model_path = os.path.expanduser("~/models/tiny-random/Qwen-Image")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = _create_tp_compatible_model(tmp_dir, base_model_path, num_attention_heads=attention_heads)
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.tokenizer_path = os.path.join(model_path, "tokenizer")
        config.actor_rollout_ref.model.algorithm = "dual_grpo"  # change rollout pipeline
        config.actor_rollout_ref.rollout.name = "vllm_omni"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.enforce_eager = True
        config.actor_rollout_ref.rollout.step_execution = False
        # Keep the 2-GPU TP smoke light; CI EOFError on worker launch is usually OOM.
        # Keep enough inference steps for sde_window_range=[0, 5] / sde_window_size=2.
        config.actor_rollout_ref.rollout.n = 2
        config.actor_rollout_ref.rollout.pipeline.height = 256
        config.actor_rollout_ref.rollout.pipeline.width = 256
        config.actor_rollout_ref.rollout.pipeline.num_inference_steps = 10
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.rollout.llm_calculate_log_probs = True  # new
        config.actor_rollout_ref.rollout.max_new_tokens = 20  # new
        config.actor_rollout_ref.rollout.temperature = 0.8  # new
        config.actor_rollout_ref.rollout.top_k = 5  # new
        config.actor_rollout_ref.rollout.top_p = 0.9  # new
        config.actor_rollout_ref.rollout.agent.num_workers = min(2, requested_gpus)
        config.actor_rollout_ref.rollout.agent.default_agent_loop = "diffusion_single_turn_agent"
        tokenizer_max_length = 1024
        prompt_template_encode_start_idx = 34
        max_length = tokenizer_max_length + prompt_template_encode_start_idx

        config.actor_rollout_ref.rollout.algo.noise_level = 1.0
        config.actor_rollout_ref.rollout.algo.sde_window_size = 2
        config.actor_rollout_ref.rollout.algo.sde_window_range = [0, 5]

        config.actor_rollout_ref.rollout.pipeline.true_cfg_scale = 4.0
        config.actor_rollout_ref.rollout.pipeline.max_sequence_length = max_length
        config.actor_rollout_ref.rollout.nnodes = 1

        config.reward.reward_manager.name = "naive"
        config.trainer.n_gpus_per_node = requested_gpus

        config.data.max_prompt_length = max_length
        config.actor_rollout_ref.rollout.max_model_len = max_length

        config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size

        # Smoke: prefer local FLASH_ATTN over product-default Hub FA3 (cf. FSDP engine test).
        from tests.utils.smoke_attention import resolve_smoke_attention_backends

        attn_backend, rollout_attn_backend = resolve_smoke_attention_backends()
        config.actor_rollout_ref.model.attn_backend = attn_backend
        config.actor_rollout_ref.rollout.rollout_attn_backend = rollout_attn_backend

        yield config


@pytest.mark.parametrize("agent_reward_loop", [False, True])
def test_single_turn(init_config, agent_reward_loop: bool):
    """Smoke-test CompositeAgentLoopWorker end-to-end on Qwen-Image."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        }
    )
    try:
        AgentLoopManager.agent_loop_workers_class = ray.remote(CompositeAgentLoopWorker)
        llm_server_manager = LLMServerManager.create(config=init_config)
        agent_loop_manager = AgentLoopManager.create(
            config=init_config,
            llm_client=llm_server_manager.get_client(),
            reward_loop_worker_handles=[_FakeRewardLoopWorkerHandle(), _FakeARRewardLoopWorkerHandle()]
            if agent_reward_loop
            else None,
        )

        system_prompt = (
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
        )
        user_prompts = [
            "Generate a traffic light where none of the lights are green.",
            "A fruit basket containing only apples, no oranges.",
        ]

        raw_prompts = []
        for user_prompt in user_prompts:
            raw_prompts.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        raw_negative_prompts = []
        for user_prompt in user_prompts:
            raw_negative_prompts.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": " "},
                ]
            )

        batch = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array(raw_prompts),
                "raw_negative_prompt": np.array(raw_negative_prompts),
                "data_source": np.array(["jpeg_compressibility"] * len(raw_prompts)),
                "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * len(raw_prompts)),
            },
        )
        n = init_config.actor_rollout_ref.rollout.n
        batch = batch.repeat(n)
        batch.meta_info["global_steps"] = 0
        result = agent_loop_manager.generate_sequences(prompts=batch)
        batch_size = len(raw_prompts) * n
        assert len(result) == batch_size

        expected_batch_keys = [
            "responses",
            "all_latents",
            "all_timesteps",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "rollout_log_probs",
            "rollout_llm_log_probs",
        ]
        expected_non_tensor_batch_keys = ["text_encoder_responses"]
        if agent_reward_loop:
            expected_batch_keys += ["rm_scores", "llm_rm_scores"]
            expected_non_tensor_batch_keys += ["dit_msg", "ar_msg"]

        for key in expected_batch_keys:
            assert key in result.batch, f"Key {key} not found in result batch with keys {list(result.batch.keys())}."

        for key in expected_non_tensor_batch_keys:
            assert key in result.non_tensor_batch, (
                f"Key {key} not found in result non-tensor batch with keys {list(result.non_tensor_batch.keys())}."
            )

        height = init_config.actor_rollout_ref.rollout.pipeline.height
        width = init_config.actor_rollout_ref.rollout.pipeline.width
        max_new_tokens = init_config.actor_rollout_ref.rollout.max_new_tokens

        _assert_text_encoder_outputs(result, batch_size=batch_size, max_token_len=max_new_tokens)
        _assert_qwen_image_outputs(result, batch_size=batch_size, height=height, width=width)

        num_turns = result.non_tensor_batch["__num_turns__"]
        assert np.all(num_turns == 2)
    finally:
        ray.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
