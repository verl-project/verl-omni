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

"""GPU integration test for deterministic rollout seeding through vLLM-omni."""

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

from verl_omni.agent_loop import DiffusionAgentLoopWorker

from ..utils.gpu_test_topology import resolve_diffusion_agent_loop_gpu_topology

MODEL_PATH = os.path.expanduser("~/models/tiny-random/Qwen-Image")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason=f"tiny model missing at {MODEL_PATH}"),
]


def _create_tp_compatible_model(parent_dir, src_model_path, num_attention_heads=2):
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


def _make_prompt_batch(num_prompts: int = 1) -> DataProto:
    system_prompt = (
        "Describe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:"
    )
    user_prompts = [
        "A photo of cute cat with long fur and big eyes.",
        "A photo of cute dog with short hair.",
    ][:num_prompts]

    raw_prompts = []
    raw_negative_prompts = []
    for user_prompt in user_prompts:
        raw_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        raw_negative_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": " "},
            ]
        )

    return DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "raw_negative_prompt": np.array(raw_negative_prompts),
            "data_source": np.array(["jpeg_compressibility"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * len(raw_prompts)),
        },
    )


@pytest.fixture
def seed_rollout_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config")):
        config = compose(config_name="diffusion_trainer")

    requested_gpus, tp_size, attention_heads = resolve_diffusion_agent_loop_gpu_topology(default_num_gpus=1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = _create_tp_compatible_model(tmp_dir, MODEL_PATH, num_attention_heads=attention_heads)
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.tokenizer_path = os.path.join(model_path, "tokenizer")
        config.actor_rollout_ref.rollout.name = "vllm_omni"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.enforce_eager = True
        config.actor_rollout_ref.rollout.n = 4
        config.actor_rollout_ref.rollout.pipeline.num_inference_steps = 10
        config.actor_rollout_ref.rollout.calculate_log_probs = True
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
        config.reward.reward_manager.name = "image"
        config.trainer.n_gpus_per_node = requested_gpus
        config.data.max_prompt_length = max_length
        config.actor_rollout_ref.rollout.max_model_len = max_length
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
        yield config


def _initial_latents(result: DataProto) -> torch.Tensor:
    """Return the first denoising latent for every rollout row."""
    return result.batch["all_latents"][:, 0].detach().cpu()


def test_rollout_seed_reproducible_and_diverse_via_agent_loop(seed_rollout_config):
    """End-to-end rollout seeding through vLLM-omni agent loop.

    - Same ``rollout_seed`` + batch -> bit-identical initial latents across reruns.
    - Distinct rollout indices within one step -> distinct initial latents.
    """
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
        AgentLoopManager.agent_loop_workers_class = ray.remote(DiffusionAgentLoopWorker)
        llm_server_manager = LLMServerManager.create(config=seed_rollout_config)
        agent_loop_manager = AgentLoopManager.create(
            config=seed_rollout_config,
            llm_client=llm_server_manager.get_client(),
        )

        n = seed_rollout_config.actor_rollout_ref.rollout.n
        batch = _make_prompt_batch(num_prompts=1).repeat(n)
        batch.meta_info["global_steps"] = 1
        batch.meta_info["rollout_seed"] = 42

        first = agent_loop_manager.generate_sequences(prompts=batch)
        second = agent_loop_manager.generate_sequences(prompts=batch)

        latents_first = _initial_latents(first)
        latents_second = _initial_latents(second)
        assert latents_first.shape[0] == n
        assert torch.equal(latents_first, latents_second), (
            "identical rollout_seed and batch must reproduce initial latents on GPU"
        )

        for i in range(n):
            for j in range(i + 1, n):
                assert not torch.equal(latents_first[i], latents_first[j]), (
                    f"rollout indices {i} and {j} must not share the same initial latent"
                )
    finally:
        ray.shutdown()
