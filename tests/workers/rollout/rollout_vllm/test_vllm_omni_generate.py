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
E2E test for vLLMOmniHttpServer generate flow.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py -v -s
    python tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode

from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

MODEL_PATH = Path(os.path.expanduser("~/models/tiny-random/Qwen-Image"))


# ---------------------------------------------------------------------
#                👇 Test Helper Functions & Fixtures 👇
# ---------------------------------------------------------------------

_MIN_PROMPT_TOKENS = 35


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"), trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        f"The pipeline drops the first 34 chat‑template prefix tokens; "
        f"use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


def _assert_non_empty_tensor(value, field_name: str) -> None:
    assert value is not None, f"{field_name} should not be None"
    assert isinstance(value, torch.Tensor), f"{field_name} should be a torch.Tensor, got {type(value).__name__}"
    assert value.numel() > 0, f"{field_name} should not be empty"


def _assert_flow_grpo_step_execution_contract(output: DiffusionOutput) -> None:
    """Validate the FlowGRPO trajectory contract in step-execution mode.

    vLLMOmniHttpServer maps all_log_probs to DiffusionOutput.log_probs
    and the remaining custom_output fields to DiffusionOutput.extra_fields.
    """
    expected_extra_fields = {
        "all_latents",
        "all_timesteps",
        "prompt_embeds",
        "prompt_embeds_mask",
        "negative_prompt_embeds",
        "negative_prompt_embeds_mask",
    }

    missing_fields = expected_extra_fields - set(output.extra_fields)
    assert not missing_fields, f"Missing FlowGRPO step-execution fields: {sorted(missing_fields)}"

    required_tensors = {
        "all_latents": output.extra_fields["all_latents"],
        "all_log_probs": output.log_probs,
        "all_timesteps": output.extra_fields["all_timesteps"],
        "prompt_embeds": output.extra_fields["prompt_embeds"],
        "prompt_embeds_mask": output.extra_fields["prompt_embeds_mask"],
    }
    for field_name, value in required_tensors.items():
        _assert_non_empty_tensor(value, field_name)

    # This test does not provide negative_prompt_ids, so True-CFG is disabled.
    # The keys must still be preserved while their values remain None.
    assert output.extra_fields["negative_prompt_embeds"] is None
    assert output.extra_fields["negative_prompt_embeds_mask"] is None

    all_latents = output.extra_fields["all_latents"]
    all_log_probs = output.log_probs
    all_timesteps = output.extra_fields["all_timesteps"]
    prompt_embeds = output.extra_fields["prompt_embeds"]
    prompt_embeds_mask = output.extra_fields["prompt_embeds_mask"]

    # The server removes the per-request batch dimension.
    assert all_latents.shape[0] == all_timesteps.shape[0] + 1
    assert all_log_probs.shape[0] == all_timesteps.shape[0]
    assert prompt_embeds.shape[:-1] == prompt_embeds_mask.shape


@pytest.fixture
def init_server(request):
    """Create and launch a vLLMOmniHttpServer Ray actor with Qwen/Qwen-Image."""
    model_path = MODEL_PATH
    step_execution = getattr(request, "param", False)

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        },
        ignore_reinit_error=True,
    )

    # Smoke: prefer local FLASH_ATTN over product-default Hub FA3.
    from tests.utils.smoke_attention import resolve_smoke_attention_backends

    attn_backend, rollout_attn_backend = resolve_smoke_attention_backends()

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 16 if step_execution else 256,
            "step_execution": step_execution,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 4,
            "rollout_attn_backend": rollout_attn_backend,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": model_path,
            "tokenizer_path": os.path.join(model_path, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "attn_backend": attn_backend,
            "algorithm": "flow_grpo",
        }
    )

    ServerCls = ray.remote(vLLMOmniHttpServer)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=10,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


def test_generate(init_server):
    """Concurrent generate() calls covering basic output, logprobs, and multi-request correctness."""
    server = init_server

    prompts = [
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline",
        "a fluffy orange cat sitting on a wooden windowsill looking outside at "
        "a garden full of colorful flowers on a bright sunny afternoon",
        "a majestic mountain landscape covered with fresh white snow under a "
        "clear blue sky with pine trees in the foreground and a frozen lake",
        "a futuristic city at night with neon lights glowing on tall glass "
        "skyscrapers and flying vehicles soaring between the buildings",
    ]

    refs = []
    for i, prompt in enumerate(prompts):
        rid = f"test_{i}_{uuid4().hex[:8]}"
        ref = server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": i == 0,  # first request includes logprobs
            },
            request_id=rid,
        )
        refs.append(ref)

    results = ray.get(refs, timeout=600)

    for i, output in enumerate(results):
        assert isinstance(output, DiffusionOutput), f"Request {i}: expected DiffusionOutput"
        assert len(output.diffusion_output) == 3, f"Request {i}: expected 3 channels (CHW)"
        h, w = len(output.diffusion_output[0]), len(output.diffusion_output[0][0])
        assert h > 0 and w > 0, f"Request {i}: image dimensions must be positive"
        assert output.stop_reason in ("completed", "aborted", None), f"Request {i}: unexpected stop_reason"
        assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0, f"Request {i}: pixel values must be in [0, 1]"

    # Verify logprobs for the first request
    lp = results[0].log_probs
    assert lp is not None, "log_probs should be present when logprobs=True"
    if isinstance(lp, torch.Tensor):
        assert lp.numel() > 0
    else:
        assert len(lp) > 0

    print(f"All {len(prompts)} concurrent requests returned valid DiffusionOutput")


@pytest.mark.parametrize(
    "init_server",
    [True],
    indirect=True,
    ids=["step-execution"],
)
def test_flow_grpo_step_execution_contract(init_server):
    """Verify FlowGRPO trajectory outputs with step_execution=True."""
    prompt = (
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline"
    )

    output = ray.get(
        init_server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": True,
            },
            request_id=f"step_execution_{uuid4().hex[:8]}",
        ),
        timeout=600,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3
    assert output.stop_reason in ("completed", "aborted", None)

    _assert_flow_grpo_step_execution_contract(output)
