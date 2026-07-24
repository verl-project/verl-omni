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

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any
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

_MIN_PROMPT_TOKENS = 35
# Keep smoke generates cheap; engine launch dominates wall time.
_SMOKE_STEPS = 2
_SMOKE_SIZE = 256

_PROMPTS = [
    "a beautiful sunset over the ocean with vibrant orange and purple clouds "
    "reflecting on the calm water surface near a rocky coastline",
    "a fluffy orange cat sitting on a wooden windowsill looking outside at "
    "a garden full of colorful flowers on a bright sunny afternoon",
    "a majestic mountain landscape covered with fresh white snow under a "
    "clear blue sky with pine trees in the foreground and a frozen lake",
    "a futuristic city at night with neon lights glowing on tall glass "
    "skyscrapers and flying vehicles soaring between the buildings",
]

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"), trust_remote_code=True)
    return _TOKENIZER


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(
        _get_tokenizer().apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
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


def _build_rollout_cfg(*, step_execution: bool = False) -> Any:
    from tests.utils.smoke_attention import resolve_smoke_attention_backends

    _, rollout_attn_backend = resolve_smoke_attention_backends()
    cfg: dict[str, Any] = {
        "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
        "name": "vllm_omni",
        "mode": "async",
        "tensor_model_parallel_size": 1,
        "data_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_num_batched_tokens": 8192,
        # Request-level packing vs step-wise continuous batching are mutually exclusive.
        "max_num_seqs": 16 if step_execution else 8,
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
        "n": 1,
        "rollout_attn_backend": rollout_attn_backend,
        "pipeline": {
            "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
            "height": _SMOKE_SIZE,
            "width": _SMOKE_SIZE,
            "num_inference_steps": _SMOKE_STEPS,
        },
    }
    if not step_execution:
        cfg["engine_kwargs"] = {
            "vllm_omni": {
                "request_batch_max_wait_ms": 10.0,
            }
        }
    return OmegaConf.create(cfg)


def _build_model_cfg(*, attn_backend: str | None = None) -> Any:
    from tests.utils.smoke_attention import resolve_smoke_attention_backends

    resolved_attn_backend, _ = resolve_smoke_attention_backends()
    model_path = MODEL_PATH
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": model_path,
            "tokenizer_path": os.path.join(model_path, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "attn_backend": attn_backend or resolved_attn_backend,
            "algorithm": "flow_grpo",
        }
    )


def _launch_server(*, step_execution: bool = False):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
            }
        },
        ignore_reinit_error=True,
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
        max_concurrency=16,
    ).remote(
        config=_build_rollout_cfg(step_execution=step_execution),
        model_config=_build_model_cfg(),
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )
    ray.get(server.launch_server.remote())
    return server


def _shutdown_server() -> None:
    ray.shutdown()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def init_server():
    """Module-scoped request-level (step_execution=false) server."""
    server = _launch_server(step_execution=False)
    yield server
    _shutdown_server()


@pytest.fixture
def init_step_execution_server():
    """Function-scoped step-execution server (cannot share with request-level)."""
    server = _launch_server(step_execution=True)
    yield server
    _shutdown_server()


def _generate_concurrent(
    server,
    prompts: list[str],
    *,
    logprobs_first_only: bool = True,
    sampling_overrides: dict[str, Any] | None = None,
) -> list[DiffusionOutput]:
    sampling_overrides = sampling_overrides or {}
    refs = []
    for i, prompt in enumerate(prompts):
        rid = f"test_{i}_{uuid4().hex[:8]}"
        sampling_params = {
            "num_inference_steps": _SMOKE_STEPS,
            "true_cfg_scale": 1.0,
            "height": _SMOKE_SIZE,
            "width": _SMOKE_SIZE,
            "logprobs": (i == 0) if logprobs_first_only else True,
            **sampling_overrides,
        }
        refs.append(
            server.generate.remote(
                prompt_ids=_tokenize_prompt(prompt),
                sampling_params=sampling_params,
                request_id=rid,
            )
        )
    return ray.get(refs, timeout=600)


def _assert_valid_diffusion_output(output: DiffusionOutput, *, index: int, expect_logprobs: bool = False) -> None:
    assert isinstance(output, DiffusionOutput), f"Request {index}: expected DiffusionOutput"
    assert len(output.diffusion_output) == 3, f"Request {index}: expected 3 channels (CHW)"
    h, w = len(output.diffusion_output[0]), len(output.diffusion_output[0][0])
    assert h > 0 and w > 0, f"Request {index}: image dimensions must be positive"
    assert output.stop_reason in ("completed", "aborted", None), f"Request {index}: unexpected stop_reason"
    assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0, f"Request {index}: pixel values must be in [0, 1]"
    if expect_logprobs:
        lp = output.log_probs
        assert lp is not None, f"Request {index}: log_probs should be present when logprobs=True"
        if isinstance(lp, torch.Tensor):
            assert lp.numel() > 0
        else:
            assert len(lp) > 0


def test_generate(init_server):
    """Concurrent generate() covering basic output, logprobs, and multi-request correctness."""
    results = _generate_concurrent(init_server, _PROMPTS, logprobs_first_only=True)

    for i, output in enumerate(results):
        _assert_valid_diffusion_output(output, index=i, expect_logprobs=(i == 0))

    print(f"All {len(_PROMPTS)} concurrent requests returned valid DiffusionOutput")


def test_generate_request_level_batch(init_server):
    """Concurrent generate under request-level batching (max_num_seqs>1 + wait_ms)."""
    results = _generate_concurrent(
        init_server,
        _PROMPTS,
        logprobs_first_only=False,
    )

    assert len(results) == len(_PROMPTS)
    for i, output in enumerate(results):
        _assert_valid_diffusion_output(output, index=i, expect_logprobs=True)

    print(f"All {len(_PROMPTS)} request-level-batched generates returned valid DiffusionOutput")


def test_flow_grpo_step_execution_contract(init_step_execution_server):
    """Verify FlowGRPO trajectory outputs with step_execution=True."""
    prompt = (
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline"
    )

    output = ray.get(
        init_step_execution_server.generate.remote(
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
