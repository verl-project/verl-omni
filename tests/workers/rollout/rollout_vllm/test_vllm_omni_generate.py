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


def _build_rollout_cfg() -> Any:
    from tests.utils.smoke_attention import resolve_smoke_attention_backends

    _, rollout_attn_backend = resolve_smoke_attention_backends()
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 8,
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
            "engine_kwargs": {
                "vllm_omni": {
                    "request_batch_max_wait_ms": 10.0,
                }
            },
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": _SMOKE_SIZE,
                "width": _SMOKE_SIZE,
                "num_inference_steps": _SMOKE_STEPS,
            },
        }
    )


def _build_model_cfg() -> Any:
    model_path = MODEL_PATH
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": model_path,
            "tokenizer_path": os.path.join(model_path, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "algorithm": "flow_grpo",
        }
    )


@pytest.fixture(scope="module")
def init_server():
    """Module-scoped server shared by generate smokes (launch dominates runtime)."""
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
        config=_build_rollout_cfg(),
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
    yield server
    ray.shutdown()


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
    """Concurrent generate() calls covering basic output, logprobs, and multi-request correctness."""
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
