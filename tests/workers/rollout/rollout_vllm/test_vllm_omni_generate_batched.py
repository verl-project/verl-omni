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
E2E test for vLLMOmniHttpServer.generate_batched (one B=N forward).

Mirrors :mod:`tests.workers.rollout.rollout_vllm.test_vllm_omni_generate`
but exercises the batched API used by FlowGRPO when
``actor_rollout_ref.rollout.enable_batched_diffusion=True`` and
``actor_rollout_ref.rollout.n > 1``: one engine request produces
``num_outputs_per_prompt`` samples via a single ``QwenImagePipelineWithLogProb``
forward, and the server splits the result into per-sample
:class:`DiffusionOutput` records.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_batched.py -v -s
    python tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_batched.py
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

_MIN_PROMPT_TOKENS = 35


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"), trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        f"The pipeline drops the first 34 chat-template prefix tokens; "
        f"use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServer Ray actor with Qwen/Qwen-Image."""
    model_path = MODEL_PATH

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
            "max_num_seqs": 256,
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
            "enable_batched_diffusion": True,
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


def _assert_valid_diffusion_output(output, *, expect_log_probs: bool, tag: str) -> None:
    """Shared shape/sanity assertions for a single per-sample DiffusionOutput."""
    assert isinstance(output, DiffusionOutput), f"{tag}: expected DiffusionOutput, got {type(output)!r}"
    assert len(output.diffusion_output) == 3, f"{tag}: expected 3 channels (CHW)"
    h, w = len(output.diffusion_output[0]), len(output.diffusion_output[0][0])
    assert h > 0 and w > 0, f"{tag}: image dimensions must be positive"
    assert output.stop_reason in ("completed", "aborted", None), f"{tag}: unexpected stop_reason"
    assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0, f"{tag}: pixel values must be in [0, 1]"
    if expect_log_probs:
        lp = output.log_probs
        assert lp is not None, f"{tag}: log_probs should be present when logprobs=True"
        if isinstance(lp, torch.Tensor):
            assert lp.numel() > 0, f"{tag}: log_probs tensor must be non-empty"
        else:
            assert len(lp) > 0, f"{tag}: log_probs sequence must be non-empty"


def test_generate_batched(init_server):
    """A single ``generate_batched`` call returns ``num_outputs_per_prompt`` per-sample outputs.

    This is the exact path used by ``DiffusionAgentLoopWorker`` when
    ``enable_batched_diffusion=True`` and ``rollout.n > 1``: every prompt
    group is collapsed into one engine request that runs a single
    ``B=num_outputs_per_prompt`` transformer forward pass.
    """
    server = init_server
    num_outputs_per_prompt = 4

    prompt_ids = _tokenize_prompt(
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline"
    )

    outputs = ray.get(
        server.generate_batched.remote(
            prompt_ids=prompt_ids,
            sampling_params={
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": True,
            },
            request_id=f"batched_{uuid4().hex[:8]}",
            num_outputs_per_prompt=num_outputs_per_prompt,
        ),
        timeout=600,
    )

    assert isinstance(outputs, list), f"expected list[DiffusionOutput], got {type(outputs)!r}"
    assert len(outputs) == num_outputs_per_prompt, f"expected {num_outputs_per_prompt} samples, got {len(outputs)}"
    for i, out in enumerate(outputs):
        _assert_valid_diffusion_output(out, expect_log_probs=True, tag=f"sample {i}")

    print(f"generate_batched returned {num_outputs_per_prompt} valid DiffusionOutputs")


def test_generate_batched_n1_matches_generate(init_server):
    """``generate_batched(num_outputs_per_prompt=1)`` returns the same shape as ``generate``.

    Guards the refactor that re-implemented ``generate`` on top of
    ``_generate_engine_call(num_outputs_per_prompt=1)``.
    """
    server = init_server
    prompt_ids = _tokenize_prompt(
        "a fluffy orange cat sitting on a wooden windowsill looking outside at "
        "a garden full of colorful flowers on a bright sunny afternoon"
    )
    sampling_params = {
        "num_inference_steps": 10,
        "true_cfg_scale": 4.0,
        "height": 512,
        "width": 512,
        "logprobs": True,
    }

    single = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=f"single_{uuid4().hex[:8]}",
        ),
        timeout=600,
    )
    batched = ray.get(
        server.generate_batched.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=f"batched_n1_{uuid4().hex[:8]}",
            num_outputs_per_prompt=1,
        ),
        timeout=600,
    )

    _assert_valid_diffusion_output(single, expect_log_probs=True, tag="generate")
    assert isinstance(batched, list) and len(batched) == 1, "generate_batched(n=1) must return a 1-element list"
    _assert_valid_diffusion_output(batched[0], expect_log_probs=True, tag="generate_batched[0]")


def test_generate_batched_invalid_n(init_server):
    """``num_outputs_per_prompt < 1`` raises ``ValueError`` before touching the engine."""
    server = init_server
    prompt_ids = _tokenize_prompt(
        "a majestic mountain landscape covered with fresh white snow under a "
        "clear blue sky with pine trees in the foreground and a frozen lake"
    )
    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        ray.get(
            server.generate_batched.remote(
                prompt_ids=prompt_ids,
                sampling_params={"num_inference_steps": 10, "height": 512, "width": 512},
                request_id=f"bad_{uuid4().hex[:8]}",
                num_outputs_per_prompt=0,
            ),
            timeout=60,
        )
    assert "num_outputs_per_prompt" in str(exc_info.value)
