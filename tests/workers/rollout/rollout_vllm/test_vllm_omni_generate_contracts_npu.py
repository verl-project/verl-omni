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
"""Ascend ports of the GPU rollout request-batch and step-execution contracts.

Ports the remaining cases of ``test_vllm_omni_generate.py`` (GPU core ID 1)
that ``test_vllm_omni_generate_npu.py`` (NPU Test 0) does not cover: the
GPU-side ``test_generate`` stays skipped upstream and sleep/wake remains an
NPU-specific addition. The contract assertions are imported verbatim from
the GPU module; only server launch, model path, and attention backends are
adapted to the pinned Ascend stack.
"""

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

# Reuse the GPU contract assertions unchanged so both devices enforce the
# same output contract.
from tests.workers.rollout.rollout_vllm.test_vllm_omni_generate import (
    _PROMPTS,
    _assert_flow_grpo_step_execution_contract,
    _assert_named_media_output,
    _assert_non_empty_tensor,
    _assert_training_step_execution_contract,
    _assert_valid_diffusion_output,
)
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

MODEL_PATH = Path(os.path.expanduser(os.getenv("MODEL_PATH", "~/.cache/modelscope/hub/models/tiny-random/Qwen-Image")))

_MIN_PROMPT_TOKENS = 35
# Keep smoke generates cheap; engine launch dominates wall time (GPU parity).
_SMOKE_STEPS = 2
_SMOKE_SIZE = 256

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
        "The pipeline drops the first 34 chat-template prefix tokens; "
        "use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


def _require_visible_npus() -> None:
    import torch_npu  # noqa: F401

    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        pytest.fail("These contract smokes require at least one visible Ascend NPU.")
    for component in ("model_index.json", "transformer/config.json", "tokenizer/tokenizer_config.json"):
        if not (MODEL_PATH / component).is_file():
            pytest.fail(f"Missing local Qwen-Image component: {MODEL_PATH / component}")


def _build_rollout_cfg(*, step_execution: bool = False) -> Any:
    cfg: dict[str, Any] = {
        "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
        "name": "vllm_omni",
        "mode": "async",
        "tensor_model_parallel_size": 1,
        "data_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_num_batched_tokens": 2048,
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
        "seed": 42,
        "full_determinism": False,
        "rollout_attn_backend": "TORCH_SDPA",
        "pipeline": {
            "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
            "height": _SMOKE_SIZE,
            "width": _SMOKE_SIZE,
            "num_inference_steps": _SMOKE_STEPS,
            "requested_outputs": ["image_preview", "image_latent"],
        },
    }
    if not step_execution:
        cfg["engine_kwargs"] = {
            "vllm_omni": {
                "request_batch_max_wait_ms": 10.0,
            }
        }
    return OmegaConf.create(cfg)


def _build_model_cfg(*, algorithm: str = "flow_grpo") -> Any:
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": MODEL_PATH,
            "tokenizer_path": os.path.join(MODEL_PATH, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "attn_backend": "_native_npu",
            "algorithm": algorithm,
        }
    )


def _launch_server(*, step_execution: bool = False, algorithm: str = "flow_grpo"):
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    first_device = visible_devices.split(",")[0]
    ray.init(
        runtime_env={
            "env_vars": {
                "DEVICE_NAME": "npu",
                "TOKENIZERS_PARALLELISM": "true",
                "VLLM_LOGGING_LEVEL": "WARN",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "HCCL_NPU_SOCKET_PORT_RANGE": os.getenv("HCCL_NPU_SOCKET_PORT_RANGE") or "auto",
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
                "HCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=16,
    ).remote(
        config=_build_rollout_cfg(step_execution=step_execution),
        model_config=_build_model_cfg(algorithm=algorithm),
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices=first_device,
    )
    ray.get(server.launch_server.remote())
    return server


def _shutdown_server() -> None:
    ray.shutdown()
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


@pytest.fixture(scope="module")
def init_server():
    """Module-scoped request-level (step_execution=false) server."""
    _require_visible_npus()
    server = _launch_server(step_execution=False)
    yield server
    _shutdown_server()


@pytest.fixture
def init_step_execution_server():
    """Function-scoped step-execution server (cannot share with request-level)."""
    _require_visible_npus()
    server = _launch_server(step_execution=True)
    yield server
    _shutdown_server()


@pytest.fixture
def init_training_step_execution_server(request):
    """Function-scoped NFT/DPO step-execution server."""
    _require_visible_npus()
    server = _launch_server(step_execution=True, algorithm=request.param)
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
        rid = f"npu_test_{i}_{uuid4().hex[:8]}"
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
    return ray.get(refs, timeout=900)


@pytest.mark.parametrize("output_type", ["image", "latent"])
def test_generate_request_level_batch(init_server, output_type):
    """Concurrent generate under request-level batching (max_num_seqs>1 + wait_ms)."""
    results = _generate_concurrent(
        init_server,
        _PROMPTS,
        logprobs_first_only=False,
        sampling_overrides={"output_type": output_type, "requested_outputs": ["image_preview", "image_latent"]},
    )

    assert len(results) == len(_PROMPTS)
    for i, output in enumerate(results):
        _assert_named_media_output(output, output_type=output_type)
        if output_type != "latent":
            _assert_valid_diffusion_output(output, index=i, expect_logprobs=True)
        else:
            _assert_non_empty_tensor(output.log_probs, "log_probs")

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
            request_id=f"npu_step_execution_{uuid4().hex[:8]}",
        ),
        timeout=900,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3
    assert output.stop_reason in ("completed", "aborted", None)

    _assert_named_media_output(output)
    _assert_flow_grpo_step_execution_contract(output)


@pytest.mark.parametrize(
    "init_training_step_execution_server",
    ["diffusion_nft"],
    indirect=True,
    ids=["diffusion-nft-step-execution"],
)
def test_diffusion_nft_step_execution_contract(init_training_step_execution_server):
    """Verify DiffusionNFT final-latent outputs with step_execution=True."""
    prompt = (
        "a detailed watercolor painting of a quiet forest stream surrounded by "
        "mossy stones and tall green trees under soft morning light"
    )
    output = ray.get(
        init_training_step_execution_server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={"num_inference_steps": 10, "height": 512, "width": 512},
            request_id=f"npu_diffusion_nft_step_execution_{uuid4().hex[:8]}",
        ),
        timeout=900,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3
    _assert_named_media_output(output)
    _assert_training_step_execution_contract(
        output,
        algorithm="DiffusionNFT",
        include_train_timesteps=True,
    )


@pytest.mark.parametrize(
    "init_training_step_execution_server",
    ["dpo"],
    indirect=True,
    ids=["dpo-step-execution"],
)
def test_dpo_step_execution_contract(init_training_step_execution_server):
    """Verify online DPO final-latent outputs with step_execution=True."""
    prompt = (
        "a cinematic photograph of a red tram crossing a rainy city street at "
        "night with reflections from warm shop lights on the pavement"
    )
    output = ray.get(
        init_training_step_execution_server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={"num_inference_steps": 10, "height": 512, "width": 512},
            request_id=f"npu_dpo_step_execution_{uuid4().hex[:8]}",
        ),
        timeout=900,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3
    _assert_named_media_output(output)
    _assert_training_step_execution_contract(
        output,
        algorithm="DPO",
        include_train_timesteps=False,
    )
