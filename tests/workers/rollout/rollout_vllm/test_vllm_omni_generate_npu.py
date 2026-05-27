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
NPU smoke test for vLLM-Omni rollout.
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


def _resolve_diffusion_npu_topology(default_num_npus: int = 1) -> tuple[int, int]:
    requested_npus = max(1, int(os.getenv("NUM_NPUS", str(default_num_npus))))
    tp_size = min(2, requested_npus)
    return requested_npus, tp_size


def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"), trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS
    return token_ids


@pytest.fixture
def init_server():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            }
        },
        ignore_reinit_error=True,
    )

    requested_npus, tp_size = _resolve_diffusion_npu_topology()
    model_path = MODEL_PATH

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": tp_size,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.3,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 64,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": True,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 2,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 4,
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
        }
    )
    model_cfg.architecture = "QwenImageTransformer2DModel"

    ServerCls = ray.remote(vLLMOmniHttpServer)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "HCCL_CUMEM_ENABLE": "0",
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
        gpus_per_node=requested_npus,
        nnodes=1,
        cuda_visible_devices=",".join(str(i) for i in range(requested_npus)),
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


def test_generate_and_sleep_wakeup(init_server):
    server = init_server

    prompt = (
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline"
    )
    request_id = f"npu_{uuid4().hex[:8]}"

    output = ray.get(
        server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={
                "num_inference_steps": 4,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": True,
            },
            request_id=request_id,
        ),
        timeout=600,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3
    assert output.stop_reason in ("completed", "aborted", None)
    assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0
    assert output.log_probs is not None

    ray.get(server.sleep.remote())
    ray.get(server.wake_up.remote())

    output_2 = ray.get(
        server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={
                "num_inference_steps": 4,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
            },
            request_id=f"npu_{uuid4().hex[:8]}",
        ),
        timeout=600,
    )

    assert isinstance(output_2, DiffusionOutput)
    assert len(output_2.diffusion_output) == 3
    assert output_2.stop_reason in ("completed", "aborted", None)
