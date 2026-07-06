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
Multi-server FlowGRPO routing benchmark.

Compares rollout replica routing policies under the same OCR burst pattern used
in training (train_batch_size x rollout.n concurrent generates).

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_rollout_server_routing_perf.py -v -s

Env overrides:
    FLOWGRPO_ROUTING_NUM_REPLICAS       (default: 4)
    FLOWGRPO_ROUTING_NUM_PROMPTS        (default: 32)
    FLOWGRPO_ROUTING_N                  (default: 16)
    FLOWGRPO_ROUTING_POLICY             (default: prompt_uid_affinity)
    FLOWGRPO_ROUTING_COMPARE_POLICIES   (default: 1, run least_inflight then uid affinity)
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf
from verl.workers.rollout.replica import RolloutMode

from verl_omni.workers.rollout.omni_llm_server import OmniLLMServerClient
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.request_routing import OmniRequestLoadBalancer
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

_flowgrpo_spec = importlib.util.spec_from_file_location(
    "flowgrpo_batch_helpers",
    Path(__file__).with_name("test_vllm_omni_request_batch_flowgrpo.py"),
)
_flowgrpo_helpers = importlib.util.module_from_spec(_flowgrpo_spec)
assert _flowgrpo_spec.loader is not None
_flowgrpo_spec.loader.exec_module(_flowgrpo_helpers)

DEFAULT_MODEL_PATH = _flowgrpo_helpers.DEFAULT_MODEL_PATH
DEFAULT_OCR_PARQUET = _flowgrpo_helpers.DEFAULT_OCR_PARQUET
REQUEST_BATCH_MAX_WAIT_MS = _flowgrpo_helpers.REQUEST_BATCH_MAX_WAIT_MS
REQUEST_BATCH_MIN_SIZE = _flowgrpo_helpers.REQUEST_BATCH_MIN_SIZE
flowgrpo_training_sampling_params = _flowgrpo_helpers.flowgrpo_training_sampling_params
load_flowgrpo_ocr_samples = _flowgrpo_helpers.load_flowgrpo_ocr_samples
parse_batch_verify_log = _flowgrpo_helpers.parse_batch_verify_log
summarize_batch_histogram = _flowgrpo_helpers.summarize_batch_histogram
tokenize_chat_messages = _flowgrpo_helpers.tokenize_chat_messages

NUM_REPLICAS = int(os.environ.get("FLOWGRPO_ROUTING_NUM_REPLICAS", "4"))
NUM_PROMPTS = int(os.environ.get("FLOWGRPO_ROUTING_NUM_PROMPTS", "32"))
ROLLOUT_N = int(os.environ.get("FLOWGRPO_ROUTING_N", "16"))
ROUTING_POLICY = os.environ.get("FLOWGRPO_ROUTING_POLICY", "prompt_uid_affinity")
COMPARE_POLICIES = os.environ.get("FLOWGRPO_ROUTING_COMPARE_POLICIES", "1") == "1"
MAX_NUM_SEQS = int(os.environ.get("FLOWGRPO_REQUEST_BATCH_MAX_NUM_SEQS", "32"))
STEP_EXECUTION = os.environ.get("FLOWGRPO_ROUTING_STEP_EXECUTION", "0") == "1"
NUM_INFERENCE_STEPS = int(os.environ.get("FLOWGRPO_ROUTING_NUM_INFERENCE_STEPS", "10"))
_MIN_PROMPT_TOKENS = 35

_NUM_REQS_RE = re.compile(r"num_reqs=(\d+)")


def _build_rollout_cfg() -> Any:
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": float(os.environ.get("FLOWGRPO_ROUTING_GPU_MEM_UTIL", "0.45")),
            "max_num_batched_tokens": 8192,
            "max_num_seqs": MAX_NUM_SEQS,
            "dtype": "bfloat16",
            "load_format": "safetensors",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": False,
            "disable_log_stats": True,
            "calculate_log_probs": True,
            "step_execution": STEP_EXECUTION,
            "engine_kwargs": {
                "vllm_omni": {
                    "step_execution": STEP_EXECUTION,
                }
            },
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": 4.0,
                "max_sequence_length": 256,
            },
            "algo": {
                "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutAlgoConfig",
                "noise_level": 1.2,
                "sde_type": "sde",
                "sde_window_size": 2,
                "sde_window_range": [0, 5],
            },
            "server_routing": {
                "_target_": "verl_omni.workers.config.RolloutServerRoutingConfig",
                "policy": ROUTING_POLICY,
            },
        }
    )


def _build_model_cfg() -> Any:
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": str(DEFAULT_MODEL_PATH),
            "tokenizer_path": str(DEFAULT_MODEL_PATH / "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "algorithm": "flow_grpo",
        }
    )


def _build_trainer_cfg(policy: str) -> Any:
    rollout_cfg = _build_rollout_cfg()
    rollout_cfg.server_routing.policy = policy
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": rollout_cfg,
                "model": _build_model_cfg(),
            }
        }
    )


async def _run_routed_burst(
    *,
    client: OmniLLMServerClient,
    tokenized: list[tuple[list[int], list[int]]],
    num_prompts: int,
    rollout_n: int,
) -> float:
    async def _one_generate(prompt_idx: int, rollout_idx: int, prompt_ids: list[int], neg_ids: list[int]):
        uid = f"flowgrpo-uid-{prompt_idx}"
        return await client.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            negative_prompt_ids=neg_ids,
            sampling_params=flowgrpo_training_sampling_params(seed=1000 + prompt_idx * rollout_n + rollout_idx),
            routing_key=uid,
        )

    tasks = []
    for prompt_idx in range(num_prompts):
        prompt_ids, neg_ids = tokenized[prompt_idx % len(tokenized)]
        for rollout_idx in range(rollout_n):
            tasks.append(_one_generate(prompt_idx, rollout_idx, prompt_ids, neg_ids))

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    for output in results:
        assert isinstance(output, DiffusionOutput)
        assert output.diffusion_output is not None
    return elapsed


@pytest.fixture(scope="module")
def multi_server_cluster(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.device_count() < NUM_REPLICAS:
        pytest.skip(f"requires >= {NUM_REPLICAS} CUDA devices")
    if not DEFAULT_MODEL_PATH.is_dir() or not DEFAULT_OCR_PARQUET.is_file():
        pytest.skip("model or OCR parquet missing")

    verify_log = tmp_path_factory.mktemp("routing_perf") / "flowgrpo_batch_verify.log"
    os.environ["FLOWGRPO_BATCH_VERIFY_LOG"] = str(verify_log)

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "FLOWGRPO_BATCH_VERIFY_LOG": str(verify_log),
            }
        },
        ignore_reinit_error=True,
    )

    rollout_cfg = _build_rollout_cfg()
    model_cfg = _build_model_cfg()
    ServerCls = ray.remote(vLLMOmniHttpServer)
    servers = []
    server_map: dict[str, ray.actor.ActorHandle] = {}

    for replica_rank in range(NUM_REPLICAS):
        server = ServerCls.options(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": "0",
                    "FLOWGRPO_BATCH_VERIFY_LOG": str(verify_log),
                }
            },
            max_concurrency=NUM_PROMPTS * ROLLOUT_N + 8,
        ).remote(
            config=rollout_cfg,
            model_config=model_cfg,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=replica_rank,
            node_rank=0,
            gpus_per_node=NUM_REPLICAS,
            nnodes=1,
            cuda_visible_devices=str(replica_rank),
        )
        servers.append(server)
        server_map[f"replica-{replica_rank}"] = server

    ray.get([server.launch_server.remote() for server in servers])

    yield servers, server_map, verify_log
    ray.shutdown()


@pytest.mark.parametrize("policy", ["prompt_uid_affinity"])
def test_flowgrpo_routing_policy_performance(multi_server_cluster, policy: str):
    servers, server_map, verify_log = multi_server_cluster
    del servers  # accessed via LB

    samples = load_flowgrpo_ocr_samples(DEFAULT_OCR_PARQUET, num_samples=NUM_PROMPTS)
    tokenized = [
        (
            tokenize_chat_messages(DEFAULT_MODEL_PATH, sample["raw_prompt"]),
            tokenize_chat_messages(DEFAULT_MODEL_PATH, sample["raw_negative_prompt"]),
        )
        for sample in samples
    ]

    policies = ["least_inflight", "prompt_uid_affinity"] if COMPARE_POLICIES else [policy]
    results: list[dict[str, Any]] = []

    for routing_policy in policies:
        verify_log.write_text("")
        lb = OmniRequestLoadBalancer.remote(servers=server_map, policy=routing_policy)
        trainer_cfg = _build_trainer_cfg(routing_policy)
        client = OmniLLMServerClient(config=trainer_cfg, load_balancer_handle=lb)

        elapsed = asyncio.run(
            _run_routed_burst(
                client=client,
                tokenized=tokenized,
                num_prompts=NUM_PROMPTS,
                rollout_n=ROLLOUT_N,
            )
        )

        hist = parse_batch_verify_log(verify_log)
        num_reqs_hist = hist["num_reqs"]
        max_num_reqs = max(num_reqs_hist) if num_reqs_hist else 0
        lb_status = ray.get(lb.get_status.remote())

        row = {
            "policy": routing_policy,
            "elapsed_s": elapsed,
            "max_num_reqs": max_num_reqs,
            "total_requests": NUM_PROMPTS * ROLLOUT_N,
            "lb_status": lb_status,
            "num_reqs_hist": num_reqs_hist,
        }
        results.append(row)

        print(
            f"\n=== Routing benchmark ({routing_policy}) ===\n"
            f"replicas={NUM_REPLICAS} prompts={NUM_PROMPTS} rollout_n={ROLLOUT_N}\n"
            f"wall_time_s={elapsed:.2f}\n"
            f"max_num_reqs={max_num_reqs}\n"
            f"lb_status={lb_status}\n"
            f"num_reqs histogram:\n{summarize_batch_histogram(num_reqs_hist)}\n"
        )

    if COMPARE_POLICIES and len(results) == 2:
        least = next(r for r in results if r["policy"] == "least_inflight")
        affinity = next(r for r in results if r["policy"] == "prompt_uid_affinity")
        speedup = least["elapsed_s"] / affinity["elapsed_s"] if affinity["elapsed_s"] > 0 else 0.0
        print(
            "\n=== Policy comparison ===\n"
            f"least_inflight: {least['elapsed_s']:.2f}s, max_num_reqs={least['max_num_reqs']}\n"
            f"prompt_uid_affinity: {affinity['elapsed_s']:.2f}s, max_num_reqs={affinity['max_num_reqs']}\n"
            f"affinity_speedup={speedup:.2f}x\n"
        )
        assert affinity["max_num_reqs"] >= least["max_num_reqs"], (
            "prompt_uid_affinity should co-locate rollout.n copies and improve fused batch sizes"
        )

    final = results[-1]
    if not STEP_EXECUTION:
        assert final["max_num_reqs"] >= 8, (
            f"Expected fused batches under {final['policy']}, got max_num_reqs={final['max_num_reqs']}"
        )
