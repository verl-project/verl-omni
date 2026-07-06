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
Debug/regression test for vllm-omni request-wise batching under FlowGRPO OCR.

Replays OCR train-parquet prompts, negative prompts, and FlowGRPO sampling
params, then fires a burst of concurrent ``vLLMOmniHttpServer.generate()``
calls — the same ingress path as ``DiffusionSingleTurnAgentLoop``.

Instrumentation in ``QwenImagePipelineWithLogProb`` writes per-forward batch
sizes to ``FLOWGRPO_BATCH_VERIFY_LOG``. This test parses that log and fails
with a histogram when fused batch sizes stay too small.

Usage:
    FLOWGRPO_BATCH_VERIFY_LOG=/tmp/flowgrpo_batch_verify_test.log \\
    pytest tests/workers/rollout/rollout_vllm/test_vllm_omni_request_batch_flowgrpo.py -v -s

Optional env overrides:
    FLOWGRPO_REQUEST_BATCH_MODEL_PATH
    FLOWGRPO_REQUEST_BATCH_OCR_PARQUET
    FLOWGRPO_REQUEST_BATCH_NUM_CONCURRENT  (default: 32, train_batch_size)
    FLOWGRPO_REQUEST_BATCH_MAX_NUM_SEQS    (default: 32)
    FLOWGRPO_MIN_EXPECTED_MAX_BATCH        (default: 8; smoke: 1)
    FLOWGRPO_REQUEST_BATCH_MAX_WAIT_MS   (default: 250)
    FLOWGRPO_REQUEST_BATCH_MIN_SIZE        (default: 0, auto max_num_seqs//2)
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode

from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

DEFAULT_MODEL_PATH = Path(os.environ.get("FLOWGRPO_REQUEST_BATCH_MODEL_PATH", "/home/public/yx/models/Qwen/Qwen-Image"))
DEFAULT_OCR_PARQUET = Path(
    os.environ.get(
        "FLOWGRPO_REQUEST_BATCH_OCR_PARQUET",
        "/home/public/yx/data/ocr/qwen_image/train.parquet",
    )
)
NUM_CONCURRENT = int(os.environ.get("FLOWGRPO_REQUEST_BATCH_NUM_CONCURRENT", "32"))
MAX_NUM_SEQS = int(os.environ.get("FLOWGRPO_REQUEST_BATCH_MAX_NUM_SEQS", "32"))
MIN_EXPECTED_MAX_BATCH = int(os.environ.get("FLOWGRPO_MIN_EXPECTED_MAX_BATCH", "8"))
REQUEST_BATCH_MAX_WAIT_MS = float(os.environ.get("FLOWGRPO_REQUEST_BATCH_MAX_WAIT_MS", "250"))
REQUEST_BATCH_MIN_SIZE = int(os.environ.get("FLOWGRPO_REQUEST_BATCH_MIN_SIZE", "0"))
_MIN_PROMPT_TOKENS = 35

_NUM_REQS_RE = re.compile(r"num_reqs=(\d+)")
_FUSED_B_RE = re.compile(r"fused_B=(\d+)")


def _messages_from_parquet_cell(cell: Any) -> list[dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in cell]


def load_flowgrpo_ocr_samples(parquet_path: Path, *, num_samples: int) -> list[dict[str, Any]]:
    """Load OCR parquet rows in the same message shape used by RLHFDataset."""
    df = pd.read_parquet(parquet_path)
    samples: list[dict[str, Any]] = []
    for idx in range(min(num_samples, len(df))):
        row = df.iloc[idx]
        samples.append(
            {
                "raw_prompt": _messages_from_parquet_cell(row["prompt"]),
                "raw_negative_prompt": _messages_from_parquet_cell(row["negative_prompt"]),
                "ground_truth": row["reward_model"]["ground_truth"],
            }
        )
    return samples


def tokenize_chat_messages(model_path: Path, messages: list[dict[str, str]]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path / "tokenizer",
        trust_remote_code=True,
    )
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        "Qwen-Image drops the first 34 chat-template prefix tokens."
    )
    return token_ids


def flowgrpo_training_sampling_params(*, seed: int) -> dict[str, Any]:
    """Sampling params aligned with ``run_qwen_image_ocr_lora.sh`` rollout."""
    return {
        "height": 512,
        "width": 512,
        "num_inference_steps": 10,
        "true_cfg_scale": 4.0,
        "max_sequence_length": 256,
        "noise_level": 1.2,
        "sde_type": "sde",
        "sde_window_size": 2,
        "sde_window_range": (0, 5),
        "logprobs": True,
        "seed": seed,
    }


def parse_batch_verify_log(log_path: Path) -> dict[str, Counter[int]]:
    """Parse ``[flowgrpo_reqbatch]`` lines into histograms."""
    num_reqs_hist: Counter[int] = Counter()
    fused_b_hist: Counter[int] = Counter()
    if not log_path.exists():
        return {"num_reqs": num_reqs_hist, "fused_B": fused_b_hist}

    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            if "num_reqs=" in line:
                for match in _NUM_REQS_RE.finditer(line):
                    num_reqs_hist[int(match.group(1))] += 1
            if "fused_B=" in line:
                for match in _FUSED_B_RE.finditer(line):
                    fused_b_hist[int(match.group(1))] += 1
    return {"num_reqs": num_reqs_hist, "fused_B": fused_b_hist}


def summarize_batch_histogram(hist: Counter[int]) -> str:
    if not hist:
        return "(empty)"
    lines = [f"  size={size}: {count}" for size, count in sorted(hist.items())]
    return "\n".join(lines)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_dir(), reason=f"model missing at {DEFAULT_MODEL_PATH}"),
    pytest.mark.skipif(not DEFAULT_OCR_PARQUET.is_file(), reason=f"OCR parquet missing at {DEFAULT_OCR_PARQUET}"),
]


@pytest.fixture
def flowgrpo_request_batch_server(tmp_path):
    """Launch one vLLMOmniHttpServer with FlowGRPO OCR rollout settings."""
    verify_log = tmp_path / "flowgrpo_batch_verify.log"
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

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
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
            "engine_kwargs": {
                "vllm_omni": {
                    "step_execution": False,
                    "request_batch_max_wait_ms": REQUEST_BATCH_MAX_WAIT_MS,
                    "request_batch_min_size": REQUEST_BATCH_MIN_SIZE,
                }
            },
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
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
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": str(DEFAULT_MODEL_PATH),
            "tokenizer_path": str(DEFAULT_MODEL_PATH / "tokenizer"),
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
                "FLOWGRPO_BATCH_VERIFY_LOG": str(verify_log),
            }
        },
        max_concurrency=NUM_CONCURRENT + 4,
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

    yield server, verify_log

    ray.shutdown()


def test_flowgrpo_concurrent_request_batch_sizes(flowgrpo_request_batch_server):
    """Burst concurrent FlowGRPO OCR requests and report fused scheduler batch sizes."""
    server, verify_log = flowgrpo_request_batch_server
    samples = load_flowgrpo_ocr_samples(DEFAULT_OCR_PARQUET, num_samples=NUM_CONCURRENT)
    tokenized = [
        (
            tokenize_chat_messages(DEFAULT_MODEL_PATH, sample["raw_prompt"]),
            tokenize_chat_messages(DEFAULT_MODEL_PATH, sample["raw_negative_prompt"]),
        )
        for sample in samples
    ]

    refs = []
    for i, (prompt_ids, negative_prompt_ids) in enumerate(tokenized):
        rid = f"flowgrpo_batch_{i}_{uuid4().hex[:8]}"
        refs.append(
            server.generate.remote(
                prompt_ids=prompt_ids,
                negative_prompt_ids=negative_prompt_ids,
                sampling_params=flowgrpo_training_sampling_params(seed=1000 + i),
                request_id=rid,
            )
        )

    results = ray.get(refs, timeout=3600)
    for i, output in enumerate(results):
        assert isinstance(output, DiffusionOutput), f"request {i}: expected DiffusionOutput"
        assert output.diffusion_output is not None

    hist = parse_batch_verify_log(verify_log)
    num_reqs_hist = hist["num_reqs"]
    fused_b_hist = hist["fused_B"]
    max_num_reqs = max(num_reqs_hist) if num_reqs_hist else 0
    max_fused_b = max(fused_b_hist) if fused_b_hist else 0

    print(
        "\n=== FlowGRPO request-batch debug summary ===\n"
        f"concurrent_requests={NUM_CONCURRENT}\n"
        f"max_num_seqs={MAX_NUM_SEQS}\n"
        f"request_batch_max_wait_ms={REQUEST_BATCH_MAX_WAIT_MS}\n"
        f"request_batch_min_size={REQUEST_BATCH_MIN_SIZE}\n"
        f"verify_log={verify_log}\n"
        f"max_num_reqs_seen={max_num_reqs}\n"
        f"max_fused_B_seen={max_fused_b}\n"
        "num_reqs histogram:\n"
        f"{summarize_batch_histogram(num_reqs_hist)}\n"
        "fused_B histogram:\n"
        f"{summarize_batch_histogram(fused_b_hist)}\n"
    )

    assert num_reqs_hist, (
        f"No [flowgrpo_reqbatch] lines in {verify_log}. "
        "Ensure QwenImagePipelineWithLogProb batch instrumentation is active."
    )

    assert max_num_reqs >= MIN_EXPECTED_MAX_BATCH, (
        "vllm-omni request-wise batching did not reach the expected fused batch size "
        f"under FlowGRPO OCR ingress (max_num_reqs={max_num_reqs}, "
        f"max_fused_B={max_fused_b}, max_num_seqs={MAX_NUM_SEQS}, "
        f"concurrent={NUM_CONCURRENT}).\n"
        "num_reqs histogram:\n"
        f"{summarize_batch_histogram(num_reqs_hist)}\n"
        "fused_B histogram:\n"
        f"{summarize_batch_histogram(fused_b_hist)}\n"
        "This reproduces training ingress: OCR parquet prompts, true_cfg_scale=4.0, "
        "negative prompts, per-request seeds, and burst concurrent HTTP generates."
    )
