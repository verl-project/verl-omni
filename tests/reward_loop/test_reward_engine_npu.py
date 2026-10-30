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
"""Ascend port of test_visual_reward_manager.py::test_reward_model_genrm."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def npu_reward_checkpoint(monkeypatch, tmp_path_factory):
    """Prepare a reusable checkpoint and a fresh model-inspection cache.

    NPU_REWARD_MODEL_PATH selects a test-owned checkpoint directory. The builder
    rejects unrelated or incomplete contents. A fresh VLLM_CACHE_ROOT exercises
    cold inspection; monkeypatch restores the environment after the test.
    """
    try:
        requested = int(os.environ.get("NUM_NPUS", "8"))
    except ValueError:
        pytest.fail("NUM_NPUS must be an integer.")
    if requested < 2 or requested % 2:
        pytest.fail("NUM_NPUS must be even and at least two.")

    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or ",".join(str(i) for i in range(requested))
    env_vars = {
        "DEVICE_NAME": "npu",
        "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
        "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES") or visible_devices,
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE") or "auto",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ASCEND_ENABLE_NZ": "0",
        "VLLM_CACHE_ROOT": str(tmp_path_factory.mktemp("reward-registry")),
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available() or requested > torch.npu.device_count():
        pytest.fail("The standalone reward smoke requires at least NUM_NPUS visible Ascend NPUs.")

    from tests.reward_loop.build_qwen25_vl_reward_tiny_random import ensure_tiny_qwen25_vl_reward_checkpoint

    model_path = Path(
        os.environ.get("NPU_REWARD_MODEL_PATH", "~/.cache/verl-omni-npu-smoke/Qwen2.5-VL-Reward-NPU")
    ).expanduser()
    model_path = ensure_tiny_qwen25_vl_reward_checkpoint(model_path).resolve()
    print(f"Reward model-inspection cache: {env_vars['VLLM_CACHE_ROOT']}")
    return requested, model_path


def test_standalone_reward_engine_npu(npu_reward_checkpoint):
    """Start dedicated TP=2 reward replicas and score three synthetic images.

    Check every /v1/models endpoint, finite (3, 1) OCR scores in [0, 1], and one
    text response per image. Random weights exercise execution, not OCR quality.
    Servers stay resident; reward offload and sleep/wake are outside this case.
    """
    import ray
    import requests
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.protocol import DataProto
    from verl.utils import hf_tokenizer
    from verl.workers.rollout.replica import RolloutMode

    requested, model_path = npu_reward_checkpoint
    if ray.is_initialized():
        pytest.fail("Run this test without an existing Ray connection.")

    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "verl_omni/trainer/config"), version_base="1.3"):
        config = compose(config_name="diffusion_trainer")
    config.trainer.device = "npu"
    config.trainer.n_gpus_per_node = requested
    config.actor_rollout_ref.model.path = str(model_path)
    config.actor_rollout_ref.model.tokenizer_path = str(model_path)
    config.reward.num_workers = 1
    config.reward.custom_reward_function.path = str(repo_root / "verl_omni/utils/reward_score/genrm_ocr.py")
    config.reward.custom_reward_function.name = "compute_score_ocr"
    reward = config.reward.reward_model
    reward.enable = True
    reward.enable_resource_pool = True
    reward.n_gpus_per_node = requested
    reward.nnodes = 1
    reward.model_path = str(model_path)
    rollout = reward.rollout
    rollout.name = "vllm"
    rollout.tensor_model_parallel_size = 2
    rollout.dtype = "bfloat16"
    rollout.load_format = "safetensors"
    rollout.gpu_memory_utilization = 0.25
    rollout.enforce_eager = True
    rollout.free_cache_engine = False
    rollout.skip_tokenizer_init = False
    rollout.max_model_len = 512
    rollout.max_num_batched_tokens = 512
    rollout.max_num_seqs = 1
    rollout.prompt_length = 256
    rollout.response_length = 8
    rollout.enable_prefix_caching = False
    with open_dict(rollout.engine_kwargs):
        rollout.engine_kwargs.vllm = {
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "mm_processor_kwargs": {"min_pixels": 56 * 56, "max_pixels": 112 * 112},
        }

    env_vars = {
        key: os.environ[key]
        for key in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "VLLM_CACHE_ROOT",
            "PYTHONPATH",
            "HCCL_NPU_SOCKET_PORT_RANGE",
        )
        if key in os.environ
    }
    env_vars.update(
        {
            "DEVICE_NAME": "npu",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        }
    )
    ray.init(
        address="local",
        num_cpus=min(os.cpu_count() or 16, 32),
        include_dashboard=False,
        runtime_env={"env_vars": env_vars},
    )
    try:
        assert ray.cluster_resources().get("NPU", 0) >= requested
        manager = RewardLoopManager(config)
        engine = manager.reward_model_manager
        assert engine is not None and engine.resource_pool is None
        assert len(engine.rollout_replicas) == requested // 2
        for replica in engine.rollout_replicas:
            assert replica.is_reward_model and replica.rollout_mode == RolloutMode.STANDALONE
        with requests.Session() as session:
            session.trust_env = False
            for address in engine.server_addresses:
                response = session.get(f"http://{address}/v1/models", timeout=30)
                response.raise_for_status()
                assert str(model_path) in {entry["id"] for entry in response.json()["data"]}

        tokenizer = hf_tokenizer(str(model_path))
        prompt = torch.tensor(tokenizer.encode("Read the text in this image."), dtype=torch.long)
        batch_size = 3
        data = DataProto.from_dict(
            tensors={
                "input_ids": prompt.unsqueeze(0).repeat(batch_size, 1),
                "responses": (torch.arange(batch_size * 3 * 56 * 56) % 256)
                .to(torch.uint8)
                .reshape(batch_size, 3, 56, 56),
            },
            non_tensors={
                "data_source": ["ocr"] * batch_size,
                "reward_model": [{"ground_truth": "OCR"}] * batch_size,
                "media_kind": ["image"] * batch_size,
                "extra_info": [{} for _ in range(batch_size)],
            },
        )
        output = manager.compute_rm_score(data)
        scores = output.batch["rm_scores"]
        assert scores.shape == (batch_size, 1)
        assert torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        responses = output.non_tensor_batch["genrm_response"]
        assert len(responses) == batch_size and all(isinstance(text, str) for text in responses)
        print(f"Standalone reward replicas: {len(engine.rollout_replicas)}; scored images: {batch_size}")
        print(f"OCR scores: {scores.flatten().tolist()}; model responses: {responses.tolist()}")
    finally:
        ray.shutdown()
