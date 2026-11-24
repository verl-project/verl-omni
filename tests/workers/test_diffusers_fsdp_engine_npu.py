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
"""Ascend port of test_diffusers_fsdp_engine.py's FSDP inference/train case."""

import os
from pathlib import Path

import pytest


def test_diffusers_fsdp_engine_npu(monkeypatch):
    """Run Qwen-Image LoRA inference and a finite-gradient FSDP update on NPUs."""
    import ray
    import torch
    import torch_npu  # noqa: F401
    from verl.utils import tensordict_utils as tu

    from tests.workers.test_diffusers_fsdp_engine import create_training_config, run_diffusers_engine_batches

    if not torch.npu.is_available():
        pytest.fail("This smoke test requires Ascend NPUs and torch-npu.")
    available = torch.npu.device_count()
    try:
        requested = int(os.getenv("NUM_NPUS", str(available)))
    except ValueError:
        pytest.fail("NUM_NPUS must be an integer.")
    if not 2 <= requested <= available:
        pytest.fail(f"Need 2 <= NUM_NPUS <= visible NPU count ({available}), got {requested}.")

    model_path = Path(os.getenv("MODEL_PATH", "~/.cache/modelscope/hub/models/tiny-random/Qwen-Image")).expanduser()
    for component in ("model_index.json", "transformer/config.json", "tokenizer/tokenizer_config.json"):
        if not (model_path / component).is_file():
            pytest.fail(f"Missing local Qwen-Image component: {model_path / component}")

    monkeypatch.setenv("DEVICE_NAME", "npu")
    runtime_env = {
        "env_vars": {
            "DEVICE_NAME": "npu",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "ASCEND_RT_VISIBLE_DEVICES": os.getenv(
                "ASCEND_RT_VISIBLE_DEVICES", ",".join(str(i) for i in range(requested))
            ),
            "HCCL_NPU_SOCKET_PORT_RANGE": os.getenv("HCCL_NPU_SOCKET_PORT_RANGE") or "auto",
        }
    }
    runtime_env["env_vars"]["ASCEND_VISIBLE_DEVICES"] = (
        os.getenv("ASCEND_VISIBLE_DEVICES") or runtime_env["env_vars"]["ASCEND_RT_VISIBLE_DEVICES"]
    )
    ray.init(address="local", runtime_env=runtime_env)
    try:
        if ray.cluster_resources().get("NPU", 0) < requested:
            pytest.fail(f"Ray must advertise at least {requested} NPU resources: {ray.cluster_resources()}")
        training_config, actor_config = create_training_config(
            model_type="diffusion_model",
            strategy="fsdp",
            device_count=requested,
            model=str(model_path),
            sequence_parallel_size=1,
            attn_backend="_native_npu",
        )
        assert training_config.engine_config.fsdp_size == requested
        assert training_config.engine_config.ulysses_sequence_parallel_size == 1
        inference_output, training_output = run_diffusers_engine_batches(
            training_config, actor_config, requested, device_name="npu"
        )
        log_probs = inference_output["log_probs"]
        assert log_probs.numel() > 0
        assert torch.isfinite(log_probs).all(), "Engine inference returned nonfinite log probabilities."
        metrics = tu.get(training_output, "metrics")
        for name in ("loss", "grad_norm"):
            assert name in metrics, f"Missing training metric: {name}"
            assert torch.isfinite(torch.as_tensor(metrics[name])).all(), f"Nonfinite {name}: {metrics[name]}"
    finally:
        ray.shutdown()
