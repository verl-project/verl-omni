# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU checks that the generated deploy config forwards text-encoder TP.

Regression coverage for the bug where ``text_encoder_tp_size`` was passed as a
top-level engine flag and silently dropped by vLLM-Omni's stage/parallel-config
resolution, leaving the diffusion text encoder unsharded (resident on DiT rank
0). The rollout config now carries ``text_encoder_tp_size`` and it must land in
every generated stage so it reaches ``DiffusionParallelConfig``.
"""

import types
from unittest.mock import MagicMock

import yaml
from verl.utils.device import get_visible_devices_keyword

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy


def _run_write_deploy_config(
    monkeypatch, *, tensor_parallel_size, text_encoder_tp_size=1, include_text_encoder_tp_size=True
):
    config_kwargs = {"tensor_model_parallel_size": tensor_parallel_size}
    if include_text_encoder_tp_size:
        config_kwargs["text_encoder_tp_size"] = text_encoder_tp_size
    fake_self = types.SimpleNamespace(
        config=types.SimpleNamespace(**config_kwargs),
    )
    adapter = MagicMock()
    adapter.build_stage_configs.return_value = [types.SimpleNamespace(stage_id=0)]
    adapter.get_pipeline_id.return_value = "minimax_h3"
    adapter.get_stage_engine_extras.return_value = {}
    monkeypatch.setenv(get_visible_devices_keyword(), "0,1,2,3")

    engine_kwargs: dict = {}
    ARStrategy(fake_self)._write_deploy_config(engine_kwargs, "minimax_h3", adapter, "t2av")
    with open(engine_kwargs["deploy_config"]) as f:
        return yaml.safe_load(f)


def test_deploy_config_stage_carries_sharded_text_encoder_tp_size(monkeypatch):
    deploy = _run_write_deploy_config(monkeypatch, tensor_parallel_size=4, text_encoder_tp_size=4)
    assert deploy["stages"], "expected at least one generated stage"
    for stage in deploy["stages"]:
        assert stage["tensor_parallel_size"] == 4
        assert stage["text_encoder_tp_size"] == 4


def test_deploy_config_stage_defaults_text_encoder_tp_size_to_one(monkeypatch):
    deploy = _run_write_deploy_config(monkeypatch, tensor_parallel_size=4, text_encoder_tp_size=1)
    for stage in deploy["stages"]:
        assert stage["text_encoder_tp_size"] == 1


def test_deploy_config_stage_defaults_when_config_lacks_text_encoder_tp_size(monkeypatch):
    # The AR/omni RolloutConfig has no text_encoder_tp_size; _write_deploy_config must not
    # crash and should default the stage field to 1 (regression for the Qwen3-Omni e2e).
    deploy = _run_write_deploy_config(monkeypatch, tensor_parallel_size=2, include_text_encoder_tp_size=False)
    for stage in deploy["stages"]:
        assert stage["text_encoder_tp_size"] == 1
