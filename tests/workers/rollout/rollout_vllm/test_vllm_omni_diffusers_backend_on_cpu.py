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

"""Rollout pipeline selection for vLLM-Omni's diffusers backend, without GPU workers."""

from argparse import Namespace
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from vllm_omni.engine.arg_utils import OmniEngineArgs

import verl_omni.pipelines  # noqa: F401  # registers the rollout pipelines
from verl_omni.pipelines.sd3_flow_grpo import (
    StableDiffusion3DiffusersPipelineWithLogProb,
    StableDiffusion3PipelineWithLogProb,
)
from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as strategy_module


def _path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _strategy(omni_kwargs, *, algorithm="flow_grpo", lora_rank=0, lora=None):
    config = DiffusionRolloutConfig(engine_kwargs={"vllm_omni": omni_kwargs})
    model_config = SimpleNamespace(
        architecture="StableDiffusion3Pipeline",
        algorithm=algorithm,
        lora_rank=lora_rank,
        lora_alpha=16,
        lora_adapter_path=None,
        lora=lora or {},
        policy_state_adapters=("default",),
    )
    return strategy_module.DiffusionStrategy(SimpleNamespace(config=config, model_config=model_config))


def _prepare(monkeypatch, strategy) -> dict:
    monkeypatch.setattr(strategy_module, "import_external_libs", lambda _: None)
    args = Namespace(tensor_parallel_size=1, max_num_seqs=4)
    engine_args = asdict(OmniEngineArgs.from_cli_args(args))
    strategy.prepare_engine_args(engine_args, args)
    return engine_args


@pytest.mark.parametrize(
    "omni_kwargs,expected",
    [
        ({}, StableDiffusion3PipelineWithLogProb),
        ({"diffusion_load_format": "default"}, StableDiffusion3PipelineWithLogProb),
        ({"diffusion_load_format": "diffusers"}, StableDiffusion3DiffusersPipelineWithLogProb),
        ({"diffusion-load-format": "diffusers"}, StableDiffusion3DiffusersPipelineWithLogProb),
    ],
)
def test_load_format_selects_the_rollout_pipeline(monkeypatch, omni_kwargs, expected):
    strategy = _strategy(omni_kwargs)
    engine_args = _prepare(monkeypatch, strategy)
    assert engine_args["custom_pipeline_args"] == {"pipeline_class": _path(expected)}
    assert engine_args["max_num_seqs"] == 4  # both pipelines batch requests
    assert strategy._diffusion_io_spec() is expected.diffusion_io_spec


def test_diffusers_backend_without_a_registered_pipeline_is_refused(monkeypatch):
    strategy = _strategy({"diffusion_load_format": "diffusers"}, algorithm="dpo")
    with pytest.raises(NotImplementedError, match="diffusers backend is registered"):
        _prepare(monkeypatch, strategy)


def test_diffusers_backend_refuses_unmerged_lora(monkeypatch):
    with pytest.raises(ValueError, match="lora.merge=True"):
        _prepare(monkeypatch, _strategy({"diffusion_load_format": "diffusers"}, lora_rank=8))
    merged = _strategy({"diffusion_load_format": "diffusers"}, lora_rank=8, lora={"merge": True})
    engine_args = _prepare(monkeypatch, merged)
    assert engine_args["custom_pipeline_args"]["pipeline_class"] == _path(StableDiffusion3DiffusersPipelineWithLogProb)
    # The native pipeline keeps its LoRA layers.
    _prepare(monkeypatch, _strategy({}, lora_rank=8))
