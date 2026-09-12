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

"""Exercise the fused diffusion ETP path without starting workers or loading weights."""

import os
from argparse import Namespace
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as server_module
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as strategy_module


def _prepare(monkeypatch, *, algorithm="diffusion_nft", typed=1, cli=None, parallel=None):
    monkeypatch.setattr(strategy_module, "import_external_libs", lambda _: None)
    monkeypatch.setattr(strategy_module.VllmOmniPipelineBase, "get_pipeline_path", lambda **_: "test.Pipeline")
    monkeypatch.setattr(
        strategy_module.VllmOmniPipelineBase, "get_class", lambda **_: SimpleNamespace(supports_request_batch=True)
    )
    config = DiffusionRolloutConfig(tensor_model_parallel_size=4, text_encoder_tp_size=typed)
    server = SimpleNamespace(
        config=config, model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm=algorithm)
    )
    args = Namespace(tensor_parallel_size=4, text_encoder_tp_size=cli)
    engine_args = asdict(OmniEngineArgs.from_cli_args(args))
    if parallel is not None:
        engine_args["parallel_config"] = parallel
    strategy_module.DiffusionStrategy(server).prepare_engine_args(engine_args, args)
    return engine_args


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
@pytest.mark.parametrize("typed,cli,expected", [(1, None, 1), (4, None, 4), (1, 4, 4), (4, 4, 4), (1, 1, 1)])
def test_text_encoder_tp_reaches_fused_diffusion_config(monkeypatch, algorithm, typed, cli, expected):
    engine_args = _prepare(monkeypatch, algorithm=algorithm, typed=typed, cli=cli)
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)
    parallel = stages[0]["engine_args"]["parallel_config"]
    od_config = OmniDiffusionConfig(parallel_config=parallel)
    assert od_config.parallel_config.tensor_parallel_size == 4
    assert od_config.parallel_config.text_encoder_tp_size == expected


@pytest.mark.parametrize("cli", [0, 2, 8])
def test_legacy_cli_rejects_invalid_encoder_groups(monkeypatch, cli):
    with pytest.raises(ValueError, match="text_encoder_tp_size"):
        _prepare(monkeypatch, cli=cli)


def test_conflicting_typed_and_legacy_etp_fails(monkeypatch):
    with pytest.raises(ValueError, match="Conflicting text_encoder_tp_size"):
        _prepare(monkeypatch, typed=4, cli=1)


@pytest.mark.parametrize("as_object", [False, True])
def test_explicit_parallel_config_keeps_etp_and_other_fields(monkeypatch, as_object):
    parallel = {"tensor_parallel_size": 4, "text_encoder_tp_size": 4, "ring_degree": 2}
    if as_object:
        parallel = DiffusionParallelConfig.from_dict(parallel)
    engine_args = _prepare(monkeypatch, parallel=parallel)
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)
    od_config = OmniDiffusionConfig(parallel_config=stages[0]["engine_args"]["parallel_config"])
    assert od_config.parallel_config.text_encoder_tp_size == 4
    assert od_config.parallel_config.ring_degree == 2


def test_missing_nested_etp_is_filled_without_mutating_input(monkeypatch):
    parallel = {"tensor_parallel_size": 4, "ring_degree": 2}
    engine_args = _prepare(monkeypatch, typed=4, parallel=parallel)
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)
    assert stages[0]["engine_args"]["parallel_config"].text_encoder_tp_size == 4
    assert "text_encoder_tp_size" not in parallel


def test_conflicting_parallel_config_fails(monkeypatch):
    with pytest.raises(ValueError, match="Conflicting text_encoder_tp_size"):
        _prepare(monkeypatch, typed=4, parallel={"tensor_parallel_size": 4, "text_encoder_tp_size": 1})


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
@pytest.mark.parametrize("typed,cli", [(4, None), (1, 4)])
async def test_run_server_preserves_etp_at_engine_boundary(monkeypatch, algorithm, typed, cli):
    monkeypatch.setattr(strategy_module, "import_external_libs", lambda _: None)
    monkeypatch.setattr(strategy_module.VllmOmniPipelineBase, "get_pipeline_path", lambda **_: None)
    monkeypatch.setattr(server_module, "get_non_ephemeral_free_port", lambda *_: 12345)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    server = SimpleNamespace(
        config=DiffusionRolloutConfig(tensor_model_parallel_size=4, text_encoder_tp_size=typed),
        model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm=algorithm),
    )
    server._generate_strategy = strategy_module.DiffusionStrategy(server)
    captured = {}

    class EngineBoundaryReached(Exception):
        pass

    def capture_engine(**kwargs):
        captured.update(kwargs)
        raise EngineBoundaryReached

    monkeypatch.setattr(server_module, "AsyncOmni", capture_engine)
    with pytest.raises(EngineBoundaryReached):
        await server_module.vLLMOmniHttpServer.run_server(
            server, Namespace(tensor_parallel_size=4, text_encoder_tp_size=cli)
        )
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(captured)
    od_config = OmniDiffusionConfig(parallel_config=stages[0]["engine_args"]["parallel_config"])
    assert od_config.parallel_config.tensor_parallel_size == 4
    assert od_config.parallel_config.text_encoder_tp_size == 4
