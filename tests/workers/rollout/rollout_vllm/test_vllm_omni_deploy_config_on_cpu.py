# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU checks for vLLM-Omni deploy configs, replica allocation and SP rank mapping."""

import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout import utils as verl_rollout_utils

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.workers.config import DiffusionRolloutConfig
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

    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="thinker_only"):
            return [
                types.SimpleNamespace(
                    stage_id=0,
                    final_output=True,
                    final_output_type="audio",
                    sampling_constraints={},
                )
            ]

        @classmethod
        def get_pipeline_id(cls, pipeline_mode="thinker_only"):
            return "minimax_h3"

    monkeypatch.setenv(get_visible_devices_keyword(), "0,1,2,3")

    engine_kwargs: dict = {}
    ARStrategy(fake_self)._write_deploy_config(engine_kwargs, "minimax_h3", Adapter, "t2av")
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


def test_sp_does_not_require_a_verl_patch(monkeypatch):
    monkeypatch.delattr(verl_rollout_utils, "get_rollout_sequence_parallel_size", raising=False)
    config = DiffusionRolloutConfig(name="vllm_omni", tensor_model_parallel_size=1, ulysses_degree=4)
    assert config.tensor_model_parallel_size == 1


@pytest.mark.parametrize("tp,usp,ring", [(2, 1, 1), (1, 4, 1), (2, 2, 1), (1, 1, 4), (1, 8, 1)])
@pytest.mark.parametrize("standalone", [False, True])
async def test_manager_replica_and_worker_slices(monkeypatch, tp, usp, ring, standalone):
    from verl_omni.workers.rollout import replica as replica_module
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    class RecordingReplica(vLLMOmniReplica):
        async def launch_servers(self):
            self._server_handle = f"handle-{self.replica_rank}"
            self._server_address = f"address-{self.replica_rank}"

        async def init_standalone(self):
            # No Ray cluster: record the resource shape consumed by the inherited method.
            self.resource_shape = [self.gpus_per_replica_node] * self.nnodes
            await self.launch_servers()

    rollout = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=tp,
        ulysses_degree=usp,
        ring_degree=ring,
        n_gpus_per_node=4,
        nnodes=2,
    )
    workers = list(range(8))
    group = None if standalone else SimpleNamespace(world_size=8, workers=workers)
    manager = replica_module.DiffusionLLMServerManager(
        SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout, model=None)),
        worker_group=group,
    )
    manager.rollout_replica_class = RecordingReplica
    monkeypatch.setattr(replica_module.RLInsightLogger, "enabled", lambda: False)
    await manager._initialize_llm_servers()
    width = tp * usp * ring
    assert len(manager.get_replicas()) == 8 // width
    for rank, replica in enumerate(manager.get_replicas()):
        assert replica.world_size == width
        assert replica.config.tensor_model_parallel_size == tp
        assert replica.nnodes == max(1, width // 4)
        if standalone:
            assert sum(replica.resource_shape) == width
        else:
            assert replica.workers == workers[rank * width : (rank + 1) * width]
    assert len(manager.server_handles) == len(manager.server_addresses) == 8 // width


@pytest.mark.parametrize("tp,usp,ring", [(2, 1, 1), (1, 4, 1), (2, 2, 1), (1, 1, 4), (1, 8, 1)])
@pytest.mark.parametrize("explicit_replica", [-1, 5])
def test_registered_adapter_rank_and_ipc_mapping(monkeypatch, tp, usp, ring, explicit_replica):
    import verl.workers.rollout.base as base
    import verl.workers.rollout.vllm_rollout.vllm_rollout as upstream

    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniServerAdapter

    monkeypatch.setattr(base, "omega_conf_to_dataclass", lambda config: config)
    monkeypatch.setattr(upstream.ray, "get_runtime_context", lambda: SimpleNamespace(get_job_id=lambda: "test"))
    monkeypatch.setattr(upstream, "is_support_ipc", lambda: True)
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "4")
    rollout = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=tp,
        ulysses_degree=usp,
        ring_degree=ring,
    )
    assert base.get_rollout_class("vllm_omni", "async") is vLLMOmniServerAdapter
    width = tp * usp * ring
    for rank in range(8 if explicit_replica == -1 else width):
        monkeypatch.setenv("RANK", str(rank))
        adapter = vLLMOmniServerAdapter(rollout, None, None, replica_rank=explicit_replica)
        replica_rank = rank // width if explicit_replica == -1 else explicit_replica
        local = rank % width
        assert (adapter.replica_rank, adapter.rollout_rank, adapter.node_rank) == (replica_rank, local, local // 4)
        assert adapter._has_server == (local == 0)
        assert adapter.config.tensor_model_parallel_size == tp
        assert adapter.zmq_handle == f"ipc:///tmp/rl-colocate-zmq-test-replica-{replica_rank}-rank-{local % 4}.sock"


async def test_sp_one_manager_delegates(monkeypatch):
    from verl.workers.rollout.llm_server import LLMServerManager

    from verl_omni.workers.rollout.replica import DiffusionLLMServerManager

    original = AsyncMock()
    monkeypatch.setattr(LLMServerManager, "_initialize_llm_servers", original)
    manager = object.__new__(DiffusionLLMServerManager)
    manager.rollout_config = DiffusionRolloutConfig()
    await manager._initialize_llm_servers(start_rank=3)
    original.assert_awaited_once_with(start_rank=3)


async def test_sp_pool_divisibility_fails_before_launch():
    from verl_omni.workers.rollout.replica import DiffusionLLMServerManager

    rollout = DiffusionRolloutConfig(name="vllm_omni", tensor_model_parallel_size=1, ulysses_degree=4)
    manager = DiffusionLLMServerManager(
        SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout, model=None)),
        worker_group=SimpleNamespace(world_size=6),
    )
    manager.rollout_replica_class = Mock(side_effect=AssertionError("must not launch"))
    with pytest.raises(ValueError, match="divisible"):
        await manager._initialize_llm_servers()
    manager.rollout_replica_class.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "vllm"},
        {"data_parallel_size": 2},
        {"engine_kwargs": {"vllm_omni": {"output_mode": "ar"}}},
        {"disaggregation": {"enabled": True}},
    ],
)
def test_sp_rejects_unsupported_modes(overrides):
    config = dict(name="vllm_omni", tensor_model_parallel_size=1, ulysses_degree=4)
    config.update(overrides)
    with pytest.raises(ValueError, match="sequence parallelism"):
        DiffusionRolloutConfig(**config)


def test_http_profiler_uses_sp_footprint(monkeypatch):
    from verl.utils.profiler import ProfilerConfig
    from verl.utils.profiler.config import TorchProfilerToolConfig

    from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as module

    server = object.__new__(module.vLLMOmniHttpServer)
    server.config = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=1,
        ulysses_degree=4,
        profiler=ProfilerConfig(tool="torch", ranks=[4]),
    )
    server.replica_rank = 1
    server.replica_world_size = 1
    server.profiler_controller = SimpleNamespace(
        config=ProfilerConfig(tool="torch", ranks=[4]),
        tool_config=TorchProfilerToolConfig(),
    )
    server._generate_strategy = SimpleNamespace(post_init=Mock())
    monkeypatch.setenv("VERL_ZMQ_BASE_VISIBLE_DEVICES", "")
    monkeypatch.setattr(module.vLLMHttpServer, "_post_init", lambda *_: None)
    server._post_init("0,1,2,3")
    assert server.replica_world_size == 4
    assert server.profiler_controller.config.ranks == [1]
    assert server.config.profiler.ranks == [4]


async def test_manager_start_rank_and_metrics(monkeypatch):
    from verl.workers.config.rollout import PrometheusConfig

    from verl_omni.workers.rollout import replica as replica_module

    rollout = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=1,
        ulysses_degree=4,
        nnodes=1,
        n_gpus_per_node=8,
        disable_log_stats=False,
        prometheus=PrometheusConfig(enable=True),
    )
    cfg = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout, model=None))
    manager = replica_module.DiffusionLLMServerManager(cfg, start_rank=3)
    manager.rollout_replica_class = Mock(
        side_effect=lambda **kw: SimpleNamespace(
            replica_rank=kw["replica_rank"],
            init_standalone=AsyncMock(),
            _server_handle=f"h{kw['replica_rank']}",
            _server_address=f"a{kw['replica_rank']}",
        )
    )
    prometheus, insight = Mock(), Mock()
    monkeypatch.setattr(replica_module, "update_prometheus_config", prometheus)
    monkeypatch.setattr(replica_module.RLInsightLogger, "enabled", lambda: True)
    monkeypatch.setattr(replica_module.RLInsightLogger, "register_rollout_metrics", insight)
    await manager._initialize_llm_servers()
    assert [r.replica_rank for r in manager.get_replicas()] == [3, 4]
    prometheus.assert_called_once_with(rollout.prometheus, ["a3", "a4"], "vllm_omni")
    assert insight.call_args.kwargs["labels"] == [{"replica": 3}, {"replica": 4}]


async def test_inherited_standalone_allocates_all_sp_ranks(monkeypatch):
    from verl.workers.rollout import replica as upstream

    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    config = DiffusionRolloutConfig(name="vllm_omni", tensor_model_parallel_size=1, ulysses_degree=8)
    replica = vLLMOmniReplica(2, config, None, gpus_per_node=4)
    pool = object()
    pool_manager = Mock(
        return_value=SimpleNamespace(
            create_resource_pool=Mock(),
            resource_pool_dict={"rollout_pool_2": pool},
        )
    )
    worker_group = Mock(return_value=SimpleNamespace(workers=list(range(8))))
    monkeypatch.setattr(upstream, "ResourcePoolManager", pool_manager)
    monkeypatch.setattr(upstream, "RayWorkerGroup", worker_group)
    monkeypatch.setattr(upstream, "get_device_name", lambda: "cpu")
    replica.get_ray_class_with_init_args = Mock(return_value="worker-class")
    replica.launch_servers = AsyncMock()
    await replica.init_standalone()
    assert pool_manager.call_args.kwargs["resource_pool_spec"] == {"rollout_pool_2": [4, 4]}
    assert worker_group.call_args.kwargs["resource_pool"] is pool
    assert replica.workers == list(range(8))
    replica.launch_servers.assert_awaited_once()


def test_ar_adapter_keeps_parent_state(monkeypatch):
    import verl.workers.rollout.base as base
    import verl.workers.rollout.vllm_rollout.vllm_rollout as upstream
    from verl.workers.config import RolloutConfig

    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniServerAdapter

    monkeypatch.setattr(base, "omega_conf_to_dataclass", lambda config: config)
    monkeypatch.setattr(upstream.ray, "get_runtime_context", lambda: SimpleNamespace(get_job_id=lambda: "test"))
    monkeypatch.setattr(upstream, "is_support_ipc", lambda: True)
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    config = RolloutConfig(name="vllm_omni", tensor_model_parallel_size=2)
    parent = upstream.ServerAdapter(config, None, None)
    local = vLLMOmniServerAdapter(config, None, None)
    assert vars(local) == vars(parent)


@pytest.mark.parametrize("module_name", ["ray_diffusion_trainer", "v1.trainer_base", "v1.trainer_separate_async"])
def test_diffusion_trainers_select_local_manager(module_name):
    import importlib

    from verl_omni.workers.rollout.replica import DiffusionLLMServerManager

    trainer = importlib.import_module(f"verl_omni.trainer.diffusion.{module_name}")
    assert trainer.LLMServerManager is DiffusionLLMServerManager
