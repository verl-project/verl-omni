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
"""CPU contracts for named reward deployments and their lifecycle."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.protocol import DataProto

from verl_omni.reward_loop import deployment as deployment_module
from verl_omni.reward_loop.deployment import (
    MultiRewardModelManager,
    NativeRewardDeployment,
    NativeRewardExecutor,
    PickScoreEngineAdapter,
    RewardExecutorSpec,
    _prepare_engine_config,
    accelerator_workers_enabled,
    build_engine_reward_executors,
    reward_is_enabled,
    reward_pool_is_separate,
    reward_role_required,
    streaming_reward_enabled,
    validate_reward_deployment_terms,
)
from verl_omni.reward_loop.reward_loop import (
    OmniRewardLoopManager,
    OmniRewardLoopWorker,
)


def _config(deployments=None):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.reward_model.enable = False
    config.reward.deployments = OmegaConf.create(deployments or {})
    return config


def test_engine_deployments_require_parent_pool():
    config = _config(
        {
            "ocr": {"backend": "engine"},
            "pickscore": {"backend": "engine"},
        }
    )

    with pytest.raises(ValueError, match="require a parent resource pool"):
        MultiRewardModelManager(config)


def test_mixed_engine_and_native_deployments_share_one_parent_pool(monkeypatch):
    config = _config(
        {
            "ocr": {"backend": "engine", "model_path": "/models/ocr"},
            "pickscore": {
                "backend": "native",
                "adapter": "pickscore",
                "placement": {"devices": [0, 1, 2, 3]},
            },
        }
    )
    manager = object.__new__(MultiRewardModelManager)
    manager.config = config
    manager.resource_pool = SimpleNamespace(world_size=10)
    observed = {}

    def fake_split(pool, sizes):
        observed["pool"] = pool
        observed["sizes"] = sizes
        return ["engine-pool", "native-pool", "unused-pool"]

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    engine_pools, native_pool = manager._split_deployment_resource_pools(
        [("ocr", config.reward.deployments.ocr)],
        [("pickscore", config.reward.deployments.pickscore)],
        config.reward.reward_model,
    )

    assert observed == {"pool": manager.resource_pool, "sizes": [2, 4, 4]}
    assert engine_pools == {"ocr": "engine-pool"}
    assert native_pool == "native-pool"


def test_multi_reward_model_manager_splits_parent_pool(monkeypatch):
    manager = object.__new__(MultiRewardModelManager)
    manager.resource_pool = SimpleNamespace(world_size=8)
    parent = manager.resource_pool
    observed = {}

    def fake_split(pool, sizes):
        observed["pool"] = pool
        observed["sizes"] = sizes
        return [f"sub-{index}" for index in range(len(sizes))]

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    entries = [
        (
            "pickscore",
            {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2},
        ),
        (
            "ocr",
            {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2},
        ),
    ]
    base_config = {"rollout": {"tensor_model_parallel_size": 1}}

    result = manager._split_engine_resource_pool(entries, base_config)

    assert observed == {"pool": parent, "sizes": [4, 4]}
    assert result == {"pickscore": "sub-0", "ocr": "sub-1"}


def test_multi_reward_model_manager_rejects_parent_pool_overcommit():
    manager = object.__new__(MultiRewardModelManager)
    manager.resource_pool = SimpleNamespace(world_size=4)
    entries = [
        ("one", {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2}),
        ("two", {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 1}),
    ]

    with pytest.raises(ValueError, match="request 6 devices"):
        manager._split_engine_resource_pool(entries, {"rollout": {"tensor_model_parallel_size": 1}})


def test_multi_reward_model_manager_binds_each_engine_to_its_sub_pool(monkeypatch):
    config = _config(
        {
            "pickscore": {
                "backend": "engine",
                "model_path": "/models/pickscore",
                "rollout": {"tensor_model_parallel_size": 2},
            },
            "ocr": {
                "backend": "engine",
                "model_path": "/models/ocr",
                "rollout": {"tensor_model_parallel_size": 2},
            },
        }
    )
    parent_pool = SimpleNamespace(world_size=4)
    observed = []

    def fake_split(pool, sizes):
        assert pool is parent_pool
        assert sizes == [2, 2]
        return ["pickscore-pool", "ocr-pool"]

    class FakeEngineDeployment:
        def __init__(self, name, deployment, base_config, resource_pool, fallback_model):
            del deployment, base_config, fallback_model
            observed.append((name, resource_pool))
            self._spec = RewardExecutorSpec(name, "engine", None, f"{name}:8000", {})

        @property
        def executor_spec(self):
            return self._spec

        def wake_up(self):
            return None

        def sleep(self):
            return None

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    monkeypatch.setattr(deployment_module, "EngineRewardDeployment", FakeEngineDeployment)

    manager = MultiRewardModelManager(config, resource_pool=parent_pool)

    assert observed == [("pickscore", "pickscore-pool"), ("ocr", "ocr-pool")]
    assert set(manager.reward_executor_specs) == {"pickscore", "ocr"}


def test_engine_deployment_requires_trainer_parent_pool():
    config = _config({"pickscore": {"backend": "engine"}})
    assert reward_is_enabled(config)
    assert reward_role_required(config)
    assert not reward_pool_is_separate(config)
    assert not streaming_reward_enabled(config)

    config.reward.reward_model.enable_resource_pool = True
    assert reward_role_required(config)
    assert reward_pool_is_separate(config)
    assert not streaming_reward_enabled(config)


def test_native_only_deployment_uses_parent_pool_and_batch_scoring():
    config = _config(
        {"pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0]}}}
    )

    assert reward_is_enabled(config)
    assert reward_role_required(config)
    assert not streaming_reward_enabled(config)


def test_accelerator_worker_setting_keeps_the_legacy_alias():
    config = _config()

    assert not accelerator_workers_enabled(config)

    config.reward.accelerator_workers.enabled = True
    assert accelerator_workers_enabled(config)

    config.reward.accelerator_workers.enabled = False
    config.reward.custom_reward_function.use_accelerator = True
    assert accelerator_workers_enabled(config)


def test_engine_deployment_rejects_legacy_per_deployment_pool_switch():
    config = _config({"ocr": {"backend": "engine", "enable_resource_pool": True}})

    with pytest.raises(ValueError, match="must not set enable_resource_pool"):
        MultiRewardModelManager(config, resource_pool=SimpleNamespace(world_size=1))


def test_native_pickscore_adapter_is_selected_by_default():
    deployment = NativeRewardDeployment(
        "pickscore",
        OmegaConf.create({"backend": "native", "adapter": "pickscore", "model_path": "/models/pickscore"}),
    )

    assert deployment.executor_spec.executor_config["scorer"] == (
        "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeScorer"
    )
    assert deployment.executor_spec.model_path == "/models/pickscore"


@pytest.mark.parametrize(
    ("deployments", "message"),
    [
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": []},
                }
            },
            "non-empty list",
        ),
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": [0, 0]},
                }
            },
            "duplicate",
        ),
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": [-1]},
                }
            },
            "non-negative integers",
        ),
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": [0]},
                    "rollout": {"tensor_model_parallel_size": 2},
                }
            },
            "does not support engine resource fields",
        ),
    ],
)
def test_native_deployment_rejects_invalid_placement(deployments, message):
    with pytest.raises(ValueError, match=message):
        MultiRewardModelManager(_config(deployments), resource_pool=SimpleNamespace(world_size=8))


def test_native_deployment_rejects_overlapping_device_assignments():
    config = _config(
        {
            "pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0, 1]}},
            "hpsv3": {
                "backend": "native",
                "executor": {"scorer": "tests.fake:HpsScorer"},
                "placement": {"devices": [1, 2]},
            },
        }
    )

    with pytest.raises(ValueError, match="overlaps index 1"):
        MultiRewardModelManager(config, resource_pool=SimpleNamespace(world_size=8))


def test_native_device_assignments_size_the_native_subpool(monkeypatch):
    config = _config(
        {
            "pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0, 1]}},
            "hpsv3": {
                "backend": "native",
                "executor": {"scorer": "tests.fake:HpsScorer"},
                "placement": {"devices": [4, 5]},
            },
        }
    )
    manager = object.__new__(MultiRewardModelManager)
    manager.config = config
    manager.resource_pool = SimpleNamespace(world_size=8)
    manager.native_device_assignments = manager._validate_native_device_assignments(
        [(name, deployment) for name, deployment in config.reward.deployments.items()]
    )
    observed = {}

    def fake_split(pool, sizes):
        observed["pool"] = pool
        observed["sizes"] = sizes
        return ["native-pool", "unused-pool"]

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    engine_pools, native_pool = manager._split_deployment_resource_pools(
        [], list(config.reward.deployments.items()), config.reward.reward_model
    )

    assert observed == {"pool": manager.resource_pool, "sizes": [6, 2]}
    assert engine_pools == {}
    assert native_pool == "native-pool"
    assert manager.native_device_assignments == {"pickscore": (0, 1), "hpsv3": (4, 5)}


def test_native_deployments_create_isolated_worker_groups(monkeypatch):
    config = _config(
        {
            "pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0, 1]}},
            "hpsv3": {
                "backend": "native",
                "executor": {"scorer": "tests.fake:HpsScorer"},
                "placement": {"devices": [2, 3]},
            },
        }
    )
    config.reward.reward_functions = OmegaConf.create(
        {
            "pickscore": {"deployment": "pickscore"},
            "hpsv3": {"deployment": "hpsv3"},
        }
    )
    specs = {
        "pickscore": RewardExecutorSpec("pickscore", "native", "/models/pickscore", None, {}),
        "hpsv3": RewardExecutorSpec("hpsv3", "native", "/models/hpsv3", None, {}),
    }
    manager = object.__new__(OmniRewardLoopManager)
    manager.config = config
    manager.reward_router_address = None
    manager.multi_reward_model_manager = SimpleNamespace(
        deployments={"pickscore": object(), "hpsv3": object()},
        reward_executor_specs=specs,
        native_device_assignments={"pickscore": (0, 1), "hpsv3": (2, 3)},
    )
    observed = []

    def create_native_workers(group_config, group_specs, bundle_indices, name_prefix):
        observed.append((group_config, group_specs, bundle_indices, name_prefix))
        return [f"{name_prefix}-worker"]

    monkeypatch.setattr("verl_omni.reward_loop.reward_loop.ray.remote", lambda cls: cls)
    manager._create_node_affinity_workers = lambda *args: pytest.fail("no shared worker group expected")
    manager._create_native_workers = create_native_workers

    manager._init_reward_loop_workers()

    assert manager._reward_worker_groups == {
        "pickscore": ["native_reward_loop_worker_pickscore-worker"],
        "hpsv3": ["native_reward_loop_worker_hpsv3-worker"],
    }
    assert manager.reward_loop_workers == [
        "native_reward_loop_worker_pickscore-worker",
        "native_reward_loop_worker_hpsv3-worker",
    ]
    assert [(set(group_specs), tuple(bundle_indices), name_prefix) for _, group_specs, bundle_indices, name_prefix in observed] == [
        ({"pickscore"}, (0, 1), "native_reward_loop_worker_pickscore"),
        ({"hpsv3"}, (2, 3), "native_reward_loop_worker_hpsv3"),
    ]
    assert [set(group_config.reward.reward_functions) for group_config, *_ in observed] == [
        {"pickscore"},
        {"hpsv3"},
    ]


def test_engine_config_fills_the_default_rollout_name():
    config = _config()
    engine_config = _prepare_engine_config(
        OmegaConf.create({"backend": "engine", "model_path": "/models/clip"}),
        config.reward.reward_model,
    )

    assert engine_config.enable is True
    assert engine_config.rollout.name == "vllm"
    assert "backend" not in engine_config


def test_pickscore_engine_requires_explicit_logit_scale():
    spec = RewardExecutorSpec(
        name="pickscore",
        backend="engine",
        model_path="/models/pickscore",
        router_address="router:8000",
        executor_config={"adapter": "pickscore"},
    )

    with pytest.raises(ValueError, match="requires executor.logit_scale"):
        build_engine_reward_executors({"pickscore": spec})


@pytest.mark.parametrize(
    ("deployments", "term", "message"),
    [
        ({}, {"deployment": "missing"}, "unknown deployment"),
        (
            {"native": {"backend": "native", "executor": {"scorer": "unused:Unused"}}},
            {"deployment": "native", "path": "tests.fake.py", "name": "score"},
            "scores directly",
        ),
        (
            {"engine": {"backend": "engine"}},
            {"deployment": "engine"},
            "needs path/name",
        ),
        (
            {"pickscore": {"backend": "engine", "adapter": "pickscore"}},
            {"deployment": "pickscore", "path": "tests.fake.py", "name": "score"},
            "scores directly",
        ),
    ],
)
def test_reward_deployment_terms_fail_fast(deployments, term, message):
    config = _config(deployments)
    config.reward.reward_functions = OmegaConf.create({"term": term})

    with pytest.raises(ValueError, match=message):
        validate_reward_deployment_terms(config)


class _FakeScorer:
    instances = []

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.closed = False
        self.__class__.instances.append(self)

    def score(self, prompts, images):
        assert prompts == ["prompt"]
        assert len(images) == 1
        return torch.tensor([0.75])

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_native_executor_wakes_scores_and_sleeps(monkeypatch):
    spec = RewardExecutorSpec(
        name="native",
        backend="native",
        model_path="/models/native",
        router_address=None,
        executor_config={"scorer": "tests.fake:FakeScorer"},
    )
    executor = NativeRewardExecutor(spec)
    _FakeScorer.instances.clear()
    monkeypatch.setattr(deployment_module, "_load_native_scorer", lambda _: _FakeScorer)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    await executor.wake_up()
    result = await executor.score("prompt", torch.zeros(3, 2, 2, dtype=torch.uint8))
    await executor.sleep()

    assert result == {"score": 0.75, "pickscore_raw": 0.75}
    assert len(_FakeScorer.instances) == 1
    assert _FakeScorer.instances[0].model_path == "/models/native"
    assert _FakeScorer.instances[0].device == torch.device("cpu", 0)
    assert _FakeScorer.instances[0].closed
    assert executor._scorer is None


@pytest.mark.asyncio
async def test_native_executor_waits_for_inflight_score_before_sleep(monkeypatch):
    class _BlockingScorer:
        release = False
        closed = False

        def __init__(self, **kwargs):
            del kwargs

        def score(self, prompts, images):
            del prompts, images
            import time

            while not self.release:
                time.sleep(0.01)
            return torch.tensor([0.5])

        def close(self):
            self.closed = True

    spec = RewardExecutorSpec(
        name="native",
        backend="native",
        model_path=None,
        router_address=None,
        executor_config={"scorer": "tests.fake:BlockingScorer"},
    )
    executor = NativeRewardExecutor(spec)
    _BlockingScorer.release = False
    monkeypatch.setattr(deployment_module, "_load_native_scorer", lambda _: _BlockingScorer)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    score_task = asyncio.create_task(executor.score("prompt", torch.zeros(3, 2, 2, dtype=torch.uint8)))
    await asyncio.sleep(0.02)
    sleep_task = asyncio.create_task(executor.sleep())
    await asyncio.sleep(0.02)
    assert not sleep_task.done()
    _BlockingScorer.release = True
    assert await score_task == {"score": 0.5, "pickscore_raw": 0.5}
    await sleep_task
    assert executor._scorer is None


@pytest.mark.asyncio
async def test_native_executor_does_not_serialize_async_scorer_calls(monkeypatch):
    class _BatchingScorer:
        instances = []

        def __init__(self, **kwargs):
            del kwargs
            self._entered = 0
            self._second_request = asyncio.Event()
            self.closed = False
            self.__class__.instances.append(self)

        async def score(self, prompts, images):
            del prompts, images
            self._entered += 1
            if self._entered == 2:
                self._second_request.set()
            await self._second_request.wait()
            return [0.5]

        async def close(self):
            self.closed = True

    spec = RewardExecutorSpec(
        name="native",
        backend="native",
        model_path=None,
        router_address=None,
        executor_config={"scorer": "tests.fake:BatchingScorer"},
    )
    executor = NativeRewardExecutor(spec)
    _BatchingScorer.instances.clear()
    monkeypatch.setattr(deployment_module, "_load_native_scorer", lambda _: _BatchingScorer)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    results = await asyncio.wait_for(
        asyncio.gather(
            executor.score("first", torch.zeros(3, 2, 2, dtype=torch.uint8)),
            executor.score("second", torch.zeros(3, 2, 2, dtype=torch.uint8)),
        ),
        timeout=1,
    )
    await executor.sleep()

    assert results == [
        {"score": 0.5, "pickscore_raw": 0.5},
        {"score": 0.5, "pickscore_raw": 0.5},
    ]
    assert _BatchingScorer.instances[0]._entered == 2
    assert _BatchingScorer.instances[0].closed


@pytest.mark.asyncio
async def test_pickscore_engine_adapter_posts_openai_embedding_payloads(monkeypatch):
    requests = []

    class _Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {"data": [{"embedding": [1.0, 0.0]}]}

    class _Session:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, url, json):
            requests.append((url, json))
            return _Response()

    monkeypatch.setattr(deployment_module.aiohttp, "ClientSession", _Session)
    adapter = PickScoreEngineAdapter("router:8000", "pickscore", logit_scale=98.0, score_divisor=26.0)

    result = await adapter.score("prompt", torch.zeros(3, 2, 2, dtype=torch.uint8))

    assert result == {"score": pytest.approx(98.0 / 26.0), "pickscore_raw": pytest.approx(98.0 / 26.0)}
    assert [url for url, _ in requests] == ["http://router:8000/v1/embeddings"] * 2
    assert requests[0][1] == {"model": "pickscore", "input": "prompt", "encoding_format": "float"}
    assert requests[1][1]["encoding_format"] == "float"
    assert requests[1][1]["input"][0]["content"][0]["type"] == "image_url"


@pytest.mark.asyncio
async def test_streaming_worker_sleeps_native_models_after_one_request(monkeypatch):
    worker = object.__new__(OmniRewardLoopWorker)
    executor = SimpleNamespace(sleep=AsyncMock())
    worker.native_reward_executors = {"native": executor}
    worker._native_batch_active = False

    async def compute_score(_self, data):
        return {"reward_score": data}

    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.RewardLoopWorker.compute_score",
        compute_score,
    )

    assert await worker.compute_score("streaming") == {"reward_score": "streaming"}
    executor.sleep.assert_awaited_once_with()


def test_deployment_manager_rejects_legacy_and_named_models():
    config = _config({"pickscore": {"backend": "native", "adapter": "pickscore"}})
    config.reward.reward_model.enable = True

    with pytest.raises(ValueError, match="cannot be combined"):
        MultiRewardModelManager(config)


def test_native_deployment_requires_allocated_native_resource_pool():
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = SimpleNamespace(
        native_resource_pool=None,
    )

    with pytest.raises(ValueError, match="require an allocated native resource pool"):
        manager._create_native_workers(
            config=SimpleNamespace(),
            specs={},
            bundle_indices=[0],
            name_prefix="native_reward_loop_worker_pickscore",
        )


def test_named_deployment_groups_merge_scores_and_extra_info(monkeypatch):
    class _Worker:
        def __init__(self, outputs):
            self._outputs = outputs
            self.compute_score_batch = SimpleNamespace(remote=lambda data: self._outputs[: len(data)])

    class _RewardManager:
        @staticmethod
        def assemble_rm_scores(data, scores):
            del data
            return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(2, 3, 2, 2, dtype=torch.uint8)},
        non_tensors={"data_source": ["a", "b"]},
    )
    manager = object.__new__(OmniRewardLoopManager)
    manager.reward_manager_cls = _RewardManager
    manager._reward_worker_groups = {
        "shared": [
            _Worker(
                [
                    {"reward_score": 0.25, "reward_extra_info": {"reward/ocr": 0.25, "reward/combined": 0.25}},
                    {"reward_score": 0.5, "reward_extra_info": {"reward/ocr": 0.5, "reward/combined": 0.5}},
                ]
            )
        ],
        "pickscore": [
            _Worker(
                [
                    {
                        "reward_score": 0.75,
                        "reward_extra_info": {"reward/pickscore": 0.75, "reward/combined": 0.75},
                    },
                    {
                        "reward_score": 1.0,
                        "reward_extra_info": {"reward/pickscore": 1.0, "reward/combined": 1.0},
                    },
                ]
            )
        ],
    }
    monkeypatch.setattr("verl_omni.reward_loop.reward_loop.ray.get", lambda refs: refs)

    result = manager._compute_named_deployment_scores(data)

    assert torch.equal(result.batch["rm_scores"], torch.tensor([[1.0], [1.5]]))
    assert result.non_tensor_batch["reward/ocr"].tolist() == [0.25, 0.5]
    assert result.non_tensor_batch["reward/pickscore"].tolist() == [0.75, 1.0]
    assert result.non_tensor_batch["reward/combined"].tolist() == [1.0, 1.5]
    assert result.meta_info["reward_extra_keys"] == ["reward/ocr", "reward/pickscore", "reward/combined"]
