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
"""CPU-native reward placement and executor contracts."""

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop import reward_loop as loop_module
from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import MultiRewardModelManager, NativeManagedRewardModel
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec, parse_reward_model_config, reward_role_required


def test_cpu_native_role_resolves_placement_interpolations():
    config = OmegaConf.create(
        {
            "resource": "cpu",
            "reward": {
                "reward_model": {"enable": False},
                "models": {"quality": {"backend": "native", "placement": {"resource": "${resource}"}}},
            },
        }
    )
    assert not reward_role_required(config)
    config.resource = "accelerator"
    assert reward_role_required(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["thread", "async", "awaitable"])
async def test_cancelled_native_inference_keeps_model_until_actual_completion(monkeypatch, kind):
    entered, release, finished, closed = (Event() for _ in range(4))

    async def async_infer(self, value):
        entered.set()
        try:
            async with asyncio.timeout(3):
                while not release.is_set():
                    await asyncio.sleep(0.001)
            return value + 1
        finally:
            finished.set()

    class Model:
        def __init__(self, **kwargs):
            del kwargs

        def infer(self, value):
            if kind == "awaitable":
                return async_infer(self, value)
            entered.set()
            try:
                if not release.wait(3):
                    raise TimeoutError("Controlled inference was not released")
                return value + 1
            finally:
                finished.set()

        def close(self):
            assert finished.is_set()
            closed.set()

    if kind == "async":
        Model.infer = async_infer
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: Model)
    executor = NativeRewardExecutor(
        RewardModelSpec(name="quality", backend="native", device_type="cpu", executor_config={"model": "test:model"})
    )
    await executor.wake_up()
    caller = asyncio.create_task(executor.infer(1))
    sleeping = None
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        caller.cancel()
        await asyncio.wait({caller}, timeout=0.04)
        assert not caller.done()
        assert executor._inflight == 1
        assert not finished.is_set()
        sleeping = asyncio.create_task(executor.sleep())
        await asyncio.wait({sleeping}, timeout=0.02)
        assert not sleeping.done()
        assert not closed.is_set()
        release.set()
        await asyncio.wait_for(sleeping, 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(caller), 1)
        assert finished.is_set() and closed.is_set()
        assert executor._inflight == 0 and executor._model is None
    finally:
        release.set()
        await asyncio.gather(caller, *([sleeping] if sleeping else []), return_exceptions=True)
        await executor.sleep()


@pytest.mark.asyncio
async def test_cpu_native_executor_uses_unindexed_cpu_device(monkeypatch):
    spec = RewardModelSpec(
        name="quality",
        backend="native",
        model_path="/models/quality",
        executor_config={"model": "tests.fake:CpuModel"},
        device_type="cpu",
    )
    executor = NativeRewardExecutor(spec)
    closed = []
    model = SimpleNamespace(infer=lambda value: value + 1, close=lambda: closed.append(True))

    def build_model(model_path, device):
        assert model_path == "/models/quality"
        assert device == torch.device("cpu")
        return model

    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: build_model)
    monkeypatch.setattr(executor_module, "_empty_accelerator_cache", lambda: pytest.fail("CPU sleep cleared GPU cache"))
    await executor.wake_up()
    assert await executor.infer(1) == 2
    await executor.sleep()
    assert closed == [True]
    assert executor._model is None


@pytest.mark.asyncio
async def test_native_executor_explicit_cpu_device_does_not_probe_accelerator(monkeypatch):
    executor = NativeRewardExecutor(
        RewardModelSpec(
            name="quality",
            backend="native",
            executor_config={"model": "tests.fake:CpuModel", "kwargs": {"device": "cpu"}},
            device_type="cpu",
        )
    )
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: lambda device: SimpleNamespace(device=device))
    monkeypatch.setattr(executor_module, "get_device_name", lambda: pytest.fail("Explicit device was ignored"))
    await executor.wake_up()
    assert executor._model.device == "cpu"


@pytest.mark.parametrize(
    "placement, message",
    [
        ({"resource": "gpu"}, "placement.resource"),
        ({"resource": "cpu", "cpus_per_worker": 0}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": -1}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": True}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": 1.5}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": "2"}, "positive integer"),
        ({"resource": "accelerator", "cpus_per_worker": 2}, "only supported for resource='cpu'"),
        ({"resource": "cpu", "num_replicas": 0}, "placement.num_replicas"),
        ({"resource": "cpu", "num_replicas": True}, "placement.num_replicas"),
        ({"resource": "cpu", "num_replicas": 1.5}, "placement.num_replicas"),
        ({"resource": "cpu", "num_replicas": None}, "placement.num_replicas"),
        ({"resource": "cpu", "devices": [0]}, "remove placement.devices"),
        ({"resource": "accelerator", "num_replicas": 2}, "only supported for resource='cpu'"),
    ],
)
def test_cpu_native_rejects_invalid_resource_reservations(placement, message):
    with pytest.raises(ValueError, match=message):
        parse_reward_model_config(
            "quality",
            {
                "backend": "native",
                "placement": {
                    **({"num_replicas": 1} if placement["resource"] == "cpu" else {"devices": [0]}),
                    **placement,
                },
                "executor": {"model": "tests.fake:CpuModel"},
            },
        )


@pytest.mark.parametrize(
    "resource,device,valid",
    [
        ("cpu", "cpu", True),
        ("cpu", torch.device("cpu"), True),
        ("cpu", "cpu:0", True),
        ("cpu", "cuda:0", False),
        ("cpu", torch.device("cuda:1"), False),
        ("accelerator", "cpu", False),
        ("accelerator", "cuda:0", True),
        ("accelerator", "meta", False),
        ("cpu", "invalid-device", False),
        ("cpu", None, False),
        ("cpu", 0, False),
    ],
)
def test_explicit_native_device_matches_reserved_resource(monkeypatch, resource, device, valid):
    from verl_omni.workers.config import reward as config_module

    monkeypatch.setattr(config_module, "get_device_name", lambda: "cuda")
    value = {
        "backend": "native",
        "placement": {"resource": resource, **({"num_replicas": 2} if resource == "cpu" else {"devices": [0]})},
        "executor": {"model": "tests.fake:CpuModel", "kwargs": {"device": device}},
    }
    if valid:
        parsed = parse_reward_model_config("quality", value)
        assert parsed.executor.kwargs["device"] == device
    else:
        with pytest.raises(ValueError, match=r"executor.kwargs.device"):
            MultiRewardModelManager(
                OmegaConf.create(
                    {"reward": {"models": {"quality": value}, "reward_model": {"enable": False}}},
                    flags={"allow_objects": True},
                )
            )
    assert not torch.cuda.is_initialized()


@pytest.mark.asyncio
async def test_cpu_phase_submission_failure_drains_accepted_rpc_before_sleep():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    accepted = []

    async def score(data):
        entered.set()
        await release.wait()
        calls.append("score_finished")
        return [{"reward_score": 1.0} for _ in range(len(data))]

    def submit(data):
        request = asyncio.create_task(score(data))
        accepted.append(request)
        return request

    def fail_submission(data):
        raise ValueError("intentional RPC submission failure")

    async def wake_up():
        calls.append("wake_up")

    async def sleep():
        calls.append("sleep")

    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager._reward_dispatch_batch_sizes = {}
    manager._reward_worker_groups = {
        "quality": [
            SimpleNamespace(compute_score_batch=SimpleNamespace(remote=submit)),
            SimpleNamespace(compute_score_batch=SimpleNamespace(remote=fail_submission)),
        ]
    }
    manager.multi_reward_model_manager = SimpleNamespace(models={"quality": object()}, wake_up=wake_up, sleep=sleep)
    scoring = asyncio.create_task(manager.async_compute_rm_score(DataProto.from_dict(tensors={"id": torch.arange(2)})))
    try:
        await entered.wait()
        await asyncio.sleep(0)
        returned_before_release = scoring.done()
    finally:
        release.set()
        result = (await asyncio.gather(scoring, return_exceptions=True))[0]
        await asyncio.gather(*accepted)
    assert not returned_before_release, "A submission failure returned before an accepted RPC finished"
    assert isinstance(result, ValueError)
    assert "intentional RPC submission failure" in str(result)
    assert calls == ["wake_up", "score_finished", "sleep"]


def test_shared_workers_start_before_native_cpus_are_reserved(monkeypatch):
    model_config = {
        "backend": "native",
        "placement": {"resource": "cpu", "num_replicas": 1},
        "executor": {"model": "tests.fake:CpuModel"},
    }
    model = NativeManagedRewardModel("quality", model_config)
    config = OmegaConf.create(
        {
            "reward": {
                "num_workers": 1,
                "models": {"quality": model_config},
                "reward_functions": {
                    "quality": {"path": "tests.fake.py", "name": "quality"},
                    "rule": {"path": "tests.fake.py", "name": "rule"},
                },
                "reward_model": {"enable": False},
            }
        }
    )
    calls = []

    def ready():
        calls.append("shared_ready")

    shared = SimpleNamespace(__ray_ready__=SimpleNamespace(remote=ready))

    def reserve_native(*args):
        calls.append("native_cpu_reserved")
        return [object()]

    manager = object.__new__(OmniRewardLoopManager)
    manager.config = config
    manager.reward_router_address = None
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"quality": model},
        reward_model_specs={"quality": model.spec},
        bind_native_workers=lambda name, workers: model.bind_workers(workers),
    )
    manager._create_node_affinity_workers = lambda *args: [shared]
    manager._create_native_workers = reserve_native
    monkeypatch.setattr(loop_module.ray, "remote", lambda cls: cls)
    monkeypatch.setattr(loop_module.ray, "get", lambda refs: refs)
    manager._init_reward_loop_workers()
    assert calls == ["shared_ready", "native_cpu_reserved"]
