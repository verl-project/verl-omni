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
import time
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop import reward_loop as loop_module
from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import NativeManagedRewardModel
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
@pytest.mark.parametrize("device_type", ["cpu", "accelerator"])
async def test_native_executor_explicit_device_does_not_probe_accelerator(monkeypatch, device_type):
    executor = NativeRewardExecutor(
        RewardModelSpec(
            name="quality",
            backend="native",
            executor_config={"model": "tests.fake:CpuModel", "kwargs": {"device": "cpu"}},
            device_type=device_type,
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
    ],
)
def test_cpu_native_rejects_invalid_resource_reservations(placement, message):
    with pytest.raises(ValueError, match=message):
        parse_reward_model_config(
            "quality",
            {
                "backend": "native",
                "placement": {"devices": [0], **placement},
                "executor": {"model": "tests.fake:CpuModel"},
            },
        )


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


def test_shared_worker_identity_is_recorded_before_native_cpus_are_reserved(monkeypatch):
    model_config = {
        "backend": "native",
        "placement": {"resource": "cpu", "devices": [0]},
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
    identity = {"pid": 17, "created": 23.0, "node_id": "owned-node"}

    def capture(callback):
        calls.append("shared_identity")
        return identity

    shared = SimpleNamespace(__ray_call__=SimpleNamespace(remote=capture))

    def reserve_native(*args):
        calls.append("native_cpu_reserved")
        return [object()]

    manager = object.__new__(OmniRewardLoopManager)
    manager.config = config
    manager.reward_router_address = None
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"quality": model},
        reward_model_specs={"quality": model.spec},
        native_device_assignments={"quality": (0,)},
        bind_native_workers=lambda name, workers: model.bind_workers(workers),
    )
    manager._create_node_affinity_workers = lambda *args: [shared]
    manager._create_native_workers = reserve_native
    monkeypatch.setattr(loop_module.ray, "remote", lambda cls: cls)
    monkeypatch.setattr(loop_module.ray, "get", lambda refs: refs)
    manager._init_reward_loop_workers()
    assert calls == ["shared_identity", "native_cpu_reserved"]
    assert manager._shared_worker_process_identities == {id(shared): identity}


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["cancel", "timeout", "rpc_error", "submission_error"])
@pytest.mark.parametrize("confirmed", [False, True])
async def test_never_returning_rpc_has_bounded_cleanup_and_retains_model_until_stopped(monkeypatch, trigger, confirmed):
    calls = []
    entered = asyncio.Event()
    request = None

    async def score(data):
        entered.set()
        await asyncio.Event().wait()

    def submit(data):
        nonlocal request
        request = asyncio.create_task(score(data))
        return request

    async def failed(data):
        raise ValueError("scoring failed")

    def failed_submit(data):
        raise ValueError("submission failed")

    worker = SimpleNamespace(compute_score_batch=SimpleNamespace(remote=submit))
    workers = [worker]
    if trigger not in ("cancel", "timeout"):
        workers.append(
            SimpleNamespace(
                compute_score_batch=SimpleNamespace(remote=failed if trigger == "rpc_error" else failed_submit)
            )
        )
    model = NativeManagedRewardModel(
        "quality",
        {
            "backend": "native",
            "placement": {"resource": "cpu", "devices": [0]},
            "executor": {"model": "tests.fake:CpuModel"},
        },
    )
    model.bind_workers(workers)
    model._worker_process_identities[id(worker)] = {"owned": True}

    async def wake():
        pass

    async def sleep():
        assert calls == ["terminate", "stopped"]
        assert worker not in model._workers
        calls.append("sleep")

    def terminate(actor, identity, timeout):
        assert actor is worker
        assert identity == {"owned": True}
        calls.append("terminate")
        time.sleep(0.005)
        if not confirmed:
            raise TimeoutError("worker termination unconfirmed")
        calls.append("stopped")

    monkeypatch.setattr(loop_module, "_SCORING_DRAIN_TIMEOUT", 0.02)
    monkeypatch.setattr(loop_module, "terminate_actor_and_wait", terminate)
    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager._reward_worker_groups = {"quality": workers}
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"quality": model},
        wake_up=wake,
        sleep=sleep,
        has_engine_model=False,
    )
    operation = manager.async_compute_rm_score(DataProto.from_dict(tensors={"id": torch.arange(2)}))
    scoring = asyncio.create_task(asyncio.wait_for(operation, timeout=0.02) if trigger == "timeout" else operation)
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    started = time.monotonic()
    if trigger == "cancel":
        scoring.cancel()
        await asyncio.sleep(0.005)
        scoring.cancel()
    completed, _ = await asyncio.wait({scoring}, timeout=0.25)
    assert completed, "Cleanup deadline must bound a scoring RPC that never returns"
    result = (await asyncio.gather(scoring, return_exceptions=True))[0]
    assert time.monotonic() - started < 0.25
    assert not manager._score_lock.locked()
    assert request.cancelled()
    if confirmed:
        assert calls == ["terminate", "stopped", "sleep"]
        expected_error = {"cancel": asyncio.CancelledError, "timeout": TimeoutError}.get(trigger, ValueError)
        assert isinstance(result, expected_error)
    else:
        assert calls == ["terminate"]
        assert isinstance(result, RuntimeError)
        assert "termination could not be confirmed" in str(result)
    with pytest.raises(RuntimeError, match="recreate the reward manager"):
        await manager.async_compute_rm_score(DataProto.from_dict(tensors={"id": torch.arange(2)}))
