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
"""Real-Ray integration checks for CPU-native reward workers."""

import asyncio
import os
import time
from types import SimpleNamespace
from uuid import uuid4

import psutil
import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from verl import DataProto

from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager, OmniRewardLoopWorker
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec, parse_reward_model_config


class _CpuModel:
    instances = []

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.closed = False
        self.__class__.instances.append(self)

    def infer(self, value):
        return value + 1

    def close(self):
        self.closed = True


class _TorchCpuModel:
    def __init__(self, device):
        self.weight = torch.tensor(1.0, device=device)
        self.instance_id = str(uuid4())

    def infer(self, image):
        return {
            "value": float(image.flatten()[0].to(self.weight.device) + self.weight),
            "device": str(self.weight.device),
            "instance_id": self.instance_id,
        }


async def reward_torch_cpu(reward_model, solution_image, ground_truth):
    if ground_truth in {"fail", "controlled-fail"}:
        raise ValueError("intentional scoring failure")
    return {
        "score": (output := await reward_model.infer(solution_image))["value"],
        "device": output["device"],
        "instance_id": output["instance_id"],
        "pid": os.getpid(),
        "cpus": ray.get_runtime_context().get_assigned_resources().get("CPU", 0),
        "gpus": ray.get_runtime_context().get_assigned_resources().get("GPU", 0),
    }


def reward_rule(**kwargs):
    return 0.25


class _RestartableRewardWorker(OmniRewardLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._delayed_started = asyncio.Event()
        self._delayed_finished = asyncio.Event()
        self._release_delay = asyncio.Event()
        self._release_failure = asyncio.Event()
        self._failure_seen = asyncio.Event()
        self._delayed_succeeded = False

    def worker_pid(self):
        return os.getpid()

    async def compute_score(self, data):
        ground_truth = data.non_tensor_batch["reward_model"][0]["ground_truth"]
        if ground_truth == "controlled-fail":
            await self._release_failure.wait()
        if ground_truth == "delay":
            self._delayed_started.set()
            await self._release_delay.wait()
        try:
            result = await super().compute_score(data)
            if ground_truth == "delay":
                self._delayed_succeeded = True
            return result
        except Exception:
            self._failure_seen.set()
            raise
        finally:
            if ground_truth == "delay":
                self._delayed_finished.set()

    async def wait_for_delayed_sample(self):
        await self._delayed_started.wait()

    async def wait_for_failure(self):
        await self._failure_seen.wait()

    def release_failure(self):
        self._release_failure.set()

    def release_delay(self):
        self._release_delay.set()

    async def delayed_sample_state(self):
        await self._delayed_finished.wait()
        executor = self.native_reward_executors["quality"]
        return self._delayed_succeeded, executor._model is None, executor._inflight


async def wait_for_restarted_worker(worker, previous_pid, timeout=120):
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            pid = await asyncio.wait_for(worker.worker_pid.remote(), timeout=remaining)
            if pid != previous_pid:
                return pid
        except ray.exceptions.ActorUnavailableError:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError("CPU reward actor did not restart")


def _full_config(tmp_path, offload):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(tmp_path)
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.actor_rollout_ref.model.tokenizer_path = str(tmp_path)
    config.reward.reward_model.enable = False
    config.reward.reward_manager.name = "MultiVisualRewardManager"
    config.reward.num_workers = 1
    config.reward.models = {
        "quality": {
            "backend": "native",
            "offload": offload,
            "placement": {"resource": "cpu", "devices": [3, 7], "cpus_per_worker": 1},
            "executor": {"model": f"{__name__}:_TorchCpuModel"},
        }
    }
    config.reward.reward_functions = {
        "quality": {
            "path": __file__,
            "name": "reward_torch_cpu",
            "weight": 0.5,
            "required": True,
        },
        "rule": {"path": __file__, "name": "reward_rule"},
    }
    return config


def _full_data():
    return DataProto.from_dict(
        tensors={"responses": torch.arange(1, 4, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 8, 8)},
        non_tensors={
            "data_source": ["test"] * 3,
            "reward_model": [{"ground_truth": "ok"}] * 3,
            "extra_info": [{}] * 3,
        },
    )


@ray.remote
class _RealRayCpuWorker:
    def __init__(self, config, router_address, specs):
        self.config = config
        self.router_address = router_address
        self.specs = specs

    def placement(self):
        resources = ray.get_runtime_context().get_assigned_resources()
        return self.router_address is None, set(self.specs), resources.get("CPU", 0), resources.get("GPU", 0)

    def run_native_lifecycle(self):
        executor = NativeRewardExecutor(self.specs["quality"])

        async def run():
            await executor.wake_up()
            model = executor._model
            result = await executor.infer(41)
            await executor.sleep()
            return str(model.device), result, model.closed, executor._model is None

        return asyncio.run(run())


def test_cpu_native_workers_run_with_real_ray_cpu_resources():
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        config = SimpleNamespace(reward=SimpleNamespace(num_workers=2))
        spec = RewardModelSpec(
            name="quality",
            backend="native",
            model_path="cpu-test",
            executor_config={"model": f"{_CpuModel.__module__}:_CpuModel"},
            device_type="cpu",
        )
        placement = parse_reward_model_config(
            "quality",
            {
                "backend": "native",
                "placement": {"resource": "cpu", "devices": [3, 7], "cpus_per_worker": 2},
                "executor": {"model": f"{_CpuModel.__module__}:_CpuModel"},
            },
        ).placement
        manager = object.__new__(OmniRewardLoopManager)
        manager.multi_reward_model_manager = SimpleNamespace(
            models={"quality": SimpleNamespace(placement=placement)},
            native_resource_pools={},
        )
        manager.reward_loop_workers_class = _RealRayCpuWorker
        workers = manager._create_native_workers(config, {"quality": spec}, "quality", "cpu_native_real_ray_test")
        assert ray.get([worker.placement.remote() for worker in workers]) == [
            (True, {"quality"}, 2, 0),
            (True, {"quality"}, 2, 0),
        ]
        assert ray.get([worker.run_native_lifecycle.remote() for worker in workers]) == [
            ("cpu", 42, True, True),
            ("cpu", 42, True, True),
        ]
    finally:
        if started_ray:
            ray.shutdown()


@pytest.mark.parametrize("offload", [True, False])
def test_full_cpu_reward_manager_scoring_and_real_actor_restart(tmp_path, monkeypatch, offload):
    from verl_omni.reward_loop import cpu_reward_workers

    original_builder = cpu_reward_workers.build_cpu_reward_workers

    def restartable_workers(**kwargs):
        kwargs["reward_loop_workers_class"] = ray.remote(max_restarts=1)(_RestartableRewardWorker)
        return original_builder(**kwargs)

    monkeypatch.setattr(cpu_reward_workers, "build_cpu_reward_workers", restartable_workers)
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        manager = OmniRewardLoopManager(_full_config(tmp_path, offload))
        data = _full_data()
        first = manager.compute_rm_score(data)
        second = manager.compute_rm_score(data)
        for result in [first, second]:
            torch.testing.assert_close(result.batch["rm_scores"], torch.tensor([[1.25], [1.75], [2.25]]))
            assert result.non_tensor_batch["reward/quality/device"].tolist() == ["cpu"] * 3
            assert result.non_tensor_batch["reward/quality/cpus"].tolist() == [1] * 3
            assert result.non_tensor_batch["reward/quality/gpus"].tolist() == [0] * 3
            assert "reward/quality/errors" not in result.non_tensor_batch
        first_ids = first.non_tensor_batch["reward/quality/instance_id"]
        second_ids = second.non_tensor_batch["reward/quality/instance_id"]
        assert (first_ids != second_ids).all() if offload else (first_ids == second_ids).all()

        worker = manager._reward_worker_groups["quality"][0]
        ray.kill(worker, no_restart=False)
        asyncio.run(wait_for_restarted_worker(worker, first.non_tensor_batch["reward/quality/pid"][0]))
        restarted = manager.compute_rm_score(data)
        torch.testing.assert_close(restarted.batch["rm_scores"], first.batch["rm_scores"])
        assert restarted.non_tensor_batch["reward/quality/pid"][0] != first.non_tensor_batch["reward/quality/pid"][0]

        failed_data = _full_data()
        failed_data.non_tensor_batch["reward_model"][0] = {"ground_truth": "fail"}
        with pytest.raises(ray.exceptions.RayTaskError, match="intentional scoring failure"):
            manager.compute_rm_score(failed_data)
        recovered = manager.compute_rm_score(data)
        torch.testing.assert_close(recovered.batch["rm_scores"], first.batch["rm_scores"])
    finally:
        ray.shutdown()


@pytest.mark.parametrize("separate_replica", [False, True])
def test_cpu_phase_failure_drains_accepted_samples_before_sleep(tmp_path, monkeypatch, separate_replica):
    from verl_omni.reward_loop import cpu_reward_workers

    original_builder = cpu_reward_workers.build_cpu_reward_workers

    def controlled_workers(**kwargs):
        kwargs["reward_loop_workers_class"] = ray.remote(_RestartableRewardWorker)
        return original_builder(**kwargs)

    monkeypatch.setattr(cpu_reward_workers, "build_cpu_reward_workers", controlled_workers)
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        manager = OmniRewardLoopManager(_full_config(tmp_path, True))
        data = DataProto.concat([_full_data(), _full_data()[2:]])
        data.non_tensor_batch["reward_model"][0] = {"ground_truth": "controlled-fail"}
        data.non_tensor_batch["reward_model"][2 if separate_replica else 1] = {"ground_truth": "delay"}
        failing_worker, peer = manager._reward_worker_groups["quality"]
        delayed_worker = peer if separate_replica else failing_worker

        async def run():
            scoring = asyncio.create_task(manager.async_compute_rm_score(data))
            try:
                await asyncio.wait_for(delayed_worker.wait_for_delayed_sample.remote(), timeout=30)
                await failing_worker.release_failure.remote()
                await failing_worker.wait_for_failure.remote()
                completed, _ = await asyncio.wait({scoring}, timeout=0.2)
            finally:
                await delayed_worker.release_delay.remote()
                result = (await asyncio.gather(scoring, return_exceptions=True))[0]
            state = await delayed_worker.delayed_sample_state.remote()
            assert not completed, "The phase returned while an accepted sample was still waiting to infer"
            assert isinstance(result, ray.exceptions.RayTaskError)
            assert "intentional scoring failure" in str(result)
            assert state == (True, True, 0), "Accepted inference must finish successfully before model unloading"

        asyncio.run(run())
        recovered = manager.compute_rm_score(_full_data())
        torch.testing.assert_close(recovered.batch["rm_scores"], torch.tensor([[1.25], [1.75], [2.25]]))
    finally:
        ray.shutdown()


def test_cpu_phase_cancellation_drains_accepted_samples_before_sleep(tmp_path, monkeypatch):
    from verl_omni.reward_loop import cpu_reward_workers

    original_builder = cpu_reward_workers.build_cpu_reward_workers

    def controlled_workers(**kwargs):
        kwargs["reward_loop_workers_class"] = ray.remote(_RestartableRewardWorker)
        return original_builder(**kwargs)

    monkeypatch.setattr(cpu_reward_workers, "build_cpu_reward_workers", controlled_workers)
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        manager = OmniRewardLoopManager(_full_config(tmp_path, True))
        data = _full_data()
        data.non_tensor_batch["reward_model"][1] = {"ground_truth": "delay"}
        worker = manager._reward_worker_groups["quality"][0]

        async def run():
            scoring = asyncio.create_task(manager.async_compute_rm_score(data))
            try:
                await asyncio.wait_for(worker.wait_for_delayed_sample.remote(), timeout=30)
                for _ in range(2):
                    scoring.cancel()
                    completed, _ = await asyncio.wait({scoring}, timeout=0.2)
                    if completed:
                        break
            finally:
                await worker.release_delay.remote()
                result = (await asyncio.gather(scoring, return_exceptions=True))[0]
            state = await worker.delayed_sample_state.remote()
            assert not completed, "Cancellation returned before accepted RPCs completed"
            assert isinstance(result, asyncio.CancelledError)
            assert state == (True, True, 0)

        asyncio.run(run())
        recovered = manager.compute_rm_score(_full_data())
        torch.testing.assert_close(recovered.batch["rm_scores"], torch.tensor([[1.25], [1.75], [2.25]]))
    finally:
        ray.shutdown()


@pytest.mark.parametrize("stall_shared_worker", [False, True])
def test_cpu_scoring_that_never_returns_exits_after_repeated_cancellation(tmp_path, monkeypatch, stall_shared_worker):
    from verl_omni.reward_loop import cpu_reward_workers, reward_loop

    original_builder = cpu_reward_workers.build_cpu_reward_workers

    def controlled_workers(**kwargs):
        # Production has no restart policy; fault cleanup always disables restart.
        kwargs["reward_loop_workers_class"] = ray.remote(_RestartableRewardWorker)
        return original_builder(**kwargs)

    monkeypatch.setattr(cpu_reward_workers, "build_cpu_reward_workers", controlled_workers)
    if stall_shared_worker:
        original_shared = OmniRewardLoopManager._create_node_affinity_workers

        def controlled_shared(manager, *args, **kwargs):
            original_class = manager.reward_loop_workers_class
            manager.reward_loop_workers_class = ray.remote(_RestartableRewardWorker)
            try:
                return original_shared(manager, *args, **kwargs)
            finally:
                manager.reward_loop_workers_class = original_class

        monkeypatch.setattr(OmniRewardLoopManager, "_create_node_affinity_workers", controlled_shared)
    monkeypatch.setattr(reward_loop, "_SCORING_DRAIN_TIMEOUT", 0.05)
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        manager = OmniRewardLoopManager(_full_config(tmp_path, True))
        data = _full_data()
        data.non_tensor_batch["reward_model"][1] = {"ground_truth": "delay"}
        worker = manager._reward_worker_groups["quality"][0]
        shared_worker = manager._reward_worker_groups["shared"][0]
        shared_pid = manager._shared_worker_process_identities[id(shared_worker)]["pid"]
        pid = ray.get(worker.worker_pid.remote())

        async def run():
            scoring = asyncio.create_task(manager.async_compute_rm_score(data))
            await asyncio.wait_for(worker.wait_for_delayed_sample.remote(), timeout=30)
            if stall_shared_worker:
                await asyncio.wait_for(shared_worker.wait_for_delayed_sample.remote(), timeout=30)
            started = time.monotonic()
            scoring.cancel()
            while not scoring.done() and time.monotonic() - started < 40:
                await asyncio.sleep(0.05)
                scoring.cancel()
            assert scoring.done(), "Repeated cancellation must not reset the cleanup deadline"
            result = (await asyncio.gather(scoring, return_exceptions=True))[0]
            assert isinstance(result, asyncio.CancelledError), result
            assert not manager._score_lock.locked()
            assert manager._scoring_unusable
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            if stall_shared_worker:
                assert not psutil.pid_exists(shared_pid) or psutil.Process(shared_pid).status() == psutil.STATUS_ZOMBIE
            with pytest.raises(RuntimeError, match="recreate the reward manager"):
                await manager.async_compute_rm_score(_full_data())

        asyncio.run(run())
    finally:
        ray.shutdown()
