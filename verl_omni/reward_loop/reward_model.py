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
"""Controller-side lifecycle and resource management for named reward models."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import OmegaConf
from verl.experimental.reward_loop.reward_model import RewardModelManager
from verl.single_controller.ray.base import split_resource_pool

from .reward_model_config import (
    EngineRewardModelConfig,
    NativeRewardModelConfig,
    RewardModelSpec,
    get_reward_model_entries,
    has_reward_models,
    parse_reward_model_config,
    to_mapping,
)

__all__ = [
    "EngineManagedRewardModel",
    "ManagedRewardModel",
    "MultiRewardModelManager",
    "NativeManagedRewardModel",
]

logger = logging.getLogger(__name__)


class MultiRewardModelManager:
    """Create and lifecycle-manage all configured named reward models."""

    def __init__(self, config, resource_pool=None):
        if has_reward_models(config) and config.reward.reward_model.get("enable", False):
            raise ValueError("reward.reward_model.enable cannot be combined with reward.models")
        self.config = config
        self.resource_pool = resource_pool
        self.models: dict[str, ManagedRewardModel] = {}
        self.engine_resource_pools: dict[str, Any] = {}
        self.native_resource_pool = None

        entries = [
            (name, parse_reward_model_config(name, model)) for name, model in get_reward_model_entries(config).items()
        ]
        base_config = config.reward.reward_model
        fallback_model = base_config.get("model_path")
        engine_entries = [(name, model) for name, model in entries if isinstance(model, EngineRewardModelConfig)]
        native_entries = [(name, model) for name, model in entries if isinstance(model, NativeRewardModelConfig)]

        self.native_device_assignments = self._validate_native_device_assignments(native_entries)
        engine_pools, self.native_resource_pool = self._split_model_resource_pools(
            engine_entries, native_entries, base_config
        )
        self.engine_resource_pools = engine_pools

        for name, model in entries:
            if isinstance(model, EngineRewardModelConfig):
                self.models[name] = EngineManagedRewardModel(
                    name, model, base_config, engine_pools.get(name), fallback_model
                )
            else:
                self.models[name] = NativeManagedRewardModel(name, model)

    @property
    def reward_model_specs(self) -> dict[str, RewardModelSpec]:
        return {name: model.executor_spec for name, model in self.models.items()}

    @property
    def has_engine_model(self) -> bool:
        return any(isinstance(model, EngineManagedRewardModel) for model in self.models.values())

    def get_reward_model_spec(self, name: str) -> RewardModelSpec:
        try:
            return self.reward_model_specs[name]
        except KeyError as exc:
            raise ValueError(f"Unknown reward model {name!r}") from exc

    def bind_native_workers(self, name: str, workers) -> None:
        model = self.models.get(name)
        if not isinstance(model, NativeManagedRewardModel):
            raise ValueError(f"Reward model {name!r} is not a native model")
        model.bind_workers(workers)

    async def wake_up(self) -> None:
        """Wake independent reward models concurrently."""
        await asyncio.gather(*(model.wake_up() for model in self.models.values()))

    async def sleep(self) -> None:
        """Attempt to sleep every model and report the first lifecycle error."""
        models = list(reversed(self.models.values()))
        results = await asyncio.gather(*(model.sleep() for model in models), return_exceptions=True)
        errors = []
        for model, result in zip(models, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(result)
                logger.error("Failed to sleep reward model %s: %s", model.name, result)
        if errors:
            raise errors[0]

    @staticmethod
    def _validate_native_device_assignments(
        native_entries: list[tuple[str, NativeRewardModelConfig]],
    ) -> dict[str, tuple[int, ...]]:
        """Validate only cross-model placement overlap."""
        assignments: dict[str, tuple[int, ...]] = {}
        claimed_devices: dict[int, str] = {}
        for name, model in native_entries:
            for device in model.placement.devices:
                if device in claimed_devices:
                    raise ValueError(
                        f"Native reward model {name!r} placement.devices overlaps index {device} "
                        f"already assigned to {claimed_devices[device]!r}"
                    )
                claimed_devices[device] = name
            assignments[name] = tuple(model.placement.devices)
        return assignments

    def _split_engine_resource_pool(self, engine_entries, base_config) -> dict[str, Any]:
        """Return the engine-only view of the parent resource-pool split."""
        engine_pools, _ = self._split_model_resource_pools(engine_entries, [], base_config)
        return engine_pools

    def _split_model_resource_pools(self, engine_entries, native_entries, base_config):
        if not engine_entries and not native_entries:
            return {}, None
        if self.resource_pool is None:
            raise ValueError("Named reward models require a parent resource pool selected by the trainer")

        engine_requested_sizes = []
        for _, model in engine_entries:
            engine_requested_sizes.append(model.requested_resource_size(base_config))

        native_requested = 0
        if native_entries:
            native_requested = (
                max(device for devices in self.native_device_assignments.values() for device in devices) + 1
            )

        requested_total = sum(engine_requested_sizes) + native_requested
        if requested_total > self.resource_pool.world_size:
            raise ValueError(
                f"Named reward models request {requested_total} devices, but the parent reward pool has only "
                f"{self.resource_pool.world_size}"
            )

        split_sizes = list(engine_requested_sizes)
        if native_requested:
            split_sizes.append(native_requested)
        if requested_total < self.resource_pool.world_size:
            split_sizes.append(self.resource_pool.world_size - requested_total)
        sub_pools = split_resource_pool(self.resource_pool, split_sizes)
        engine_count = len(engine_entries)
        engine_pools = {name: pool for (name, _), pool in zip(engine_entries, sub_pools[:engine_count], strict=True)}
        native_pool = sub_pools[engine_count] if native_requested else None
        return engine_pools, native_pool


class ManagedRewardModel(ABC):
    """A named reward model with an explicit asynchronous lifecycle."""

    def __init__(self, spec: RewardModelSpec, offload: bool):
        self.spec = spec
        self.offload = offload

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def executor_spec(self) -> RewardModelSpec:
        return self.spec

    @abstractmethod
    async def wake_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def sleep(self) -> None:
        raise NotImplementedError


class EngineManagedRewardModel(ManagedRewardModel):
    """Engine-backed model owned by the upstream ``RewardModelManager``."""

    def __init__(self, name: str, model, base_config, resource_pool, fallback_model=None):
        if not isinstance(model, EngineRewardModelConfig):
            model = parse_reward_model_config(name, model)
        offload = model.resolved_offload
        config = _prepare_engine_config(model, base_config, fallback_model)
        self.reward_model_manager = RewardModelManager(config, resource_pool)
        super().__init__(
            RewardModelSpec(
                name=name,
                backend="engine",
                model_path=config.model_path,
                router_address=self.reward_model_manager.get_router_address(),
            ),
            offload=offload,
        )

    async def wake_up(self) -> None:
        if self.offload:
            await asyncio.to_thread(self.reward_model_manager.wake_up)

    async def sleep(self) -> None:
        if self.offload:
            await asyncio.to_thread(self.reward_model_manager.sleep)


class NativeManagedRewardModel(ManagedRewardModel):
    """Native model state owned by explicitly placed reward workers."""

    def __init__(self, name: str, model):
        if not isinstance(model, NativeRewardModelConfig):
            model = parse_reward_model_config(name, model)
        offload = model.resolved_offload
        executor_config = {"model": model.executor.model, "kwargs": model.executor.kwargs}

        super().__init__(
            RewardModelSpec(
                name=name,
                backend="native",
                model_path=model.model_path,
                executor_config=executor_config,
            ),
            offload=offload,
        )
        self._workers = None
        self._resident = False

    def bind_workers(self, workers) -> None:
        self._workers = list(workers)

    async def _run_worker_lifecycle(self, method: str) -> None:
        if self._workers is None:
            raise RuntimeError(f"Native reward model {self.name!r} has no bound workers")
        refs = [getattr(worker, method).remote(self.name) for worker in self._workers]
        await asyncio.gather(*refs)

    async def wake_up(self) -> None:
        if not self.offload and self._resident:
            return
        await self._run_worker_lifecycle("wake_up_reward_model")
        self._resident = True

    async def sleep(self) -> None:
        if not self.offload:
            return
        await self._run_worker_lifecycle("sleep_reward_model")
        self._resident = False


def _prepare_engine_config(model, base_config, fallback_model=None):
    if not isinstance(model, EngineRewardModelConfig):
        model = parse_reward_model_config("engine", model)
    offload = model.resolved_offload
    config = OmegaConf.merge(
        OmegaConf.create(to_mapping(base_config)),
        OmegaConf.create(model.to_engine_overrides()),
    )
    config.enable = True
    if config.get("model_path") is None:
        config.model_path = fallback_model
    if config.get("model_path") is None:
        raise ValueError("Engine reward model requires model_path")
    if config.get("rollout") is None:
        raise ValueError("Engine reward model requires rollout config")
    config.rollout.free_cache_engine = offload
    config.rollout.enable_sleep_mode = offload
    if OmegaConf.is_missing(config.rollout, "name") or config.rollout.get("name") == "???":
        config.rollout.name = "vllm"
    engine_kwargs = config.rollout.get("engine_kwargs") or {}
    vllm_kwargs = engine_kwargs.get("vllm") or {}
    if config.rollout.name == "vllm" and vllm_kwargs.get("runner") == "pooling":
        worker_extension_cls = "verl_omni.reward_loop.vllm_worker.PoolingRewardModelWorkerExtension"
        if vllm_kwargs.get("worker_extension_cls") is None:
            config.rollout.engine_kwargs.vllm.worker_extension_cls = worker_extension_cls
    return config
