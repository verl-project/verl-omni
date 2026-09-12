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
"""Worker-side executors for named reward models."""

from __future__ import annotations

import asyncio
import gc
import importlib
import inspect
from typing import Any

import torch
from verl.utils.device import get_device_id, get_device_name

from .reward_model_config import RewardModelSpec, is_engine_backend

__all__ = [
    "EngineRewardExecutor",
    "NativeRewardExecutor",
    "build_engine_reward_executors",
    "build_native_reward_executors",
]


class EngineRouterClient:
    """Expose one engine router to a configured reward function."""

    def __init__(self, router_address: str, model_path: str):
        self.router_address = router_address
        self.model_path = model_path

    def reward_kwargs(self) -> dict[str, str]:
        return {
            "reward_router_address": self.router_address,
            "model_name": self.model_path,
        }


class EngineRewardExecutor:
    """Worker-side request contract for one engine-backed reward model."""

    def __init__(self, spec: RewardModelSpec):
        self.spec = spec
        self._client = EngineRouterClient(
            router_address=spec.router_address,
            model_path=spec.model_path,
        )

    def reward_kwargs(self) -> dict[str, str]:
        return self._client.reward_kwargs()


class NativeRewardExecutor:
    """Worker-local model owner that exposes inference, never scoring."""

    def __init__(self, spec: RewardModelSpec):
        if spec.backend != "native":
            raise ValueError(f"NativeRewardExecutor requires a native spec, got {spec.backend!r}")
        self.spec = spec
        self._model: Any | None = None
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()

    async def wake_up(self) -> None:
        async with self._lock:
            if self._model is not None:
                return
            model_cls = _load_native_model(self.spec.executor_config["model"])
            kwargs = dict(self.spec.executor_config.get("kwargs", {}))
            if self.spec.model_path is not None:
                kwargs.setdefault("model_path", self.spec.model_path)
            kwargs.setdefault("device", torch.device(get_device_name(), get_device_id()))
            self._model = model_cls(**kwargs)

    def reward_kwargs(self) -> dict[str, Any]:
        return {"reward_model": self}

    async def infer(self, *args, **kwargs):
        """Run model inference while protecting the wake/sleep boundary."""
        async with self._lock:
            if self._model is None:
                raise RuntimeError(f"Native reward model {self.spec.name!r} is not awake")
            self._inflight += 1
            self._idle.clear()
        try:
            infer_fn = getattr(self._model, "infer", None)
            if infer_fn is None:
                raise TypeError(f"Native reward model {type(self._model).__name__!r} must define infer()")
            if inspect.iscoroutinefunction(infer_fn):
                result = await infer_fn(*args, **kwargs)
            else:
                result = await asyncio.to_thread(infer_fn, *args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def sleep(self) -> None:
        while True:
            await self._idle.wait()
            async with self._lock:
                if self._inflight:
                    continue
                model, self._model = self._model, None
                if model is not None:
                    close = getattr(model, "close", None)
                    if close is not None:
                        result = await close() if inspect.iscoroutinefunction(close) else await asyncio.to_thread(close)
                        if inspect.isawaitable(result):
                            await result
                gc.collect()
                _empty_accelerator_cache()
                return


def build_engine_reward_executors(specs: dict[str, RewardModelSpec]) -> dict[str, EngineRewardExecutor]:
    """Build worker-side router clients for engine-backed model specs."""
    return {name: EngineRewardExecutor(spec) for name, spec in specs.items() if is_engine_backend(spec.backend)}


def build_native_reward_executors(specs: dict[str, RewardModelSpec]) -> dict[str, NativeRewardExecutor]:
    """Build worker-local executors for native model specs."""
    return {name: NativeRewardExecutor(spec) for name, spec in specs.items() if spec.backend == "native"}


def _load_native_model(model_path: str):
    module_path, class_name = model_path.rsplit(":", 1)
    if module_path.startswith("pkg://"):
        module_path = module_path[len("pkg://") :].replace("/", ".")
    if "/" not in module_path and not module_path.endswith(".py"):
        return getattr(importlib.import_module(module_path), class_name)

    from verl.utils.import_utils import load_extern_object

    return load_extern_object(module_path=module_path, object_name=class_name)


def _empty_accelerator_cache() -> None:
    accelerator = getattr(torch, get_device_name(), None)
    empty_cache = getattr(accelerator, "empty_cache", None)
    if callable(empty_cache) and getattr(accelerator, "is_available", lambda: False)():
        empty_cache()
