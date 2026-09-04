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
"""Managed reward-model deployments used by :mod:`verl_omni.reward_loop`.

The upstream ``RewardModelManager`` remains the owner of one engine-backed
reward model. ``MultiRewardModelManager`` owns the parent resource pool and
splits it between those single-model managers; native models remain worker-local.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import importlib
import inspect
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch
from omegaconf import OmegaConf
from PIL import Image
from verl.experimental.reward_loop.reward_model import RewardModelManager
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)

_ENGINE_BACKENDS = {"engine"}
_NATIVE_BACKENDS = {"native"}


def has_reward_deployments(config) -> bool:
    """Return whether the new deployment directory has at least one entry."""
    deployments = config.reward.get("deployments", {})
    return bool(deployments)


def reward_is_enabled(config) -> bool:
    """Return whether either legacy or deployment reward computation is enabled."""
    reward_model = config.reward.get("reward_model", {})
    return bool(reward_model.get("enable", False) or has_reward_deployments(config))


def has_engine_reward_deployments(config) -> bool:
    return any(
        is_engine_backend(deployment.get("backend"))
        for deployment in (config.reward.get("deployments", {}) or {}).values()
    )


def is_engine_backend(backend: str | None) -> bool:
    return backend in _ENGINE_BACKENDS


def has_native_reward_deployments(config) -> bool:
    return any(
        deployment.get("backend") in _NATIVE_BACKENDS
        for deployment in (config.reward.get("deployments", {}) or {}).values()
    )


def validate_reward_deployment_terms(config) -> None:
    """Check that multi-reward terms and their named deployments agree.

    Validate before any ``RewardModelManager`` is constructed: a typo in a
    term must not first allocate an engine replica group and only then fail in
    a worker. Native scorers own their whole score operation, while a generic
    engine deployment needs the term's existing function to turn its router
    response into a reward. ``pickscore`` is the engine adapter currently
    supplied by verl-omni and therefore scores directly.
    """
    deployments = config.reward.get("deployments", {}) or {}
    for term_name, term in (config.reward.get("reward_functions", {}) or {}).items():
        deployment_name = term.get("deployment")
        if deployment_name is None:
            continue
        if deployment_name not in deployments:
            raise ValueError(f"Reward term {term_name!r} references unknown deployment {deployment_name!r}")

        deployment = deployments[deployment_name]
        has_function = term.get("path") is not None
        backend = deployment.get("backend")
        adapter = deployment.get("adapter") or _coerce_mapping(deployment.get("executor")).get("adapter")
        if backend in _NATIVE_BACKENDS and has_function:
            raise ValueError(
                f"Native reward deployment {deployment_name!r} scores directly; "
                f"remove path/name from reward term {term_name!r}"
            )
        if is_engine_backend(backend) and adapter is None and not has_function:
            raise ValueError(
                f"Engine reward deployment {deployment_name!r} needs path/name in reward term {term_name!r} "
                "to consume its router"
            )
        if is_engine_backend(backend) and adapter == "pickscore" and has_function:
            raise ValueError(
                f"PickScore engine deployment {deployment_name!r} scores directly; "
                f"remove path/name from reward term {term_name!r}"
            )


def reward_role_required(config) -> bool:
    """Whether the reward loop needs the trainer-selected parent pool.

    Every named deployment is managed from the trainer-selected ``global_pool``
    or ``reward_pool``.  This includes native deployments: their worker
    placement must use the native subpool selected by
    ``MultiRewardModelManager`` rather than implicitly borrowing the actor
    pool.
    """
    return bool(config.reward.reward_model.get("enable", False) or has_reward_deployments(config))


def reward_pool_is_separate(config) -> bool:
    """Whether the legacy reward-model role uses a dedicated resource pool."""
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def streaming_reward_enabled(config) -> bool:
    """Whether workers can score while the agent rollout is still streaming."""
    if not reward_is_enabled(config):
        return True
    if has_engine_reward_deployments(config):
        # The controller owns engine wake/sleep around ``compute_rm_score``;
        # streaming workers have no controller callback at request time.
        return False
    if has_native_reward_deployments(config):
        # A named native deployment has its own explicitly placed worker
        # group.  The upstream streaming interface accepts one worker list,
        # so it cannot fan one request out to all native deployment groups.
        # Native rewards deliberately retain ordinary batch scoring.
        return False
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def accelerator_workers_enabled(config) -> bool:
    """Whether custom reward workers should use the accelerator placement helper."""
    reward = config.reward
    if hasattr(reward, "get"):
        accelerator_workers = reward.get("accelerator_workers", {}) or {}
        legacy_custom = reward.get("custom_reward_function", {}) or {}
    else:
        accelerator_workers = getattr(reward, "accelerator_workers", {}) or {}
        legacy_custom = getattr(reward, "custom_reward_function", {}) or {}

    def value(mapping, key, default=False):
        if hasattr(mapping, "get"):
            return mapping.get(key, default)
        return getattr(mapping, key, default)

    return bool(value(accelerator_workers, "enabled") or value(legacy_custom, "use_accelerator"))


def _coerce_mapping(value) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return OmegaConf.to_container(value, resolve=False)


def _load_native_scorer(scorer_path: str):
    """Load a native scorer from ``module:Class`` or an existing file path.

    The public configuration uses Python module names so the same deployment
    works in every Ray worker without depending on a controller-local relative
    path.  File paths remain supported through verl's existing loader.
    """
    module_path, class_name = scorer_path.rsplit(":", 1)
    if module_path.startswith("pkg://"):
        module_path = module_path[len("pkg://") :].replace("/", ".")
    if "/" not in module_path and not module_path.endswith(".py"):
        return getattr(importlib.import_module(module_path), class_name)

    from verl.utils.import_utils import load_extern_object

    return load_extern_object(module_path=module_path, object_name=class_name)


def _empty_accelerator_cache() -> None:
    """Release cached allocations for the current supported accelerator."""
    accelerator = getattr(torch, get_device_name(), None)
    empty_cache = getattr(accelerator, "empty_cache", None)
    if callable(empty_cache) and getattr(accelerator, "is_available", lambda: False)():
        empty_cache()


def _prepare_engine_config(deployment, base_config, fallback_model=None):
    """Build the exact config expected by upstream ``RewardModelManager``."""
    config = OmegaConf.merge(
        OmegaConf.create(_coerce_mapping(base_config)),
        OmegaConf.create(_coerce_mapping(deployment)),
    )
    for key in ("backend", "executor", "name", "enable_resource_pool", "adapter"):
        if key in config:
            del config[key]
    config.enable = True
    if config.get("model_path") is None:
        config.model_path = fallback_model
    if config.get("model_path") is None:
        raise ValueError("Engine reward deployment requires model_path")
    if config.get("rollout") is None:
        raise ValueError("Engine reward deployment requires rollout config")
    if OmegaConf.is_missing(config.rollout, "name") or config.rollout.get("name") == "???":
        config.rollout.name = "vllm"
    return config


@dataclass(frozen=True)
class RewardExecutorSpec:
    """Static metadata shared with every reward-loop worker."""

    name: str
    backend: str
    model_path: str | None
    router_address: str | None
    executor_config: dict[str, Any]


class RewardDeployment(ABC):
    """A model deployment with an explicit wake/score/sleep lifecycle."""

    def __init__(self, spec: RewardExecutorSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def executor_spec(self) -> RewardExecutorSpec:
        return self.spec

    @abstractmethod
    def wake_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def sleep(self) -> None:
        raise NotImplementedError


class EngineRewardDeployment(RewardDeployment):
    """Engine deployment owned by the existing ``verl.RewardModelManager``."""

    def __init__(self, name: str, deployment, base_config, resource_pool, fallback_model=None):
        config = _prepare_engine_config(deployment, base_config, fallback_model)
        executor_config = _coerce_mapping(deployment.get("executor"))
        self.reward_model_manager = RewardModelManager(config, resource_pool)
        super().__init__(
            RewardExecutorSpec(
                name=name,
                backend="engine",
                model_path=config.model_path,
                router_address=self.reward_model_manager.get_router_address(),
                executor_config={
                    **executor_config,
                    "adapter": deployment.get("adapter") or executor_config.get("adapter"),
                },
            )
        )

    def wake_up(self) -> None:
        self.reward_model_manager.wake_up()

    def sleep(self) -> None:
        self.reward_model_manager.sleep()


class NativeRewardDeployment(RewardDeployment):
    """Native model state that is owned by every accelerator reward worker."""

    def __init__(self, name: str, deployment):
        executor_config = _coerce_mapping(deployment.get("executor"))
        scorer = executor_config.get("scorer")
        adapter = deployment.get("adapter") or executor_config.get("adapter")
        if scorer is None and adapter == "pickscore":
            scorer = "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeScorer"
            executor_config["scorer"] = scorer
        if not scorer:
            raise ValueError(f"Native reward deployment {name!r} requires executor.scorer")
        super().__init__(
            RewardExecutorSpec(
                name=name,
                backend="native",
                model_path=deployment.get("model_path"),
                router_address=None,
                executor_config=executor_config,
            )
        )

    def wake_up(self) -> None:
        return None

    def sleep(self) -> None:
        return None


class MultiRewardModelManager:
    """Create and lifecycle-manage all configured reward deployments.

    The manager lives in the trainer/controller process. It receives one
    trainer-selected parent pool and splits it into disjoint engine and native
    subpools before constructing one upstream ``RewardModelManager`` per engine
    deployment. Native deployments are represented here and instantiated
    lazily by ``NativeRewardExecutor`` in reward-loop workers placed on the
    native subpool.
    """

    def __init__(self, config, resource_pool=None):
        if has_reward_deployments(config) and config.reward.reward_model.get("enable", False):
            raise ValueError("reward.reward_model.enable cannot be combined with reward.deployments")
        self.config = config
        self.resource_pool = resource_pool
        self.deployments: dict[str, RewardDeployment] = {}
        self.engine_resource_pools: dict[str, Any] = {}
        self.native_resource_pool = None
        entries = config.reward.get("deployments", {}) or {}
        fallback_model = config.reward.reward_model.get("model_path")
        base_config = config.reward.reward_model
        engine_entries = [
            (name, deployment) for name, deployment in entries.items() if is_engine_backend(deployment.get("backend"))
        ]
        for name, deployment in engine_entries:
            if "enable_resource_pool" in deployment:
                raise ValueError(
                    f"Engine reward deployment {name!r} must not set enable_resource_pool; "
                    "select the parent pool with reward.reward_model.enable_resource_pool instead"
                )
        native_entries = [
            (name, deployment) for name, deployment in entries.items() if deployment.get("backend") in _NATIVE_BACKENDS
        ]
        self.native_device_assignments = self._validate_native_device_assignments(native_entries)
        engine_pools, self.native_resource_pool = self._split_deployment_resource_pools(
            engine_entries, native_entries, base_config
        )
        self.engine_resource_pools = engine_pools
        for name, deployment in entries.items():
            backend = deployment.get("backend")
            if is_engine_backend(backend):
                self.deployments[name] = EngineRewardDeployment(
                    name, deployment, base_config, engine_pools.get(name), fallback_model
                )
            elif backend in _NATIVE_BACKENDS:
                self.deployments[name] = NativeRewardDeployment(name, deployment)
            else:
                supported = ", ".join(sorted(_ENGINE_BACKENDS | _NATIVE_BACKENDS))
                raise ValueError(
                    f"Reward deployment {name!r} has unsupported backend {backend!r}; expected one of {supported}"
                )

    @staticmethod
    def _rollout_world_size(deployment, base_config) -> int:
        rollout = _coerce_mapping(deployment.get("rollout"))
        base_rollout = _coerce_mapping(base_config.get("rollout"))
        merged = {**base_rollout, **rollout}
        values = [
            int(merged.get("tensor_model_parallel_size", 1)),
            int(merged.get("data_parallel_size", 1)),
            int(merged.get("pipeline_model_parallel_size", 1)),
        ]
        if any(value <= 0 for value in values):
            raise ValueError("Engine reward rollout parallel sizes must be positive")
        replicas = int(deployment.get("replicas", 1))
        if replicas <= 0:
            raise ValueError("Engine reward deployment replicas must be positive")
        return replicas * values[0] * values[1] * values[2]

    def _split_engine_resource_pool(self, engine_entries, base_config) -> dict[str, Any]:
        """Backward-compatible engine-only view of the parent-pool split."""
        engine_pools, _ = self._split_deployment_resource_pools(engine_entries, [], base_config)
        return engine_pools

    @staticmethod
    def _validate_native_device_assignments(native_entries) -> dict[str, tuple[int, ...]]:
        """Validate native worker placement and return native-pool indices.

        ``placement.devices`` intentionally addresses *bundles in the native
        subpool*, never a physical CUDA/NPU index.  Ray can remap visible
        devices inside an actor, whereas placement-group bundle indices remain
        stable.  A listed bundle hosts one complete native model instance and
        cannot be shared by another native deployment.
        """
        assignments: dict[str, tuple[int, ...]] = {}
        claimed_devices: dict[int, str] = {}
        engine_fields = {
            "replicas",
            "rollout",
            "n_gpus_per_node",
            "nnodes",
            "tensor_model_parallel_size",
            "data_parallel_size",
            "pipeline_model_parallel_size",
            "expert_parallel_size",
            "enable_resource_pool",
        }
        for name, deployment in native_entries:
            unsupported = sorted(field for field in engine_fields if deployment.get(field) is not None)
            if unsupported:
                fields = ", ".join(unsupported)
                raise ValueError(
                    f"Native reward deployment {name!r} does not support engine resource fields: {fields}"
                )

            placement = deployment.get("placement")
            if placement is not None and not isinstance(placement, dict):
                placement = OmegaConf.to_container(placement, resolve=False)
            if not isinstance(placement, dict) or "devices" not in placement:
                raise ValueError(
                    f"Native reward deployment {name!r} requires placement.devices as native-pool bundle indices"
                )
            devices = placement["devices"]
            if not isinstance(devices, list) or not devices:
                raise ValueError(f"Native reward deployment {name!r} placement.devices must be a non-empty list")

            normalized_devices = []
            for device in devices:
                if isinstance(device, bool) or not isinstance(device, int) or device < 0:
                    raise ValueError(
                        f"Native reward deployment {name!r} placement.devices must contain non-negative integers"
                    )
                if device in normalized_devices:
                    raise ValueError(
                        f"Native reward deployment {name!r} placement.devices contains duplicate index {device}"
                    )
                if device in claimed_devices:
                    raise ValueError(
                        f"Native reward deployment {name!r} placement.devices overlaps index {device} "
                        f"already assigned to {claimed_devices[device]!r}"
                    )
                normalized_devices.append(device)
                claimed_devices[device] = name
            assignments[name] = tuple(normalized_devices)
        return assignments

    def _split_deployment_resource_pools(self, engine_entries, native_entries, base_config):
        """Split the selected parent pool between engine and native workers.

        Engine deployments reserve the number of bundles needed by their
        rollout replicas. Native deployments share one *parent* native
        subpool, then bind individual deployments to the explicit bundle
        indices in ``placement.devices``. Any unrequested bundles are
        returned as an unused trailing split so ``split_resource_pool``
        receives the complete parent-pool size it requires.
        """
        if not engine_entries and not native_entries:
            return {}, None
        engine_requested_sizes = []
        if self.resource_pool is None:
            raise ValueError("Reward deployments require a parent resource pool selected by the trainer")

        for name, deployment in engine_entries:
            explicit_gpus = deployment.get("n_gpus_per_node")
            explicit_nodes = deployment.get("nnodes")
            if (explicit_gpus is None) != (explicit_nodes is None):
                raise ValueError(f"Engine reward deployment {name!r} must set both n_gpus_per_node and nnodes")
            if explicit_gpus is not None:
                requested = int(explicit_gpus) * int(explicit_nodes)
            else:
                requested = self._rollout_world_size(deployment, base_config)
            if requested <= 0:
                raise ValueError(f"Engine reward deployment {name!r} resource allocation must be positive")
            rollout_world_size = self._rollout_world_size(deployment, base_config)
            if requested < rollout_world_size or requested % rollout_world_size:
                raise ValueError(
                    f"Engine reward deployment {name!r} allocation ({requested}) must be a multiple of "
                    f"rollout world size ({rollout_world_size})"
                )
            engine_requested_sizes.append(requested)

        native_requested = 0
        if native_entries:
            assignments = getattr(self, "native_device_assignments", None)
            if assignments is None:
                assignments = self._validate_native_device_assignments(native_entries)
            native_requested = max(device for devices in assignments.values() for device in devices) + 1

        parent_world_size = self.resource_pool.world_size
        requested_total = sum(engine_requested_sizes) + native_requested
        if requested_total > parent_world_size:
            raise ValueError(
                f"Reward deployments request {requested_total} devices, but the parent reward pool has only "
                f"{parent_world_size}"
            )
        split_sizes = [*engine_requested_sizes]
        if native_requested:
            split_sizes.append(native_requested)
        if requested_total < parent_world_size:
            split_sizes = [*split_sizes, parent_world_size - requested_total]
        sub_pools = split_resource_pool(self.resource_pool, split_sizes)
        engine_count = len(engine_entries)
        engine_pools = {
            name: pool for (name, _), pool in zip(engine_entries, sub_pools[:engine_count], strict=True)
        }
        native_pool = sub_pools[engine_count] if native_requested else None
        return engine_pools, native_pool

    @property
    def reward_executor_specs(self) -> dict[str, RewardExecutorSpec]:
        return {name: deployment.executor_spec for name, deployment in self.deployments.items()}

    @property
    def has_engine_deployment(self) -> bool:
        return any(isinstance(deployment, EngineRewardDeployment) for deployment in self.deployments.values())

    def get_reward_executor_spec(self, name: str) -> RewardExecutorSpec:
        try:
            return self.reward_executor_specs[name]
        except KeyError as exc:
            raise ValueError(f"Unknown reward deployment {name!r}") from exc

    def wake_up(self) -> None:
        for deployment in self.deployments.values():
            deployment.wake_up()

    def sleep(self) -> None:
        errors = []
        for deployment in reversed(list(self.deployments.values())):
            try:
                deployment.sleep()
            except Exception as exc:  # pragma: no cover - only relevant to live engine failures
                errors.append(exc)
                logger.exception("Failed to sleep reward deployment %s", deployment.name)
        if errors:
            raise errors[0]


def _to_pil_image(image) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image


def _image_data_url(image) -> str:
    image = _to_pil_image(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class PickScoreEngineAdapter:
    """Use a vLLM CLIP embedding endpoint to calculate the PickScore formula.

    vLLM's ``CLIPEmbeddingModel`` intentionally ignores the checkpoint
    ``logit_scale`` parameter.  Keep the scale explicit in the deployment
    configuration rather than claiming that an arbitrary PickScore checkpoint
    is numerically identical to the Transformers implementation.
    """

    def __init__(self, router_address: str, model_path: str, logit_scale: float, score_divisor: float = 26.0):
        self.router_address = router_address
        self.model_path = model_path
        self.logit_scale = logit_scale
        self.score_divisor = score_divisor

    async def score(self, prompt: str, image) -> dict[str, float]:
        image_url = _image_data_url(image)
        text_payload = {
            "model": self.model_path,
            "input": prompt,
            "encoding_format": "float",
        }
        image_payload = {
            "model": self.model_path,
            "input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}],
            "encoding_format": "float",
        }
        url = f"http://{self.router_address}/v1/embeddings"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(url, json=text_payload) as response:
                response.raise_for_status()
                text_result = await response.json()
            async with session.post(url, json=image_payload) as response:
                response.raise_for_status()
                image_result = await response.json()
        text = torch.tensor(text_result["data"][0]["embedding"], dtype=torch.float32)
        image_embedding = torch.tensor(image_result["data"][0]["embedding"], dtype=torch.float32)
        cosine = torch.nn.functional.cosine_similarity(text.unsqueeze(0), image_embedding.unsqueeze(0)).item()
        raw_score = self.logit_scale * cosine / self.score_divisor
        return {"score": raw_score, "pickscore_raw": raw_score}


class EngineRouterAdapter:
    """Adapt one named engine router to an existing reward function.

    Many engine-backed rewards already have their task-specific logic in a
    regular reward function (for example, OCR calls chat completions and then
    applies its own string metric). This adapter gives that function the
    selected deployment's router arguments instead of the former global router.
    """

    def __init__(self, router_address: str, model_path: str):
        self.router_address = router_address
        self.model_path = model_path

    def reward_kwargs(self) -> dict[str, str]:
        return {
            "reward_router_address": self.router_address,
            "model_name": self.model_path,
        }


class EngineRewardExecutor:
    """Worker-side executor for one engine-backed reward deployment.

    The deployment/controller side owns router replicas and lifecycle through
    ``RewardModelManager``.  This executor owns only the worker-side request
    contract: generic reward functions receive router kwargs, while a semantic
    adapter such as PickScore can turn endpoint responses into a score.
    """

    def __init__(self, spec: RewardExecutorSpec):
        self.spec = spec
        adapter = spec.executor_config.get("adapter")
        if adapter == "pickscore":
            if "logit_scale" not in spec.executor_config:
                raise ValueError(
                    f"PickScore engine deployment {spec.name!r} requires executor.logit_scale; "
                    "vLLM does not load the checkpoint logit_scale."
                )
            self._adapter = PickScoreEngineAdapter(
                router_address=spec.router_address,
                model_path=spec.model_path,
                logit_scale=float(spec.executor_config["logit_scale"]),
                score_divisor=float(spec.executor_config.get("score_divisor", 26.0)),
            )
        elif adapter is None:
            self._adapter = EngineRouterAdapter(
                router_address=spec.router_address,
                model_path=spec.model_path,
            )
        else:
            raise ValueError(f"Unsupported engine reward adapter {adapter!r} for deployment {spec.name!r}")

    def reward_kwargs(self) -> dict[str, str]:
        reward_kwargs = getattr(self._adapter, "reward_kwargs", None)
        if reward_kwargs is None:
            raise RuntimeError(f"Engine deployment {self.spec.name!r} does not accept a reward function")
        return reward_kwargs()

    async def score(self, prompt: str, image) -> dict[str, float]:
        score = getattr(self._adapter, "score", None)
        if score is None:
            raise RuntimeError(f"Engine deployment {self.spec.name!r} requires a reward function")
        return await score(prompt, image)


class NativeRewardExecutor:
    """Worker-local native-model owner with safe wake/sleep semantics."""

    def __init__(self, spec: RewardExecutorSpec):
        if spec.backend != "native":
            raise ValueError(f"NativeRewardExecutor requires a native spec, got {spec.backend!r}")
        self.spec = spec
        self._scorer: Any | None = None
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()

    async def wake_up(self) -> None:
        async with self._lock:
            await self._wake_up_locked()

    async def _wake_up_locked(self) -> None:
        """Load this deployment's scorer while the lifecycle lock is held."""
        if self._scorer is not None:
            return
        scorer_cls = _load_native_scorer(self.spec.executor_config["scorer"])
        kwargs = dict(self.spec.executor_config.get("kwargs", {}))
        if self.spec.model_path is not None:
            kwargs.setdefault("model_path", self.spec.model_path)
        # Ray normally rewrites the visible-device environment for an actor.
        # Its accelerator ID is therefore a physical allocation ID, whereas
        # torch needs the process-local current index (often ``0``). Use
        # verl's platform helper rather than feeding the physical ID to
        # ``torch.device``.
        kwargs.setdefault("device", torch.device(get_device_name(), get_device_id()))
        self._scorer = scorer_cls(**kwargs)

    async def score(self, prompt: str, image) -> dict:
        async with self._lock:
            self._inflight += 1
            self._idle.clear()
            try:
                await self._wake_up_locked()
            except BaseException:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()
                raise
        try:
            scorer = self._scorer
            pil_image = _to_pil_image(image)
            score_fn = getattr(scorer, "score", None)
            if score_fn is None:
                result = await asyncio.get_running_loop().run_in_executor(None, scorer, prompt, pil_image)
            elif inspect.iscoroutinefunction(score_fn):
                # A native scorer may implement its own batching policy.  For
                # example, PickScoreNativeScorer queues these one-sample calls
                # and performs one CLIP forward for the ready batch.
                result = await score_fn([prompt], [pil_image])
            else:
                # The upstream reward loop fans a batch out into concurrent
                # coroutines.  Do not serialize those calls here: a scorer
                # that needs batching or framework-specific synchronization
                # owns that policy itself.
                result = await asyncio.get_running_loop().run_in_executor(None, score_fn, [prompt], [pil_image])
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, torch.Tensor):
                result = result.tolist()
            if isinstance(result, (list, tuple)):
                score = float(result[0])
            else:
                score = float(result)
            return {"score": score, "pickscore_raw": score}
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def sleep(self) -> None:
        while True:
            await self._idle.wait()
            async with self._lock:
                # A new score can begin after ``wait`` returns. Holding the
                # same lock that increments ``_inflight`` makes this recheck
                # and scorer removal one atomic lifecycle transition.
                if self._inflight:
                    continue
                scorer, self._scorer = self._scorer, None
                if scorer is not None:
                    close = getattr(scorer, "close", None)
                    if close is not None:
                        if inspect.iscoroutinefunction(close):
                            result = await close()
                        else:
                            result = await asyncio.get_running_loop().run_in_executor(None, close)
                        if inspect.isawaitable(result):
                            await result
                gc.collect()
                _empty_accelerator_cache()
                return


def build_engine_reward_executors(specs: dict[str, RewardExecutorSpec]) -> dict[str, EngineRewardExecutor]:
    """Construct per-worker engine executors for named engine deployments."""
    return {name: EngineRewardExecutor(spec) for name, spec in specs.items() if is_engine_backend(spec.backend)}


def build_native_reward_executors(specs: dict[str, RewardExecutorSpec]) -> dict[str, NativeRewardExecutor]:
    """Construct one worker-local executor for each named native deployment."""
    return {name: NativeRewardExecutor(spec) for name, spec in specs.items() if spec.backend == "native"}
