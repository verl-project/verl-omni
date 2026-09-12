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
"""Configuration contracts for named reward models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf
from verl.base_config import BaseConfig

__all__ = [
    "EngineRewardModelConfig",
    "NativeRewardModelConfig",
    "NativeRewardModelExecutorConfig",
    "RewardModelConfig",
    "RewardModelPlacementConfig",
    "RewardModelSpec",
    "accelerator_workers_enabled",
    "get_reward_model_entries",
    "has_engine_reward_models",
    "has_native_reward_models",
    "has_reward_models",
    "is_engine_backend",
    "parse_reward_model_placement",
    "parse_reward_model_config",
    "resolve_reward_model_name",
    "reward_is_enabled",
    "reward_pool_is_separate",
    "reward_role_required",
    "streaming_reward_enabled",
    "to_mapping",
    "validate_reward_model_terms",
]

_ENGINE_BACKENDS = {"engine"}
_NATIVE_BACKENDS = {"native"}


@dataclass
class RewardModelPlacementConfig(BaseConfig):
    """Placement-group bundle indices assigned to one native reward model."""

    devices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.devices, list) or not self.devices:
            raise ValueError("Native reward model placement.devices must be a non-empty list")
        if any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in self.devices):
            raise ValueError("Native reward model placement.devices must contain non-negative integers")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("Native reward model placement.devices must not contain duplicates")

    @classmethod
    def from_mapping(cls, name: str, value) -> RewardModelPlacementConfig:
        placement = to_mapping(value)
        _reject_unknown_fields(name, placement, {"devices"}, prefix="placement.")
        if "devices" not in placement:
            raise ValueError(f"Native reward model {name!r} requires placement.devices as native-pool bundle indices")
        try:
            return cls(devices=placement["devices"])
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Native reward model {name!r} {exc}") from exc


@dataclass
class NativeRewardModelExecutorConfig(BaseConfig):
    """Import contract for one worker-local native reward model."""

    model: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or ":" not in self.model or not all(self.model.rsplit(":", 1)):
            raise ValueError("Native reward model requires executor.model")
        if not isinstance(self.kwargs, dict):
            raise TypeError("Native reward model executor.kwargs must be a mapping")

    @classmethod
    def from_mapping(cls, name: str, value) -> NativeRewardModelExecutorConfig:
        executor = to_mapping(value)
        _reject_unknown_fields(name, executor, {"model", "kwargs"}, prefix="executor.")
        return cls(
            model=executor.get("model", ""),
            kwargs=to_mapping(executor.get("kwargs")),
        )


@dataclass
class RewardModelConfig(BaseConfig):
    """Fields shared by every named reward model."""

    name: str = ""
    backend: str = ""
    offload: bool | None = None
    model_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Reward model name must be a non-empty string")
        if self.offload is not None and not isinstance(self.offload, bool):
            raise ValueError("Reward model offload must be a boolean")
        if self.model_path is not None and (not isinstance(self.model_path, str) or not self.model_path):
            raise ValueError("Reward model model_path must be a non-empty string or null")

    @property
    def resolved_offload(self) -> bool:
        return self.offload if self.offload is not None else True


@dataclass
class EngineRewardModelConfig(RewardModelConfig):
    """Complete user-facing schema for one engine-backed reward model."""

    replicas: int = 1
    n_gpus_per_node: int | None = None
    nnodes: int | None = None
    rollout: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend != "engine":
            raise ValueError(f"Engine reward model {self.name!r} requires backend='engine'")
        _validate_positive_int(self.replicas, f"Engine reward model {self.name!r} replicas")
        if (self.n_gpus_per_node is None) != (self.nnodes is None):
            raise ValueError(f"Engine reward model {self.name!r} must set both n_gpus_per_node and nnodes")
        if self.n_gpus_per_node is not None:
            _validate_positive_int(
                self.n_gpus_per_node,
                f"Engine reward model {self.name!r} n_gpus_per_node",
            )
            _validate_positive_int(self.nnodes, f"Engine reward model {self.name!r} nnodes")
        if not isinstance(self.rollout, dict):
            raise TypeError(f"Engine reward model {self.name!r} rollout must be a mapping")
        # These existing rollout switches are accepted as aliases for the
        # backend-neutral offload field, but conflicting values fail early.
        alias_values = [self.rollout[key] for key in ("free_cache_engine", "enable_sleep_mode") if key in self.rollout]
        values = ([self.offload] if self.offload is not None else []) + alias_values
        if any(not isinstance(value, bool) for value in values):
            raise ValueError("Reward model offload must be a boolean")
        if len(set(values)) > 1:
            raise ValueError("Reward model offload conflicts with rollout sleep settings")

    @classmethod
    def from_mapping(cls, name: str, value) -> EngineRewardModelConfig:
        model = to_mapping(value)
        allowed = {
            "backend",
            "offload",
            "model_path",
            "replicas",
            "n_gpus_per_node",
            "nnodes",
            "rollout",
        }
        _reject_unknown_fields(name, model, allowed)
        return cls(
            name=name,
            backend=model.get("backend", ""),
            offload=model.get("offload"),
            model_path=model.get("model_path"),
            replicas=model.get("replicas", 1),
            n_gpus_per_node=model.get("n_gpus_per_node"),
            nnodes=model.get("nnodes"),
            rollout=to_mapping(model.get("rollout")),
        )

    @property
    def resolved_offload(self) -> bool:
        alias_values = [self.rollout[key] for key in ("free_cache_engine", "enable_sleep_mode") if key in self.rollout]
        values = ([self.offload] if self.offload is not None else []) + alias_values
        return values[0] if values else True

    def rollout_world_size(self, base_config) -> int:
        base_rollout = to_mapping(base_config.get("rollout"))
        merged = {**base_rollout, **self.rollout}
        parallel_sizes = []
        for field_name in (
            "tensor_model_parallel_size",
            "data_parallel_size",
            "pipeline_model_parallel_size",
        ):
            value = merged.get(field_name, 1)
            _validate_positive_int(value, f"Engine reward model {self.name!r} rollout.{field_name}")
            parallel_sizes.append(value)
        return self.replicas * parallel_sizes[0] * parallel_sizes[1] * parallel_sizes[2]

    def requested_resource_size(self, base_config) -> int:
        world_size = self.rollout_world_size(base_config)
        requested = self.n_gpus_per_node * self.nnodes if self.n_gpus_per_node is not None else world_size
        if requested < world_size or requested % world_size:
            raise ValueError(
                f"Engine reward model {self.name!r} allocation ({requested}) must be a multiple of "
                f"rollout world size ({world_size})"
            )
        return requested

    def to_engine_overrides(self) -> dict[str, Any]:
        values = {
            "model_path": self.model_path,
            "n_gpus_per_node": self.n_gpus_per_node,
            "nnodes": self.nnodes,
            "rollout": self.rollout,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass
class NativeRewardModelConfig(RewardModelConfig):
    """Complete user-facing schema for one worker-local native reward model."""

    placement: RewardModelPlacementConfig | None = None
    executor: NativeRewardModelExecutorConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend != "native":
            raise ValueError(f"Native reward model {self.name!r} requires backend='native'")
        if not isinstance(self.placement, RewardModelPlacementConfig):
            raise ValueError(f"Native reward model {self.name!r} requires placement.devices")
        if not isinstance(self.executor, NativeRewardModelExecutorConfig):
            raise ValueError(f"Native reward model {self.name!r} requires executor.model")

    @classmethod
    def from_mapping(cls, name: str, value) -> NativeRewardModelConfig:
        model = to_mapping(value)
        allowed = {"backend", "offload", "model_path", "placement", "executor"}
        _reject_unknown_fields(name, model, allowed)
        return cls(
            name=name,
            backend=model.get("backend", ""),
            offload=model.get("offload"),
            model_path=model.get("model_path"),
            placement=RewardModelPlacementConfig.from_mapping(name, model.get("placement")),
            executor=NativeRewardModelExecutorConfig.from_mapping(name, model.get("executor")),
        )


@dataclass
class RewardModelSpec(BaseConfig):
    """Static model metadata copied to reward-loop workers."""

    name: str = ""
    backend: str = ""
    model_path: str | None = None
    router_address: str | None = None
    executor_config: dict[str, Any] = field(default_factory=dict)


def to_mapping(value) -> dict[str, Any]:
    """Convert an OmegaConf mapping to a plain dictionary without resolving values."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if OmegaConf.is_config(value):
        result = OmegaConf.to_container(value, resolve=False)
        if isinstance(result, dict):
            return result
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a mapping, got {type(value).__name__}")


def get_reward_model_entries(config):
    """Return configured named reward models."""
    return config.reward.get("models", {}) or {}


def has_reward_models(config) -> bool:
    """Return whether at least one named reward model is configured."""
    return bool(get_reward_model_entries(config))


def is_engine_backend(backend: str | None) -> bool:
    """Return whether a backend name selects the managed engine path."""
    return backend in _ENGINE_BACKENDS


def has_engine_reward_models(config) -> bool:
    """Return whether any configured named model uses the engine backend."""
    return any(is_engine_backend(model.get("backend")) for model in get_reward_model_entries(config).values())


def has_native_reward_models(config) -> bool:
    """Return whether any configured named model uses the native backend."""
    return any(model.get("backend") in _NATIVE_BACKENDS for model in get_reward_model_entries(config).values())


def resolve_reward_model_name(term_name: str, term, models) -> str | None:
    """Resolve an explicit model reference or the same-name shorthand."""
    model_name = term.get("model")
    if model_name is None and term_name in models:
        model_name = term_name
    return model_name


def validate_reward_model_terms(config) -> None:
    """Validate named-model references before allocating model resources."""
    models = get_reward_model_entries(config)
    for term_name, term in (config.reward.get("reward_functions", {}) or {}).items():
        model_name = resolve_reward_model_name(term_name, term, models)
        if model_name is None:
            continue
        if model_name not in models:
            raise ValueError(f"Reward term {term_name!r} references unknown model {model_name!r}")
        has_function = term.get("path") is not None and term.get("name") is not None
        if not has_function:
            raise ValueError(
                f"Reward model {model_name!r} needs path/name in reward term {term_name!r} "
                "to turn model output into a score"
            )


def reward_is_enabled(config) -> bool:
    """Return whether either the legacy or named-model reward path is enabled."""
    reward_model = config.reward.get("reward_model", {})
    return bool(reward_model.get("enable", False) or has_reward_models(config))


def reward_role_required(config) -> bool:
    """Whether the reward loop needs the trainer-selected parent resource pool."""
    return bool(config.reward.reward_model.get("enable", False) or has_reward_models(config))


def reward_pool_is_separate(config) -> bool:
    """Return whether reward models use the dedicated parent resource pool."""
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def streaming_reward_enabled(config) -> bool:
    """Whether the current reward path can run inside streaming rollout workers."""
    if not reward_is_enabled(config):
        return True
    if has_engine_reward_models(config) or has_native_reward_models(config):
        return False
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def accelerator_workers_enabled(config) -> bool:
    """Whether existing custom reward workers use accelerator placement."""
    reward = config.reward
    accelerator_workers = reward.get("accelerator_workers", {}) or {}
    custom_reward = reward.get("custom_reward_function", {}) or {}
    return bool(accelerator_workers.get("enabled", False) or custom_reward.get("use_accelerator", False))


def parse_reward_model_placement(name: str, value) -> RewardModelPlacementConfig:
    """Build and validate one native model placement schema."""
    return RewardModelPlacementConfig.from_mapping(name, value)


def parse_reward_model_config(name: str, value) -> EngineRewardModelConfig | NativeRewardModelConfig:
    """Parse and validate one named reward model before resource allocation."""
    model = to_mapping(value)
    backend = model.get("backend")
    if backend == "engine":
        return EngineRewardModelConfig.from_mapping(name, model)
    if backend == "native":
        return NativeRewardModelConfig.from_mapping(name, model)
    raise ValueError(f"Reward model {name!r} has unsupported backend {backend!r}; expected one of engine, native")


def _reject_unknown_fields(
    name: str,
    value: dict[str, Any],
    allowed: set[str],
    *,
    prefix: str = "",
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        fields = ", ".join(f"{prefix}{field}" for field in unknown)
        raise ValueError(f"Reward model {name!r} has unsupported fields: {fields}")


def _validate_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
