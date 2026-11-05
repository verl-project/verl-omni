# Copyright 2026 verl-omni contributors
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

"""Worker-side dispatch for registered training-step extensions."""

from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from verl_omni.workers.utils.step_extension_transport import SELECTOR_KEY


@dataclass(frozen=True)
class WorkerOperations:
    """Separate operations by the native worker RPC delivery mode."""

    stage: Mapping[str, Callable]
    control: Mapping[str, Callable]
    commit: Mapping[str, Callable]

    def __post_init__(self):
        for mode in ("stage", "control", "commit"):
            operations = dict(getattr(self, mode))
            if not operations or any(
                type(key) is not str or not key or not callable(fn) for key, fn in operations.items()
            ):
                raise ValueError(f"Invalid {mode} operations")
            object.__setattr__(self, mode, MappingProxyType(operations))


_WORKER_EXTENSIONS: dict[str, WorkerOperations] = {}
_STEP_EXTENSIONS: dict[str, tuple[Callable, Callable]] = {}
_STEP_EXTENSION_MODES: dict[str, frozenset[str]] = {}


def register_step_extension(name: str, enabled: Callable, factory: Callable, *, trainer_modes: frozenset[str]) -> None:
    """Register one opt-in training-step extension and its constructor."""
    if type(name) is not str or not name or not callable(enabled) or not callable(factory):
        raise ValueError("Invalid training-step extension registration")
    if name in _STEP_EXTENSIONS:
        raise ValueError(f"Training-step extension already registered: {name}")
    if (
        not isinstance(trainer_modes, frozenset)
        or not trainer_modes
        or any(type(mode) is not str or not mode for mode in trainer_modes)
    ):
        raise ValueError("Training-step extension needs supported trainer modes")
    _STEP_EXTENSIONS[name] = (enabled, factory)
    _STEP_EXTENSION_MODES[name] = trainer_modes


def _select_step_extension(config):
    """Resolve opt-ins without constructing a trainer or starting workers."""
    name = config.trainer.get("v1", {}).get("step_extension")
    if name is not None and (type(name) is not str or not name or name not in _STEP_EXTENSIONS):
        raise ValueError(f"Unknown training-step extension: {name!r}")
    selected = []
    for registered_name, (enabled, factory) in _STEP_EXTENSIONS.items():
        active = enabled(config)
        if type(active) is not bool:
            raise ValueError(f"Training-step extension selector {registered_name} must return a boolean")
        if active or name == registered_name:
            selected.append((registered_name, factory))
    if len(selected) > 1:
        raise ValueError(f"Conflicting training-step extensions: {[name for name, _ in selected]}")
    return selected[0] if selected else None


def validate_step_extension_mode(config, *, uses_v1: bool) -> None:
    """Reject an active extension before Ray starts if the trainer cannot run it."""
    selected = _select_step_extension(config)
    if selected is None:
        return
    name, _ = selected
    mode = config.trainer.get("v1", {}).get("trainer_mode")
    modes = _STEP_EXTENSION_MODES[name]
    if not uses_v1 or mode not in modes:
        raise ValueError(f"Training-step extension {name} requires v1 trainer mode in {sorted(modes)}, got {mode!r}")


def resolve_step_extension(trainer):
    """Construct the selected extension, or leave the native path untouched."""
    selected = _select_step_extension(trainer.config)
    return selected[1](trainer) if selected else None


def register_worker_extension(name: str, operations: WorkerOperations) -> None:
    """Register one extension's explicitly enumerated worker operations."""
    if type(name) is not str or not name or not isinstance(operations, WorkerOperations):
        raise ValueError("Invalid worker extension registration")
    if name in _WORKER_EXTENSIONS:
        raise ValueError(f"Worker extension already registered: {name}")
    _WORKER_EXTENSIONS[name] = operations


def invoke_worker_extension(name: str, mode: str, operation: str, worker, payload):
    """Call one operation without retrying an unknown worker outcome."""
    if type(name) is not str or type(mode) is not str or type(operation) is not str:
        raise ValueError("Invalid worker extension dispatch key")
    if mode not in ("stage", "control", "commit"):
        raise ValueError("Invalid worker extension dispatch mode")
    extension = _WORKER_EXTENSIONS.get(name)
    if extension is None:
        raise ValueError(f"Unknown worker extension: {name}")
    handler = getattr(extension, mode).get(operation)
    if handler is None:
        raise ValueError(f"Unknown {mode} operation: {operation}")
    return handler(worker, payload)


def with_worker_extension_selector(metadata, name: str, operation: str):
    """Return a distinct batch metadata carrier with a validated selector."""
    if type(name) is not str or not name or type(operation) is not str or not operation:
        raise ValueError("Invalid worker extension selector")
    if SELECTOR_KEY in metadata.extra_info or (metadata.fields is not None and SELECTOR_KEY in metadata.fields):
        raise ValueError("Caller-supplied worker extension selector is forbidden")
    result = copy(metadata)
    result.extra_info = {**metadata.extra_info, SELECTOR_KEY: (name, operation)}
    return result
