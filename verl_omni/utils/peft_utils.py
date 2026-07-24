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

"""Helpers for controlling PEFT adapters on wrapped or manually injected models."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_NO_ADAPTER_LOADED = "No adapter loaded"


def _is_no_adapter_loaded(exc: ValueError) -> bool:
    return _NO_ADAPTER_LOADED in str(exc)


def _iter_child_modules(model: Any) -> Iterator[Any]:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return iter(())
    return (module for module in modules() if module is not model)


def set_peft_adapter(model: Any, adapter_name: str = "default") -> int:
    """Activate a PEFT adapter, falling back to manually injected child modules.

    Returns the number of objects whose ``set_adapter`` hook was called.
    """

    set_adapter = getattr(model, "set_adapter", None)
    if callable(set_adapter):
        try:
            set_adapter(adapter_name)
            return 1
        except ValueError as exc:
            if not _is_no_adapter_loaded(exc):
                raise

    activated = 0
    for module in _iter_child_modules(model):
        module_set_adapter = getattr(module, "set_adapter", None)
        if not callable(module_set_adapter):
            continue
        try:
            module_set_adapter(adapter_name)
        except ValueError as exc:
            if not _is_no_adapter_loaded(exc):
                raise
            continue
        activated += 1
    return activated


def _set_child_adapters_enabled(model: Any, enabled: bool) -> int:
    updated = 0
    for module in _iter_child_modules(model):
        enable_adapters = getattr(module, "enable_adapters", None)
        if not callable(enable_adapters):
            continue
        enable_adapters(enabled)
        updated += 1
    return updated


def disable_peft_adapters(model: Any) -> int:
    """Disable PEFT adapters on manually injected child modules."""

    return _set_child_adapters_enabled(model, False)


def enable_peft_adapters(model: Any) -> int:
    """Enable PEFT adapters on manually injected child modules."""

    return _set_child_adapters_enabled(model, True)


@contextmanager
def peft_adapters_disabled(model: Any):
    """Temporarily disable PEFT adapters, supporting top-level and injected LoRA."""

    disable_adapters = getattr(model, "disable_adapters", None)
    enable_adapters = getattr(model, "enable_adapters", None)
    if callable(disable_adapters) and callable(enable_adapters):
        try:
            disable_adapters()
        except ValueError as exc:
            if not _is_no_adapter_loaded(exc):
                raise
        else:
            try:
                yield
            finally:
                enable_adapters()
            return

    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        try:
            maybe_context = disable_adapter()
        except ValueError as exc:
            if not _is_no_adapter_loaded(exc):
                raise
        else:
            if hasattr(maybe_context, "__enter__") and hasattr(maybe_context, "__exit__"):
                with maybe_context:
                    yield
                return

    disabled = disable_peft_adapters(model)
    if disabled == 0:
        raise RuntimeError("The model does not expose PEFT adapter hooks to disable.")
    try:
        yield
    finally:
        enable_peft_adapters(model)
