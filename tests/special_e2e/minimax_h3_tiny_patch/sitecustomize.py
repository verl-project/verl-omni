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
"""Process-local vLLM-Omni compatibility patch for the tiny H3 E2E model.

Python imports ``sitecustomize`` during interpreter startup. The smoke runner
adds this directory to ``PYTHONPATH`` for its trainer and inherited Ray worker
processes only, so the installed vLLM-Omni package is never modified. The H3
encoder already constructs its attention projections from Qwen3-VL config;
its final width assertion and the conditioning contract must use that tiny width.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_CONFIG_ENV = "VERL_OMNI_MINIMAX_H3_TINY_TEXT_CONFIG"
_TARGET_MODULES = {
    "vllm_omni.diffusion.models.minimax_h3.encoder": "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "vllm_omni.model_executor.models.minimax_h3.conditioning": "MINIMAX_H3_TEXT_HIDDEN_SIZE",
}


def _configured_hidden_size() -> int | None:
    config_path = os.environ.get(_CONFIG_ENV)
    if not config_path:
        return None
    config = json.loads(Path(config_path).read_text())
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"{config_path} does not contain a Qwen3-VL text_config mapping")
    hidden_size = int(text_config["hidden_size"])
    if hidden_size < 1:
        raise ValueError(f"Qwen3-VL hidden_size must be positive, got {hidden_size}")
    return hidden_size


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: Any, hidden_size: int) -> None:
        self._wrapped = wrapped
        self._hidden_size = hidden_size

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_module(module, self._hidden_size)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hidden_size: int) -> None:
        self._hidden_size = hidden_size

    def find_spec(self, fullname: str, path=None, target=None):
        del target
        if fullname not in _TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot locate {fullname} for the tiny E2E patch")
        spec.loader = _PatchLoader(spec.loader, self._hidden_size)
        return spec


def _patch_module(module: ModuleType, hidden_size: int) -> None:
    constant = _TARGET_MODULES[module.__name__]
    if not hasattr(module, constant):
        raise RuntimeError(f"{module.__name__} no longer exposes {constant}; update the tiny E2E patch")
    setattr(module, constant, hidden_size)


def _install() -> None:
    hidden_size = _configured_hidden_size()
    if hidden_size is None:
        return
    for name in _TARGET_MODULES:
        loaded = sys.modules.get(name)
        if loaded is not None:
            _patch_module(loaded, hidden_size)
    sys.meta_path.insert(0, _PatchFinder(hidden_size))


_install()
