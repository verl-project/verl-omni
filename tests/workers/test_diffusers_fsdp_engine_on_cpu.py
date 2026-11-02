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
"""CPU-only wiring tests for Diffusers FSDP engines."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
from verl.workers.engine.base import EngineRegistry

from verl_omni.workers.engine.fsdp import diffusers_impl


@pytest.mark.parametrize(
    "model_type,expected_class_name",
    [
        ("diffusion_model", "PPODiffusersFSDPEngine"),
        ("diffusion_dpo_model", "DPODiffusersFSDPEngine"),
        ("diffusion_nft_model", "NFTDiffusersFSDPEngine"),
    ],
)
def test_engine_registered_via_decorator(model_type, expected_class_name):
    """``@EngineRegistry.register`` decorator metadata via AST inspection."""
    tree = ast.parse(Path(diffusers_impl.__file__).read_text())
    class_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == expected_class_name:
            class_def = node
            break

    assert class_def is not None, f"{expected_class_name} class not found"

    decorator_found = False
    for decorator in class_def.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "EngineRegistry"
            and func.attr == "register"
        ):
            continue
        kwargs_found = 0
        for kw in decorator.keywords:
            if kw.arg == "model_type" and isinstance(kw.value, ast.Constant):
                if kw.value.value == model_type:
                    kwargs_found += 1
            if kw.arg == "backend" and isinstance(kw.value, ast.List):
                vals = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                if set(vals) == {"fsdp", "fsdp2"}:
                    kwargs_found += 1
            if kw.arg == "device" and isinstance(kw.value, ast.List):
                vals = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                if set(vals) == {"cuda", "npu", "xpu", "mlu"}:
                    kwargs_found += 1
        assert kwargs_found == 3, (
            f"@EngineRegistry.register missing expected kwargs (model_type/backend/device); found {kwargs_found}/3"
        )
        decorator_found = True
        break

    assert decorator_found, f"@EngineRegistry.register decorator not found on {expected_class_name}"


@pytest.mark.parametrize(
    "model_type,expected_class_name",
    [
        ("diffusion_model", "PPODiffusersFSDPEngine"),
        ("diffusion_dpo_model", "DPODiffusersFSDPEngine"),
        ("diffusion_nft_model", "NFTDiffusersFSDPEngine"),
    ],
)
def test_engine_registered_for_fsdp_and_fsdp2_backends(model_type, expected_class_name):
    """``EngineRegistry`` has each diffusion model registered for fsdp and fsdp2."""
    engines = EngineRegistry._engines
    assert model_type in engines, f"'{model_type}' not in EngineRegistry._engines. Keys: {list(engines)}"
    diffusion_registry = engines[model_type]

    for backend in ("fsdp", "fsdp2"):
        assert backend in diffusion_registry, (
            f"'{backend}' not registered for {model_type}; registered: {list(diffusion_registry)}"
        )
        for device in ("cuda", "npu", "xpu", "mlu"):
            entry = diffusion_registry[backend].get(device)
            assert entry is not None, f"No engine for {model_type}/{backend}/{device}"
            assert entry.__name__ == expected_class_name, (
                f"Engine for {model_type}/{backend}/{device} is {entry.__name__}, expected {expected_class_name}"
            )


@pytest.mark.parametrize(
    "model_type,expected_class_name",
    [
        ("diffusion_model", "PPODiffusersFSDPEngine"),
        ("diffusion_dpo_model", "DPODiffusersFSDPEngine"),
        ("diffusion_nft_model", "NFTDiffusersFSDPEngine"),
    ],
)
@pytest.mark.parametrize("device", ["cuda", "npu", "xpu", "mlu"])
def test_engine_get_engine_cls_resolves_to_diffusers_fsdp_engine(model_type, expected_class_name, device):
    """``EngineRegistry.get_engine_cls`` returns the expected Diffusers FSDP engine."""
    with patch("verl.workers.engine.base.get_device_name", return_value=device):
        for backend in ("fsdp", "fsdp2"):
            cls = EngineRegistry.get_engine_cls(model_type, backend)
            assert cls.__name__ == expected_class_name, (
                f"get_engine_cls('{model_type}', '{backend}') returned {cls.__name__}, expected {expected_class_name}"
            )
