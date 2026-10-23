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
"""CPU tests for Qwen-Image Dual-GRPO composite AR module setup."""

from types import SimpleNamespace

import torch

from verl_omni.pipelines.model_base import configure_composite_ar_module
from verl_omni.pipelines.qwen_image_dual_grpo.ar_module_adapter import qwen2_vl_base_forward


def _module_with_nested_visual() -> torch.nn.Module:
    module = torch.nn.Module()
    module.device = torch.device("cpu")
    module.model = torch.nn.Module()
    module.model.visual = torch.nn.Linear(4, 4)
    return module


def _assert_dummy_visual(visual, device: torch.device) -> None:
    assert callable(visual)
    assert not isinstance(visual, torch.nn.Module)
    out = visual(torch.zeros(2, 3), grid_thw=torch.zeros(1, 3))
    torch.testing.assert_close(out, torch.tensor([0.0]))
    assert out.device == device


def _dual_grpo_config():
    return SimpleNamespace(architecture="QwenImagePipeline", algorithm="dual_grpo", external_lib=None)


def test_configure_ar_module_strips_nested_visual_and_patches_forward():
    module = _module_with_nested_visual()
    configure_composite_ar_module(module, _dual_grpo_config())
    assert module.model.forward.__func__ is qwen2_vl_base_forward
    _assert_dummy_visual(module.model.visual, module.device)


def test_configure_ar_module_strips_top_level_visual():
    module = torch.nn.Module()
    module.device = torch.device("cpu")
    module.visual = torch.nn.Linear(4, 4)
    module.model = torch.nn.Module()
    configure_composite_ar_module(module, _dual_grpo_config())
    _assert_dummy_visual(module.visual, module.device)
    assert isinstance(module.model, torch.nn.Module)


def test_configure_ar_module_warns_when_visual_is_missing(caplog):
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    configure_composite_ar_module(module, _dual_grpo_config())
    assert module.model.forward.__func__ is qwen2_vl_base_forward
    assert "vision tower not found" in caplog.text


def test_unregistered_pipelines_leave_the_ar_module_alone():
    for architecture, algorithm in (
        ("QwenImagePipeline", "flow_grpo"),
        ("QwenImageEditPlusPipeline", "flow_grpo"),
    ):
        module = _module_with_nested_visual()
        visual = module.model.visual
        configure_composite_ar_module(
            module,
            SimpleNamespace(architecture=architecture, algorithm=algorithm, external_lib=None),
        )
        assert module.model.visual is visual
        assert getattr(module.model.forward, "__func__", None) is not qwen2_vl_base_forward
