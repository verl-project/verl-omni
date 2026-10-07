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
"""Upstream factory hooks preserve named media without changing ordinary inference."""

import inspect
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from verl_omni.pipelines.rollout_postprocessing import install_rollout_postprocessor


def test_factory_installation_is_idempotent_and_scoped_to_selected_module():
    native_processor = Mock(side_effect=lambda pixels: pixels + 1)
    factory = Mock(return_value=native_processor)
    module = ModuleType("upstream_pipeline")
    module.get_postprocessor = factory
    other = ModuleType("other_pipeline")
    other.get_postprocessor = factory

    install_rollout_postprocessor(module, "get_postprocessor")
    installed = module.get_postprocessor
    install_rollout_postprocessor(module, "get_postprocessor")
    assert module.get_postprocessor is installed
    assert other.get_postprocessor is factory
    processor = installed(od_config="config")
    factory.assert_called_once_with("config")

    pixels = torch.zeros(1, 3, 2, 4)
    torch.testing.assert_close(processor(pixels), pixels + 1)
    native_processor.assert_called_once_with(pixels)
    named = {"payload": {"image": {"image_preview": pixels}}, "metadata": {"media_artifacts": {}}}
    assert processor(named) is named
    assert native_processor.call_count == 1


def test_factory_wrapper_preserves_request_sampling_signature_and_training_metadata():
    seen = []

    def postprocess(pixels, sampling_params=None):
        seen.append(sampling_params)
        return pixels + 1

    module = ModuleType("video_pipeline")
    module.get_postprocessor = lambda config: postprocess
    install_rollout_postprocessor(module, "get_postprocessor")
    processor = module.get_postprocessor(None)
    assert inspect.signature(processor) == inspect.signature(postprocess)
    result = processor(
        {"payload": {"video": torch.zeros(1)}, "metadata": {"rl": {"sentinel": 7}}},
        sampling_params="sampling",
    )
    assert seen == ["sampling"]
    assert result["metadata"]["rl"] == {"sentinel": 7}
    torch.testing.assert_close(result["payload"]["video"], torch.ones(1))


def test_unknown_factory_is_not_silently_ignored():
    with pytest.raises(AttributeError, match="missing_factory"):
        install_rollout_postprocessor(ModuleType("upstream_pipeline"), "missing_factory")
