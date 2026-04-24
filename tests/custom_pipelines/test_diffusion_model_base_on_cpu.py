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
"""CPU tests for DiffusionModelBase registration and dispatch."""

import pytest

from verl_omni.custom_pipelines.base import DiffusionModelBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(architecture: str, external_lib=None) -> DiffusionModelConfig:
    """Build a minimal DiffusionModelConfig without hitting __post_init__ model loading."""
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", architecture)
    object.__setattr__(cfg, "external_lib", external_lib)
    return cfg


# ---------------------------------------------------------------------------
# DiffusionModelBase registry
# ---------------------------------------------------------------------------


class TestDiffusionModelBaseRegistry:
    def test_register_and_retrieve(self):
        @DiffusionModelBase.register("_TestArch_CPU")
        class _Impl(DiffusionModelBase):
            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, module, model_config, *args, **kwargs):
                pass

        cfg = _make_model_config("_TestArch_CPU")
        assert DiffusionModelBase.get_class(cfg) is _Impl

    def test_get_class_unknown_architecture_raises(self):
        cfg = _make_model_config("__DoesNotExist__")
        with pytest.raises(NotImplementedError, match="No diffusion model registered"):
            DiffusionModelBase.get_class(cfg)

    def test_register_decorator_returns_class_unchanged(self):
        @DiffusionModelBase.register("_TestReturnArch_CPU")
        class _Impl(DiffusionModelBase):
            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, module, model_config, *args, **kwargs):
                pass

        # Decorator must return the original class
        assert _Impl.__name__ == "_Impl"
        assert issubclass(_Impl, DiffusionModelBase)
