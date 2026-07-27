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

from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(architecture: str, external_lib=None, algorithm: str = "flow_grpo") -> DiffusionModelConfig:
    """Build a minimal DiffusionModelConfig without hitting __post_init__ model loading."""
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", architecture)
    object.__setattr__(cfg, "external_lib", external_lib)
    object.__setattr__(cfg, "algorithm", algorithm)
    return cfg


# ---------------------------------------------------------------------------
# DiffusionModelBase registry
# ---------------------------------------------------------------------------


class TestDiffusionModelBaseRegistry:
    def test_register_and_retrieve(self):
        @DiffusionModelBase.register("_TestArch_CPU", algorithm="flow_grpo")
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
        @DiffusionModelBase.register("_TestReturnArch_CPU", algorithm="flow_grpo")
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


class TestVllmOmniPipelineBaseRegistry:
    def test_builtin_qwen_rollout_algorithms_registered(self):
        pytest.importorskip("vllm_omni")
        from verl_omni.pipelines.qwen_image_diffusion_nft.vllm_omni_rollout_adapter import (
            QwenImageDiffusionNFTPipeline,
        )
        from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

        assert VllmOmniPipelineBase.get_class("QwenImagePipeline", "flow_grpo") is QwenImagePipelineWithLogProb
        assert VllmOmniPipelineBase.get_class("QwenImagePipeline", "diffusion_nft") is QwenImageDiffusionNFTPipeline

    def test_diffusion_nft_rollout_does_not_override_sde_trajectory_loop(self):
        pytest.importorskip("vllm_omni")
        from verl_omni.pipelines.qwen_image_diffusion_nft.vllm_omni_rollout_adapter import (
            QwenImageDiffusionNFTPipeline,
        )
        from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

        assert "diffuse" in QwenImagePipelineWithLogProb.__dict__
        assert "diffuse" not in QwenImageDiffusionNFTPipeline.__dict__

    def test_sd3_latent_postprocess_is_defined_by_sd3_adapter(self, monkeypatch):
        pytest.importorskip("vllm_omni")
        import torch

        from verl_omni.pipelines.sd3_flow_grpo import vllm_omni_rollout_adapter

        image_outputs = []

        def image_postprocess_factory(od_config):
            def image_postprocess(output):
                image_outputs.append(output)
                return "decoded"

            return image_postprocess

        monkeypatch.setattr(
            vllm_omni_rollout_adapter,
            "_SD3_IMAGE_POST_PROCESS_FUNC",
            image_postprocess_factory,
        )

        assert (
            vllm_omni_rollout_adapter.pipeline_sd3.get_sd3_image_post_process_func
            is vllm_omni_rollout_adapter.get_latent_post_process_func
        )

        postprocess = vllm_omni_rollout_adapter.get_latent_post_process_func(od_config=None)
        latent = torch.zeros(1, 16, 2, 2)
        image = torch.zeros(1, 3, 2, 2)
        assert postprocess(latent) is latent
        assert postprocess(image) == "decoded"
        assert len(image_outputs) == 1
        assert image_outputs[0] is image
