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
"""CPU tests for MixGRPO pipeline registration and SDE window positioning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import verl_omni.pipelines  # noqa: F401 — trigger pipeline registration
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_mix_grpo.diffusers_training_adapter import QwenImageMixGRPO
from verl_omni.pipelines.qwen_image_mix_grpo.vllm_omni_rollout_adapter import (
    QwenImageMixGRPOPipelineWithLogProb,
)
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


def _make_model_config(algorithm: str = "mix_grpo") -> DiffusionModelConfig:
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "QwenImagePipeline")
    object.__setattr__(cfg, "algorithm", algorithm)
    object.__setattr__(cfg, "external_lib", None)
    return cfg


def _make_request(extra_args: dict) -> SimpleNamespace:
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


class TestMixGRPORegistry:
    def test_vllm_omni_rollout_pipeline_registered(self):
        pipeline_cls = VllmOmniPipelineBase.get_class("QwenImagePipeline", "mix_grpo")
        assert pipeline_cls is QwenImageMixGRPOPipelineWithLogProb

    def test_training_adapter_registered(self):
        model_config = _make_model_config()
        assert DiffusionModelBase.get_class(model_config) is QwenImageMixGRPO


class TestMixGRPOWindowPositioning:
    def test_random_without_seed_is_noop(self):
        extra = {
            "sample_strategy": "random",
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
            "global_steps": 3,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
        assert extra["sde_window_range"] == [0, 5]

    def test_random_with_seed_pins_window(self):
        extra = {
            "sample_strategy": "random",
            "sde_window_seed": 42,
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
            "global_steps": 7,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})

        start, end = extra["sde_window_range"]
        assert end - start == 2
        assert 0 <= start <= 3

        # Same seed + global_steps must be deterministic across calls.
        extra_copy = dict(extra)
        req_copy = _make_request(extra_copy)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req_copy, {})
        assert extra_copy["sde_window_range"] == extra["sde_window_range"]

    def test_random_seed_changes_with_global_steps(self):
        def pinned_start(global_steps: int) -> int:
            extra = {
                "sample_strategy": "random",
                "sde_window_seed": 99,
                "sde_window_size": 2,
                "sde_window_range": [0, 10],
                "global_steps": global_steps,
            }
            req = _make_request(extra)
            QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
            return extra["sde_window_range"][0]

        assert pinned_start(0) != pinned_start(1)

    def test_progressive_slides_window(self):
        extra = {
            "sample_strategy": "progressive",
            "sde_window_size": 2,
            "sde_window_range": [0, 8],
            "iters_per_group": 2,
            "global_steps": 4,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
        assert extra["sde_window_range"] == [4, 6]

    def test_progressive_clamps_at_envelope_end(self):
        extra = {
            "sample_strategy": "progressive",
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
            "iters_per_group": 1,
            "global_steps": 100,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
        assert extra["sde_window_range"] == [3, 5]

    def test_progressive_without_window_size_is_noop(self):
        extra = {
            "sample_strategy": "progressive",
            "sde_window_range": [0, 5],
            "global_steps": 10,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
        assert extra["sde_window_range"] == [0, 5]

    def test_unknown_strategy_is_noop(self):
        extra = {
            "sample_strategy": "fixed",
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
            "global_steps": 3,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
        assert extra["sde_window_range"] == [0, 5]

    def test_window_size_from_kwargs(self):
        extra = {
            "sample_strategy": "progressive",
            "sde_window_range": [0, 6],
            "iters_per_group": 1,
            "global_steps": 2,
        }
        req = _make_request(extra)
        QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(
            req, {"sde_window_size": 2, "sde_window_range": [0, 6]}
        )
        assert extra["sde_window_range"] == [4, 6]

    @pytest.mark.parametrize("strategy", ["random", "progressive"])
    def test_invalid_window_raises(self, strategy: str):
        extra = {
            "sample_strategy": strategy,
            "sde_window_size": 6,
            "sde_window_range": [0, 5],
            "global_steps": 0,
        }
        if strategy == "random":
            extra["sde_window_seed"] = 1
        req = _make_request(extra)
        with pytest.raises(ValueError, match="does not fit"):
            QwenImageMixGRPOPipelineWithLogProb._maybe_make_progressive_window(req, {})
