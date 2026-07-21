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
"""CPU tests for omni prepare_model_inputs dispatch."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_pipeline_modules():
    sys.modules.setdefault("verl_omni", types.ModuleType("verl_omni"))
    sys.modules.setdefault("verl_omni.pipelines", types.ModuleType("verl_omni.pipelines"))
    sys.modules.setdefault("verl_omni.pipelines.qwen3_omni", types.ModuleType("verl_omni.pipelines.qwen3_omni"))
    workers_config = types.ModuleType("verl_omni.workers.config")
    workers_config.DiffusionModelConfig = object
    sys.modules["verl_omni.workers.config"] = workers_config

    _load_module(
        "verl_omni.pipelines.model_base",
        REPO_ROOT / "verl_omni" / "pipelines" / "model_base.py",
    )
    _load_module(
        "verl_omni.pipelines.qwen3_omni.thinker_training_adapter",
        REPO_ROOT / "verl_omni" / "pipelines" / "qwen3_omni" / "thinker_training_adapter.py",
    )
    return _load_module(
        "verl_omni.pipelines.utils",
        REPO_ROOT / "verl_omni" / "pipelines" / "utils.py",
    )


@dataclass
class _FakeModelConfig:
    architecture: str = "Qwen3OmniMoeForConditionalGeneration"
    model_stage: str = "thinker"
    external_lib: str | None = None


@pytest.fixture(scope="module")
def prepare_omni_model_inputs():
    return _bootstrap_pipeline_modules().prepare_omni_model_inputs


class TestPrepareOmniModelInputs:
    def test_extracts_standard_text_keys(self, prepare_omni_model_inputs):
        micro_batch = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "labels": torch.tensor([[-100, 2, 3]]),
                "position_ids": torch.zeros(1, 3, 3),
            },
            batch_size=[1],
        )
        model_inputs = prepare_omni_model_inputs(_FakeModelConfig(), micro_batch)
        assert set(model_inputs) == {"input_ids", "attention_mask", "labels", "position_ids"}

    def test_squeezes_mrope_position_ids(self, prepare_omni_model_inputs):
        micro_batch = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "labels": torch.tensor([[-100, 2]]),
                "position_ids": torch.zeros(1, 3, 1, 2),
            },
            batch_size=[1],
        )
        model_inputs = prepare_omni_model_inputs(_FakeModelConfig(), micro_batch)
        assert model_inputs["position_ids"].shape == (1, 3, 2)

    def test_drops_zero_multimodal_rows_and_casts_dtype(self, prepare_omni_model_inputs):
        micro_batch = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]]),
                "labels": torch.tensor([[-100, 2], [-100, 4]]),
                "position_ids": torch.zeros(2, 3, 2),
                "pixel_values": torch.tensor(
                    [
                        [[1.0, 2.0]],
                        [[0.0, 0.0]],
                    ]
                ),
                "image_grid_thw": torch.tensor(
                    [
                        [[1, 2, 2]],
                        [[0, 0, 0]],
                    ]
                ),
            },
            batch_size=[2],
        )
        model_inputs = prepare_omni_model_inputs(_FakeModelConfig(), micro_batch, dtype=torch.bfloat16)
        assert model_inputs["pixel_values"].shape[0] == 1
        assert model_inputs["image_grid_thw"].shape[0] == 1
        assert model_inputs["pixel_values"].dtype == torch.bfloat16

    def test_preserves_all_micro_batch_keys(self, prepare_omni_model_inputs):
        micro_batch = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "image_mask": torch.tensor([[False, True]]),
                "reference_chosen_logps": torch.tensor([0.1]),
            },
            batch_size=[1],
        )
        model_inputs = prepare_omni_model_inputs(_FakeModelConfig(), micro_batch)
        assert set(model_inputs) == set(micro_batch.keys())
