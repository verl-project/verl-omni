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
"""Shared fixtures for the diffusion teacher runtime tests."""

import json

import pytest

from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config.diffusion import DiffusionModelConfig


@pytest.fixture
def fake_sd3_checkpoint(tmp_path):
    """Factory for a diffusers layout carrying just what is read off disk.

    ``DiffusionModelConfig.__post_init__`` reads ``model_index.json``, and the
    scheduler checks read ``scheduler/scheduler_config.json``; nothing else in a
    real checkpoint matters to the config or scheduler tiers.
    """

    def _make(name="teacher", class_name="StableDiffusion3Pipeline", **scheduler_kwargs):
        checkpoint = tmp_path / name
        (checkpoint / "scheduler").mkdir(parents=True)
        (checkpoint / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
        FlowMatchSDEDiscreteScheduler(**scheduler_kwargs).save_pretrained(checkpoint / "scheduler")
        return str(checkpoint)

    return _make


@pytest.fixture
def diffusion_model_config():
    """Factory for a DiffusionModelConfig over a fake checkpoint."""

    def _make(path, **overrides):
        kwargs = {
            "path": path,
            "algorithm": "flow_grpo",
            "attn_backend": "native",
            "load_tokenizer": False,
        }
        kwargs.update(overrides)
        return DiffusionModelConfig(**kwargs)

    return _make
