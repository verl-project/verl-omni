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
"""Pytest options for rollout GPU tests."""

from __future__ import annotations

import os

import pytest

from .train_rollout_consistency_helpers import QwenImageConsistencyProfile, resolve_qwen_image_profile


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--qwen-image-model",
        action="store",
        default=os.environ.get("QWEN_IMAGE_MODEL", "tiny"),
        choices=("tiny", "full"),
        help=(
            "Qwen-Image variant for train-rollout consistency tests. "
            "Default: tiny. Override with env QWEN_IMAGE_MODEL or QWEN_IMAGE_MODEL_PATH (full only)."
        ),
    )
    parser.addoption(
        "--qwen-image-recipe",
        action="store",
        default=os.environ.get("QWEN_IMAGE_RECIPE", "default"),
        choices=("default", "ocr"),
        help=(
            "FlowGRPO recipe for train-rollout consistency tests. "
            "'ocr' matches run_qwen_image_ocr_lora.sh SDE/LoRA settings and uses hybrid weight sync."
        ),
    )
    parser.addoption(
        "--use-fa3",
        action="store_true",
        default=os.environ.get("TRAIN_ROLLOUT_USE_FA3", "0") == "1",
        help=(
            "Use FA3 on both sides: actor attn_backend=_flash_3_varlen_hub and "
            "rollout DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN (fa3-fwd)."
        ),
    )
    parser.addoption(
        "--weight-sync-steps",
        action="store",
        default=os.environ.get("WEIGHT_SYNC_STEPS", "1"),
        type=int,
        help=(
            "Number of LoRA perturb + hybrid weight-sync cycles in the weight-sync consistency test. "
            "Default: 1 (baseline check + one post-sync check)."
        ),
    )


@pytest.fixture(scope="module")
def qwen_image_consistency_profile(request: pytest.FixtureRequest) -> QwenImageConsistencyProfile:
    variant = request.config.getoption("--qwen-image-model")
    recipe = request.config.getoption("--qwen-image-recipe")
    profile = resolve_qwen_image_profile(variant, recipe=recipe)
    if request.config.getoption("--use-fa3"):
        return QwenImageConsistencyProfile(**{**profile.__dict__, "attn_backend": "_flash_3_varlen_hub"})
    return profile


@pytest.fixture(scope="module")
def ray_runtime(request: pytest.FixtureRequest):
    import ray

    env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
    }
    if request.config.getoption("--use-fa3"):
        env_vars["DIFFUSION_ATTENTION_BACKEND"] = "FLASH_ATTN"

    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": env_vars},
            ignore_reinit_error=True,
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
