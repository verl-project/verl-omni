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

"""Static and deterministic checks for the SD3.5 V1 OCR named-model migration."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl_omni.workers.config.reward import parse_reward_model_config, validate_reward_model_terms

RECIPE = Path(__file__).parents[2] / "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh"


@pytest.mark.parametrize("executor_override", [None, "mp"])
def test_sd35_v1_ocr_recipe_composes_named_engine_model(tmp_path, executor_override):
    workspace = tmp_path / "workspace with spaces"
    overrides = ["trainer.logger=[console]"]
    if executor_override is not None:
        overrides.append(
            f"++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.distributed_executor_backend={executor_override}"
        )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python3() { printf "%s\\0" "$@"; }; export -f python3; bash "$@"',
            "bash",
            str(RECIPE),
            *overrides,
        ],
        env={**os.environ, "OCR_WORKSPACE": str(workspace), "FA3": "0"},
        check=True,
        capture_output=True,
    )
    args = result.stdout.decode().rstrip("\0").split("\0")
    assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]
    config_dir = RECIPE.parents[3] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=args[2:])

    validate_reward_model_terms(config)
    model = parse_reward_model_config("ocr", config.reward.models.ocr)
    assert model.backend == "engine"
    assert model.model_path == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert model.resolved_offload is False
    assert model.rollout["tensor_model_parallel_size"] == 1
    assert config.reward.reward_model.enable is False
    assert config.reward.reward_manager.name == "MultiVisualRewardManager"
    assert config.reward.reward_functions.ocr.name == "compute_score_ocr"
    assert config.reward.reward_functions.ocr.required is True
    assert config.reward.reward_functions.ocr.use_rollout_sampling_params is True
    assert config.trainer.use_v1 is True
    assert config.trainer.v1.trainer_mode == "sync"
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    if executor_override is not None:
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.distributed_executor_backend == executor_override
        )
    assert list(config.trainer.logger) == ["console"]
    assert config.data.train_files == str(workspace / "data/ocr/sd3/train.parquet")
