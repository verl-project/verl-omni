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
"""CPU composition parity tests for the V1 OCR GenRM named-engine migration.

Maintained V1 OCR recipes register a named ``reward.models.<name>`` engine and
bind ``reward.reward_functions.<term>`` to it instead of using the deprecated
``reward.reward_model.enable=True`` + ``reward.custom_reward_function`` path.
This test composes every V1 OCR recipe without a GPU and checks its engine,
placement, offload lifecycle, judge output budget, and reward-term bindings.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from verl.utils.import_utils import load_extern_object

from verl_omni.reward_loop.reward_manager.visual import _sampling_params_from_rollout
from verl_omni.reward_loop.reward_model import _prepare_engine_config
from verl_omni.utils.reward_score.genrm_ocr import DEFAULT_GRM_PROMPT
from verl_omni.workers.config.reward import (
    EngineRewardModelConfig,
    parse_reward_model_config,
    resolve_reward_model_name,
    validate_reward_model_terms,
    warn_deprecated_reward_config,
)

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "verl_omni/trainer/config"
EXAMPLES_DIR = ROOT / "examples"

OCR_SCORER_PATH = "pkg://verl_omni.utils.reward_score.genrm_ocr"
MIXED_SCORER_PATH = "examples/diffusionopd_trainer/sd35/mixed_task_reward.py"

EXPECTED_MAX_TOKENS = 256
LEGACY_DEFAULT_GRM_PROMPT = (
    "Please output only the text content from the image without any additional descriptions or formatting."
)
CUSTOM_USER_PROMPT = "Read the text exactly."


@dataclass(frozen=True)
class TermCase:
    """One migrated reward term and the contract it must keep."""

    name: str
    path: str
    scorer_name: str
    model: str
    weight: float = 1.0
    user_prompt: str | None = None


@dataclass(frozen=True)
class RecipeCase:
    """One V1 OCR recipe and its expected named-engine reward contract."""

    case_id: str
    recipe: str
    model_name: str
    model_path: str
    offload: bool
    n_gpus_per_node: int
    nnodes: int
    tensor_parallel_size: int
    enable_resource_pool: bool
    parent_pool_world_size: int
    terms: tuple[TermCase, ...]
    gpu_memory_utilization: float
    enforce_eager: bool
    max_model_len: int | None = None
    trainer_mode: str = "sync"
    reward_num_workers: int = 1


def _ocr_term() -> TermCase:
    return TermCase("ocr", OCR_SCORER_PATH, "compute_score_ocr", "ocr")


def _colocated_qwen3_vl(case_id: str, recipe: str, *, enforce_eager: bool = True) -> RecipeCase:
    return RecipeCase(
        case_id=case_id,
        recipe=recipe,
        model_name="ocr",
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        offload=True,
        n_gpus_per_node=4,
        nnodes=1,
        tensor_parallel_size=4,
        enable_resource_pool=False,
        parent_pool_world_size=4,
        terms=(_ocr_term(),),
        gpu_memory_utilization=0.5,
        enforce_eager=enforce_eager,
    )


def _dedicated_sd35(
    case_id: str,
    recipe: str,
    *,
    trainer_mode: str = "sync",
    terms: tuple[TermCase, ...] | None = None,
) -> RecipeCase:
    return RecipeCase(
        case_id=case_id,
        recipe=recipe,
        model_name="ocr",
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        offload=False,
        n_gpus_per_node=1,
        nnodes=1,
        tensor_parallel_size=1,
        enable_resource_pool=True,
        parent_pool_world_size=1,
        terms=(_ocr_term(),) if terms is None else terms,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        trainer_mode=trainer_mode,
    )


RECIPES = (
    _colocated_qwen3_vl(
        "flowgrpo_qwen_image",
        "examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh",
    ),
    _colocated_qwen3_vl(
        "diffusionnft_qwen_image",
        "examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh",
    ),
    _colocated_qwen3_vl(
        "online_dpo_qwen_image",
        "examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora_v1.sh",
        enforce_eager=False,
    ),
    _colocated_qwen3_vl(
        "flowdppo_qwen_image",
        "examples/flowdppo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh",
    ),
    _colocated_qwen3_vl(
        "grpoguard_qwen_image",
        "examples/grpoguard_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh",
    ),
    _colocated_qwen3_vl(
        "flowgrpo_boogu_image",
        "examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora_v1.sh",
    ),
    _dedicated_sd35(
        "diffusionopd_sd35_ocr_distill",
        "examples/diffusionopd_trainer/sd35/run_sd35_medium_ocr_distill.sh",
    ),
    _dedicated_sd35(
        "flowgrpo_sd35_separate_async",
        "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh",
        trainer_mode="separate_async",
    ),
    _dedicated_sd35(
        "flowgrpo_sd35_delta",
        "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_v1_separate_async_delta.sh",
        trainer_mode="separate_async",
    ),
    _dedicated_sd35(
        "diffusionopd_sd35_mopd",
        "examples/diffusionopd_trainer/sd35/run_sd35_medium_mopd_distill.sh",
        terms=(TermCase("mixed", MIXED_SCORER_PATH, "compute_score_mixed", "ocr"),),
    ),
    _dedicated_sd35(
        "awm_sd35",
        "examples/awm_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh",
    ),
    _dedicated_sd35(
        "dgpo_sd35",
        "examples/dgpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh",
    ),
    RecipeCase(
        case_id="dgpo_sd35_shared_judge",
        recipe="examples/dgpo_trainer/sd35/run_sd35_medium_ocr_shared_judge.sh",
        model_name="judge",
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        offload=False,
        n_gpus_per_node=1,
        nnodes=1,
        tensor_parallel_size=1,
        enable_resource_pool=True,
        parent_pool_world_size=1,
        terms=(
            TermCase("ocr_default", OCR_SCORER_PATH, "compute_score_ocr", "judge", weight=0.5),
            TermCase(
                "ocr_custom",
                OCR_SCORER_PATH,
                "compute_score_ocr",
                "judge",
                weight=0.5,
                user_prompt=CUSTOM_USER_PROMPT,
            ),
        ),
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_model_len=8192,
    ),
    _dedicated_sd35(
        "flowgrpo_sd35",
        "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh",
    ),
)


def _compose_recipe(case: RecipeCase, tmp_path: Path):
    """Run a recipe with a ``python3`` stub and compose its Hydra config."""
    recipe = ROOT / case.recipe
    workspace = tmp_path / "workspace"
    env = {
        **os.environ,
        "OCR_WORKSPACE": str(workspace),
        "FA3": "0",
        # Keep Boogu's ``hf download`` branch out of the CPU test.
        "MODEL_PATH": "dummy/model",
        # OPD recipes require explicit teacher checkpoints at shell level.
        "TEACHER_PATH": str(tmp_path / "teacher"),
        "OCR_TEACHER_PATH": str(tmp_path / "ocr-teacher"),
        "AES_TEACHER_PATH": str(tmp_path / "aes-teacher"),
        "GENEVAL_TEACHER_PATH": str(tmp_path / "geneval-teacher"),
        "CUSTOM_USER_PROMPT": CUSTOM_USER_PROMPT,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python3() { printf "%s\\0" "$@"; }; export -f python3; bash "$@"',
            "bash",
            str(recipe),
            "trainer.logger=[console]",
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    tokens = result.stdout.decode().rstrip("\0").split("\0")
    starts = [
        index
        for index in range(len(tokens) - 1)
        if tokens[index] == "-m" and tokens[index + 1] == "verl_omni.trainer.main_diffusion_v1"
    ]
    assert starts, f"{case.recipe} did not invoke the V1 trainer"
    overrides = tokens[starts[-1] + 2 :]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


@pytest.mark.parametrize("case", RECIPES, ids=lambda case: case.case_id)
def test_ocr_v1_recipe_keeps_legacy_reward_contract(case: RecipeCase, tmp_path: Path):
    recipe_text = (ROOT / case.recipe).read_text()
    assert "reward.custom_reward_function" not in recipe_text
    assert "reward.reward_model.enable=True" not in recipe_text
    assert "reward.reward_model.model_path=" not in recipe_text
    assert "reward.reward_model.rollout." not in recipe_text
    assert "+reward.models." in recipe_text

    config = _compose_recipe(case, tmp_path)
    assert config.trainer.use_v1 is True
    assert config.trainer.v1.trainer_mode == case.trainer_mode

    base = config.reward.reward_model
    assert base.enable is False
    assert base.model_path is None
    assert base.enable_resource_pool == case.enable_resource_pool
    assert config.reward.reward_manager.name == "MultiVisualRewardManager"
    assert config.reward.aggregation == "weighted_sum"
    assert config.reward.custom_reward_function.path is None
    assert config.reward.num_workers == case.reward_num_workers

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warn_deprecated_reward_config(config)
    deprecation_warnings = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings == []

    validate_reward_model_terms(config)
    assert list(config.reward.models) == [case.model_name]
    model = parse_reward_model_config(case.model_name, config.reward.models[case.model_name])
    assert isinstance(model, EngineRewardModelConfig)
    assert model.backend == "engine"
    assert model.model_path == case.model_path
    assert model.resolved_offload == case.offload
    assert model.n_gpus_per_node == case.n_gpus_per_node
    assert model.nnodes == case.nnodes
    assert model.rollout["name"] == "vllm"
    assert model.rollout["response_length"] == EXPECTED_MAX_TOKENS
    assert model.rollout["tensor_model_parallel_size"] == case.tensor_parallel_size
    assert model.rollout.get("data_parallel_size", 1) == 1
    assert model.rollout.get("pipeline_model_parallel_size", 1) == 1
    assert "free_cache_engine" not in model.rollout
    assert "enable_sleep_mode" not in model.rollout

    # The parent pool and named engine request must preserve replica count.
    assert base.rollout.get("data_parallel_size", 1) == 1
    assert base.rollout.get("pipeline_model_parallel_size", 1) == 1
    requested = model.requested_resource_size(base)
    assert requested == case.parent_pool_world_size
    old_replicas = case.parent_pool_world_size // case.tensor_parallel_size
    new_replicas = requested // case.tensor_parallel_size
    assert new_replicas == old_replicas

    prepared = _prepare_engine_config(model, base)
    assert prepared.enable is True
    assert prepared.model_path == case.model_path
    assert prepared.rollout.name == "vllm"
    assert prepared.rollout.response_length == EXPECTED_MAX_TOKENS
    assert prepared.rollout.tensor_model_parallel_size == case.tensor_parallel_size
    assert prepared.rollout.gpu_memory_utilization == case.gpu_memory_utilization
    assert prepared.rollout.enforce_eager == case.enforce_eager
    assert prepared.rollout.max_model_len == case.max_model_len
    assert prepared.rollout.free_cache_engine is case.offload
    assert prepared.rollout.enable_sleep_mode is case.offload

    effective_rollout = ChainMap(model.rollout, base.rollout)
    assert _sampling_params_from_rollout(effective_rollout) == {"max_tokens": EXPECTED_MAX_TOKENS}

    assert set(config.reward.reward_functions) == {term.name for term in case.terms}
    total_weight = 0.0
    for term in case.terms:
        entry = config.reward.reward_functions[term.name]
        assert entry["path"] == term.path
        assert entry["name"] == term.scorer_name
        assert entry["weight"] == pytest.approx(term.weight)
        assert entry["required"] is True
        assert entry["use_rollout_sampling_params"] is True
        assert resolve_reward_model_name(term.name, entry, config.reward.models) == term.model
        if term.path == OCR_SCORER_PATH:
            assert callable(load_extern_object(term.path, term.scorer_name))
        else:
            assert (ROOT / term.path).is_file()
        if term.user_prompt is None:
            assert "user_prompt" not in entry
            assert "system_prompt" not in entry
        else:
            assert entry["user_prompt"] == term.user_prompt
        total_weight += term.weight
    assert total_weight == pytest.approx(1.0)
    assert DEFAULT_GRM_PROMPT == LEGACY_DEFAULT_GRM_PROMPT


def test_no_v1_ocr_recipe_uses_legacy_global_reward_model():
    legacy_recipes = []
    for path in sorted(EXAMPLES_DIR.rglob("*.sh")):
        text = path.read_text()
        is_v1 = "main_diffusion_v1" in text or "use_v1=true" in text or "use_v1=True" in text
        is_ocr = "compute_score_ocr" in text or "genrm_ocr" in text or "mixed_task_reward" in text
        if is_v1 and is_ocr and ("reward.reward_model.enable=True" in text or "reward.custom_reward_function." in text):
            legacy_recipes.append(str(path.relative_to(ROOT)))
    assert legacy_recipes == []
