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
"""CPU regression checks for the V1 OCR GenRM named-engine migration.

Maintained V1 OCR recipes and test harnesses register a named
``reward.models.<name>`` engine and bind ``reward.reward_functions.<term>`` to
it instead of using the deprecated ``reward.reward_model.enable=True`` +
``reward.custom_reward_function`` path. This suite composes every V1 OCR shell
entry without a GPU and checks its engine, placement, offload lifecycle, judge
output budget, scorer loadability, and reward-term bindings.
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
    reward_is_enabled,
    streaming_reward_enabled,
    validate_reward_model_terms,
    warn_deprecated_reward_config,
)

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "verl_omni/trainer/config"

OCR_SCORER_PATH = "pkg://verl_omni.utils.reward_score.genrm_ocr"
JPEG_SCORER_PATH = "pkg://verl_omni.utils.reward_score.jpeg_compressibility"
MIXED_SCORER_PATH = "examples/diffusionopd_trainer/sd35/mixed_task_reward.py"

EXPECTED_MAX_TOKENS = 256
HARNESS_MAX_TOKENS = 32
LEGACY_DEFAULT_GRM_PROMPT = (
    "Please output only the text content from the image without any additional descriptions or formatting."
)
CUSTOM_USER_PROMPT = "Read the text exactly."
TINY_REWARD_MODEL_PATH = "/tmp/tiny-random-qwen3-vl"


@dataclass(frozen=True)
class TermCase:
    """One migrated reward term and the contract it must keep."""

    name: str
    path: str
    scorer_name: str
    model: str | None
    weight: float = 1.0
    required: bool = True
    use_rollout_sampling_params: bool = True
    user_prompt: str | None = None


@dataclass(frozen=True)
class RecipeCase:
    """One V1 OCR script and its expected named-engine reward contract."""

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
    response_length: int = EXPECTED_MAX_TOKENS
    max_model_len: int | None = None
    prompt_length: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    full_determinism: bool = False
    seed: int | None = None
    trainer_mode: str = "sync"
    reward_num_workers: int = 1


def _ocr_term(*, weight: float = 1.0) -> TermCase:
    return TermCase("ocr", OCR_SCORER_PATH, "compute_score_ocr", "ocr", weight=weight)


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


def _nightly_qwen_image(case_id: str, recipe: str) -> RecipeCase:
    return RecipeCase(
        case_id=case_id,
        recipe=recipe,
        model_name="ocr",
        model_path=TINY_REWARD_MODEL_PATH,
        offload=True,
        n_gpus_per_node=1,
        nnodes=1,
        tensor_parallel_size=1,
        enable_resource_pool=False,
        parent_pool_world_size=1,
        terms=(_ocr_term(),),
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        response_length=HARNESS_MAX_TOKENS,
        prompt_length=256,
        do_sample=False,
        temperature=0.0,
        full_determinism=True,
        seed=42,
    )


def _special_e2e_qwen_image(
    case_id: str,
    recipe: str,
    *,
    n_gpus_per_node: int,
    terms: tuple[TermCase, ...],
) -> RecipeCase:
    return RecipeCase(
        case_id=case_id,
        recipe=recipe,
        model_name="ocr",
        model_path=TINY_REWARD_MODEL_PATH,
        offload=True,
        n_gpus_per_node=n_gpus_per_node,
        nnodes=1,
        tensor_parallel_size=1,
        enable_resource_pool=False,
        parent_pool_world_size=n_gpus_per_node,
        terms=terms,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        response_length=HARNESS_MAX_TOKENS,
        prompt_length=256,
    )


FLOWGRPO_SD35_CASE = _dedicated_sd35(
    "flowgrpo_sd35",
    "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh",
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
    FLOWGRPO_SD35_CASE,
    _nightly_qwen_image(
        "nightly_flowgrpo_qwen_image",
        "tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh",
    ),
    _nightly_qwen_image(
        "nightly_diffusionnft_qwen_image",
        "tests/nightly/qwen_image_diffusion_nft/run_qwen_image_diffusion_nft.sh",
    ),
    _nightly_qwen_image(
        "nightly_online_dpo_qwen_image",
        "tests/nightly/qwen_image_online_dpo/run_qwen_image_online_dpo.sh",
    ),
    _special_e2e_qwen_image(
        "special_e2e_flowgrpo_qwen_image",
        "tests/special_e2e/run_flowgrpo_qwen_image.sh",
        n_gpus_per_node=4,
        terms=(
            TermCase(
                "jpeg",
                JPEG_SCORER_PATH,
                "compute_score",
                None,
                weight=0.5,
                required=False,
                use_rollout_sampling_params=False,
            ),
            _ocr_term(weight=0.5),
        ),
    ),
    _special_e2e_qwen_image(
        "special_e2e_online_dpo_qwen_image",
        "tests/special_e2e/run_online_dpo_qwen_image.sh",
        n_gpus_per_node=2,
        terms=(_ocr_term(),),
    ),
)

_PYTHON_CAPTURE_SHIM = (
    "python3() { "
    'if [[ "${1:-}" == "-m" && "${2:-}" == "verl_omni.trainer.main_diffusion_v1" ]]; then '
    "shift 2; "
    'elif [[ "${1:-}" == */tests/nightly/run.py ]]; then '
    "shift 1; "
    "else return 0; fi; "
    'if [[ -n "${HYDRA_CAPTURE_DIR:-}" ]]; then '
    'idx=$(find "${HYDRA_CAPTURE_DIR}" -maxdepth 1 -name "*.overrides" | wc -l | tr -d " "); '
    'printf "%s\\0" "$@" > "${HYDRA_CAPTURE_DIR}/${idx}.overrides"; '
    "else "
    'printf "%s\\0" "$@" > "${HYDRA_CAPTURE_FILE}"; '
    "fi; "
    "}; "
    "nvidia-smi() { return 0; }; "
    "export -f python3 nvidia-smi; "
    'bash "$@"'
)


def _run_capture_env(
    case: RecipeCase,
    tmp_path: Path,
    *,
    workspace: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved_workspace = workspace if workspace is not None else tmp_path / "workspace"
    env = {
        **{k: v for k, v in os.environ.items() if k != "REWARD_MODEL_PATH"},
        "OCR_WORKSPACE": str(resolved_workspace),
        "FA3": "0",
        "ATTN_BACKEND": "native",
        "ROLLOUT_ATTN_BACKEND": "TORCH_SDPA",
        # Keep Boogu's ``hf download`` branch out of the CPU test.
        "MODEL_PATH": "dummy/model",
        "DATA_DIR": str(tmp_path / "data"),
        "OUTPUT_ROOT": str(tmp_path / "debug_dumps"),
        # Capture only the primary OCR run by default for nightly harnesses.
        "L3_REFRESH_BASELINE": "1",
        # OPD recipes require explicit teacher checkpoints at shell level.
        "TEACHER_PATH": str(tmp_path / "teacher"),
        "OCR_TEACHER_PATH": str(tmp_path / "ocr-teacher"),
        "AES_TEACHER_PATH": str(tmp_path / "aes-teacher"),
        "GENEVAL_TEACHER_PATH": str(tmp_path / "geneval-teacher"),
        "CUSTOM_USER_PROMPT": CUSTOM_USER_PROMPT,
    }
    if case.model_path == TINY_REWARD_MODEL_PATH:
        env["REWARD_MODEL_PATH"] = TINY_REWARD_MODEL_PATH
    if extra_env:
        env.update(extra_env)
    return env


def _compose_recipe(
    case: RecipeCase,
    tmp_path: Path,
    *,
    workspace: Path | None = None,
    extra_overrides: tuple[str, ...] = ("trainer.logger=[console]",),
    extra_env: dict[str, str] | None = None,
):
    """Run a recipe or harness with a ``python3`` stub and compose its Hydra config."""
    recipe = ROOT / case.recipe
    capture_file = tmp_path / f"{case.case_id}.hydra_overrides"
    env = _run_capture_env(
        case,
        tmp_path,
        workspace=workspace,
        extra_env={"HYDRA_CAPTURE_FILE": str(capture_file), **(extra_env or {})},
    )
    subprocess.run(
        ["bash", "-c", _PYTHON_CAPTURE_SHIM, "bash", str(recipe), *extra_overrides],
        env=env,
        check=True,
        capture_output=True,
    )
    assert capture_file.is_file(), f"{case.recipe} did not invoke the V1 trainer"
    raw = capture_file.read_bytes()
    overrides = [token.decode() for token in raw.rstrip(b"\0").split(b"\0") if token]
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
    assert model.rollout["response_length"] == case.response_length
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
    assert prepared.rollout.response_length == case.response_length
    assert prepared.rollout.tensor_model_parallel_size == case.tensor_parallel_size
    assert prepared.rollout.gpu_memory_utilization == case.gpu_memory_utilization
    assert prepared.rollout.enforce_eager == case.enforce_eager
    assert prepared.rollout.max_model_len == case.max_model_len
    if case.prompt_length is not None:
        assert prepared.rollout.prompt_length == case.prompt_length
    if case.do_sample is not None:
        assert prepared.rollout.do_sample is case.do_sample
    if case.temperature is not None:
        assert prepared.rollout.temperature == pytest.approx(case.temperature)
    assert prepared.rollout.full_determinism is case.full_determinism
    if case.seed is not None:
        assert prepared.rollout.seed == case.seed
    assert prepared.rollout.free_cache_engine is case.offload
    assert prepared.rollout.enable_sleep_mode is case.offload

    effective_rollout = ChainMap(model.rollout, base.rollout)
    expected_sampling_params: dict[str, object] = {"max_tokens": case.response_length}
    if case.temperature is not None and case.temperature != 1.0:
        expected_sampling_params["temperature"] = case.temperature
        expected_sampling_params["top_p"] = 1.0
    if case.full_determinism:
        expected_sampling_params["seed"] = 42 if case.seed is None else case.seed
    assert _sampling_params_from_rollout(effective_rollout) == expected_sampling_params

    assert set(config.reward.reward_functions) == {term.name for term in case.terms}
    total_weight = 0.0
    for term in case.terms:
        entry = config.reward.reward_functions[term.name]
        assert entry["path"] == term.path
        assert entry["name"] == term.scorer_name
        assert entry["weight"] == pytest.approx(term.weight)
        assert bool(entry.get("required", False)) is term.required
        assert bool(entry.get("use_rollout_sampling_params", False)) is term.use_rollout_sampling_params
        assert resolve_reward_model_name(term.name, entry, config.reward.models) == term.model
        scorer_path = term.path if term.path.startswith("pkg://") else str(ROOT / term.path)
        assert callable(load_extern_object(scorer_path, term.scorer_name))
        if term.user_prompt is None:
            assert "user_prompt" not in entry
            assert "system_prompt" not in entry
        else:
            assert entry["user_prompt"] == term.user_prompt
        total_weight += term.weight
    assert total_weight == pytest.approx(1.0)
    assert DEFAULT_GRM_PROMPT == LEGACY_DEFAULT_GRM_PROMPT


@pytest.mark.parametrize("executor_override", [None, "mp"])
def test_sd35_v1_ocr_recipe_handles_spaces_in_workspace_and_executor_override(
    tmp_path: Path,
    executor_override: str | None,
):
    workspace = tmp_path / "workspace with spaces"
    overrides = ["trainer.logger=[console]"]
    if executor_override is not None:
        overrides.append(
            f"++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.distributed_executor_backend={executor_override}"
        )

    config = _compose_recipe(
        FLOWGRPO_SD35_CASE,
        tmp_path,
        workspace=workspace,
        extra_overrides=tuple(overrides),
    )

    validate_reward_model_terms(config)
    model = parse_reward_model_config("ocr", config.reward.models.ocr)
    assert model.backend == "engine"
    assert model.model_path == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert model.resolved_offload is False
    assert model.rollout["tensor_model_parallel_size"] == 1
    assert model.rollout["response_length"] == EXPECTED_MAX_TOKENS
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


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in RECIPES
        if case.case_id in {"special_e2e_flowgrpo_qwen_image", "special_e2e_online_dpo_qwen_image"}
    ],
    ids=lambda case: case.case_id,
)
def test_special_e2e_ocr_scripts_inherit_npu_weight_nz_guard(case: RecipeCase, tmp_path: Path):
    """Ensure lib_npu_e2e.sh's base reward rollout override still reaches named engines."""
    config = _compose_recipe(
        case,
        tmp_path,
        extra_overrides=(
            "trainer.logger=[console]",
            "++reward.reward_model.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0",
        ),
    )
    model = parse_reward_model_config(case.model_name, config.reward.models[case.model_name])
    prepared = _prepare_engine_config(model, config.reward.reward_model)
    assert prepared.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode == 0


def test_nightly_flowgrpo_trajectory_ab_uses_rollout_rule_reward(tmp_path: Path):
    """Ensure #933's trajectory A/B sub-runs drop the OCR engine and stay rollout-side."""
    case = next(c for c in RECIPES if c.case_id == "nightly_flowgrpo_qwen_image")
    capture_dir = tmp_path / "ab_captures"
    capture_dir.mkdir()
    env = _run_capture_env(
        case,
        tmp_path,
        extra_env={
            "HYDRA_CAPTURE_DIR": str(capture_dir),
            "L3_TRAJECTORY_AB_ONLY": "1",
            "L3_REFRESH_BASELINE": "0",
        },
    )
    subprocess.run(
        ["bash", "-c", _PYTHON_CAPTURE_SHIM, "bash", str(ROOT / case.recipe)],
        env=env,
        check=True,
        capture_output=True,
    )
    capture_files = sorted(capture_dir.glob("*.overrides"), key=lambda path: int(path.stem))
    assert len(capture_files) == 2

    configs = []
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        for capture_file in capture_files:
            raw = capture_file.read_bytes()
            overrides = [token.decode() for token in raw.rstrip(b"\0").split(b"\0") if token]
            configs.append(compose(config_name="diffusion_trainer", overrides=overrides))

    assert [cfg.trainer.v1.worker_side_trajectory for cfg in configs] == [False, True]
    for cfg in configs:
        assert cfg.reward.reward_model.enable is False
        assert dict(cfg.reward.models) == {}
        assert reward_is_enabled(cfg) is False
        assert streaming_reward_enabled(cfg) is True
        assert cfg.reward.reward_manager.name == "MultiVisualRewardManager"
        assert list(cfg.reward.reward_functions) == ["jpeg"]
        assert cfg.reward.reward_functions.jpeg.path == JPEG_SCORER_PATH
        assert cfg.reward.reward_functions.jpeg.name == "compute_score"
        assert cfg.reward.reward_functions.jpeg.weight == pytest.approx(1.0)
        assert cfg.reward.reward_functions.jpeg.required is True
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            warn_deprecated_reward_config(cfg)
        assert [w for w in recorded if issubclass(w.category, DeprecationWarning)] == []


def test_no_v1_ocr_recipe_uses_legacy_global_reward_model():
    legacy_recipes = []
    for path in sorted(ROOT.rglob("*.sh")):
        rel_parts = path.relative_to(ROOT).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        text = path.read_text()
        is_v1 = "main_diffusion_v1" in text or "use_v1=true" in text or "use_v1=True" in text
        is_ocr = "compute_score_ocr" in text or "genrm_ocr" in text or "mixed_task_reward" in text
        if is_v1 and is_ocr and ("reward.reward_model.enable=True" in text or "reward.custom_reward_function." in text):
            legacy_recipes.append(str(path.relative_to(ROOT)))
    assert legacy_recipes == []
