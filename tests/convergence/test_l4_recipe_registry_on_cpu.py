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
"""L4 recipe registry semantics.

Everything here runs on CPU with synthetic fixtures: no checkpoint, dataset, or
GPU is touched.  The fixtures are deliberately small dictionaries rather than
real YAML files so that each rejection rule is tested in isolation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from tests.convergence.recipe_registry import (
    RecipeError,
    check_preconditions,
    load_recipe,
    load_registry,
    validate_recipe_dict,
)
from tests.convergence.run_convergence import (
    RunnerError,
    render_overrides,
    render_placeholder,
    resolved_paths,
    safelisted_env,
)

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"


def valid_recipe() -> dict:
    """Return a minimal, fully valid L4 recipe document (synthetic)."""
    return {
        "schema_version": 1,
        "layer": "L4",
        "case_id": "synthetic_case",
        "title": "synthetic L4 case",
        "algorithm": "flow_grpo",
        "precision": "bf16",
        "recipe": {
            "launcher": "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh",
            "overrides": ["data.train_files={dataset:train}", "data.seed=42"],
        },
        "models": {
            "policy": {
                "kind": "real",
                "source": "stabilityai/stable-diffusion-3.5-medium",
                "revision": "main",
                "local_path": "/models/sd35",
                "env_path": "L4_SYNTHETIC_MODEL",
            }
        },
        "dataset": {
            "kind": "real",
            "train": "train.parquet",
            "val": "test.parquet",
            "env_root": "L4_SYNTHETIC_DATA",
        },
        "hardware": {
            "min_gpus": 1,
            "min_gpu_memory_gb": 1,
            "gpu_architectures": ["sm89"],
        },
        "budget": {"timeout_minutes": 5, "total_training_steps": 4},
        "metrics": {
            "warmup_steps": 1,
            "final_window": 1,
            "min_points": 3,
            "tracked": [{"name": "critic/rewards/mean", "direction": "higher"}],
        },
        "convergence": {"atol": 0.0, "rtol": 0.1, "min_improvement": -1.0, "require_finite": True},
        "baseline": {"artifact_name": "l4-convergence-baseline", "branch": "main"},
    }


def write_registry(tmp_path: Path, documents: list[dict], names: list[str] | None = None) -> Path:
    """Write synthetic recipe documents into a temporary registry directory."""
    registry_dir = tmp_path / "recipes"
    registry_dir.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(documents):
        stem = names[index] if names else f"case_{index}"
        (registry_dir / f"{stem}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return registry_dir


def test_valid_recipe_has_no_issues() -> None:
    assert validate_recipe_dict(valid_recipe()) == []


def test_shipped_recipes_are_valid_and_unique() -> None:
    registry = load_registry(RECIPES_DIR)
    assert set(registry) == {"sd35_medium_flowgrpo", "qwen_image_flowgrpo", "qwen3_omni_thinker_gspo"}
    assert all(recipe.release_gate for recipe in registry.values())
    # The launchers must be real example scripts shipped by the repository.
    repo_root = RECIPES_DIR.parents[2]
    for recipe in registry.values():
        assert (repo_root / recipe.launcher).is_file(), recipe.launcher


@pytest.mark.parametrize("missing_key", ["case_id", "layer", "algorithm", "precision", "metrics", "baseline"])
def test_missing_required_top_level_key_is_rejected(missing_key: str) -> None:
    document = valid_recipe()
    document.pop(missing_key)
    issues = validate_recipe_dict(document)
    assert any(issue.path == missing_key and "required" in issue.message for issue in issues)


def test_synthetic_weights_are_rejected() -> None:
    document = valid_recipe()
    document["models"]["policy"]["source"] = "Qwen/Qwen-Image-tiny-random"
    issues = validate_recipe_dict(document)
    assert any("synthetic" in issue.message for issue in issues)


def test_non_real_model_kind_is_rejected() -> None:
    document = valid_recipe()
    document["models"]["policy"]["kind"] = "tiny"
    issues = validate_recipe_dict(document)
    assert any("must be 'real'" in issue.message for issue in issues)


def test_non_real_dataset_is_rejected() -> None:
    document = valid_recipe()
    document["dataset"]["kind"] = "synthetic"
    issues = validate_recipe_dict(document)
    assert any(issue.path == "dataset.kind" for issue in issues)


def test_unknown_direction_is_rejected() -> None:
    document = valid_recipe()
    document["metrics"]["tracked"][0]["direction"] = "sideways"
    issues = validate_recipe_dict(document)
    assert any(issue.path.endswith(".direction") for issue in issues)


def test_duplicate_tracked_metric_is_rejected() -> None:
    document = valid_recipe()
    document["metrics"]["tracked"].append({"name": "critic/rewards/mean", "direction": "higher"})
    issues = validate_recipe_dict(document)
    assert any("duplicate tracked metric" in issue.message for issue in issues)


def test_all_optional_metrics_are_rejected() -> None:
    document = valid_recipe()
    document["metrics"]["tracked"][0]["optional"] = True
    issues = validate_recipe_dict(document)
    assert any("at least one metric must be required" in issue.message for issue in issues)


def test_min_points_smaller_than_scoring_window_is_rejected() -> None:
    document = valid_recipe()
    document["metrics"]["warmup_steps"] = 5
    document["metrics"]["final_window"] = 5
    document["metrics"]["min_points"] = 6
    issues = validate_recipe_dict(document)
    assert any("must be >= warmup_steps + final_window" in issue.message for issue in issues)


def test_unknown_precision_is_rejected() -> None:
    document = valid_recipe()
    document["precision"] = "int8"
    issues = validate_recipe_dict(document)
    assert any(issue.path == "precision" for issue in issues)


def test_unsupported_schema_version_is_rejected() -> None:
    document = valid_recipe()
    document["schema_version"] = 99
    issues = validate_recipe_dict(document)
    assert any(issue.path == "schema_version" for issue in issues)


def test_duplicate_case_id_across_files_is_rejected(tmp_path: Path) -> None:
    document = valid_recipe()
    registry_dir = write_registry(tmp_path, [document, copy.deepcopy(document)], names=["a", "b"])
    with pytest.raises(RecipeError) as error:
        load_registry(registry_dir)
    assert "duplicate case_id" in str(error.value)


def test_registry_without_gated_recipe_is_rejected(tmp_path: Path) -> None:
    document = valid_recipe()
    document["release_gate"] = False
    registry_dir = write_registry(tmp_path, [document])
    with pytest.raises(RecipeError) as error:
        load_registry(registry_dir)
    assert "no release_gate recipe" in str(error.value)


def test_empty_registry_directory_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RecipeError):
        load_registry(empty)


def test_missing_recipe_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RecipeError):
        load_recipe(tmp_path / "nope.yaml")


def test_recipe_sha256_is_content_addressed(tmp_path: Path) -> None:
    document = valid_recipe()
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    first = load_recipe(path)
    second = load_recipe(path)
    assert first.sha256 == second.sha256
    document["budget"]["total_training_steps"] = 5
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert load_recipe(path).sha256 != first.sha256


# --------------------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------------------


def _fake_gpu(index: int, *, total: int, free: int, arch: str = "sm89") -> dict:
    return {
        "index": str(index),
        "name": "NVIDIA L20",
        "memory_total_mib": total,
        "memory_free_mib": free,
        "compute_cap": f"{arch[2]}.{arch[3]}",
        "architecture": arch,
    }


def test_preconditions_report_missing_assets_precisely(tmp_path: Path) -> None:
    recipe = load_recipe(Path(_write_single(tmp_path, valid_recipe())))
    report = check_preconditions(
        recipe,
        repo_root=tmp_path,
        env={},
        gpus=[_fake_gpu(0, total=1024, free=1024)],
    )
    assert report.status == "skipped"
    reason = report.reason
    assert "launcher" in reason
    assert "model:policy" in reason
    assert "dataset:train" in reason
    assert "dataset:val" in reason


def test_preconditions_ready_when_assets_and_gpu_exist(tmp_path: Path) -> None:
    document = valid_recipe()
    document["recipe"]["launcher"] = "launcher.sh"
    (tmp_path / "launcher.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.parquet").write_bytes(b"x")
    (data_dir / "test.parquet").write_bytes(b"x")
    recipe = load_recipe(Path(_write_single(tmp_path, document)))
    env = {"L4_SYNTHETIC_MODEL": str(model_dir), "L4_SYNTHETIC_DATA": str(data_dir)}
    report = check_preconditions(
        recipe,
        repo_root=tmp_path,
        env=env,
        gpus=[_fake_gpu(4, total=46068, free=46068)],
    )
    assert report.status == "ready", report.reason
    assert report.reason == ""


def test_preconditions_skip_when_gpu_memory_is_too_small(tmp_path: Path) -> None:
    document = valid_recipe()
    document["hardware"]["min_gpu_memory_gb"] = 80
    document["recipe"]["launcher"] = "launcher.sh"
    (tmp_path / "launcher.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.parquet").write_bytes(b"x")
    (data_dir / "test.parquet").write_bytes(b"x")
    recipe = load_recipe(Path(_write_single(tmp_path, document)))
    env = {"L4_SYNTHETIC_MODEL": str(model_dir), "L4_SYNTHETIC_DATA": str(data_dir)}
    report = check_preconditions(
        recipe,
        repo_root=tmp_path,
        env=env,
        gpus=[_fake_gpu(0, total=46068, free=46068)],
    )
    assert report.status == "skipped"
    assert "gpu" in report.reason


def test_preconditions_require_gpus_to_be_idle(tmp_path: Path) -> None:
    """A present-but-busy GPU must not be treated as available."""
    document = valid_recipe()
    document["recipe"]["launcher"] = "launcher.sh"
    (tmp_path / "launcher.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.parquet").write_bytes(b"x")
    (data_dir / "test.parquet").write_bytes(b"x")
    recipe = load_recipe(Path(_write_single(tmp_path, document)))
    env = {"L4_SYNTHETIC_MODEL": str(model_dir), "L4_SYNTHETIC_DATA": str(data_dir)}
    report = check_preconditions(
        recipe,
        repo_root=tmp_path,
        env=env,
        gpus=[_fake_gpu(0, total=46068, free=100)],
    )
    assert report.status == "skipped"
    assert "idle=0" in report.reason


def test_preconditions_skip_without_any_gpu(tmp_path: Path) -> None:
    document = valid_recipe()
    document["recipe"]["launcher"] = "launcher.sh"
    (tmp_path / "launcher.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.parquet").write_bytes(b"x")
    (data_dir / "test.parquet").write_bytes(b"x")
    recipe = load_recipe(Path(_write_single(tmp_path, document)))
    env = {"L4_SYNTHETIC_MODEL": str(model_dir), "L4_SYNTHETIC_DATA": str(data_dir)}
    report = check_preconditions(recipe, repo_root=tmp_path, env=env, gpus=[])
    assert report.status == "skipped"
    assert "no GPU visible" in report.reason


def _write_single(tmp_path: Path, document: dict) -> str:
    path = tmp_path / "single.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------
# Placeholders and environment safety
# --------------------------------------------------------------------------------------


def test_resolved_paths_uses_env_override_then_local_path(tmp_path: Path) -> None:
    document = valid_recipe()
    document["dataset"]["env_root"] = "L4_SYNTHETIC_DATA"
    recipe = load_recipe(Path(_write_single(tmp_path, document)))
    resolved = resolved_paths(recipe, {"L4_SYNTHETIC_MODEL": "/from/env", "L4_SYNTHETIC_DATA": "/data/root"})
    assert resolved["model:policy"] == "/from/env"
    assert resolved["dataset:train"] == "/data/root/train.parquet"
    assert resolved["dataset:val"] == "/data/root/test.parquet"

    fallback = resolved_paths(recipe, {})
    assert fallback["model:policy"] == "/models/sd35"
    assert fallback["dataset:train"] == "train.parquet"


def test_render_overrides_substitutes_placeholders(tmp_path: Path) -> None:
    recipe = load_recipe(Path(_write_single(tmp_path, valid_recipe())))
    resolved = {"dataset:train": "/data/train.parquet", "model:policy": "/models/sd35"}
    rendered = render_overrides(recipe, resolved)
    assert rendered == ["data.train_files=/data/train.parquet", "data.seed=42"]


def test_unknown_placeholder_is_rejected() -> None:
    with pytest.raises(RunnerError):
        render_placeholder("{model:missing}", {"model:policy": "/x"})


def test_placeholder_is_not_applied_inside_plain_values() -> None:
    assert render_placeholder("data.seed=42", {}) == "data.seed=42"


def test_embedded_placeholders_keep_their_surrounding_text() -> None:
    """A literal ``{case_dir}/checkpoints`` must become a real path, not a brace directory."""
    resolved = {"case_dir": "/out/current/case", "output_root": "/out"}
    assert render_placeholder("{case_dir}/checkpoints", resolved) == "/out/current/case/checkpoints"
    assert render_placeholder("{output_root}/x/{case_dir}/y", resolved) == "/out/x//out/current/case/y"
    # Hydra dict values with spaces are not placeholders and must survive untouched.
    tokenizers = "{clip: {path: tokenizer, max_length: 77}}"
    assert render_placeholder(tokenizers, resolved) == tokenizers


def test_output_placeholders_keep_run_state_out_of_the_repo(tmp_path: Path) -> None:
    recipe = load_recipe(Path(_write_single(tmp_path, valid_recipe())))
    resolved = resolved_paths(recipe, {}, tmp_path / "out")
    assert resolved["output_root"] == str(tmp_path / "out")
    assert resolved["case_dir"] == str(tmp_path / "out" / "current" / "synthetic_case")
    # Without an explicit output root the placeholders stay absent, so a recipe
    # that depends on them fails loudly instead of writing into the repository.
    assert "output_root" not in resolved_paths(recipe, {})


def test_safelisted_env_drops_secrets() -> None:
    env = {
        "L4_MODEL_PATH": "/models/x",
        "HF_HOME": "/hf",
        "HF_TOKEN": "hf_secret",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "PATH": "/usr/bin",
    }
    safe = safelisted_env(env)
    assert safe["L4_MODEL_PATH"] == "/models/x"
    assert safe["HF_HOME"] == "/hf"
    assert safe["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "HF_TOKEN" not in safe
    assert "AWS_SECRET_ACCESS_KEY" not in safe
    assert "PATH" not in safe
    assert json.dumps(safe)  # the manifest payload must stay JSON serializable
