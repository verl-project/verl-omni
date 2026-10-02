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
"""Deriving `min_improvement` from a reviewed baseline.

The point of these tests is that the tool never invents a threshold: it either
derives one from a measured improvement, or it refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.convergence.derive_thresholds import (
    EXIT_CANNOT_DERIVE,
    EXIT_OK,
    ThresholdError,
    apply_to_recipe,
    derive_min_improvement,
    main,
    pick_driving_metric,
)
from tests.convergence.recipe_registry import load_recipe

REWARD = "critic/rewards/mean"


def recipe_document(*, min_improvement: float = 0.0, optional_first: bool = False) -> dict:
    return {
        "schema_version": 1,
        "layer": "L4",
        "case_id": "synthetic_derive_case",
        "title": "synthetic derive case",
        "algorithm": "flow_grpo",
        "precision": "bf16",
        "recipe": {"launcher": "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh"},
        "models": {
            "policy": {
                "kind": "real",
                "source": "stabilityai/stable-diffusion-3.5-medium",
                "revision": "main",
            }
        },
        "dataset": {"kind": "real", "train": "train.parquet", "val": "test.parquet"},
        "hardware": {"min_gpus": 1, "min_gpu_memory_gb": 1, "gpu_architectures": ["sm89"]},
        "budget": {"timeout_minutes": 5, "total_training_steps": 8},
        "metrics": {
            "warmup_steps": 1,
            "final_window": 2,
            "min_points": 4,
            "tracked": [
                {"name": REWARD, "direction": "higher", "optional": optional_first},
                {"name": "actor/loss", "direction": "lower", "optional": True},
            ],
        },
        "convergence": {"atol": 0.02, "rtol": 0.05, "min_improvement": min_improvement, "require_finite": True},
        "baseline": {"artifact_name": "l4-convergence-baseline", "branch": "main"},
    }


def write_recipe(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def baseline_payload(
    *,
    improvement: float | None = 0.10,
    metric: str = REWARD,
    final_mean: float = 0.40,
    first_mean: float = 0.30,
) -> dict:
    summary = {
        "direction": "higher",
        "num_points": 8,
        "final_window_mean": final_mean,
        "first_window_mean": first_mean,
    }
    if improvement is not None:
        summary["improvement"] = improvement
    return {"schema_version": 1, "layer": "L4", "case_id": "synthetic_derive_case", "summary": {metric: summary}}


def test_proposes_a_fraction_of_the_measured_improvement(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, recipe_document()))
    proposal = derive_min_improvement(recipe, baseline_payload(improvement=0.10), retention=0.5)
    assert proposal.metric == REWARD
    assert proposal.baseline_improvement == pytest.approx(0.10)
    assert proposal.proposed_min_improvement == pytest.approx(0.05)
    assert proposal.current_min_improvement == 0.0


def test_retention_defaults_to_half() -> None:
    recipe = load_recipe(Path(__file__).resolve().parent / "recipes" / "sd35_medium_flowgrpo.yaml")
    proposal = derive_min_improvement(recipe, baseline_payload(improvement=0.20))
    assert proposal.retention == 0.5
    assert proposal.proposed_min_improvement == pytest.approx(0.10)


def test_refuses_when_the_baseline_did_not_improve(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, recipe_document()))
    with pytest.raises(ThresholdError) as error:
        derive_min_improvement(recipe, baseline_payload(improvement=-0.01))
    assert "did not improve" in str(error.value)


def test_refuses_when_improvement_was_not_measured(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, recipe_document()))
    with pytest.raises(ThresholdError) as error:
        derive_min_improvement(recipe, baseline_payload(improvement=None))
    assert "could not measure" in str(error.value)


def test_refuses_when_the_driving_metric_is_missing(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, recipe_document()))
    with pytest.raises(ThresholdError) as error:
        derive_min_improvement(recipe, baseline_payload(metric="critic/score/mean"))
    assert "no summary for the driving metric" in str(error.value)


def test_refuses_an_out_of_range_retention(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, recipe_document()))
    for retention in (0.0, -0.5, 1.5):
        with pytest.raises(ThresholdError):
            derive_min_improvement(recipe, baseline_payload(), retention=retention)


def test_unknown_driving_metric_is_rejected(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, recipe_document()))
    with pytest.raises(ThresholdError) as error:
        pick_driving_metric(recipe, "not/a/metric")
    assert "is not tracked" in str(error.value)


def test_driving_metric_prefers_a_required_metric(tmp_path: Path) -> None:
    document = recipe_document(optional_first=True)
    document["metrics"]["tracked"][1]["optional"] = False
    recipe = load_recipe(write_recipe(tmp_path, document))
    assert pick_driving_metric(recipe, None)["name"] == "actor/loss"


def test_apply_only_rewrites_the_min_improvement_line(tmp_path: Path) -> None:
    document = recipe_document()
    path = write_recipe(tmp_path, document)
    recipe = load_recipe(path)
    proposal = derive_min_improvement(recipe, baseline_payload(improvement=0.10), retention=0.5)

    original = path.read_text(encoding="utf-8")
    updated = apply_to_recipe(path, proposal)
    assert updated.count("min_improvement:") == 1
    assert "min_improvement: 0.05" in updated
    # every other line is untouched
    diff_lines = [
        (before, after)
        for before, after in zip(original.splitlines(), updated.splitlines(), strict=True)
        if before != after
    ]
    assert len(diff_lines) == 1, diff_lines
    assert "min_improvement" in diff_lines[0][0]

    # the rewritten file is still a valid recipe with the new floor
    path.write_text(updated, encoding="utf-8")
    assert load_recipe(path).data["convergence"]["min_improvement"] == pytest.approx(0.05)


def test_apply_rejects_a_recipe_without_a_min_improvement_key(tmp_path: Path) -> None:
    path = tmp_path / "bare.yaml"
    path.write_text("convergence:\n  atol: 1.0\n  rtol: 0.1\n", encoding="utf-8")
    proposal = derive_min_improvement(
        load_recipe(write_recipe(tmp_path, recipe_document())), baseline_payload(improvement=0.1)
    )
    with pytest.raises(ThresholdError) as error:
        apply_to_recipe(path, proposal)
    assert "exactly one" in str(error.value)


def test_cli_writes_the_recipe_only_with_write(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    document = recipe_document()
    path = write_recipe(tmp_path, document)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload(improvement=0.20)), encoding="utf-8")

    assert main(["--recipe", str(path), "--baseline", str(baseline_path)]) == EXIT_OK
    assert load_recipe(path).data["convergence"]["min_improvement"] == 0.0  # not written without --write

    assert main(["--recipe", str(path), "--baseline", str(baseline_path), "--write"]) == EXIT_OK
    assert load_recipe(path).data["convergence"]["min_improvement"] == pytest.approx(0.10)
    assert "min_improvement 0 -> 0.1" in capsys.readouterr().out


def test_cli_exits_non_zero_when_it_cannot_derive(tmp_path: Path) -> None:
    path = write_recipe(tmp_path, recipe_document())
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload(improvement=-0.5)), encoding="utf-8")
    assert main(["--recipe", str(path), "--baseline", str(baseline_path)]) == EXIT_CANNOT_DERIVE
    assert load_recipe(path).data["convergence"]["min_improvement"] == 0.0


def test_shipped_recipes_still_hold_the_placeholder_floor() -> None:
    """The floor stays 0.0 until a reviewed baseline exists; this documents that."""
    recipes_dir = Path(__file__).resolve().parent / "recipes"
    for name in ("sd35_medium_flowgrpo", "qwen_image_flowgrpo", "qwen3_omni_thinker_gspo"):
        recipe = load_recipe(recipes_dir / f"{name}.yaml")
        assert recipe.data["convergence"]["min_improvement"] == 0.0, name
