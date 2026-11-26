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
"""CPU tests for the UniCoT agentic RL parquet builder."""

import json
from pathlib import Path

import pandas as pd
import pytest

from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl as builder


def _write_snapshot(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    """Write a minimal Hub cache snapshot containing source metadata."""
    root = tmp_path / name
    snapshot = root / "snapshots" / "0000000000000000000000000000000000000000"
    snapshot.mkdir(parents=True)
    (snapshot / "metadata.json").write_text(json.dumps(rows))
    return root


def _reflection_row(data_id: str, states: int = 2) -> dict:
    """Create a reflection trajectory with aligned image transitions."""
    inputs = [f"./images/{data_id}_{index}.png" for index in range(states)]
    outputs: list[str | None] = [f"./images/{data_id}_{index + 1}.png" for index in range(states - 1)] + [None]
    return {
        "data_id": data_id,
        "prompt": f"Reflection prompt {data_id}.",
        "eval": [f"Evaluation {index}." for index in range(states)],
        "eval_summary": [f"Summary {index}." for index in range(states)],
        "edit": ["Improve lighting."] * (states - 1) + ["Everything is good. No editing needed."],
        "input_image": inputs,
        "output_image": outputs,
    }


def _breakdown_row(data_id: str, count: int = 2) -> dict:
    """Create a plan with active subtasks and null-padded unused slots."""
    subtasks: list[str | None] = [f"Subtask {index}." for index in range(count)]
    subtasks.extend([None] * (3 - count))
    images: list[str | None] = [f"./images/{data_id}_{index}.png" for index in range(count)]
    images.extend([None] * (3 - count))
    return {
        "data_id": data_id,
        "prompt": f"Breakdown prompt {data_id}.",
        "subtasks": subtasks,
        "subtask_images": images,
    }


def _no_breakdown_row(data_id: str) -> dict:
    """Create a single-image source row without a planning requirement."""
    return {
        "data_id": data_id,
        "prompt": f"Simple prompt {data_id}.",
        "subtasks": ["No breakdown needed.", None, None],
        "subtask_images": [None, None, None],
    }


def _build(
    tmp_path: Path,
    *,
    reflection_rows: list[dict],
    breakdown_rows: list[dict],
    train_size: int | None = None,
    val_size: int | None = None,
    mix_ratio: float = 0.5,
    val_ratio: float = 0.2,
    seed: int = 7,
) -> Path:
    """Build real parquet outputs from small local source snapshots."""
    output = tmp_path / "output"
    reflection_dir = _write_snapshot(tmp_path, "reflection", reflection_rows) if reflection_rows else ""
    breakdown_dir = _write_snapshot(tmp_path, "breakdown", breakdown_rows) if breakdown_rows else ""
    builder.main_cli(
        reflection_dir=str(reflection_dir),
        breakdown_dir=str(breakdown_dir),
        local_save_dir=str(output),
        train_size=train_size,
        val_size=val_size,
        mix_ratio=mix_ratio,
        seed=seed,
        val_ratio=val_ratio,
    )
    return output


def _read(output: Path, split: str) -> pd.DataFrame:
    """Read a generated split through the parquet consumer interface."""
    return pd.read_parquet(output / f"{split}.parquet")


def test_builds_expected_agentic_schema_without_reference_leakage(tmp_path):
    """Training prompts must expose the task schema without source annotations."""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1 + i % 3) for i in range(30)],
        breakdown_rows=[_breakdown_row(f"b{i}", 1 + i % 3) for i in range(30)],
        train_size=20,
        val_size=8,
    )
    train = _read(output, "train")

    assert set(train.columns) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert {"unicot_reflection", "unicot_breakdown"}.issubset(set(train["data_source"]))
    for messages in train["prompt"]:
        messages = list(messages)
        assert [message["role"] for message in messages] == ["system", "user"]
        prompt_blob = " ".join(message["content"] for message in messages)
        assert "Summary 0." not in prompt_blob
        assert "Subtask 0." not in prompt_blob
        assert messages[1]["content"].endswith("(≤4 sentences).")


def test_references_and_weights_live_only_in_ground_truth(tmp_path):
    """Reward references and weights must stay out of prompt metadata."""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}") for i in range(20)],
        breakdown_rows=[_breakdown_row(f"b{i}", 3) for i in range(20)],
        train_size=20,
        val_size=4,
    )
    train = _read(output, "train")

    for reward_model, extra_info in zip(train["reward_model"], train["extra_info"], strict=True):
        ground_truth = reward_model["ground_truth"]
        assert all(f"w_{dim}" in ground_truth for dim in builder.REWARD_DIMS)
        assert not any(key.startswith("w_") for key in extra_info)
        if ground_truth["task_type"] == "plan":
            assert len(ground_truth["reference_subtasks"]) == 3
            assert ground_truth.get("reference_steps") is None
        else:
            assert [step["action"] for step in ground_truth["reference_steps"]] == ["continue", "stop"]
            assert ground_truth.get("reference_subtasks") is None


def test_plan_and_reflect_rows_use_task_specific_system_prompts(tmp_path):
    """Each task must receive its own tool-use protocol."""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}") for i in range(20)],
        breakdown_rows=[_breakdown_row(f"b{i}") for i in range(20)],
        train_size=20,
        val_size=4,
    )
    train = _read(output, "train")
    prompts_by_type = {
        extra["task_type"]: prompt[0]["content"]
        for prompt, extra in zip(train["prompt"], train["extra_info"], strict=True)
    }
    assert prompts_by_type["plan"] == builder.PLAN_SYSTEM_PROMPT
    assert prompts_by_type["reflect"] == builder.REFLECT_SYSTEM_PROMPT
    assert prompts_by_type["plan"] != prompts_by_type["reflect"]


def test_no_breakdown_rows_become_single_image_reflect_tasks(tmp_path):
    """No-breakdown rows must use reflection behavior while retaining corpus identity."""
    output = _build(
        tmp_path,
        reflection_rows=[],
        breakdown_rows=[_no_breakdown_row(f"n{i}") for i in range(20)],
        train_size=None,
        val_size=None,
    )
    rows = pd.concat([_read(output, "train"), _read(output, "val")])
    for reward_model in rows["reward_model"]:
        ground_truth = reward_model["ground_truth"]
        assert ground_truth["task_type"] == "reflect"
        assert ground_truth["expected_num_images"] == 1
        assert ground_truth["plan_expected"] is False
    assert set(rows["data_source"]) == {builder.REFLECT_DATA_SOURCE}
    for extra in rows["extra_info"]:
        assert extra["unicot_source"] == builder.UNICOT_BREAKDOWN_DATASET_ID


def test_rejections_are_reported_and_dropped(tmp_path):
    """Invalid source rows must be counted and excluded from parquet outputs."""
    bad = _reflection_row("bad")
    bad["edit"][0] = ""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"good{i}", 1) for i in range(20)] + [bad],
        breakdown_rows=[],
    )
    report = json.loads((output / "build_report.json").read_text())
    all_rows = pd.concat([_read(output, "train"), _read(output, "val")])

    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "bad"
    assert "bad" not in {extra["data_id"] for extra in all_rows["extra_info"]}
    assert report["source_configs"]["reflection"]["unicot_reflection_cleaner"]
    assert report["partition_id"].startswith("partition_")


def test_full_mode_uses_every_valid_row_and_splits_are_disjoint(tmp_path):
    """Uncapped builds must retain all valid rows without split overlap."""
    reflection = [_reflection_row(f"r{i}", 1 + i % 3) for i in range(40)]
    breakdown = [_breakdown_row(f"b{i}", 1 + i % 3) for i in range(40)]
    output = _build(tmp_path, reflection_rows=reflection, breakdown_rows=breakdown)
    train = _read(output, "train")
    val = _read(output, "val")
    train_ids = {(extra["unicot_source"], extra["data_id"]) for extra in train["extra_info"]}
    val_ids = {(extra["unicot_source"], extra["data_id"]) for extra in val["extra_info"]}

    assert len(train) + len(val) == 80
    assert train_ids.isdisjoint(val_ids)


def test_full_mode_is_order_independent_and_deterministic(tmp_path):
    """Reordering metadata must not change the selected rows or their indices."""
    reflection = [_reflection_row(f"r{i}", 1 + i % 3) for i in range(30)]
    breakdown = [_breakdown_row(f"b{i}", 1 + i % 3) for i in range(30)]
    first = _build(tmp_path / "first", reflection_rows=reflection, breakdown_rows=breakdown)
    second = _build(
        tmp_path / "second",
        reflection_rows=list(reversed(reflection)),
        breakdown_rows=list(reversed(breakdown)),
    )

    assert list(_read(first, "train")["extra_info"]) == list(_read(second, "train")["extra_info"])
    assert list(_read(first, "val")["extra_info"]) == list(_read(second, "val")["extra_info"])


def test_size_caps_apply_requested_mix_ratio(tmp_path):
    """Capped splits must honor both the requested count and task mix."""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
        breakdown_rows=[_breakdown_row(f"b{i}") for i in range(40)],
        train_size=20,
        val_size=8,
        mix_ratio=0.25,
    )
    train = _read(output, "train")
    reflect_count = sum(extra["task_type"] == "reflect" for extra in train["extra_info"])
    assert len(train) == 20
    assert reflect_count == 5


def test_requires_a_source_and_valid_ratios(tmp_path):
    """Missing sources and out-of-range mix ratios must fail before building."""
    common = {
        "reflection_dir": "",
        "breakdown_dir": "",
        "local_save_dir": str(tmp_path),
        "train_size": None,
        "val_size": None,
        "mix_ratio": 0.5,
        "seed": 7,
        "val_ratio": 0.2,
    }
    with pytest.raises(SystemExit):
        builder.main_cli(**common)

    reflection_dir = _write_snapshot(tmp_path, "reflection", [_reflection_row("r")])
    common["reflection_dir"] = str(reflection_dir)
    common["mix_ratio"] = 1.1
    with pytest.raises(SystemExit):
        builder.main_cli(**common)


def test_mismatched_transition_uris_are_rejected(tmp_path):
    """Broken output-to-input image chains must appear in the rejection report."""
    bad = _reflection_row("bad", states=2)
    bad["output_image"][0] = "./images/not_the_next_input.png"
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"good{i}", 1) for i in range(20)] + [bad],
        breakdown_rows=[],
    )
    report = json.loads((output / "build_report.json").read_text())
    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "bad"
    assert report["rejections"][0]["reason"] == "transition_hash_mismatch"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_empty_reflection_image_uris_are_rejected(tmp_path, blank):
    """Blank or null image references must be rejected before row emission."""
    bad = _reflection_row("blank", states=2)
    bad["input_image"][0] = blank
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"good{i}", 1) for i in range(20)] + [bad],
        breakdown_rows=[],
    )
    report = json.loads((output / "build_report.json").read_text())
    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "blank"
    assert report["rejections"][0]["reason"] == "missing_image"


def test_hub_refs_main_beats_lexicographically_later_snapshot(tmp_path):
    """Snapshot selection must follow the main ref rather than SHA ordering."""
    root = tmp_path / "reflection"
    stale_sha = "ffffffffffffffffffffffffffffffffffffffff"
    main_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    stale = root / "snapshots" / stale_sha
    current = root / "snapshots" / main_sha
    stale.mkdir(parents=True)
    current.mkdir(parents=True)
    (stale / "metadata.json").write_text(json.dumps([_reflection_row("stale", 1)]))
    (current / "metadata.json").write_text(json.dumps([_reflection_row(f"keep{i}", 1) for i in range(20)]))
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(f"{main_sha}\n")

    output = tmp_path / "output"
    builder.main_cli(
        reflection_dir=str(root),
        breakdown_dir="",
        local_save_dir=str(output),
        train_size=None,
        val_size=None,
        mix_ratio=0.5,
        seed=7,
        val_ratio=0.2,
    )
    ids = {extra["data_id"] for extra in pd.concat([_read(output, "train"), _read(output, "val")])["extra_info"]}
    assert "stale" not in ids
    assert "keep0" in ids


def test_capped_size_fails_closed_when_mix_cannot_be_met(tmp_path):
    """An unavailable task quota must abort rather than silently change the mix."""
    with pytest.raises(SystemExit, match="cannot meet mix_ratio"):
        _build(
            tmp_path,
            reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
            breakdown_rows=[],
            train_size=20,
            val_size=None,
            mix_ratio=0.5,
        )


def test_capped_size_fails_closed_when_pool_is_smaller_than_requested(tmp_path):
    """An oversized cap must abort rather than silently undershoot."""
    with pytest.raises(SystemExit, match="requested 100"):
        _build(
            tmp_path,
            reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
            breakdown_rows=[],
            train_size=100,
            val_size=None,
            mix_ratio=1.0,
        )


def test_build_report_records_requested_and_actual_size(tmp_path):
    """The persisted report must distinguish requested and emitted row counts."""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
        breakdown_rows=[_breakdown_row(f"b{i}") for i in range(40)],
        train_size=20,
        val_size=8,
        mix_ratio=0.25,
    )
    report = json.loads((output / "build_report.json").read_text())
    train = report["splits"]["train"]
    assert train["requested_size"] == 20
    assert train["actual_size"] == 20
    assert train["total"] == 20


@pytest.mark.parametrize("source", ["reflection", "breakdown"])
@pytest.mark.parametrize("invalid", [None, "bad", 42, [], True])
def test_non_object_rows_are_reported_without_losing_valid_rows(tmp_path, source, invalid):
    """Malformed JSON entries must not abort either source's build."""
    reflection = [_reflection_row("r")]
    breakdown = [_breakdown_row("b"), _no_breakdown_row("simple")]
    (reflection if source == "reflection" else breakdown).append(invalid)
    output = _build(tmp_path, reflection_rows=reflection, breakdown_rows=breakdown)
    report = json.loads((output / "build_report.json").read_text())
    assert report["rejection_count"] == 1
    assert report["rejections"] == [{"data_id": "", "reason": "invalid_field_type", "field": None}]
    rows = pd.concat([_read(output, "train"), _read(output, "val")])
    assert {extra["data_id"] for extra in rows["extra_info"]} == {"r", "b", "simple"}


@pytest.mark.parametrize("override", [False, True])
def test_report_weights_match_all_emitted_ground_truth(tmp_path, monkeypatch, override):
    """The report must capture defaults and overrides for every task and source."""
    expected = {"w_reflect": 1.0, "w_plan": 1.0, "w_format": 1.0, "w_tool": 1.0, "w_result": 1.0}
    for dim in ("REFLECT", "PLAN", "FORMAT", "TOOL", "RESULT"):
        monkeypatch.delenv(f"RPCO_W_{dim}", raising=False)
    if override:
        monkeypatch.setenv("RPCO_W_REFLECT", " 0.25 ")
        monkeypatch.setenv("RPCO_W_PLAN", "2.5")
        monkeypatch.setenv("RPCO_W_RESULT", "0")
        expected.update(w_reflect=0.25, w_plan=2.5, w_result=0.0)
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row("r")],
        breakdown_rows=[_breakdown_row("b"), _no_breakdown_row("simple")],
    )
    report = json.loads((output / "build_report.json").read_text())
    assert report["reward_weights"] == expected
    rows = pd.concat([_read(output, "train"), _read(output, "val")])
    assert len(rows) == 3
    for reward in rows["reward_model"]:
        assert {key: reward["ground_truth"][key] for key in expected} == expected


@pytest.mark.parametrize("dim", builder.REWARD_DIMS)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e309"])
def test_build_rejects_nonfinite_reward_weights(tmp_path, monkeypatch, dim, value):
    """Reject non-finite overrides before emitting ground-truth weights."""
    env_name = f"RPCO_W_{dim.upper()}"
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValueError, match=env_name):
        _build(tmp_path, reflection_rows=[_reflection_row("r")], breakdown_rows=[_breakdown_row("b")])
