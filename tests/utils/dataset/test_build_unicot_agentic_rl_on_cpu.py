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
    root = tmp_path / name
    snapshot = root / "snapshots" / "0000000000000000000000000000000000000000"
    snapshot.mkdir(parents=True)
    (snapshot / "metadata.json").write_text(json.dumps(rows))
    return root


def _reflection_row(data_id: str, states: int = 2) -> dict:
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
    return pd.read_parquet(output / f"{split}.parquet")


def test_builds_expected_agentic_schema_without_reference_leakage(tmp_path):
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
        assert all(f"w_{dim}" in ground_truth for dim in builder.DIMS)
        assert not any(key.startswith("w_") for key in extra_info)
        if ground_truth["task_type"] == "plan":
            assert len(ground_truth["reference_subtasks"]) == 3
            assert ground_truth.get("reference_steps") is None
        else:
            assert [step["action"] for step in ground_truth["reference_steps"]] == ["continue", "stop"]
            assert ground_truth.get("reference_subtasks") is None


def test_plan_and_reflect_rows_use_task_specific_system_prompts(tmp_path):
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


def test_rejections_are_reported_and_dropped(tmp_path):
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
