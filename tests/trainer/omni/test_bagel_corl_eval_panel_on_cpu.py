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

"""Offline evaluation contracts; no verl/serving imports."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/bagel_corl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def examples():
    return [
        dict(id=f"{task}-{i}", task=task, prompt=f"{task} prompt {i}", reference_subtasks=[str(i)], split="val")
        for task in ("reflection", "breakdown")
        for i in range(210)
    ]


def test_panel_order_invariant_and_exact_seeds(tmp_path):
    tool = load("build_eval_panel")
    rows = examples()
    first = tool.build_eval_panel(dump(tmp_path / "data.jsonl", rows), tmp_path / "a.json")
    second = tool.build_eval_panel(dump(tmp_path / "data.jsonl", rows[::-1]), tmp_path / "b.json")
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert a == b
    assert {task: sum(r["task"] == task for r in a["rows"]) for task in a["tasks"]} == {
        "reflection": 200,
        "breakdown": 200,
    }
    assert all(r["prompt_hash"] and r["reference_hash"] and r["render_seeds"] == [42, 43] for r in a["rows"])


@pytest.mark.parametrize("mutation", ["insufficient", "duplicate", "overlap"])
def test_panel_fails_closed(tmp_path, mutation):
    rows = examples()
    if mutation == "insufficient":
        rows = rows[:213]
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows.append(dict(rows[0], id="training-copy", split="train"))
    with pytest.raises(ValueError):
        load("build_eval_panel").build_eval_panel(dump(tmp_path / "data.jsonl", rows), tmp_path / "panel.json")


def test_native_unicot_metadata_and_source_prompt_overlap(tmp_path):
    native = [
        dict(
            prompt=[{"role": "user", "content": f"request {i}"}],
            extra_info={"data_id": str(i), "task_type": task, "split": "val", "raw_prompt": f"request {i}"},
            reward_model={"ground_truth": {"reference_subtasks": ["draw"]}},
        )
        for i, task in enumerate(("plan", "reflect"))
    ]
    tool = load("build_eval_panel")
    result = json.loads(
        tool.build_eval_panel(dump(tmp_path / "native.jsonl", native), tmp_path / "panel.json", per_task=1).read_text()
    )
    assert {r["task"] for r in result["rows"]} == {"breakdown", "reflection"}
    native.append(
        dict(
            native[0],
            prompt=[{"role": "system", "content": "different system"}],
            extra_info=dict(native[0]["extra_info"], data_id="train-copy", split="train"),
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        tool.build_eval_panel(dump(tmp_path / "native.jsonl", native), tmp_path / "panel.json", per_task=1)


def test_native_parquet_panel_remains_trainer_readable(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    native = [
        dict(
            prompt=[{"role": "user", "content": f"request {i}"}],
            extra_info={"data_id": str(i), "task_type": task, "split": "val", "raw_prompt": f"request {i}"},
            reward_model={"ground_truth": {"reference_subtasks": ["draw"]}},
            data_source="unicot_breakdown" if task == "plan" else "unicot_reflection",
        )
        for i, task in enumerate(("plan", "reflect"))
    ]
    output = load("build_eval_panel").build_eval_panel(
        dump(tmp_path / "native.jsonl", native), tmp_path / "panel.parquet", per_task=1
    )
    rows = pq.read_table(output).to_pylist()
    assert {r["extra_info"]["data_id"] for r in rows} == {"0", "1"}
    assert all(r["extra_info"]["eval_render_seeds"] == [42, 43] for r in rows)
    assert json.loads(output.with_suffix(".panel.json").read_text())["panel_hash"]


def records(gain=0):
    return [
        dict(
            task=task,
            training_seed=seed,
            prompt_id=f"{task}-{i}",
            render_seed=r,
            kernel="sde",
            kernel_config_hash="kernel",
            scorer_version="judge-v1",
            eval_config_hash="eval",
            prompt_hash=f"prompt-{task}-{i}",
            image_quality=0.4 + gain,
            protocol_valid=1,
            autonomous_done=1,
            assisted_reflection=0,
            num_gen_calls=1,
            rewrite_gain=0,
        )
        for task in ("reflection", "breakdown")
        for seed in (1, 2, 3)
        for i in range(4)
        for r in (42, 43)
    ]


def test_comparison_pairs_clusters_per_training_seed(tmp_path):
    tool = load("compare_checkpoints")
    a = dump(tmp_path / "a.jsonl", records())
    b = dump(tmp_path / "b.jsonl", records(0.1)[::-1])
    result = tool.compare_checkpoints(a, b, n_bootstrap=100)
    assert result == tool.compare_checkpoints(a, b, n_bootstrap=100)
    assert result["aggregate"]["image_quality"]["effect"] == pytest.approx(0.1)
    assert len(result["by_training_seed"]) == 3
    assert all(
        task["image_quality"]["ci95"][0] > 0 for seed in result["by_training_seed"].values() for task in seed.values()
    )
    assert result["aggregate"]["task_weights"] == {"breakdown": 0.5, "reflection": 0.5}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "kernel", "config", "scorer", "prompt", "nan"])
def test_comparison_rejects_incomparable_records(tmp_path, mutation):
    rows = records(0.1)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        field = {
            "kernel": "kernel_config_hash",
            "config": "eval_config_hash",
            "scorer": "scorer_version",
            "prompt": "prompt_hash",
            "nan": "image_quality",
        }[mutation]
        rows[0][field] = float("nan") if mutation == "nan" else "changed"
    with pytest.raises(ValueError):
        load("compare_checkpoints").compare_checkpoints(
            dump(tmp_path / "a.jsonl", records()), dump(tmp_path / "b.jsonl", rows), n_bootstrap=20
        )
