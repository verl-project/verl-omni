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
"""CPU tests for agentic reward metric aggregation and validation tables."""

import sys
import types

import numpy as np
import pytest

from verl_omni.agent_loop.agentic_metrics_manager import (
    AgenticMetricsAgentLoopManager,
    _pair_generate_image_turns,
    aggregate_agentic_reward_metrics,
)


def test_aggregate_includes_rpco_dimensions_and_skips_absent_pr1_fields():
    metrics = aggregate_agentic_reward_metrics(
        {
            "reward_reflect": np.array([0.8, 0.9]),
            "reward_plan": np.array([0.5, 0.6]),
            "reward_format": np.array([1.0, 1.0]),
            "reward_result": np.array([1.0, 0.0]),
            "reward_done": np.array([1.0, 0.0]),
            "reward_tool_call": np.array([1.0, 1.0]),
        }
    )

    assert metrics["agentic_reward/reflect/mean"] == pytest.approx(0.85)
    assert metrics["agentic_reward/plan/max"] == pytest.approx(0.6)
    assert metrics["agentic_reward/result/min"] == pytest.approx(0.0)
    assert metrics["agentic_reward/tool_call/mean"] == pytest.approx(1.0)
    assert metrics["agentic_reward/done/mean"] == pytest.approx(0.5)
    assert "agentic_reward/correctness/mean" not in metrics
    assert "agentic_reward/aesthetics/mean" not in metrics
    assert "agentic_reward/tool/mean" not in metrics


def test_aggregate_pr1_correctness_aesthetics_when_present():
    metrics = aggregate_agentic_reward_metrics(
        {
            "reward_tool_call": np.array([1.0]),
            "reward_correctness": np.array([0.8]),
            "reward_aesthetics": np.array([0.7]),
            "reward_done": np.array([1.0]),
        }
    )
    assert metrics["agentic_reward/correctness/mean"] == pytest.approx(0.8)
    assert metrics["agentic_reward/aesthetics/max"] == pytest.approx(0.7)
    assert "agentic_reward/reflect/mean" not in metrics


def test_val_prefix_transform():
    metrics = aggregate_agentic_reward_metrics(
        {
            "reward_reflect": np.array([0.8, 0.9]),
            "reward_tool_call": np.array([1.0, 1.0]),
        }
    )
    val_metrics = {f"val_{key}": value for key, value in metrics.items()}

    assert val_metrics["val_agentic_reward/reflect/mean"] == pytest.approx(0.85)
    assert val_metrics["val_agentic_reward/tool_call/mean"] == pytest.approx(1.0)


def test_absent_and_empty_keys_are_skipped():
    assert aggregate_agentic_reward_metrics({"reward_plan": np.array([])}) == {}
    metrics = aggregate_agentic_reward_metrics({"reward_plan": np.array([0.4])})
    assert set(metrics) == {
        "agentic_reward/plan/mean",
        "agentic_reward/plan/min",
        "agentic_reward/plan/max",
    }


def test_pair_generate_image_turns_pads_either_side():
    decoded = (
        '<tool_call>{"name": "generate_image", "arguments": {"prompt": "first"}}</tool_call>\n'
        '<tool_call>{"name": "generate_image", "arguments": {"prompt": "second"}}</tool_call>'
    )
    assert _pair_generate_image_turns(decoded, ["/tmp/a.png"]) == [
        ("first", "/tmp/a.png"),
        ("second", None),
    ]
    assert _pair_generate_image_turns("", ["/tmp/a.png"]) == [("", "/tmp/a.png")]


def test_val_generations_table_accumulates_all_steps(tmp_path, monkeypatch):
    logged: list[dict] = []

    class _FakeImage:
        def __init__(self, path):
            self.path = str(path)

    class _FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.data = []

        def add_data(self, *row):
            self.data.append(list(row))

    fake_wandb = types.SimpleNamespace(
        run=object(),
        Image=_FakeImage,
        Table=_FakeTable,
        log=lambda payload, step=None, commit=True: logged.append({"payload": payload, "step": step, "commit": commit}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "2")

    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    png_a.write_bytes(b"fake")
    png_b.write_bytes(b"fake")
    manager = types.SimpleNamespace(
        _val_generations_history={},
        _log_val_generations_table=AgenticMetricsAgentLoopManager._log_val_generations_table,
    )

    AgenticMetricsAgentLoopManager._log_val_generations_table(
        manager, 0, table_key="val/generations", turn_pairs=[("p0", str(png_a))]
    )
    AgenticMetricsAgentLoopManager._log_val_generations_table(
        manager, 10, table_key="val/generations", turn_pairs=[("p10", str(png_b))]
    )
    AgenticMetricsAgentLoopManager._log_val_generations_table(
        manager, 20, table_key="val/generations", turn_pairs=[("p20", None)]
    )

    assert len(logged) == 3
    assert all(entry["commit"] is True for entry in logged)
    latest = logged[-1]["payload"]["val/generations"]
    assert [row[0] for row in latest.data] == [0, 10, 20]
    assert latest.data[0][1] == "p0"
    assert isinstance(latest.data[0][2], _FakeImage)
    assert latest.data[2][2] == ""
