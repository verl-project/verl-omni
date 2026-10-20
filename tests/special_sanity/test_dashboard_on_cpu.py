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

import importlib.util
import sys
from pathlib import Path


def _load_dashboard():
    path = Path(__file__).resolve().parents[2] / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("nightly_dashboard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_artifacts_keeps_latest_baseline_and_main_reports():
    dashboard = _load_dashboard()
    artifacts = [
        {
            "id": 1,
            "name": "l3-nightly-baseline",
            "created_at": "2026-08-01T00:00:00Z",
            "expired": False,
            "workflow_run": {"id": 11, "head_branch": "main"},
        },
        {
            "id": 2,
            "name": "l3-nightly-baseline",
            "created_at": "2026-08-24T08:42:09Z",
            "expired": False,
            "workflow_run": {"id": 22, "head_branch": "main"},
        },
        {
            "id": 3,
            "name": "l3-nightly-reports-30",
            "created_at": "2026-09-10T17:36:10Z",
            "expired": False,
            "workflow_run": {"id": 30, "head_branch": "main"},
        },
        {
            "id": 4,
            "name": "l3-nightly-reports-40",
            "created_at": "2026-09-09T17:38:14Z",
            "expired": False,
            "workflow_run": {"id": 40, "head_branch": "ci-fix"},
        },
        {
            "id": 5,
            "name": "l3-nightly-baseline-staging-30",
            "created_at": "2026-09-10T17:00:00Z",
            "expired": False,
            "workflow_run": {"id": 30, "head_branch": "main"},
        },
        {
            "id": 6,
            "name": "l3-nightly-reports-31",
            "created_at": "2026-09-08T17:52:59Z",
            "expired": True,
            "workflow_run": {"id": 31, "head_branch": "main"},
        },
    ]

    baseline, reports = dashboard.select_artifacts(artifacts, branch="main", max_reports=10)

    assert baseline is not None
    assert baseline.artifact_id == 2
    assert [item.artifact_id for item in reports] == [3]


def test_chart_payload_uses_baseline_horizontal_line_and_mean_abs_err():
    dashboard = _load_dashboard()
    rows = [
        {
            "label": "2026-09-09 01:38",
            "created_at": "2026-09-08T17:38:00Z",
            "metrics": {"perf/time_per_step": 1.5, "perf/throughput": 99.0},
            "precision": {"step1/payload.pt :: latents": 1e-5},
        },
        {
            "label": "2026-09-10 01:36",
            "created_at": "2026-09-09T17:36:00Z",
            "metrics": {"perf/time_per_step": 1.7, "perf/throughput": 80.0},
            "precision": {"step1/payload.pt :: latents": 2e-5},
        },
    ]

    payload = dashboard.build_chart_payload(rows, {"perf/time_per_step": 1.2}, "2026-08-24 16:42")

    assert payload["timestamps"] == ["2026-09-08T17:38:00Z", "2026-09-09T17:36:00Z"]
    assert [chart["title"] for chart in payload["performance"]] == ["perf/time_per_step"]
    perf = payload["performance"][0]
    assert perf["unit"] == "s"
    assert perf["values"] == [1.5, 1.7]
    assert perf["baseline"] == 1.2
    assert perf["baseline_legend"] == "baseline (2026-08-24 16:42)"
    precision = payload["precision"][0]
    assert precision["title"] == "latents"
    assert precision["values"] == [1e-5, 2e-5]


def test_precision_excludes_baseline_day_point():
    dashboard = _load_dashboard()
    rows = [
        {
            "label": "2026-08-24 16:42",
            "created_at": "2026-08-24T08:42:09Z",
            "metrics": {"perf/time_per_step": 1.2},
            "precision": {"step1/payload.pt :: latents": 0.0},
        },
        {
            "label": "2026-09-10 01:36",
            "created_at": "2026-09-09T17:36:00Z",
            "metrics": {"perf/time_per_step": 1.7},
            "precision": {"step1/payload.pt :: latents": 2e-5},
        },
    ]

    payload = dashboard.build_chart_payload(rows, {"perf/time_per_step": 1.2}, "2026-08-24 16:42")

    assert payload["performance"][0]["values"] == [1.2, 1.7]
    precision = payload["precision"][0]
    assert precision["labels"] == ["2026-09-10 01:36"]
    assert precision["values"] == [2e-5]


def test_precision_titles_drop_repeated_path_and_event():
    dashboard = _load_dashboard()
    titles = dashboard.precision_chart_titles(
        [
            "driver_forward/step_000001/rank_00000/event_000001/payload.pt :: batch.advantages",
            "driver_forward/step_000001/rank_00000/event_000001/payload.pt :: batch.all_latents",
            "driver_forward/step_000002/rank_00000/event_000002/payload.pt :: batch.advantages",
        ]
    )

    assert titles == {
        "driver_forward/step_000001/rank_00000/event_000001/payload.pt :: batch.advantages": "s1/advantages",
        "driver_forward/step_000001/rank_00000/event_000001/payload.pt :: batch.all_latents": "s1/all_latents",
        "driver_forward/step_000002/rank_00000/event_000002/payload.pt :: batch.advantages": "s2/advantages",
    }
    assert (
        dashboard.shorten_precision_key(
            "actor_forward/step_000001/rank_00001/mini_batch_idx_0002/event_000009/payload.pt :: gradients.foo"
        )
        == "actor/s1/r1/mini2/grad.foo"
    )


def test_merge_history_rows_prefers_incoming_run_and_keeps_older_points():
    dashboard = _load_dashboard()
    existing = [{"run_id": 1, "created_at": "2026-09-01T00:00:00Z", "metrics": {"perf/time_per_step": 1.0}}]
    incoming = [
        {"run_id": 1, "created_at": "2026-09-01T00:00:00Z", "metrics": {"perf/time_per_step": 1.1}},
        {"run_id": 2, "created_at": "2026-09-02T00:00:00Z", "metrics": {"perf/time_per_step": 1.2}},
    ]

    merged = dashboard.merge_history_rows(existing, incoming)

    assert [row["run_id"] for row in merged] == [1, 2]
    assert merged[0]["metrics"]["perf/time_per_step"] == 1.1


def test_render_html_embeds_line_charts_in_a_grid():
    dashboard = _load_dashboard()
    html = dashboard.render_html(
        {
            "repo": "verl-project/verl-omni",
            "layers": [{"id": "l3-nightly", "label": "L3 Nightly CI"}],
            "experiments": [
                {
                    "id": "qwen_image_flowgrpo",
                    "label": "Qwen-Image + FlowGRPO",
                    "layer": "l3-nightly",
                    "performance": [{"title": "perf/time_per_step", "values": [1.0]}],
                    "precision": [{"title": "latents", "values": [0.0]}],
                }
            ],
        }
    )

    assert 'id="performance-grid"' in html
    assert 'id="precision-grid"' in html
    assert 'id="range"' in html
    assert 'id="layers"' in html
    assert 'id="experiments"' in html
    assert 'aria-label="CI layer"' in html
    assert 'aria-label="Experiment"' in html
    assert "L3 Nightly CI" in html
    assert "layer-btn" in html
    assert "experiment-btn" in html
    assert "experiment-group" in html
    assert "L4 Weekly CI" not in html
    assert 'aria-label="Time range"' in html
    assert "VeRL-Omni CI Dashboard" in html
    assert '{days: 7, label: "7d"}' in html
    assert "grid-template-columns: repeat(auto-fit, minmax(420px, 1fr))" in html
    assert 'type: "line"' in html
