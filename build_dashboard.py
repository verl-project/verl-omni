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
"""Download L3 nightly artifacts and render a local performance/precision dashboard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BASELINE_ARTIFACT_NAME = "l3-nightly-baseline"
REPORT_ARTIFACT_PREFIX = "l3-nightly-reports-"
STAGING_ARTIFACT_PREFIX = "l3-nightly-baseline-staging-"
DISPLAY_TZ = ZoneInfo("Asia/Shanghai")
USER_AGENT = "verl-omni-l3-dashboard"
DEFAULT_HISTORY_URL = "https://verl-project.github.io/verl-omni/dashboard/pages/history.json"
KIND_SHORT = {
    "driver_forward": "driver",
    "actor_forward": "actor",
    "actor_forward_step": "actor_step",
    "lora_gradients": "lora",
}
IDENTITY_SHORT = {
    "ppo_epoch": "epoch",
    "mini_batch_idx": "mini",
    "micro_batch_idx": "micro",
    "diffusion_step": "diff",
}
LAYERS = [
    {
        "id": "l3-nightly",
        "label": "L3 Nightly CI",
    },
]
EXPERIMENTS = [
    {
        "id": "qwen_image_flowgrpo",
        "label": "Qwen-Image + FlowGRPO",
        "layer": "l3-nightly",
    },
]
PERF_CHART_KEYS = {
    "perf/time_per_step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/reward",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "timing_per_image_ms/gen",
    "timing_per_image_ms/old_log_prob",
    "timing_per_image_ms/update_actor",
    "timing_per_image_ms/adv",
}


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: int
    name: str
    created_at: str
    expired: bool
    head_branch: str | None
    workflow_run_id: int | None


def unit_for_metric(key: str) -> str | None:
    """Return the axis unit for a plotted timing metric, or None to skip it."""
    if key not in PERF_CHART_KEYS:
        return None
    if key.startswith("timing_per_image_ms/"):
        return "ms"
    return "s"


def format_run_label(created_at: str) -> str:
    parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(DISPLAY_TZ)
    return parsed.strftime("%Y-%m-%d %H:%M")


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part.split(";", 1)[0].strip().strip("<>")
    return None


def _artifact_ref(item: dict[str, Any]) -> ArtifactRef:
    workflow_run = item.get("workflow_run") or {}
    return ArtifactRef(
        artifact_id=int(item["id"]),
        name=str(item["name"]),
        created_at=str(item["created_at"]),
        expired=bool(item.get("expired")),
        head_branch=workflow_run.get("head_branch"),
        workflow_run_id=workflow_run.get("id"),
    )


def select_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    branch: str | None,
    max_reports: int,
) -> tuple[ArtifactRef | None, list[ArtifactRef]]:
    """Pick the latest baseline and recent report artifacts, newest first."""
    refs = [_artifact_ref(item) for item in artifacts]
    live = [item for item in refs if not item.expired]
    if branch:
        live = [item for item in live if item.head_branch in {None, branch}]

    baselines = sorted(
        (item for item in live if item.name == BASELINE_ARTIFACT_NAME),
        key=lambda item: item.created_at,
        reverse=True,
    )
    reports = sorted(
        (
            item
            for item in live
            if item.name.startswith(REPORT_ARTIFACT_PREFIX) and not item.name.startswith(STAGING_ARTIFACT_PREFIX)
        ),
        key=lambda item: item.created_at,
        reverse=True,
    )
    return (baselines[0] if baselines else None, reports[:max_reports])


def extract_metric_means(metrics_payload: dict[str, Any]) -> dict[str, float]:
    summary = metrics_payload.get("summary_for_compare") or {}
    means: dict[str, float] = {}
    for key, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        mean = stats.get("mean")
        if isinstance(mean, int | float):
            means[str(key)] = float(mean)
    return means


def extract_precision_errors(dump_payload: dict[str, Any]) -> dict[str, float]:
    errors: dict[str, float] = {}
    files = dump_payload.get("files") or {}
    for rel_path, file_result in files.items():
        tensors = (file_result or {}).get("tensors") or {}
        for tensor_key, metrics in tensors.items():
            if not isinstance(metrics, dict) or metrics.get("shape_mismatch"):
                continue
            value = metrics.get("mean_abs_err")
            if isinstance(value, int | float):
                errors[f"{rel_path} :: {tensor_key}"] = float(value)
    return errors


def _padded_identity(part: str) -> tuple[str, int] | None:
    name, sep, digits = part.rpartition("_")
    if not sep or not name or not digits.isdigit():
        return None
    return name, int(digits)


def shorten_precision_key(key: str, *, keep_event: bool = False) -> str:
    """Turn a dump path + tensor key into a short chart title."""
    path, sep, tensor = key.partition(" :: ")
    if not sep:
        path, tensor = key, ""
    parts = [part for part in path.replace("\\", "/").split("/") if part and part != "payload.pt"]
    display: list[str] = []
    step_num: int | None = None
    for part in parts:
        identity = _padded_identity(part)
        if identity is None:
            display.append(KIND_SHORT.get(part, part))
            continue
        name, number = identity
        if name == "rank" and number == 0:
            continue
        if name == "event" and (not keep_event or (step_num is not None and number == step_num)):
            continue
        if name == "step":
            step_num = number
            display.append(f"s{number}")
            continue
        if name == "rank":
            display.append(f"r{number}")
            continue
        if name == "event":
            display.append(f"e{number}")
            continue
        display.append(f"{IDENTITY_SHORT.get(name, name)}{number}")
    if tensor.startswith("batch."):
        tensor = tensor[len("batch.") :]
    elif tensor.startswith("gradients."):
        tensor = "grad." + tensor[len("gradients.") :]
    if display and tensor:
        return "/".join([*display, tensor])
    return tensor or "/".join(display) or key


def precision_chart_titles(keys: list[str]) -> dict[str, str]:
    """Shorten dump keys and drop a prefix shared by every series."""
    titles = {key: shorten_precision_key(key) for key in keys}
    if len(set(titles.values())) != len(titles):
        titles = {key: shorten_precision_key(key, keep_event=True) for key in keys}
    segments = [title.split("/") for title in titles.values()]
    while segments and all(len(item) > 1 for item in segments) and len({item[0] for item in segments}) == 1:
        titles = {key: "/".join(title.split("/")[1:]) for key, title in titles.items()}
        segments = [title.split("/") for title in titles.values()]
    return titles


def merge_history_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in existing + incoming:
        key = str(row.get("run_id") or row.get("created_at"))
        by_key[key] = row
    return sorted(by_key.values(), key=lambda item: str(item.get("created_at", "")))


def precision_rows(nightly_rows: list[dict[str, Any]], baseline_label: str | None) -> list[dict[str, Any]]:
    """Drop the baseline run from precision charts; it is compared against itself."""
    if not baseline_label:
        return nightly_rows
    return [row for row in nightly_rows if row.get("label") != baseline_label]


def build_chart_payload(
    nightly_rows: list[dict[str, Any]],
    baseline_means: dict[str, float],
    baseline_label: str | None,
) -> dict[str, Any]:
    """Assemble Chart.js-ready series. Nightly rows must already be sorted by time."""
    labels = [row["label"] for row in nightly_rows]
    timestamps = [str(row.get("created_at") or "") for row in nightly_rows]
    perf_keys: set[str] = set()
    for row in nightly_rows:
        perf_keys.update(key for key in row["metrics"] if unit_for_metric(key))

    performance = []
    for key in sorted(perf_keys):
        unit = unit_for_metric(key)
        assert unit is not None
        series = {
            "title": key,
            "y_label": f"Mean ({unit})",
            "unit": unit,
            "labels": labels,
            "values": [row["metrics"].get(key) for row in nightly_rows],
        }
        if key in baseline_means and baseline_label:
            series["baseline"] = baseline_means[key]
            series["baseline_legend"] = f"baseline ({baseline_label})"
        performance.append(series)

    precision_nightly_rows = precision_rows(nightly_rows, baseline_label)
    precision_labels = [row["label"] for row in precision_nightly_rows]
    precision_keys = set()
    for row in precision_nightly_rows:
        precision_keys.update(row["precision"])
    titles = precision_chart_titles(sorted(precision_keys))
    precision = [
        {
            "title": titles[key],
            "y_label": "mean_abs_err",
            "unit": "mean_abs_err",
            "labels": precision_labels,
            "values": [row["precision"].get(key) for row in precision_nightly_rows],
        }
        for key in sorted(precision_keys)
    ]
    return {"labels": labels, "timestamps": timestamps, "performance": performance, "precision": precision}


def render_html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VeRL-Omni CI Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f6f6f4;
      --fg: #1f1f1f;
      --muted: #5c5c5c;
      --line: #d9d9d4;
      --card: #ffffff;
      --accent: #1f4b99;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--fg);
      background: var(--bg);
      font: 14px/1.45 "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    header {{
      padding: 24px 24px 8px;
      border-bottom: 1px solid var(--line);
      background: var(--card);
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; font-weight: 650; }}
    h2 {{ margin: 28px 24px 12px; font-size: 16px; font-weight: 650; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 24px 12px;
    }}
    .meta, .empty {{ color: var(--muted); }}
    .empty {{ padding: 0 24px; }}
    .range {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .range button {{
      border: 1px solid var(--line);
      background: var(--bg);
      color: var(--fg);
      border-radius: 999px;
      padding: 4px 12px;
      cursor: pointer;
    }}
    .range button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .nav-hierarchy {{
      margin-bottom: 12px;
    }}
    .layers {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .layer-btn {{
      border: 2px solid var(--accent);
      background: var(--card);
      color: var(--accent);
      border-radius: 10px;
      padding: 9px 22px;
      font-size: 15px;
      font-weight: 650;
      cursor: pointer;
      line-height: 1.2;
    }}
    .layer-btn.active {{
      background: var(--accent);
      color: #fff;
      box-shadow: 0 2px 8px rgba(31, 75, 153, 0.22);
    }}
    .experiment-group {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      padding: 8px 0 0 14px;
      border-left: 3px solid #c5d4ea;
    }}
    .experiment-prefix {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.02em;
      white-space: nowrap;
    }}
    .experiments {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .experiment-btn {{
      border: 1px solid #c8d3e4;
      background: #eef2f8;
      color: #4a5f7a;
      border-radius: 999px;
      padding: 4px 12px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      line-height: 1.3;
    }}
    .experiment-btn.active {{
      background: #dce6f5;
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }}
    nav {{
      display: flex;
      gap: 16px;
      padding: 12px 24px 0;
      background: var(--card);
    }}
    nav a {{ color: var(--accent); text-decoration: none; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
      padding: 0 24px 32px;
      align-items: stretch;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 12px 12px 8px;
      min-height: 340px;
      display: flex;
      flex-direction: column;
    }}
    .card h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 600;
      word-break: break-all;
      min-height: 2.6em;
    }}
    .chart-wrap {{ position: relative; flex: 1; min-height: 260px; }}
  </style>
</head>
<body>
  <header>
    <h1>VeRL-Omni CI Dashboard</h1>
    <div class="nav-hierarchy">
      <div class="layers" id="layers" role="group" aria-label="CI layer"></div>
      <div class="experiment-group">
        <span class="experiment-prefix">Experiments</span>
        <div class="experiments" id="experiments" role="group" aria-label="Experiment"></div>
      </div>
    </div>
    <div class="toolbar">
      <div class="meta" id="meta"></div>
      <div class="range" id="range" role="group" aria-label="Time range"></div>
    </div>
  </header>
  <nav>
    <a href="#performance">Performance</a>
    <a href="#precision">Precision</a>
  </nav>
  <h2 id="performance">Performance</h2>
  <div class="grid" id="performance-grid"></div>
  <h2 id="precision">Precision (mean_abs_err)</h2>
  <div class="grid" id="precision-grid"></div>
  <script>
    const DASHBOARD = {encoded};
    const LAYERS = DASHBOARD.layers || [{{id: "l3-nightly", label: "L3 Nightly CI"}}];
    const EXPERIMENTS = DASHBOARD.experiments || [];
    const RANGE_OPTIONS = [
      {{days: 7, label: "7d"}},
      {{days: 14, label: "14d"}},
      {{days: 30, label: "30d"}},
      {{days: null, label: "All"}},
    ];
    const meta = document.getElementById("meta");
    const chartInstances = [];
    let selectedDays = 30;
    let selectedLayerId = LAYERS[0] ? LAYERS[0].id : "l3-nightly";
    let selectedExperimentId = null;

    function experimentsForLayer(layerId) {{
      return EXPERIMENTS.filter((item) => (item.layer || "l3-nightly") === layerId);
    }}

    function currentExperiment() {{
      const visible = experimentsForLayer(selectedLayerId);
      return visible.find((item) => item.id === selectedExperimentId) || visible[0] || {{}};
    }}

    function parseTime(value) {{
      if (!value) return null;
      const raw = String(value);
      const iso = raw.includes("T") ? raw : raw.replace(" ", "T") + "+08:00";
      const parsed = new Date(iso);
      return Number.isNaN(parsed.getTime()) ? null : parsed.getTime();
    }}

    function rangeCutoff(days) {{
      if (!days) return null;
      const experiment = currentExperiment();
      const stamps = experiment.timestamps || [];
      const labels = experiment.labels || [];
      let latest = null;
      for (let i = stamps.length - 1; i >= 0; i--) {{
        const t = parseTime(stamps[i]) || parseTime(labels[i]);
        if (t !== null) {{ latest = t; break; }}
      }}
      if (latest === null) latest = Date.now();
      return latest - days * 24 * 60 * 60 * 1000;
    }}

    function filterChart(chart, cutoff) {{
      const experiment = currentExperiment();
      const labels = chart.labels || experiment.labels || [];
      const stamps = experiment.timestamps || [];
      if (cutoff === null) return chart;
      const keep = labels.map((label, i) => {{
        const t = parseTime(stamps[i]) || parseTime(label);
        return t === null || t >= cutoff;
      }});
      return Object.assign({{}}, chart, {{
        labels: labels.filter((_, i) => keep[i]),
        values: (chart.values || []).filter((_, i) => keep[i]),
      }});
    }}

    function setMeta(visibleCount) {{
      const experiment = currentExperiment();
      const total = (experiment.labels || []).length;
      const rangeLabel = selectedDays ? (selectedDays + "d") : "all";
      meta.textContent = [
        DASHBOARD.repo,
        DASHBOARD.branch ? ("branch " + DASHBOARD.branch) : null,
        experiment.baseline_legend || null,
        visibleCount + "/" + total + " CI runs (" + rangeLabel + ")",
        DASHBOARD.source,
      ].filter(Boolean).join(" · ");
    }}

    function clearNode(node) {{
      while (node.firstChild) node.removeChild(node.firstChild);
    }}

    function destroyCharts() {{
      while (chartInstances.length) chartInstances.pop().destroy();
    }}

    function mountCharts(sectionId, gridId, charts, color, cutoff) {{
      const grid = document.getElementById(gridId);
      clearNode(grid);
      if (!charts.length) {{
        const empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "No " + sectionId + " series found in the downloaded reports.";
        grid.appendChild(empty);
        return;
      }}
      charts.forEach((chart, index) => {{
        const filtered = filterChart(chart, cutoff);
        const card = document.createElement("article");
        card.className = "card";
        const title = document.createElement("h3");
        title.textContent = chart.title;
        const wrap = document.createElement("div");
        wrap.className = "chart-wrap";
        const canvas = document.createElement("canvas");
        canvas.id = sectionId + "-" + index;
        wrap.appendChild(canvas);
        card.appendChild(title);
        card.appendChild(wrap);
        grid.appendChild(card);

        const datasets = [{{
          label: "nightly",
          data: filtered.values,
          borderColor: color,
          backgroundColor: color,
          spanGaps: true,
          tension: 0.15,
          pointRadius: 3,
        }}];
        if (chart.baseline !== undefined) {{
          datasets.push({{
            label: chart.baseline_legend,
            data: filtered.labels.map(() => chart.baseline),
            borderColor: "#6b6b6b",
            borderDash: [6, 4],
            pointRadius: 0,
            tension: 0,
          }});
        }}
        chartInstances.push(new Chart(canvas, {{
          type: "line",
          data: {{ labels: filtered.labels, datasets }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{ display: true, position: "bottom" }},
            }},
            scales: {{
              x: {{ title: {{ display: true, text: "Date (Asia/Shanghai)" }} }},
              y: {{ title: {{ display: true, text: chart.y_label }}, beginAtZero: false }},
            }},
          }},
        }}));
      }});
    }}

    function renderAll() {{
      destroyCharts();
      const experiment = currentExperiment();
      const cutoff = rangeCutoff(selectedDays);
      const visible = filterChart(
        {{labels: experiment.labels || [], values: experiment.labels || []}},
        cutoff
      ).labels.length;
      setMeta(visible);
      mountCharts("performance", "performance-grid", experiment.performance || [], "#1f4b99", cutoff);
      mountCharts("precision", "precision-grid", experiment.precision || [], "#8a1f1f", cutoff);
    }}

    function bindLayers() {{
      const root = document.getElementById("layers");
      clearNode(root);
      LAYERS.forEach((layer) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "layer-btn";
        button.textContent = layer.label;
        button.dataset.layerId = layer.id;
        if (layer.id === selectedLayerId) button.classList.add("active");
        button.addEventListener("click", () => {{
          selectedLayerId = layer.id;
          const visible = experimentsForLayer(selectedLayerId);
          selectedExperimentId = visible[0] ? visible[0].id : null;
          root.querySelectorAll(".layer-btn").forEach((el) => el.classList.toggle("active", el === button));
          bindExperiments();
          renderAll();
        }});
        root.appendChild(button);
      }});
    }}

    function bindExperiments() {{
      const root = document.getElementById("experiments");
      clearNode(root);
      const visible = experimentsForLayer(selectedLayerId);
      if (!selectedExperimentId && visible[0]) selectedExperimentId = visible[0].id;
      visible.forEach((experiment) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "experiment-btn";
        button.textContent = experiment.label;
        button.dataset.experimentId = experiment.id;
        if (experiment.id === selectedExperimentId) button.classList.add("active");
        button.addEventListener("click", () => {{
          selectedExperimentId = experiment.id;
          root.querySelectorAll(".experiment-btn").forEach((el) => el.classList.toggle("active", el === button));
          renderAll();
        }});
        root.appendChild(button);
      }});
    }}

    function bindRange() {{
      const root = document.getElementById("range");
      RANGE_OPTIONS.forEach((option) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = option.label;
        button.dataset.days = option.days === null ? "" : String(option.days);
        if (option.days === selectedDays) button.classList.add("active");
        button.addEventListener("click", () => {{
          selectedDays = option.days;
          root.querySelectorAll("button").forEach((el) => el.classList.toggle("active", el === button));
          renderAll();
        }});
        root.appendChild(button);
      }});
    }}

    bindLayers();
    bindExperiments();
    bindRange();
    renderAll();
  </script>
</body>
</html>
"""


def _urlopen(url: str, timeout: int = 120, *, github_api: bool = False):
    headers = {"User-Agent": USER_AGENT}
    if github_api:
        headers["Accept"] = "application/vnd.github+json"
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        # verl-project blocks classic PATs (ghp_) on the artifacts API.
        if token and not token.startswith("ghp_"):
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def list_github_artifacts(repo: str, *, max_pages: int = 15) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    url: str | None = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    pages = 0
    while url and pages < max_pages:
        pages += 1
        with _urlopen(url, github_api=True) as response:
            payload = json.loads(response.read().decode("utf-8"))
            url = _next_link(response.headers.get("Link"))
        artifacts.extend(payload.get("artifacts") or [])
    return artifacts


def nightly_link_zip_url(repo: str, artifact: ArtifactRef) -> str:
    if artifact.workflow_run_id is None:
        raise ValueError(f"Artifact {artifact.name} is missing workflow_run.id")
    return f"https://nightly.link/{repo}/actions/runs/{artifact.workflow_run_id}/{artifact.name}.zip"


def download_artifact_zip(repo: str, artifact: ArtifactRef, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = nightly_link_zip_url(repo, artifact)
    with _urlopen(url) as response:
        destination.write_bytes(response.read())


def extract_report_files(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"metrics.json", "dump_compare.json"}
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            if Path(info.filename).name in wanted:
                zipped.extract(info, output_dir)


def _write_artifact_meta(directory: Path, artifact: ArtifactRef) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / "artifact.json",
        {
            "id": artifact.artifact_id,
            "name": artifact.name,
            "created_at": artifact.created_at,
            "head_branch": artifact.head_branch,
            "workflow_run_id": artifact.workflow_run_id,
        },
    )


def _materialize_artifact(repo: str, artifact: ArtifactRef, output_dir: Path, archive: Path, refresh: bool) -> None:
    if refresh and output_dir.exists():
        shutil.rmtree(output_dir)
    if refresh or not archive.exists():
        download_artifact_zip(repo, artifact, archive)
    if not output_dir.exists():
        extract_report_files(archive, output_dir)
    _write_artifact_meta(output_dir, artifact)


def download_artifacts(
    repo: str,
    cache_dir: Path,
    *,
    branch: str | None,
    max_reports: int,
    refresh: bool,
) -> tuple[Path | None, Path]:
    artifacts = list_github_artifacts(repo)
    baseline_ref, report_refs = select_artifacts(artifacts, branch=branch, max_reports=max_reports)
    reports_root = cache_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    zips_root = cache_dir / "zips"
    baseline_dir: Path | None = None

    if baseline_ref is not None:
        baseline_dir = cache_dir / "baseline" / str(baseline_ref.artifact_id)
        archive = zips_root / f"{baseline_ref.name}-{baseline_ref.artifact_id}.zip"
        _materialize_artifact(repo, baseline_ref, baseline_dir, archive, refresh)

    for report in report_refs:
        report_dir = reports_root / str(report.artifact_id)
        archive = zips_root / f"{report.name}.zip"
        _materialize_artifact(repo, report, report_dir, archive, refresh)

    return baseline_dir, reports_root


def _latest_baseline_dir(cache_dir: Path) -> Path | None:
    root = cache_dir / "baseline"
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if not dirs:
        return None

    def sort_key(path: Path) -> str:
        meta = path / "artifact.json"
        if meta.exists():
            return str(_load_json(meta).get("created_at", ""))
        return path.name

    return max(dirs, key=sort_key)


def collect_nightly_rows(reports_root: Path, case: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not reports_root.exists():
        return rows
    for artifact_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        meta_path = artifact_dir / "artifact.json"
        if not meta_path.exists():
            continue
        meta = _load_json(meta_path)
        created_at = str(meta["created_at"])
        for metrics_path in sorted(artifact_dir.rglob("metrics.json")):
            case_name = metrics_path.parent.name
            if case and case_name != case:
                continue
            dump_path = metrics_path.with_name("dump_compare.json")
            precision = extract_precision_errors(_load_json(dump_path)) if dump_path.exists() else {}
            rows.append(
                {
                    "case": case_name,
                    "run_id": meta.get("workflow_run_id"),
                    "label": format_run_label(created_at),
                    "created_at": created_at,
                    "metrics": extract_metric_means(_load_json(metrics_path)),
                    "precision": precision,
                }
            )
    rows.sort(key=lambda item: item["created_at"])
    return rows


def load_baseline_means(baseline_dir: Path | None, case: str | None) -> tuple[dict[str, float], str | None]:
    if baseline_dir is None:
        return {}, None
    meta_path = baseline_dir / "artifact.json"
    label = format_run_label(str(_load_json(meta_path)["created_at"])) if meta_path.exists() else None
    metrics_files = sorted(baseline_dir.rglob("metrics.json"))
    if case:
        metrics_files = [path for path in metrics_files if path.parent.name == case]
    if not metrics_files:
        return {}, label
    return extract_metric_means(_load_json(metrics_files[0])), label


def load_history(history_file: Path | None, history_url: str | None) -> dict[str, Any]:
    if history_file is not None and history_file.exists():
        return _load_json(history_file)
    if not history_url:
        return {}
    try:
        with _urlopen(history_url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_experiment_history(history: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    experiments = history.get("experiments")
    if isinstance(experiments, dict) and experiment_id in experiments:
        payload = experiments[experiment_id]
        return payload if isinstance(payload, dict) else {}
    if experiment_id == EXPERIMENTS[0]["id"]:
        return {
            "rows": list(history.get("rows") or []),
            "baseline": history.get("baseline") if isinstance(history.get("baseline"), dict) else {},
        }
    return {}


def rows_for_experiment(rows: list[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("case") == experiment_id]


def build_experiment_payload(
    experiment: dict[str, str],
    *,
    nightly_rows: list[dict[str, Any]],
    history: dict[str, Any],
    baseline_dir: Path | None,
) -> dict[str, Any]:
    experiment_id = experiment["id"]
    exp_history = load_experiment_history(history, experiment_id)
    merged_rows = merge_history_rows(list(exp_history.get("rows") or []), nightly_rows)
    baseline_means, baseline_label = load_baseline_means(baseline_dir, experiment_id)
    if not baseline_means and isinstance(exp_history.get("baseline"), dict):
        baseline_means = {key: float(value) for key, value in exp_history["baseline"].get("means", {}).items()}
        baseline_label = exp_history["baseline"].get("label") or baseline_label
    charts = build_chart_payload(merged_rows, baseline_means, baseline_label)
    return {
        "id": experiment_id,
        "label": experiment["label"],
        "layer": experiment.get("layer", "l3-nightly"),
        "baseline_legend": f"baseline ({baseline_label})" if baseline_label else None,
        **charts,
    }


def build_dashboard(args: argparse.Namespace) -> Path:
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    experiments = [item for item in EXPERIMENTS if not args.case or item["id"] == args.case]
    if not experiments:
        raise ValueError(f"Unknown experiment case: {args.case}")
    if args.skip_download:
        baseline_dir = _latest_baseline_dir(cache_dir)
        reports_root = cache_dir / "reports"
        if not reports_root.exists():
            raise FileNotFoundError(f"No cached reports under {reports_root}. Run without --skip-download first.")
    else:
        baseline_dir, reports_root = download_artifacts(
            args.repo,
            cache_dir,
            branch=args.branch,
            max_reports=args.max_reports,
            refresh=args.refresh,
        )

    all_nightly_rows = collect_nightly_rows(reports_root, None)
    history = load_history(args.history_file, args.history_url)
    experiment_payloads = [
        build_experiment_payload(
            experiment,
            nightly_rows=rows_for_experiment(all_nightly_rows, experiment["id"]),
            history=history,
            baseline_dir=baseline_dir,
        )
        for experiment in experiments
    ]
    generated_at = datetime.now(tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
    payload = {
        "repo": args.repo,
        "branch": args.branch,
        "source": f"GitHub Actions artifacts via nightly.link ({REPORT_ARTIFACT_PREFIX}*, {BASELINE_ARTIFACT_NAME})",
        "generated_at": generated_at,
        "layers": LAYERS,
        "experiments": experiment_payloads,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload), encoding="utf-8")
    history_path = args.history_file or (output.parent / "history.json")
    experiments_history = dict(history.get("experiments") or {})
    if not experiments_history and history.get("rows"):
        experiments_history[EXPERIMENTS[0]["id"]] = {
            "rows": list(history.get("rows") or []),
            "baseline": history.get("baseline") if isinstance(history.get("baseline"), dict) else {},
        }
    for experiment in experiments:
        experiment_id = experiment["id"]
        exp_history = load_experiment_history(history, experiment_id)
        merged_rows = merge_history_rows(
            list(exp_history.get("rows") or []),
            rows_for_experiment(all_nightly_rows, experiment_id),
        )
        baseline_means, baseline_label = load_baseline_means(baseline_dir, experiment_id)
        if not baseline_means and isinstance(exp_history.get("baseline"), dict):
            baseline_means = {key: float(value) for key, value in exp_history["baseline"].get("means", {}).items()}
            baseline_label = exp_history["baseline"].get("label") or baseline_label
        experiments_history[experiment_id] = {
            "rows": merged_rows,
            "baseline": {"label": baseline_label, "means": baseline_means},
        }
    _write_json(history_path, {"experiments": experiments_history})
    (output.parent / ".nojekyll").write_text("", encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML dashboard from L3 nightly artifacts")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "verl-project/verl-omni"))
    parser.add_argument("--branch", default="main", help="Only use artifacts from this workflow head branch")
    parser.add_argument("--output", type=Path, default=Path("outputs/nightly_dashboard/index.html"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/nightly_dashboard/cache"))
    parser.add_argument("--history-file", type=Path, default=None)
    parser.add_argument("--history-url", default=DEFAULT_HISTORY_URL)
    parser.add_argument("--max-reports", type=int, default=30)
    parser.add_argument(
        "--case",
        default=None,
        help="Build only this experiment id (default: all configured experiments)",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download artifacts even if they are cached")
    parser.add_argument("--skip-download", action="store_true", help="Rebuild the HTML page from the local cache only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = build_dashboard(args)
    print(f"[DASHBOARD] Wrote {output}")


if __name__ == "__main__":
    main()
