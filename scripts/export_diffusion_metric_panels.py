from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt


DEFAULT_TAGS = [
    "critic/rewards/mean",
    "critic/rewards/std_mean",
    "critic/rewards/group_size",
    "critic/rewards/max",
    "critic/rewards/min",
    "critic/rewards/zero_std_ratio",
    "actor/loss",
    "actor/pg_loss",
    "actor/entropy",
    "actor/approx_kl",
    "actor/logprob",
    "response_length/mean",
]


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_long_csv(path: str) -> dict[str, list[tuple[float, float]]]:
    metrics: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"step", "tag", "value"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} is not a long-form metrics CSV; expected columns {sorted(required)}")
        for row in reader:
            step = _maybe_float(row.get("step"))
            value = _maybe_float(row.get("value"))
            tag = (row.get("tag") or "").strip()
            if step is None or value is None or not tag:
                continue
            metrics[tag].append((step, value))
    return metrics


def _load_wide_csv(path: str) -> dict[str, list[tuple[float, float]]]:
    metrics: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        step_key = None
        for candidate in ("step", "_step", "global_step", "iter", "iteration"):
            if candidate in fieldnames:
                step_key = candidate
                break
        if step_key is None:
            raise ValueError(f"{path} does not contain a step column")

        for row in reader:
            step = _maybe_float(row.get(step_key))
            if step is None:
                continue
            for key, raw in row.items():
                if key == step_key:
                    continue
                value = _maybe_float(raw)
                if value is None:
                    continue
                metrics[key].append((step, value))
    return metrics


def _load_jsonl(path: str) -> dict[str, list[tuple[float, float]]]:
    metrics: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = _maybe_float(row.get("step", row.get("_step", row.get("global_step"))))
            if step is None:
                continue
            for key, value in row.items():
                if key in {"step", "_step", "global_step"}:
                    continue
                numeric = _maybe_float(value)
                if numeric is None:
                    continue
                metrics[key].append((step, numeric))
    return metrics


def load_metrics(path: str) -> dict[str, list[tuple[float, float]]]:
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return _load_jsonl(path)

    try:
        return _load_long_csv(path)
    except ValueError:
        return _load_wide_csv(path)


def _sort_metrics(metrics: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
    return {k: sorted(v, key=lambda x: x[0]) for k, v in metrics.items()}


def select_tags(all_tags: Iterable[str], requested_tags: list[str] | None) -> list[str]:
    all_tags = list(all_tags)
    if requested_tags:
        return [tag for tag in requested_tags if tag in all_tags]
    selected = [tag for tag in DEFAULT_TAGS if tag in all_tags]
    if selected:
        return selected
    return sorted(all_tags)[:12]


def plot_panels(metrics: dict[str, list[tuple[float, float]]], tags: list[str], output_path: str, title: str | None = None) -> None:
    n = len(tags)
    cols = 3 if n > 4 else 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, tag in zip(axes, tags):
        points = metrics.get(tag, [])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, linewidth=1.8)
        ax.set_title(tag, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if xs:
            ax.set_xlim(min(xs), max(xs))

    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export diffusion training metrics to TensorBoard-like panel PNGs.")
    parser.add_argument("--input", required=True, help="Input metrics file: long-form CSV, wide CSV, or JSONL.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional explicit tag list. By default, common diffusion RL tags are selected automatically.",
    )
    parser.add_argument("--title", default="Diffusion training metrics")
    args = parser.parse_args()

    metrics = _sort_metrics(load_metrics(args.input))
    tags = select_tags(metrics.keys(), args.tags)
    if not tags:
        raise SystemExit(f"No plottable metrics found in {args.input}")
    plot_panels(metrics, tags, args.output, title=args.title)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
