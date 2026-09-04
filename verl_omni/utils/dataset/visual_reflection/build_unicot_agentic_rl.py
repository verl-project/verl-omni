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
"""Build UniCoT agentic RL train/validation parquet.

This is the GRPO application of the UniCoT parsers, not a generic dataset
loader. Invoke as::

    python -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl

The builder combines:

- UniCoT-Self-Reflection-6K as ``reflect`` rows carrying reference reflection
  states and continue/stop transitions; and
- UniCoT-Breakdown-3K as ``plan`` rows carrying reference subtasks, with
  ``No breakdown needed.`` records normalized to single-image ``reflect`` rows.

Source annotations are reward ground truth only and never appear in the model
prompt. Validation is metadata-only; the builder does not require
``images.zip`` or read image pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from verl_omni.utils.dataset.visual_reflection import VisualReflectionDataError
from verl_omni.utils.dataset.visual_reflection.contracts import derive_prompt_source_dedup_key
from verl_omni.utils.dataset.visual_reflection.partition import assign_source_splits
from verl_omni.utils.dataset.visual_reflection.unicot import (
    UNICOT_DATASET_ID,
    parse_unicot_record,
    unicot_converter_config,
)
from verl_omni.utils.dataset.visual_reflection.unicot_breakdown import (
    UNICOT_BREAKDOWN_DATASET_ID,
    breakdown_converter_config,
    parse_unicot_breakdown_record,
)

REFLECT_ABILITY = "agentic_generate_self_reflect"
PLAN_ABILITY = "agentic_plan_generate"
REFLECT_DATA_SOURCE = "unicot_reflection"
BREAKDOWN_DATA_SOURCE = "unicot_breakdown"
REWARD_DIMS = ("reflect", "plan", "format", "tool", "result")
# Public alias retained for reward/dataset consumers.
DIMS = REWARD_DIMS
MANIFEST_ID = "agentic_rl_unicot_v1"

REFLECT_SYSTEM_PROMPT = """You are a visual creation agent with two tools:
1) generate_image — create an image from a complete diffusion prompt
2) judge_image — inspect the last generated image and return structured feedback

Protocol:
1. Call generate_image with a complete prompt for the user's request.
2. Call judge_image on the last generated image.
3. Reflect briefly on the feedback. If the image needs improvement, rewrite the
   diffusion prompt and repeat. If it is good enough, finish with Done.

Always generate before judging, judge before deciding, and use no other tools."""

PLAN_SYSTEM_PROMPT = """You are a visual creation agent with two tools:
1) generate_image — create an image from a complete diffusion prompt
2) judge_image — inspect the last generated image and return structured feedback

Protocol:
1. Write a short numbered plan of at most three complete subtask image prompts.
2. Call generate_image once per planned subtask, in order.
3. After the final image, call judge_image on that image.
4. Reflect briefly on the feedback and finish with Done.

Do not judge between subtasks or generate more images than the plan lists."""

_BREVITY_SUFFIX = (
    " Keep any private thinking to one short paragraph; do not repeat the request, "
    "and keep the final reflection concise (≤4 sentences)."
)


class _TextOnlyImageResolver:
    """Validate reflection structure without materializing source image archives."""

    def __call__(
        self,
        value: Any,
        *,
        field: str = "",
        index: int = 0,
        source_record_id: str | None = None,
    ) -> dict[str, str]:
        del field, index
        uri = str(value).strip() if value is not None else ""
        if not uri:
            uri = f"<no-image>:{source_record_id or 'unknown'}"
        # A constant valid digest preserves transition-structure validation while
        # deliberately skipping pixel/hash IO in this text-only builder.
        return {"uri": uri, "sha256": "0" * 64}


def _with_brevity(prompt: str) -> str:
    return f"{prompt.rstrip()}{_BREVITY_SUFFIX}"


def _env_weight(dim: str) -> float:
    env_name = f"RPCO_W_{dim.upper()}"
    raw = os.environ.get(env_name, "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{env_name} must be a float, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return value


def _weights() -> dict[str, float]:
    return {f"w_{dim}": _env_weight(dim) for dim in REWARD_DIMS}


def _load_metadata(dataset_dir: str, dataset_id: str) -> list[dict[str, Any]]:
    root = Path(dataset_dir).expanduser()
    snapshots = sorted((root / "snapshots").glob("*/"))
    for snapshot in reversed(snapshots):
        metadata_path = snapshot / "metadata.json"
        if not metadata_path.is_file():
            continue
        data = json.loads(metadata_path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{dataset_id}: metadata.json must be a JSON list, got {type(data).__name__}")
        return data
    raise FileNotFoundError(f"{dataset_id}: no snapshot with metadata.json under {root}")


def _rejection(record: dict[str, Any], error: VisualReflectionDataError) -> dict[str, Any]:
    return {
        "data_id": str(record.get("data_id") or ""),
        "reason": error.reason.value,
        "field": error.field,
    }


def _split_record(*, dataset_id: str, data_id: str, prompt: str) -> dict[str, str]:
    return {
        "source_dataset": dataset_id,
        "source_record_id": data_id,
        "pipeline_variant": "prompt_k_turn",
        "prompt": prompt,
        "dedup_key": derive_prompt_source_dedup_key(prompt),
    }


def _parse_reflection_rows(metadata: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    resolver = _TextOnlyImageResolver()
    weights = _weights()
    for source in metadata:
        try:
            trajectory = parse_unicot_record(
                source,
                manifest_id=MANIFEST_ID,
                image_resolver=resolver,
            )
        except VisualReflectionDataError as error:
            rejections.append(_rejection(source, error))
            continue
        data_id = trajectory["source_record_id"]
        prompt = trajectory["prompt"]
        expected_num_images = len(trajectory["steps"])
        rows.append(
            {
                "data_id": data_id,
                "task_type": "reflect",
                "prompt_text": prompt,
                "expected_num_images": expected_num_images,
                "ground_truth": {
                    "user_request": prompt,
                    "task_type": "reflect",
                    "expected_num_images": expected_num_images,
                    "reference_steps": trajectory["steps"],
                    **weights,
                },
                "source_dataset": UNICOT_DATASET_ID,
                "split_record": _split_record(
                    dataset_id=UNICOT_DATASET_ID,
                    data_id=data_id,
                    prompt=prompt,
                ),
            }
        )
    return rows, rejections


def _parse_breakdown_rows(metadata: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    weights = _weights()
    for source in metadata:
        try:
            parsed = parse_unicot_breakdown_record(source, manifest_id=MANIFEST_ID)
        except VisualReflectionDataError as error:
            rejections.append(_rejection(source, error))
            continue
        ground_truth: dict[str, Any] = {
            "user_request": parsed.prompt,
            "task_type": parsed.task_type,
            "expected_num_images": parsed.expected_num_images,
            "plan_expected": parsed.plan_expected,
            **weights,
        }
        if parsed.plan_expected:
            ground_truth["reference_subtasks"] = list(parsed.subtasks)
        rows.append(
            {
                "data_id": parsed.data_id,
                "task_type": parsed.task_type,
                "prompt_text": parsed.prompt,
                "expected_num_images": parsed.expected_num_images,
                "ground_truth": ground_truth,
                "source_dataset": UNICOT_BREAKDOWN_DATASET_ID,
                "split_record": _split_record(
                    dataset_id=UNICOT_BREAKDOWN_DATASET_ID,
                    data_id=parsed.data_id,
                    prompt=parsed.prompt,
                ),
            }
        )
    return rows, rejections


def _assign_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[dict[tuple[str, str], str], str | None]:
    assignments = assign_source_splits(
        [row["split_record"] for row in rows],
        ratios={"train": 1.0 - val_ratio, "validation": val_ratio, "test": 0.0},
        seed=seed,
    )
    split_by_identity = {
        identity: "val" if assignment["split"] == "validation" else "train"
        for identity, assignment in assignments.items()
    }
    partition_ids = {assignment["partition_id"] for assignment in assignments.values()}
    partition_id = next(iter(partition_ids)) if partition_ids else None
    return split_by_identity, partition_id


def _select_rows(
    rows: list[dict[str, Any]],
    split_by_identity: dict[tuple[str, str], str],
    *,
    split: str,
    size: int | None,
    mix_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    split_rows = [row for row in rows if split_by_identity[(row["source_dataset"], row["data_id"])] == split]
    split_rows.sort(key=lambda row: (row["source_dataset"], row["data_id"]))
    if size is None:
        return split_rows

    reflect = [row for row in split_rows if row["task_type"] == "reflect"]
    plan = [row for row in split_rows if row["task_type"] == "plan"]
    reflect_count = min(len(reflect), round(size * mix_ratio))
    plan_count = min(len(plan), size - reflect_count)
    rng = random.Random(seed)
    selected = rng.sample(reflect, reflect_count) + rng.sample(plan, plan_count)
    selected.sort(key=lambda row: (row["source_dataset"], row["data_id"]))
    return selected


def _build_parquet_row(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    prompt = row["prompt_text"]
    is_plan = row["task_type"] == "plan"
    return {
        "data_source": (REFLECT_DATA_SOURCE if row["source_dataset"] == UNICOT_DATASET_ID else BREAKDOWN_DATA_SOURCE),
        "prompt": [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT if is_plan else REFLECT_SYSTEM_PROMPT},
            {"role": "user", "content": _with_brevity(prompt)},
        ],
        "ability": PLAN_ABILITY if is_plan else REFLECT_ABILITY,
        "reward_model": {"style": "rule", "ground_truth": dict(row["ground_truth"])},
        "extra_info": {
            "split": split,
            "index": index,
            "data_id": row["data_id"],
            "task_type": row["task_type"],
            "expected_num_images": row["expected_num_images"],
            "raw_prompt": prompt,
            "unicot_source": row["source_dataset"],
            "plan_expected": bool(row["ground_truth"].get("plan_expected", False)),
        },
    }


def build_rows(
    rows: list[dict[str, Any]],
    split_by_identity: dict[tuple[str, str], str],
    *,
    split: str,
    size: int | None,
    mix_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = _select_rows(
        rows,
        split_by_identity,
        split=split,
        size=size,
        mix_ratio=mix_ratio,
        seed=seed,
    )
    counts = {"reflect": 0, "plan": 0}
    parquet_rows = []
    for index, row in enumerate(selected):
        counts[row["task_type"]] += 1
        parquet_rows.append(_build_parquet_row(row, split=split, index=index))
    return parquet_rows, counts


def main_cli(
    *,
    reflection_dir: str,
    breakdown_dir: str,
    local_save_dir: str,
    train_size: int | None,
    val_size: int | None,
    mix_ratio: float,
    seed: int,
    val_ratio: float,
) -> None:
    """Build train/validation parquet files; also serves as the test entry point."""
    if not reflection_dir and not breakdown_dir:
        raise SystemExit("provide at least one of breakdown_dir / reflection_dir")
    if not 0.0 <= mix_ratio <= 1.0:
        raise SystemExit("mix_ratio must be in [0, 1]")
    if not 0.0 < val_ratio < 1.0:
        raise SystemExit("val_ratio must be in (0, 1)")
    if any(size is not None and size < 0 for size in (train_size, val_size)):
        raise SystemExit("train_size and val_size must be non-negative")

    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    if reflection_dir:
        parsed, rejected = _parse_reflection_rows(_load_metadata(reflection_dir, UNICOT_DATASET_ID))
        rows.extend(parsed)
        rejections.extend(rejected)
    if breakdown_dir:
        parsed, rejected = _parse_breakdown_rows(_load_metadata(breakdown_dir, UNICOT_BREAKDOWN_DATASET_ID))
        rows.extend(parsed)
        rejections.extend(rejected)

    split_by_identity, partition_id = _assign_splits(rows, seed=seed, val_ratio=val_ratio)
    output_dir = Path(local_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "manifest_id": MANIFEST_ID,
        "partition_id": partition_id,
        "seed": seed,
        "val_ratio": val_ratio,
        "source_configs": {
            "reflection": unicot_converter_config(),
            "breakdown": breakdown_converter_config(),
        },
        "rejections": rejections,
        "rejection_count": len(rejections),
        "splits": {},
    }
    for split, size in (("train", train_size), ("val", val_size)):
        parquet_rows, counts = build_rows(
            rows,
            split_by_identity,
            split=split,
            size=size,
            mix_ratio=mix_ratio,
            seed=seed,
        )
        dataframe = pd.DataFrame(
            parquet_rows,
            columns=["data_source", "prompt", "ability", "reward_model", "extra_info"],
        )
        destination = output_dir / f"{split}.parquet"
        dataframe.to_parquet(destination)
        report["splits"][split] = {**counts, "total": len(dataframe)}
        print(f"[INFO] {split}: wrote {len(dataframe)} rows ({counts}) to {destination}")
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UniCoT agentic RL parquet files")
    parser.add_argument("--breakdown_dir", default=os.environ.get("UNICOT_BREAKDOWN_DIR", ""))
    parser.add_argument("--reflection_dir", default=os.environ.get("UNICOT_REFLECTION_DIR", ""))
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/agentic_unicot"))
    parser.add_argument("--train_size", type=int, default=None, help="None uses the full train split")
    parser.add_argument("--val_size", type=int, default=None, help="None uses the full validation split")
    parser.add_argument(
        "--mix_ratio",
        "--reflect_ratio",
        dest="mix_ratio",
        type=float,
        default=float(os.environ.get("UNICOT_MIX_RATIO", "0.5")),
        help="Reflect fraction used only when a split size cap is supplied",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=float(os.environ.get("UNICOT_VAL_RATIO", "0.05")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("UNICOT_SPLIT_SEED", "42")))
    args = parser.parse_args()
    main_cli(
        reflection_dir=args.reflection_dir,
        breakdown_dir=args.breakdown_dir,
        local_save_dir=args.local_save_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        mix_ratio=args.mix_ratio,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
