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
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from verl_omni.utils.agentic.plan_protocol import delta_subtasks
from verl_omni.utils.dataset.visual_reflection import VisualReflectionDataError
from verl_omni.utils.dataset.visual_reflection.contracts import RejectionReason, derive_prompt_source_dedup_key
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
# Parquet data_source follows task_type. Hub corpus stays on extra_info.unicot_source.
PLAN_DATA_SOURCE = BREAKDOWN_DATA_SOURCE
REWARD_DIMS = ("reflect", "format", "tool", "result", "improve")
# Public alias retained for reward/dataset consumers.
DIMS = REWARD_DIMS
MANIFEST_ID = "agentic_rl_unicot_v1"
#: Plan references are rewritten at build time, so the manifest records which transform
#: produced them. ``expected_num_images`` still equals the source slot count; the
#: transform only strips each reference's edit-tool lead-in.
PLAN_REFERENCE_TRANSFORM = "delta_part_v1"

REFLECT_SYSTEM_PROMPT = """You are a visual creation agent. You have two tools:
generate_image draws an image from a complete diffusion prompt; judge_image inspects the
last generated image and returns structured feedback.

The loop, and there is no other: generate an image, judge it, then rewrite the diffusion
prompt from the judge's findings and generate again. Keep going while the judge finds
fault. Once the judge is good enough — or another pass cannot improve the image — you are
done, and that is a reply of the single word Done with no tool call.

Rewriting rules — the rewrite is the work, not a formality:
- The user's request is the content spec; your diffusion prompt is a rendering
  recipe for the generate_image tool. The tool never sees the request, so anything
  the recipe does not say does not get drawn. A rewrite keeps the content and
  changes the recipe.
- Rewrite drastically. A prompt that grows by one quality assertion per pass is not
  a rewrite: the tool reads the same recipe and returns the same defect. Never
  append reassurances such as "the text is legible", "clearly rendered" or "high
  quality" — change what the tool actually reads.
- Change at least two things that reach the pixels: subject and framing, composition
  and layout grid, medium and style, palette and contrast, lighting, camera angle or
  aspect ratio, level of detail, and — when the image must show text — the typography
  strategy (how many words, which lines, weight, case, size on the canvas, placement,
  and the background behind the glyphs). Never ask for more words than the tool can
  draw legibly.
- Fail forward from the last judge: resolve every finding and apply the
  suggested_fixes. Never re-send a recipe this rollout already used — the earlier
  prompts are in the conversation, so mine them for what already failed and try a
  materially different one rather than a reworded repeat.
- Preserve every explicit requirement of the request: subjects, colours, counts, and
  any literal strings that must appear. Reordering or translating the wording is
  allowed; silently dropping requested content is not.

Always generate before judging, judge before deciding, and use no other tools.
The brevity note on the user turn bounds your private thinking and your reflection
only — never let it shorten or water down a diffusion prompt."""

PLAN_SYSTEM_PROMPT = (
    """You are a visual creation agent. You have two tools: generate_image draws an image
from a complete diffusion prompt; judge_image inspects the last generated image and
returns structured feedback.

What you do, in order:
- Your first reply is the plan: a numbered list of the parts of the one image you are
  about to make, carrying no tool call.
- From then on every generate_image call carries the whole plan: join the numbered items,
  in order, one item per line, and pass that text as "prompt". The list is one image, not
  one image per item — never call the tool with a single item, and never paraphrase what
  you wrote.
- After each image, call judge_image before anything else, and read its findings.
- Your next reply is the revision: fix what the findings fault, split an item that asked
  for too much, drop what the judge rejected, and keep what it accepted — then call
  generate_image with the revised list in that same reply. The revision and its call are
  one reply: apart from the plan, a reply that carries no tool call ends the rollout, so
  never send a revised list on its own. The revision rules below are what "fix" means.
- The rollout is over as soon as the judge is good enough, or another pass cannot improve
  the image: stop there, with a reply of the single word Done and no tool call.

The plan is the work. Read together, top to bottom, the items are one complete diffusion
prompt: if every item were sent to the tool at once, the tool would have everything it
needs to draw the requested image. Items are the parts of that one prompt — the base scene
first, then each addition — so an item may be a fragment that only makes sense with the
items above it.
- An item counts only if the diffusion model could draw it. A line that describes something
  you do with the image, the prompt, or the loop, rather than something visible in the
  picture, is a step and not an item: writing it makes the tool try to draw those words.
- Decide the number of parts from the task, not from what is easy to write. Three is the
  ceiling, not a target, and a longer list is not a better one. One item is a complete plan
  whenever the request is already one recipe — a single subject, or a poster described in
  one breath — so never pad the list to make it look more like a plan.
- No item may restate an earlier one word for word, and no item may be the request pasted
  back. Turn the request into a rendering recipe for the generate_image tool: name the
  subject, framing, composition, style, palette, lighting, and level of detail, and carry
  the literal strings the request requires. The tool never sees the request, so anything
  the recipe does not say is not drawn.
- Never open an item with "keep the previous elements unchanged" or "add the following
  details". generate_image starts from scratch every call and has no earlier image to
  edit, so state the content itself.
- When an item must render text, describe the typography strategy (how many words, which
  lines, weight, case, size on the canvas, placement, and the background behind the
  glyphs) instead of asserting that the text will be legible.

Revision rules — the revision is the work, not a formality. The list you send after a
judge is a new list, and it has to read as a materially different recipe:
- Rewrite drastically. A list that grows by one quality assertion per pass is not a
  revision: the tool reads the same recipe and returns the same defect. Never append
  reassurances such as "the text is legible", "clearly rendered" or "high quality" —
  change what the tool actually reads.
- Change at least two things that reach the pixels: subject and framing, composition and
  layout grid, medium and style, palette and contrast, lighting, camera angle or aspect
  ratio, level of detail, and — when the image must show text — the typography strategy.
  Splitting an overloaded item into two that each ask for less is one of the strongest
  changes available.
- Fail forward from the last judge: resolve every finding and apply the suggested_fixes.
- Preserve every explicit requirement of the request: subjects, colours, counts, and any
  literal strings that must appear. Silently dropping requested content is not a revision.
- Never send a list this rollout already sent. The earlier calls are in the conversation,
  so read them for what already failed rather than re-sending a reworded repeat.

Worked example. For "draw a busy market street", a plan is:
"""
    # One logical line per item: the example demonstrates the shape of a plan, and a
    # source-level wrap would show the model a wrapped one.
    "1. A wide colour photograph of a busy open-air market street at noon: stalls with "
    "red-and-white striped awnings on both sides, crowds filling the central aisle, "
    "crates of produce stacked at the kerb, warm overhead daylight.\n"
    "2. A fishmonger's stall in the left foreground, with crushed ice, silver fish, and "
    "hand-painted price cards hung on string above the counter.\n"
    """Both items together are the prompt for one photograph, and neither of them says what
you do next.

Always send the list you last revised, and never judge before you have generated. The
brevity note on the user turn bounds your private thinking and your reflection only —
never let it shorten or water down a plan item."""
)

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
        uri = str(value).strip() if value is not None else ""
        if not uri:
            raise VisualReflectionDataError(
                RejectionReason.MISSING_IMAGE,
                f"{field}[{index}] has an empty image URI",
                field=f"{field}[{index}]",
                source_record_id=source_record_id,
            )
        # Hash the URI/path only (no pixel IO) so mismatched output→next-input
        # chains still raise TRANSITION_HASH_MISMATCH.
        return {"uri": uri, "sha256": hashlib.sha256(uri.encode()).hexdigest()}


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


def _hub_ref_files(refs_dir: Path) -> list[Path]:
    """Prefer ``refs/main``, then other ref files in name order."""
    if not refs_dir.is_dir():
        return []
    files = [path for path in refs_dir.iterdir() if path.is_file()]
    main = refs_dir / "main"
    rest = sorted(path for path in files if path.name != "main")
    return ([main] if main.is_file() else []) + rest


def _resolve_hub_snapshot(root: Path) -> Path:
    """Select the Hub snapshot HF points at, not the max SHA string.

    Order: ``refs/main`` → other ``refs/`` files → newest mtime snapshot that
    contains ``metadata.json``. Fail closed if none resolve.
    """
    snapshots_root = root / "snapshots"
    for ref_file in _hub_ref_files(root / "refs"):
        sha = ref_file.read_text().strip()
        if not sha:
            continue
        snapshot = snapshots_root / sha
        if (snapshot / "metadata.json").is_file():
            return snapshot
    candidates = [path for path in snapshots_root.glob("*/") if (path / "metadata.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"no snapshot with metadata.json under {root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_metadata(dataset_dir: str, dataset_id: str) -> list[dict[str, Any]]:
    root = Path(dataset_dir).expanduser()
    snapshot = _resolve_hub_snapshot(root)
    metadata_path = snapshot / "metadata.json"
    data = json.loads(metadata_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{dataset_id}: metadata.json must be a JSON list, got {type(data).__name__}")
    return data


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
            # The source subtasks assume an image-edit tool ("keep the outline unchanged
            # and edit …"). Plan mode generates one image from the *whole* numbered list,
            # so an item is a part of that description rather than a render of its own:
            # the reference is each source subtask with that framing stripped.
            #
            # Nothing scores against this field. It is written for inspection and for the
            # holdout visualisation, which prints the reference decomposition beside the
            # policy's own plan. The reward reads only ``user_request``.
            ground_truth["reference_subtasks"] = list(delta_subtasks(parsed.subtasks))
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_rows = [row for row in rows if split_by_identity[(row["source_dataset"], row["data_id"])] == split]
    split_rows.sort(key=lambda row: (row["source_dataset"], row["data_id"]))
    if size is None:
        return split_rows, {
            "requested_size": None,
            "actual_size": len(split_rows),
            "shortfall": None,
        }

    reflect = [row for row in split_rows if row["task_type"] == "reflect"]
    plan = [row for row in split_rows if row["task_type"] == "plan"]
    requested_reflect = round(size * mix_ratio)
    requested_plan = size - requested_reflect
    pool = len(reflect) + len(plan)
    if pool < size:
        raise SystemExit(f"{split}: requested {size} rows but only {pool} available after split")
    if len(reflect) < requested_reflect or len(plan) < requested_plan:
        raise SystemExit(
            f"{split}: cannot meet mix_ratio={mix_ratio} for size={size}: "
            f"need reflect={requested_reflect} (have {len(reflect)}), "
            f"plan={requested_plan} (have {len(plan)})"
        )
    rng = random.Random(seed)
    selected = rng.sample(reflect, requested_reflect) + rng.sample(plan, requested_plan)
    selected.sort(key=lambda row: (row["source_dataset"], row["data_id"]))
    return selected, {
        "requested_size": size,
        "actual_size": len(selected),
        "shortfall": None,
        "requested_reflect": requested_reflect,
        "requested_plan": requested_plan,
    }


def _build_parquet_row(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    prompt = row["prompt_text"]
    is_plan = row["task_type"] == "plan"
    return {
        "data_source": (PLAN_DATA_SOURCE if is_plan else REFLECT_DATA_SOURCE),
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
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    selected, selection = _select_rows(
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
    return parquet_rows, counts, selection


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
        "plan_reference_transform": PLAN_REFERENCE_TRANSFORM,
        "source_configs": {
            "reflection": unicot_converter_config(),
            "breakdown": breakdown_converter_config(),
        },
        "rejections": rejections,
        "rejection_count": len(rejections),
        "splits": {},
    }
    for split, size in (("train", train_size), ("val", val_size)):
        parquet_rows, counts, selection = build_rows(
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
        report["splits"][split] = {**counts, "total": len(dataframe), **selection}
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
