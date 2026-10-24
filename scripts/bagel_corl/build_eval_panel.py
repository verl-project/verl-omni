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

"""Build a deterministic balanced held-out panel from JSON(L) or parquet.

Input rows require id, task, prompt, reference_subtasks (or reference), and
split=train/val/test/holdout. Alternatively supply --training-dataset explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data["rows"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def normalize(row):
    """Accept the native UniCoT parquet schema without importing training code."""
    extra = row.get("extra_info") or {}
    task = row.get("task", extra.get("task_type"))
    task = {"plan": "breakdown", "reflect": "reflection"}.get(task, task)
    reference = row.get("reference_subtasks", row.get("reference"))
    if reference is None:
        reference = (row.get("reward_model") or {}).get("ground_truth")
    identity = row.get("id", extra.get("data_id"))
    if identity is None or "prompt" not in row or task is None:
        raise ValueError("Example requires stable id/data_id, task and prompt")
    return dict(
        id=identity,
        task=task,
        prompt=row["prompt"],
        reference=reference,
        overlap_prompt=extra.get("raw_prompt", row["prompt"]),
        split=row.get("split", extra.get("split")),
    )


def build_eval_panel(
    dataset_path: Path,
    output_path: Path,
    *,
    per_task: int = 200,
    seed: int = 42,
    training_dataset_path: Path | None = None,
    render_seeds=None,
    tasks=("breakdown", "reflection"),
) -> Path:
    if per_task < 1 or len(tasks) != 2 or len(set(tasks)) != 2:
        raise ValueError("Require positive per_task and exactly two distinct task labels")
    seeds = list(render_seeds if render_seeds is not None else [seed, seed + 1])
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) for s in seeds):
        raise ValueError("Render seeds must be unique integers")
    raw_rows = read_rows(dataset_path)
    rows = [normalize(row) for row in raw_rows]
    train = (
        [normalize(row) for row in read_rows(training_dataset_path)]
        if training_dataset_path
        else [r for r in rows if r.get("split") == "train"]
    )
    if training_dataset_path is None and any(r.get("split") not in {"train", "val", "test", "holdout"} for r in rows):
        raise ValueError("Training disjointness requires explicit split fields or --training-dataset")
    train_ids = {str(r["id"]) for r in train}
    train_prompts = {digest(r["overlap_prompt"]) for r in train}
    candidates = [r for r in rows if r.get("split") != "train"]
    seen_ids, seen_prompts = set(), set()
    grouped = {task: [] for task in tasks}
    for row in candidates:
        identity, prompt_hash = str(row["id"]), digest(row["prompt"])
        overlap_hash = digest(row["overlap_prompt"])
        if identity in seen_ids or overlap_hash in seen_prompts:
            raise ValueError(f"Duplicate example ID or prompt: {identity}")
        if identity in train_ids or overlap_hash in train_prompts:
            raise ValueError(f"Training overlap: {identity}")
        seen_ids.add(identity)
        seen_prompts.add(overlap_hash)
        if row["task"] not in grouped:
            raise ValueError(f"Unknown task: {row['task']}")
        reference = row["reference"]
        grouped[row["task"]].append(
            {
                "id": identity,
                "task": row["task"],
                "prompt": row["prompt"],
                "reference": reference,
                "prompt_hash": prompt_hash,
                "reference_hash": digest(reference),
                "render_seeds": seeds,
            }
        )
    selected = []
    for task in sorted(tasks):
        if len(grouped[task]) < per_task:
            raise ValueError(f"Task {task} has {len(grouped[task])} unique examples; require {per_task}")
        selected.extend(sorted(grouped[task], key=lambda row: digest([seed, row["id"], row["prompt_hash"]]))[:per_task])
    # Semantic hashes stay stable if the source file is reordered.
    panel = {
        "schema_version": 1,
        "selection_seed": seed,
        "per_task": per_task,
        "tasks": sorted(tasks),
        "training_ids_hash": digest(sorted(train_ids)),
        "training_prompt_hash": digest(sorted(train_prompts)),
        "rows": selected,
    }
    panel["panel_hash"] = digest(panel)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path
    if output_path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        original_by_id = {str(row["id"]): raw for row, raw in zip(rows, raw_rows, strict=True)}
        parquet_rows = []
        for row in selected:
            raw = dict(original_by_id[row["id"]])
            raw["extra_info"] = dict(raw.get("extra_info") or {})
            raw["extra_info"].update(
                eval_render_seeds=seeds,
                eval_panel_hash=panel["panel_hash"],
                eval_prompt_hash=row["prompt_hash"],
                eval_reference_hash=row["reference_hash"],
            )
            parquet_rows.append(raw)
        pq.write_table(pa.Table.from_pylist(parquet_rows), output_path)
        metadata_path = output_path.with_suffix(".panel.json")
    metadata_path.write_text(json.dumps(panel, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-seeds", type=int, nargs="+")
    parser.add_argument("--tasks", nargs=2, default=["breakdown", "reflection"])
    args = parser.parse_args()
    print(
        build_eval_panel(
            args.dataset,
            args.output,
            per_task=args.per_task,
            seed=args.seed,
            training_dataset_path=args.training_dataset,
            render_seeds=args.render_seeds,
            tasks=args.tasks,
        )
    )
