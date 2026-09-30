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
"""Rewrite the ``data_source`` column of an existing parquet dataset in place.

``data_source`` is what the trainers read as ``data.reward_fn_key`` and what the
validation trainer splices into every metric key:

    <val-core|val-aux>/<data_source>/<var_name>/<metric_name>

so it has to name the **reward** that actually scores the rows -- ``ocr`` or
``pickscore`` -- and nothing else. Older Boogu converters baked the task and the
algorithm into it (``flow_grpo/ocr_edit``), which simultaneously mislabelled
diffusionnft runs as flow_grpo and hid which reward produced the number.

This tool migrates such datasets without re-running the converter, so the image
bytes are left untouched and the rewrite is cheap. It is idempotent: rows whose
``data_source`` already matches are left byte-identical, and re-running is a no-op.

Usage::

    python examples/flowgrpo_trainer/data_process/set_data_source.py \
        --data-source pickscore \
        --dataset-dir ~/data/ocr/boogu_image_edit_pickscore

Pass ``--dry-run`` to print what would change without writing.

The reward to use is not a matter of taste -- it is fixed by the row's
``reward_model.ground_truth``, because the two rewards read it differently:

* ``pickscore`` stores the *instruction* (``pickscore_reward.py`` CLIP-encodes it
  as the prompt).
* ``ocr`` stores the *bare target word* (``genrm_ocr.py`` transcribes the
  generated image and string-compares).

``--infer`` applies that rule automatically instead of taking ``--data-source``,
which is the safe choice when migrating a directory whose provenance is unclear.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REWARD_CHOICES = ("ocr", "pickscore")
SPLITS = ("train.parquet", "test.parquet")

# genrm_ocr compares the transcription against a bare word; pickscore compares the
# image against a full instruction. The prompt is only spelled this way in the edit
# task, so its presence is a reliable discriminator.
INSTRUCTION_PREFIX = "change the text to"


def infer_reward(ground_truth: str) -> str:
    """Decide which reward the row's ``ground_truth`` was written for."""
    text = ground_truth.strip()
    if not text:
        raise ValueError("empty ground_truth: cannot infer the reward")
    return "pickscore" if text.lower().startswith(INSTRUCTION_PREFIX) else "ocr"


def resolve_targets(table: pa.Table, data_source: str | None) -> list[str]:
    """Return the ``data_source`` value to write for each row."""
    if data_source is not None:
        return [data_source] * table.num_rows
    return [infer_reward(row["reward_model"]["ground_truth"]) for row in table.to_pylist()]


def rewrite_split(path: Path, data_source: str | None, dry_run: bool) -> tuple[int, int, dict[str, str]]:
    """Rewrite one parquet file. Returns (changed, total, {old_value: new_value})."""
    table = pq.read_table(path)
    if "data_source" not in table.schema.names:
        raise ValueError(f"{path} has no data_source column; schema is {table.schema.names}")

    targets = resolve_targets(table, data_source)
    current = table.column("data_source").to_pylist()
    changed = sum(1 for old, new in zip(current, targets, strict=False) if old != new)
    transitions = {old: new for old, new in zip(current, targets, strict=False) if old != new}

    if not dry_run and changed:
        index = table.schema.get_field_index("data_source")
        table = table.set_column(index, "data_source", pa.array(targets, type=table.schema.field(index).type))
        pq.write_table(table, path)
    return changed, table.num_rows, transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, action="append", help="Repeatable.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-source", choices=REWARD_CHOICES, help="Value to write into every row.")
    group.add_argument("--infer", action="store_true", help="Derive the reward from reward_model.ground_truth.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    total_changed = 0
    for raw_dir in args.dataset_dir:
        directory = raw_dir.expanduser()
        for split in SPLITS:
            path = directory / split
            if not path.is_file():
                print(f"  {path}: missing, skipped")
                continue
            changed, total, transitions = rewrite_split(path, None if args.infer else args.data_source, args.dry_run)
            total_changed += changed
            detail = ", ".join(f"{old!r} -> {new!r}" for old, new in sorted(transitions.items())) or "already correct"
            verb = "would update" if args.dry_run else "updated"
            print(f"  {path}: {verb} {changed}/{total} rows ({detail})")

    suffix = " (dry run, nothing written)" if args.dry_run else ""
    print(f"{total_changed} rows rewritten across {len(args.dataset_dir)} dataset dir(s){suffix}")


if __name__ == "__main__":
    main()
