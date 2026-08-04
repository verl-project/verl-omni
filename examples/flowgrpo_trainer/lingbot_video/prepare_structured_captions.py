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

"""Convert pre-generated LingBot structured captions from JSONL to parquet.

Each JSONL record requires a ``caption`` field containing a JSON object (or a
JSON string).  Optional ``negative_caption`` and ``reward_model`` fields are
preserved.  The 27B prompt rewriter is intentionally not invoked here: run it
offline before this script, not in rollout workers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_records(path: Path, split: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            source = json.loads(line)
            caption = source.get("caption")
            if isinstance(caption, str):
                caption = json.loads(caption)
            if not isinstance(caption, dict | list):
                raise ValueError(f"{path}:{index + 1} has no structured `caption` JSON object/list.")
            negative_caption = source.get("negative_caption")
            if isinstance(negative_caption, str):
                negative_caption = json.loads(negative_caption)
            record = {
                "data_source": source.get("data_source", "lingbot_video/t2v"),
                "prompt": caption,
                "ability": "t2v",
                "reward_model": source.get("reward_model", {"style": "model", "ground_truth": caption}),
                "extra_info": {"split": split, "index": index},
            }
            if negative_caption is not None:
                if not isinstance(negative_caption, dict | list):
                    raise ValueError(f"{path}:{index + 1} has a non-JSON `negative_caption`.")
                record["negative_prompt"] = negative_caption
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, source in (("train", args.train_jsonl), ("val", args.val_jsonl)):
        records = _read_records(source, split)
        output = args.output_dir / f"{split}.parquet"
        pd.DataFrame(records).to_parquet(output)
        print(f"Wrote {len(records)} structured captions to {output}")


if __name__ == "__main__":
    main()
