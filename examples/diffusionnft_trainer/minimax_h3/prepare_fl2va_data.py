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
"""Convert MiniMax H3 FL2VA JSONL splits to verl-omni parquet files."""

import argparse
import json
from pathlib import Path

import pandas as pd

_FRAME_MODES = {"first": ([0], 1), "last": ([-1], 1), "first_last": ([0, -1], 2)}


def _image_names(example: dict, expected: int) -> list[str]:
    names = example.get("images")
    if names is None:
        names = [example["first_image"]]
        if expected == 2:
            names.append(example["last_image"])
    names = [str(name) for name in names]
    if len(names) != expected:
        raise ValueError(f"Expected {expected} condition image(s), got {names}.")
    return names


def _convert_split(input_dir: Path, split: str, frame_mode: str, max_samples: int) -> pd.DataFrame:
    frame_indices, expected_images = _FRAME_MODES[frame_mode]
    rows = []
    with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            prompt = str(example["prompt"]).strip()
            if not prompt:
                raise ValueError(f"Empty prompt at {split} row {index}.")
            image_paths = [input_dir / name for name in _image_names(example, expected_images)]
            missing = [str(path) for path in image_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Condition image(s) not found: {missing}")
            rows.append(
                {
                    "data_source": "minimax_h3_fl2va",
                    "prompt": [{"role": "user", "content": "<image>" * expected_images + prompt}],
                    "ability": "frame_to_audio_video",
                    "images": [{"bytes": path.read_bytes()} for path in image_paths],
                    "reward_model": {"style": "model", "ground_truth": prompt},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "frame_indices": frame_indices,
                        "source_images": [str(path.relative_to(input_dir)) for path in image_paths],
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_mode", choices=sorted(_FRAME_MODES), default="first_last")
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _convert_split(args.input_dir.expanduser(), "train", args.frame_mode, args.train_size)
    validation = _convert_split(args.input_dir.expanduser(), "test", args.frame_mode, args.val_size)
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
