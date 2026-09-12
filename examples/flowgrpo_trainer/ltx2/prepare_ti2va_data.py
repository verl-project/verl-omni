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
"""Convert single-image LTX-2.3 TI2VA JSONL splits to training parquet files."""

import argparse
import json
from pathlib import Path

import pandas as pd


def _image_name(example: dict) -> str:
    image = example.get("image")
    if image is not None:
        return str(image)
    images = example.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Each LTX-2.3 TI2VA row must contain exactly one image.")
    return str(images[0])


def _convert_split(input_dir: Path, split: str, max_samples: int) -> pd.DataFrame:
    rows = []
    jsonl_path = input_dir / f"{split}.jsonl"
    with jsonl_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            prompt = str(example["prompt"]).strip()
            if not prompt:
                raise ValueError(f"Empty prompt at {split} row {index}.")
            image_name = _image_name(example)
            image_path = input_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Condition image not found: {image_path}")
            rows.append(
                {
                    "data_source": "ltx2_ti2va",
                    "prompt": [{"role": "user", "content": f"<image>{prompt}"}],
                    "negative_prompt": [{"role": "user", "content": ""}],
                    "ability": "text_image_to_audio_video",
                    "images": [{"bytes": image_path.read_bytes()}],
                    "reward_model": {"style": "model", "ground_truth": prompt},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "source_image": image_name,
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _convert_split(input_dir, "train", args.train_size)
    validation = _convert_split(input_dir, "test", args.val_size)
    train.to_parquet(output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation samples to {output_dir}")


if __name__ == "__main__":
    main()
