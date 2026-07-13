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
"""Convert an image-edit JSONL dataset to the parquet format used by training."""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

SYSTEM_PROMPT = (
    "Describe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image that meets "
    "the user's requirements while maintaining consistency with the original input where "
    "appropriate."
)


def _load_image(image_path: Path, image_size: int) -> bytes:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
        canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _convert_split(input_dir: Path, split: str, max_samples: int, image_size: int) -> pd.DataFrame:
    jsonl_path = input_dir / f"{split}.jsonl"
    image_dir = input_dir / "images"
    rows = []
    with open(jsonl_path, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            instruction = str(example["prompt"])
            image_name = str(example["image"])
            image_path = image_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"condition image not found: {image_path}")

            rows.append(
                {
                    "data_source": "image_edit",
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Picture 1: <image>{instruction}"},
                    ],
                    "negative_prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Picture 1: <image> "},
                    ],
                    "ability": "image_edit",
                    "images": [{"bytes": _load_image(image_path, image_size)}],
                    "reward_model": {"style": "model", "ground_truth": instruction},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "instruction": instruction,
                        "image": image_name,
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
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _convert_split(args.input_dir.expanduser(), "train", args.train_size, args.image_size)
    validation = _convert_split(args.input_dir.expanduser(), "test", args.val_size, args.image_size)
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation samples to {args.output_dir}")


if __name__ == "__main__":
    main()
