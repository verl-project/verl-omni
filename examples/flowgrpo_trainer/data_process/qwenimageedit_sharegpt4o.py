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
"""Preprocess ShareGPT-4o-Image-Mini into Qwen-Image-Edit parquet format."""

import argparse
import io
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from verl.utils.hdfs_io import copy, makedirs

SYSTEM_PROMPT = (
    "You are an image editing assistant. Given the input image and the user's edit instruction, "
    "generate a new image that follows the instruction while preserving unrelated content."
)


def _load_image_bytes(image_dir: Path, image_name: str, image_size: int) -> bytes:
    image_path = image_dir / image_name
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    padded.paste(image, (left, top))
    buffer = io.BytesIO()
    padded.save(buffer, format="PNG")
    return buffer.getvalue()


def _convert_split(input_dir: Path, split: str, max_samples: int, image_size: int) -> pd.DataFrame:
    jsonl_path = input_dir / f"{split}.jsonl"
    image_dir = input_dir / "images"
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            example = json.loads(line)
            instruction = str(example["prompt"])
            image_name = str(example["image"])
            source_img = {"bytes": _load_image_bytes(image_dir, image_name, image_size)}
            rows.append(
                {
                    "data_source": "sharegpt4o_image_mini",
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>\n{instruction}"},
                    ],
                    "negative_prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "<image>\n "},
                    ],
                    "ability": "image_edit",
                    "images": [source_img],
                    "reward_model": {"style": "model", "ground_truth": instruction},
                    "extra_info": {
                        "split": split,
                        "index": len(rows),
                        "instruction": instruction,
                        "image": image_name,
                        "source_img": source_img,
                        "target_img": None,
                    },
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ShareGPT-4o-Image-Mini to Qwen-Image-Edit parquet format.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--input_dir",
        default="~/data/sharegpt4o_image_mini",
        help="Path to the ShareGPT-4o-Image-Mini dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/sharegpt4o_image_mini_qwen_image_edit",
        help="Directory to save converted parquet files.",
    )
    parser.add_argument("--train_size", type=int, default=-1, help="Maximum train samples; -1 keeps all samples.")
    parser.add_argument("--val_size", type=int, default=-1, help="Maximum validation samples; -1 keeps all samples.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize and pad condition images to this square size.",
    )

    args = parser.parse_args()
    input_dir = Path(os.path.expanduser(args.input_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _convert_split(
        input_dir=input_dir,
        split="train",
        max_samples=args.train_size,
        image_size=args.image_size,
    )
    test_df = _convert_split(
        input_dir=input_dir,
        split="test",
        max_samples=args.val_size,
        image_size=args.image_size,
    )

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(test_df)} test samples to {test_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(output_dir), dst=args.hdfs_dir)
