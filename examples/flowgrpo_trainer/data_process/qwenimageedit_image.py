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
"""Preprocess UniRL image_edit dataset into Qwen-Image-Edit parquet format.

Source format (UniRL jsonl):
    {"prompt": str, "media": [{"modality": "image", "role": "condition", "uri": "data/.../xxx.png"}],
     "metadata": {...}}

Output parquet schema matches the Qwen-Image-Edit training pipeline:
    data_source, prompt (chat messages with <image> placeholder),
    negative_prompt, ability, images ([{"bytes": PNG}]),
    reward_model, extra_info.
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

SYSTEM_PROMPT = (
    "You are an image editing assistant. Given the input image and the user's edit instruction, "
    "generate a new image that follows the instruction while preserving unrelated content."
)

DATA_SOURCE = "uniimageedit"


def _load_image_bytes(image_path: Path, image_size: int) -> bytes:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    padded.paste(image, (left, top))
    buffer = io.BytesIO()
    padded.save(buffer, format="PNG")
    return buffer.getvalue()


def _convert_split(
    input_dir: Path,
    split: str,
    max_samples: int,
    image_size: int,
    flush_every: int = 2000,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    jsonl_path = input_dir / f"{split}.jsonl"
    rows = []
    missing = 0
    processed = 0
    with open(jsonl_path) as f:
        for line in f:
            example = json.loads(line)
            instruction = str(example["prompt"])
            processed += 1

            # Resolve condition image uri (relative to input_dir)
            media = example.get("media") or []
            cond = next((m for m in media if m.get("role") == "condition"), None)
            if cond is None or "uri" not in cond:
                missing += 1
                continue
            image_uri = cond["uri"]
            image_path = input_dir / image_uri
            if not image_path.exists():
                missing += 1
                continue

            try:
                source_img = {"bytes": _load_image_bytes(image_path, image_size)}
            except Exception as e:
                print(f"[warn] failed to load {image_path}: {e}", flush=True)
                missing += 1
                continue

            if processed % 500 == 0:
                print(f"[{split}] processed={processed} kept={len(rows)} missing={missing}", flush=True)

            # Incremental flush: write partial parquet so progress is not lost
            if output_dir is not None and flush_every > 0 and len(rows) > 0 and len(rows) % flush_every == 0:
                partial = pd.DataFrame(rows)
                partial_path = output_dir / f"{split}.partial.parquet"
                partial.to_parquet(partial_path)
                print(f"[{split}] flushed {len(rows)} rows to {partial_path}", flush=True)

            meta = example.get("metadata", {}) or {}
            rows.append(
                {
                    "data_source": DATA_SOURCE,
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
                        "image": image_uri,
                        "source_img": source_img,
                        "target_img": None,
                        "global_id": meta.get("global_id"),
                        "source": meta.get("source"),
                        "edit_reward_score": meta.get("edit_reward_score"),
                    },
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if missing:
        print(f"[{split}] skipped {missing} samples with missing/invalid condition image")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UniRL image_edit dataset to Qwen-Image-Edit parquet format.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the UniRL image_edit dataset directory (contains train.jsonl, test.jsonl, data/).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/uniimageedit",
        help="Directory to save converted parquet files.",
    )
    parser.add_argument("--train_size", type=int, default=-1, help="Maximum train samples; -1 keeps all.")
    parser.add_argument("--val_size", type=int, default=-1, help="Maximum validation samples; -1 keeps all.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize and pad condition images to this square size.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _convert_split(
        input_dir=input_dir,
        split="train",
        max_samples=args.train_size,
        image_size=args.image_size,
        output_dir=output_dir,
    )
    test_df = _convert_split(
        input_dir=input_dir,
        split="test",
        max_samples=args.val_size,
        image_size=args.image_size,
        output_dir=output_dir,
    )

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(test_df)} test samples to {test_path}")
