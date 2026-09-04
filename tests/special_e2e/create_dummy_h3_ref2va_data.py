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
"""Create a tiny synthetic parquet dataset for MiniMax-H3 Ref2VA smoke tests.

Schema mirrors ``examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py``
for the image-only reference case:

    data_source: str          # "minimax_h3_ref2va"
    prompt: list[dict]        # chat-style, "<image>" prefixed
    ability: str              # "reference_to_audio_video"
    images: list[dict]        # [{"bytes": <PNG>}]
    reward_model: dict        # {"style": "rule", "ground_truth": ""}
    extra_info: dict          # {"split": str, "index": int, source_images}

Each row embeds a deterministic PNG reference image of at least 256 pixels per
side, which is the smallest short edge vLLM-Omni accepts for a Ref2VA
reference. The output is self-contained: callers do not need a source image
directory.

Usage:
    python tests/special_e2e/create_dummy_h3_ref2va_data.py \
        --local_save_dir ~/data/dummy_h3_ref2va \
        --train_size 4 --val_size 2
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

_USER_PROMPTS = [
    "Keep this reference character and animate them walking through a market with ambient chatter.",
    "Preserve the subject and light its scene as rain falls and thunder rumbles.",
    "Follow the reference pose while waves roll in and gulls cry overhead.",
    "Respect the reference object and show it floating with a soft hum in the air.",
    "Keep the reference palette while fireflies drift across a nocturnal garden.",
    "Match the reference framing but let a train pass with a fading whistle.",
]


def _reference_image_bytes(width: int, height: int, seed: int) -> bytes:
    """Return a deterministic RGB PNG suitable as a synthetic reference image."""
    y, x = np.indices((height, width), dtype=np.uint16)
    pixels = np.stack(
        [
            (x * 3 + seed * 29) % 256,
            (y * 5 + seed * 53) % 256,
            ((x // 2 + y // 3) + seed * 71) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _build_rows(split: str, size: int, image_width: int, image_height: int) -> list[dict]:
    rows = []
    for index in range(size):
        prompt = _USER_PROMPTS[index % len(_USER_PROMPTS)]
        image_bytes = _reference_image_bytes(image_width, image_height, index)
        rows.append(
            {
                "data_source": "minimax_h3_ref2va",
                "prompt": [{"role": "user", "content": f"<image>{prompt}"}],
                "ability": "reference_to_audio_video",
                "images": [{"bytes": image_bytes}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "source_images": [f"dummy_ref2va_{split}_{index}.png"],
                },
            }
        )
    return rows


def build_dummy_h3_ref2va_data(
    local_save_dir: str,
    *,
    train_size: int = 4,
    val_size: int = 2,
    image_width: int = 384,
    image_height: int = 256,
) -> tuple[str, str]:
    """Write train/test parquet splits and return their paths."""
    if min(train_size, val_size) < 0:
        raise ValueError("train_size and val_size must be non-negative.")
    if min(image_width, image_height) < 256:
        raise ValueError("MiniMax H3 Ref2VA reference-image dimensions must each be at least 256 pixels.")
    if not 0.4 <= image_width / image_height <= 2.5:
        raise ValueError("MiniMax H3 Ref2VA reference-image aspect ratio must be in [0.4, 2.5].")

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")
    pd.DataFrame(_build_rows("train", train_size, image_width, image_height)).to_parquet(train_path)
    pd.DataFrame(_build_rows("test", val_size, image_width, image_height)).to_parquet(test_path)
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy MiniMax-H3 Ref2VA parquet data for smoke tests.")
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/dummy_h3_ref2va"))
    parser.add_argument("--train_size", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    args = parser.parse_args()

    train_path, test_path = build_dummy_h3_ref2va_data(
        args.local_save_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    print(f"wrote {args.train_size} train samples -> {train_path}")
    print(f"wrote {args.val_size} test samples -> {test_path}")


if __name__ == "__main__":
    main()
