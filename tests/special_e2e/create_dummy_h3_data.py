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
"""Create self-contained MiniMax-H3 T2VA, FL2VA, and Ref2VA smoke data.

T2VA rows contain chat-style text prompts. FL2VA and Ref2VA rows additionally
embed deterministic RGB PNG images, so no source image directory is required.
The visual conditions are at least 256 pixels per side to satisfy vLLM-Omni's
MiniMax-H3 input constraints.

Usage::

    python tests/special_e2e/create_dummy_h3_data.py \
        --task all --local-save-dir ~/data/dummy_h3 \
        --train-size 4 --val-size 2
"""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
from PIL import Image

_T2VA_SYSTEM_PROMPT = (
    "Generate a short audio-video clip matching the following description. "
    "Focus on visible motion and matching ambient sound."
)
_T2VA_PROMPTS = [
    "A wooden wind chime swaying gently in the breeze on a sunny porch.",
    "Rain hitting a metal roof at night, faint distant thunder.",
    "A cat purring while curled up next to a crackling fireplace.",
    "Waves lapping against a sandy shore under a pastel sunset.",
    "A crowd cheering as fireworks burst above a city skyline.",
    "A single bird chirping in an early-morning forest clearing.",
    "Coffee brewing on a stove with soft jazz playing in the background.",
    "A train whistle in the distance while wheat sways in a field.",
]
_FL2VA_PROMPTS = [
    "The first-frame scene comes alive as a kite drifts across the sky, with soft wind ambience.",
    "Continue from this frame with waves rolling toward shore and gentle ocean sound.",
    "Animate the scene with a lantern glowing and a quiet nighttime atmosphere.",
    "The subject begins walking forward while light rain falls and distant thunder plays.",
    "Keep the composition while leaves move in a breeze and birds chirp nearby.",
    "Turn this first frame into a short clip with a train passing and its whistle fading away.",
]
_REF2VA_PROMPTS = [
    "Keep this reference character and animate them walking through a market with ambient chatter.",
    "Preserve the subject and light its scene as rain falls and thunder rumbles.",
    "Follow the reference pose while waves roll in and gulls cry overhead.",
    "Respect the reference object and show it floating with a soft hum in the air.",
    "Keep the reference palette while fireflies drift across a nocturnal garden.",
    "Match the reference framing but let a train pass with a fading whistle.",
]


def _write_splits(local_save_dir: str, train_rows: list[dict], val_rows: list[dict]) -> tuple[str, str]:
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "test.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(val_rows).to_parquet(val_path)
    return train_path, val_path


def _validate_sizes(train_size: int, val_size: int) -> None:
    if min(train_size, val_size) < 0:
        raise ValueError("train_size and val_size must be non-negative.")


def _image_bytes(width: int, height: int, seed: int, *, x_scale: int, y_scale: int) -> bytes:
    """Return a deterministic RGB PNG for a synthetic visual condition."""
    y, x = np.indices((height, width), dtype=np.uint16)
    pixels = np.stack(
        [
            (x * x_scale + seed * 29) % 256,
            (y * y_scale + seed * 53) % 256,
            ((x // 2 + y // 3) + seed * 71) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _build_t2va_rows(split: str, size: int) -> list[dict]:
    rows = []
    for index in range(size):
        prompt_text = _T2VA_PROMPTS[index % len(_T2VA_PROMPTS)]
        rows.append(
            {
                "data_source": "minimax_h3/dummy_t2av",
                "prompt": [
                    {"role": "system", "content": _T2VA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": _T2VA_SYSTEM_PROMPT},
                    {"role": "user", "content": " "},
                ],
                "ability": "t2av",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": index},
            }
        )
    return rows


def _build_fl2va_rows(split: str, size: int, image_width: int, image_height: int) -> list[dict]:
    rows = []
    for index in range(size):
        prompt = _FL2VA_PROMPTS[index % len(_FL2VA_PROMPTS)]
        rows.append(
            {
                "data_source": "minimax_h3/dummy_fl2va",
                "prompt": [{"role": "user", "content": f"<image>{prompt}"}],
                "ability": "video_generation",
                "images": [{"bytes": _image_bytes(image_width, image_height, index, x_scale=1, y_scale=2)}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": index, "frame_indices": [0]},
            }
        )
    return rows


def _build_ref2va_rows(split: str, size: int, image_width: int, image_height: int) -> list[dict]:
    rows = []
    for index in range(size):
        prompt = _REF2VA_PROMPTS[index % len(_REF2VA_PROMPTS)]
        rows.append(
            {
                "data_source": "minimax_h3_ref2va",
                "prompt": [{"role": "user", "content": f"<image>{prompt}"}],
                "ability": "reference_to_audio_video",
                "images": [{"bytes": _image_bytes(image_width, image_height, index, x_scale=3, y_scale=5)}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "source_images": [f"dummy_ref2va_{split}_{index}.png"],
                },
            }
        )
    return rows


def build_dummy_h3_t2va_data(
    local_save_dir: str,
    *,
    train_size: int = 4,
    val_size: int = 2,
) -> tuple[str, str]:
    """Write T2VA train/test parquet splits and return their paths."""
    _validate_sizes(train_size, val_size)
    return _write_splits(
        local_save_dir,
        _build_t2va_rows("train", train_size),
        _build_t2va_rows("test", val_size),
    )


def build_dummy_h3_fl2va_data(
    local_save_dir: str,
    *,
    train_size: int = 4,
    val_size: int = 2,
    image_width: int = 288,
    image_height: int = 256,
) -> tuple[str, str]:
    """Write FL2VA train/test parquet splits and return their paths."""
    _validate_sizes(train_size, val_size)
    if min(image_width, image_height) < 256:
        raise ValueError("MiniMax H3 FL2VA condition-image dimensions must each be at least 256 pixels.")
    return _write_splits(
        local_save_dir,
        _build_fl2va_rows("train", train_size, image_width, image_height),
        _build_fl2va_rows("test", val_size, image_width, image_height),
    )


def build_dummy_h3_ref2va_data(
    local_save_dir: str,
    *,
    train_size: int = 4,
    val_size: int = 2,
    image_width: int = 384,
    image_height: int = 256,
) -> tuple[str, str]:
    """Write Ref2VA train/test parquet splits and return their paths."""
    _validate_sizes(train_size, val_size)
    if min(image_width, image_height) < 256:
        raise ValueError("MiniMax H3 Ref2VA reference-image dimensions must each be at least 256 pixels.")
    if not 0.4 <= image_width / image_height <= 2.5:
        raise ValueError("MiniMax H3 Ref2VA reference-image aspect ratio must be in [0.4, 2.5].")
    return _write_splits(
        local_save_dir,
        _build_ref2va_rows("train", train_size, image_width, image_height),
        _build_ref2va_rows("test", val_size, image_width, image_height),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy MiniMax-H3 parquet data for smoke tests.")
    parser.add_argument("--task", choices=("all", "t2va", "fl2va", "ref2va"), default="all")
    parser.add_argument("--local-save-dir", default=os.path.expanduser("~/data/dummy_h3"))
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=288)
    parser.add_argument("--image-height", type=int, default=256)
    args = parser.parse_args()

    builders: dict[str, Callable[..., tuple[str, str]]] = {
        "t2va": build_dummy_h3_t2va_data,
        "fl2va": build_dummy_h3_fl2va_data,
        "ref2va": build_dummy_h3_ref2va_data,
    }
    tasks = tuple(builders) if args.task == "all" else (args.task,)
    for task in tasks:
        output_dir = os.path.join(args.local_save_dir, task) if args.task == "all" else args.local_save_dir
        kwargs = {"train_size": args.train_size, "val_size": args.val_size}
        if task != "t2va":
            kwargs.update(image_width=args.image_width, image_height=args.image_height)
        train_path, val_path = builders[task](output_dir, **kwargs)
        print(f"wrote {args.train_size} {task} train samples -> {train_path}")
        print(f"wrote {args.val_size} {task} val samples -> {val_path}")


if __name__ == "__main__":
    main()
