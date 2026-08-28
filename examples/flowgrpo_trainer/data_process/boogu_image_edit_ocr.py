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
"""Preprocess a text-editing dataset to parquet for Boogu-Image-Edit FlowGRPO.

The task is *text editing*: each sample carries a source image that already
contains rendered text, plus an instruction asking for that text to be replaced.
The reward is the existing OCR GenRM (``compute_score_ocr``), which transcribes
the rollout image and compares it to ``ground_truth`` — the text the edit was
supposed to produce. No new reward code is needed, and unlike a preference
scorer this reward *is* edit-aware for this task: the edit only scores if the
requested text actually appears.

Input layout (mirrors ``examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py``)::

    <input_dir>/train.jsonl     {"image": "0001.png", "instruction": "Change the text to \\"HELLO\\""}
    <input_dir>/test.jsonl
    <input_dir>/images/0001.png

``target_text`` may be given explicitly per row; otherwise it is taken from the
first double-quoted span of the instruction, the same convention the T2I
converter uses (``boogu_image_ocr.py::extract_solution``).

Prompt conventions differ from the Qwen-Image-Edit converter in two ways that
both silently corrupt training if copied over unchanged:

- **Both** the positive and the negative prompt use the *TI2I unified* system
  prompt. For T2I only the empty negative prompt hits that template; on the
  editing path it is the positive template too.
- The negative prompt carries **no** ``<image>`` placeholder. Upstream defaults
  to ``use_input_images_4_neg_instruct=False``, so the rollout adapter encodes
  the negative instruction text-only (``vllm_omni_rollout_adapter.py``). A
  placeholder here would never be expanded into image features.

Condition images are letterboxed onto a square canvas. Boogu-Image-Edit derives
its output resolution from the VAE-preprocessed reference (``align_res``), so
mixed source aspect ratios would produce mixed output resolutions inside a
rollout batch. A fixed square canvas pins the output to ``image_size``.
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

# Ported verbatim from BooguImagePipeline.__init__ (SYSTEM_PROMPT_4_TI2I_UNIFIED),
# the template upstream uses for the editing path. Keep in sync with
# BOOGU_SYSTEM_PROMPT_DROP in boogu_image_ocr.py.
BOOGU_SYSTEM_PROMPT_TI2I = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)


def extract_target_text(instruction: str) -> str:
    """Take the first double-quoted span, matching the T2I converter's convention."""
    parts = instruction.split('"')
    if len(parts) < 3:
        raise ValueError(
            f"cannot derive target text from instruction {instruction!r}: expected a "
            'double-quoted span (e.g. Change the text to "HELLO"), or set "target_text" on the row'
        )
    return parts[1]


def load_condition_image(image_path: Path, image_size: int) -> bytes:
    """Letterbox onto a square white canvas so every sample shares one output resolution."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
        canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def convert_split(input_dir: Path, split: str, max_samples: int, image_size: int) -> pd.DataFrame:
    jsonl_path = input_dir / f"{split}.jsonl"
    image_dir = input_dir / "images"
    rows = []
    with open(jsonl_path, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            instruction = str(example["instruction"])
            target_text = str(example.get("target_text") or extract_target_text(instruction))
            image_name = str(example["image"])
            image_path = image_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"condition image not found: {image_path}")

            rows.append(
                {
                    "data_source": "flow_grpo/ocr_edit",
                    "prompt": [
                        {"role": "system", "content": BOOGU_SYSTEM_PROMPT_TI2I},
                        {"role": "user", "content": f"Picture 1: <image>{instruction}"},
                    ],
                    # Text-only: upstream does not feed the reference image to the
                    # negative branch, so no <image> placeholder here.
                    "negative_prompt": [
                        {"role": "system", "content": BOOGU_SYSTEM_PROMPT_TI2I},
                        {"role": "user", "content": ""},
                    ],
                    "ability": "image_edit",
                    "images": [{"bytes": load_condition_image(image_path, image_size)}],
                    "reward_model": {"style": "model", "ground_truth": target_text},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "instruction": instruction,
                        "image": image_name,
                        "target_text": target_text,
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
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Square canvas edge; also the rollout output resolution (align_res). Must be a multiple of 16.",
    )
    args = parser.parse_args()

    if args.image_size % 16 or not 0 < args.image_size <= 2048:
        raise ValueError(f"--image_size must be a multiple of 16 in (0, 2048]; got {args.image_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = convert_split(args.input_dir.expanduser(), "train", args.train_size, args.image_size)
    validation = convert_split(args.input_dir.expanduser(), "test", args.val_size, args.image_size)
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation samples to {args.output_dir}")


if __name__ == "__main__":
    main()
