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
"""
Create a small synthetic parquet dataset for image-edit e2e testing.

Generates data with an images column and a <image> placeholder in the prompt,
matching RLHFDataset's multimodal input convention. Uses jpeg_compressibility
reward so no external reward model is needed.

The negative prompt is selectable because the two model families genuinely
disagree on it, and each harness should mirror its own model's real converter:

- ``with-image`` (default): the negative instruction also carries
  ``Picture 1: <image>``. This matches
  ``examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py``, so it stays the
  default to keep the Qwen-Image-Edit harness faithful to its converter.
- ``text-only``: the negative instruction is empty and references no media.
  This matches ``examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py``
  (guided TI2I encodes the negative branch without the reference image,
  ``use_input_images_4_neg_instruct=False``). The Boogu harnesses select it so
  they exercise the same row shape the real recipe trains on -- a fixture that
  only ever emitted ``with-image`` rows is why the Boogu edit e2e passed while
  the real recipe's ``text-only`` parquet could not be loaded.
"""

import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

SYSTEM_PROMPT = (
    "Describe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image that meets "
    "the user's requirements while maintaining consistency with the original input where "
    "appropriate."
)

USER_PROMPTS = [
    "Change the background color to blue",
    "Add a red hat to the character",
    "Make the image look like a watercolor painting",
    "Remove the text from the image",
    "Convert the style to 3D cartoon",
    "Add sunglasses to the person",
    "Change the season from summer to winter",
    "Make it look like a pencil sketch",
]


def _create_dummy_image(width: int = 256, height: int = 256, seed: int = 0) -> bytes:
    """Create a small random RGB image and return PNG bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Negative-prompt conventions. See the module docstring for which converter each mirrors.
NEGATIVE_PROMPT_WITH_IMAGE = "with-image"
NEGATIVE_PROMPT_TEXT_ONLY = "text-only"
NEGATIVE_PROMPT_CHOICES = (NEGATIVE_PROMPT_WITH_IMAGE, NEGATIVE_PROMPT_TEXT_ONLY)


def build_rows(
    split: str,
    n: int,
    image_width: int = 256,
    image_height: int = 256,
    negative_prompt_mode: str = NEGATIVE_PROMPT_WITH_IMAGE,
):
    if negative_prompt_mode not in NEGATIVE_PROMPT_CHOICES:
        raise ValueError(f"negative_prompt_mode must be one of {NEGATIVE_PROMPT_CHOICES}, got {negative_prompt_mode!r}")
    # `text-only` must reference no media: the row carries one condition image and the negative
    # branch consumes none, which RLHFDataset permits precisely because an `<image>` placeholder
    # there is never expanded into image features. Spelling it as `"Picture 1: <image> "` keeps
    # the placeholder count equal to the image count and so silently passes a check it should
    # not -- see the module docstring.
    negative_user_content = "Picture 1: <image> " if negative_prompt_mode == NEGATIVE_PROMPT_WITH_IMAGE else ""
    rows = []
    for i in range(n):
        prompt_text = USER_PROMPTS[i % len(USER_PROMPTS)]
        condition_img_bytes = _create_dummy_image(image_width, image_height, seed=i)
        rows.append(
            {
                "data_source": "jpeg_compressibility",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Picture 1: <image>{prompt_text}"},
                ],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": negative_user_content},
                ],
                "images": [{"bytes": condition_img_bytes}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate dummy image-edit parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_image_edit"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=4, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=4, help="Number of validation samples")
    parser.add_argument("--image-width", type=int, default=256, help="Condition image width (px)")
    parser.add_argument("--image-height", type=int, default=256, help="Condition image height (px)")
    parser.add_argument(
        "--negative-prompt-mode",
        choices=NEGATIVE_PROMPT_CHOICES,
        default=NEGATIVE_PROMPT_WITH_IMAGE,
        help=(
            "Negative-prompt convention to emit. 'with-image' mirrors the Qwen-Image-Edit "
            "converter; 'text-only' mirrors the Boogu-Image-Edit converter (see module docstring)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(
        build_rows(
            "train",
            args.train_size,
            args.image_width,
            args.image_height,
            negative_prompt_mode=args.negative_prompt_mode,
        )
    )
    val_df = pd.DataFrame(
        build_rows(
            "test",
            args.val_size,
            args.image_width,
            args.image_height,
            negative_prompt_mode=args.negative_prompt_mode,
        )
    )

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
