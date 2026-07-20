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
Create a small synthetic parquet dataset for Qwen-Image-Edit e2e testing.

Generates data with an images column and a <image> placeholder in the prompt,
matching RLHFDataset's multimodal input convention. Uses jpeg_compressibility
reward so no external reward model is needed.
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


def build_rows(split: str, n: int, image_width: int = 256, image_height: int = 256):
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
                    {"role": "user", "content": "Picture 1: <image> "},
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
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(build_rows("train", args.train_size, args.image_width, args.image_height))
    val_df = pd.DataFrame(build_rows("test", args.val_size, args.image_width, args.image_height))

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
