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
"""Create a small synthetic parquet dataset for LingBot Dense T2V e2e testing.

LingBot-Video's DiT consumes structured JSON captions rather than chat
messages, so the ``prompt`` column holds a caption dict (matching
``prepare_structured_captions.py``); the ``lingbot_dense_t2v_agent`` loop reads
it back from ``raw_prompt`` and does its own tokenization.  There is no images
column (this is text-to-video).  ``jpeg_compressibility`` is used as a
self-contained rule reward so no external reward model server is needed.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

# Tiny structured captions in the official LingBot shape (a mapping keyed by
# ``comprehensive_description`` plus a couple of auxiliary fields).  Content is
# irrelevant to the smoke test -- only the schema and the denoise/decode/reward
# path are exercised -- so these stay short.
_CAPTIONS = [
    {
        "comprehensive_description": "a red cube slowly rotating on a wooden table",
        "camera_info": "static medium shot",
        "background": "soft studio lighting",
    },
    {
        "comprehensive_description": "a paper boat drifting across a calm blue pond",
        "camera_info": "slow top-down pan",
        "background": "gentle ripples",
    },
    {
        "comprehensive_description": "a yellow balloon rising past a brick wall",
        "camera_info": "vertical tilt up",
        "background": "overcast daylight",
    },
    {
        "comprehensive_description": "a green apple rolling along a marble counter",
        "camera_info": "static close-up",
        "background": "bright kitchen",
    },
]

# A structured negative caption (same shape as the prompt) so the CFG path is
# exercised without relying on the built-in default negative prompt.
_NEGATIVE_CAPTION = {
    "universal_negative": {
        "visual_quality": ["low quality", "blurry", "jpeg artifacts"],
        "temporal_and_motion_stability": ["flickering", "motion blur", "incoherent motion"],
    }
}


def build_rows(split: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "data_source": "jpeg_compressibility",
                "prompt": _CAPTIONS[i % len(_CAPTIONS)],
                "negative_prompt": _NEGATIVE_CAPTION,
                "ability": "t2v",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy LingBot Dense T2V parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_lingbot_video"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=4, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=4, help="Number of validation samples")
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(build_rows("train", args.train_size))
    val_df = pd.DataFrame(build_rows("test", args.val_size))

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
