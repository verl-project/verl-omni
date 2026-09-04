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
"""Create a tiny synthetic parquet dataset for MiniMax-H3 t2av smoke tests.

Schema mirrors ``/dockerdata/h3smoke/data/h3_pickscore/{train,test}.parquet``:

    data_source: str          # "minimax_h3/dummy_t2av"
    prompt: list[dict]        # chat-style [{role, content}]
    negative_prompt: list[dict]
    ability: str              # "t2av"
    reward_model: dict        # {"style": "rule", "ground_truth": ""}
    extra_info: dict          # {"split": str, "index": int}

Usage:
    python tests/special_e2e/create_dummy_h3_t2av_data.py \\
        --local_save_dir ~/data/dummy_h3_t2av \\
        --train_size 4 --val_size 2
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

_SYSTEM_PROMPT = (
    "Generate a short audio-video clip matching the following description. "
    "Focus on visible motion and matching ambient sound."
)

# Prompts intentionally cover a mix of scenes so multiple rollouts differ.
_USER_PROMPTS = [
    "A wooden wind chime swaying gently in the breeze on a sunny porch.",
    "Rain hitting a metal roof at night, faint distant thunder.",
    "A cat purring while curled up next to a crackling fireplace.",
    "Waves lapping against a sandy shore under a pastel sunset.",
    "A crowd cheering as fireworks burst above a city skyline.",
    "A single bird chirping in an early-morning forest clearing.",
    "Coffee brewing on a stove with soft jazz playing in the background.",
    "A train whistle in the distance while wheat sways in a field.",
]


def _build_rows(split: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        prompt_text = _USER_PROMPTS[i % len(_USER_PROMPTS)]
        rows.append(
            {
                "data_source": "minimax_h3/dummy_t2av",
                "prompt": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": " "},
                ],
                "ability": "t2av",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def build_dummy_h3_data(local_save_dir: str, *, train_size: int = 4, val_size: int = 2) -> tuple[str, str]:
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(_build_rows("train", train_size))
    val_df = pd.DataFrame(_build_rows("test", val_size))

    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "test.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dummy MiniMax-H3 t2av parquet dataset for smoke tests.",
    )
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/dummy_h3_t2av"))
    parser.add_argument("--train_size", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=2)
    args = parser.parse_args()

    train_path, val_path = build_dummy_h3_data(args.local_save_dir, train_size=args.train_size, val_size=args.val_size)
    print(f"wrote {args.train_size} train samples -> {train_path}")
    print(f"wrote {args.val_size} val samples -> {val_path}")


if __name__ == "__main__":
    main()
