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
"""Create a small synthetic math parquet dataset for the Qwen3-Omni GSPO e2e test.

Mirrors the MATH-lighteval schema the example recipe uses (``data_source`` +
chat ``prompt`` + ``reward_model.ground_truth`` parsed by the ``dapo`` reward
manager), but with trivial arithmetic so the smoke test runs offline without
downloading a real dataset. This exercises plumbing only, not model quality.
"""

import argparse
import os

import pandas as pd

DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = "Solve the problem and put your final answer within \\boxed{}."

# (question, integer answer) pairs — trivial arithmetic, answer goes in \boxed{}.
_PROBLEMS = [
    ("What is 2 + 3?", 5),
    ("What is 7 - 4?", 3),
    ("What is 6 * 3?", 18),
    ("What is 20 / 5?", 4),
    ("What is 9 + 8?", 17),
    ("What is 12 - 7?", 5),
    ("What is 4 * 4?", 16),
    ("What is 15 + 6?", 21),
]


def build_rows(split: str, n: int):
    rows = []
    for i in range(n):
        question, answer = _PROBLEMS[i % len(_PROBLEMS)]
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "user", "content": f"{question} {INSTRUCTION}"},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": f"\\boxed{{{answer}}}"},
                "extra_info": {"split": split, "index": i, "answer": str(answer)},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate dummy math parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/math"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=32, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=8, help="Number of validation samples")
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
