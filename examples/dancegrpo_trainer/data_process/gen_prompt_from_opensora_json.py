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
Extract prompts from the Open-Sora-Plan annotation JSON file and write them to prompt.txt.

The JSON file (e.g., v1.1.0_HQ_part3.json) contains an array of records, each with a ``cap``
field (a list of caption strings). This script extracts the first caption from each record,
applies optional length and count filters, and writes one prompt per line to a text file.

Usage::

    python3 examples/dancegrpo_trainer/data_process/gen_prompt_from_opensora_json.py \\
        --input_json /path/to/v1.1.0_HQ_part3.json \\
        --output_path /path/to/prompt.txt \\
        --num_samples 500 \\
        --min_words 10 \\
        --max_words 200
"""

import argparse
import json
import os
import random


def _word_count(text: str) -> int:
    return len(text.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract prompts from Open-Sora-Plan annotation JSON."
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to the input JSON file (e.g., v1.1.0_HQ_part3.json).",
    )
    parser.add_argument(
        "--output_path",
        default="./prompt.txt",
        help="Path to the output prompt.txt file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to randomly select (default: use all that pass filters).",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=1,
        help="Minimum number of words in the prompt (default: 1).",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=None,
        help="Maximum number of words in the prompt (default: no limit).",
    )
    parser.add_argument(
        "--cap_index",
        type=int,
        default=0,
        help="Index into the ``cap`` array to use as the prompt (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling (default: no shuffling).",
    )

    args = parser.parse_args()

    # Load JSON
    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {args.input_json}")

    # Extract prompts and apply word-count filter
    prompts = []
    for item in data:
        cap = item.get("cap", [])
        if not cap or args.cap_index >= len(cap):
            continue
        text = cap[args.cap_index].strip()
        # Replace newlines with spaces so each prompt stays on a single line
        text = text.replace("\n", " ").replace("\r", " ")
        # Collapse multiple spaces into one
        text = " ".join(text.split())
        wc = _word_count(text)
        if wc < args.min_words:
            continue
        if args.max_words is not None and wc > args.max_words:
            continue
        prompts.append(text)

    print(f"After word-count filter: {len(prompts)} prompts "
          f"(min_words={args.min_words}, max_words={args.max_words})")

    if args.num_samples is not None and args.num_samples < len(prompts):
        if args.seed is not None:
            random.seed(args.seed)
            prompts = random.sample(prompts, args.num_samples)
        else:
            prompts = prompts[:args.num_samples]
        print(f"Selected {args.num_samples} samples (seed={args.seed})")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            if i > 0:
                f.write("\n")
            f.write(p)

    print(f"Wrote {len(prompts)} prompts to {args.output_path}")