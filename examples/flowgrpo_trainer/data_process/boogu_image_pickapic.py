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

"""Prepare Pick-a-Pic v1 prompts for Boogu-Image preference RL.

The default source is a prompt-only SFW export of the PickScore paper's
Pick-a-Pic v1 training set. Pairwise labels and source images are deliberately
discarded: online RL needs only the original prompt distribution.
"""

import argparse
import json
import random
import re
from pathlib import Path

import datasets
from verl.utils.hdfs_io import copy, makedirs

DEFAULT_DATASET = "CarperAI/pickapic_v1_no_images_training_sfw"
DATA_SOURCE = "pickapic_v1"

# Ported verbatim from BooguImagePipeline.__init__ (SYSTEM_PROMPT_4_T2I_UNIFIED).
BOOGU_SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)

# Boogu uses the TI2I unified template for an empty negative instruction.
BOOGU_SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)


def split_captions(
    captions: list[str],
    *,
    seed: int,
    validation_size: int,
    train_size: int | None,
) -> tuple[list[str], list[str]]:
    """Normalize, deduplicate, shuffle, and create prompt-disjoint splits."""
    normalized = {
        re.sub(r"\s+", " ", caption).strip() for caption in captions if isinstance(caption, str) and caption.strip()
    }
    eligible = sorted(caption for caption in normalized if len(caption) >= 3)
    random.Random(seed).shuffle(eligible)

    required = validation_size + (train_size or 1)
    if len(eligible) < required:
        raise ValueError(f"Need at least {required} unique prompts, but only found {len(eligible)}.")
    validation = eligible[:validation_size]
    train = eligible[validation_size:]
    if train_size is not None:
        train = train[:train_size]
    return train, validation


def build_split(captions: list[str], split: str, source: str = DEFAULT_DATASET) -> datasets.Dataset:
    rows = []
    for index, caption in enumerate(captions):
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "system", "content": BOOGU_SYSTEM_PROMPT_T2I},
                    {"role": "user", "content": caption},
                ],
                "negative_prompt": [
                    {"role": "system", "content": BOOGU_SYSTEM_PROMPT_DROP},
                    {"role": "user", "content": ""},
                ],
                "ability": "preference_alignment",
                "reward_model": {"style": "model", "ground_truth": caption},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "source": source,
                    "raw_prompt": caption,
                },
            }
        )
    return datasets.Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name or local dataset path.")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--output-dir", default="~/data/pickapic_v1/boogu_image")
    parser.add_argument("--hdfs-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-size",
        type=int,
        default=5760,
        help="Unique training prompts; 5760 is exactly 240 batches of 24. Use 0 for all remaining prompts.",
    )
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=16)
    args = parser.parse_args()
    if args.train_size < 0:
        parser.error(f"--train-size must be non-negative, got {args.train_size}")
    if args.validation_size <= 0:
        parser.error(f"--validation-size must be positive, got {args.validation_size}")

    source = datasets.load_dataset(args.dataset, split=args.source_split, num_proc=args.num_proc)
    if args.caption_column not in source.column_names:
        raise KeyError(f"Caption column {args.caption_column!r} is absent; available columns: {source.column_names}.")

    train_size = args.train_size or None
    train_captions, validation_captions = split_captions(
        source[args.caption_column],
        seed=args.seed,
        validation_size=args.validation_size,
        train_size=train_size,
    )
    train_dataset = build_split(train_captions, "train", source=args.dataset)
    validation_dataset = build_split(validation_captions, "validation", source=args.dataset)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.to_parquet(output_dir / "train.parquet")
    validation_dataset.to_parquet(output_dir / "test.parquet")
    metadata = {
        "source": args.dataset,
        "source_split": args.source_split,
        "caption_column": args.caption_column,
        "split_strategy": "normalized, deduplicated, sorted, and seeded shuffle",
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "data_source": DATA_SOURCE,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(train_dataset)} train / {len(validation_dataset)} validation prompts to {output_dir}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=str(output_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
