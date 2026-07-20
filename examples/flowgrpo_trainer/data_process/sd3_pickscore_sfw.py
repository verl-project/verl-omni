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
"""Prepare deterministic PickScore-SFW prompt parquet files for SD3.5.

PickScore is used here only as a source of text-to-image prompts. The training
recipe scores generated SD3.5 latents with DiNa-LRM rather than PickScore.
"""

import argparse
import json
import random
import re
from pathlib import Path

import datasets
from verl.utils.hdfs_io import copy, makedirs


DEFAULT_DATASET = "CarperAI/pickapic_v1_no_images_training_sfw"
DATA_SOURCE = "flow_grpo/pickscore_sfw"


def read_prompt_file(path: Path) -> list[str]:
    """Read a provided one-prompt-per-line split without changing its order."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt split does not exist: {path}")
    captions = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not captions:
        raise ValueError(f"Prompt split is empty: {path}")
    return captions


def split_captions(
    captions: list[str],
    *,
    seed: int,
    test_size: int,
    min_spaces: int,
    train_size: int | None = None,
) -> tuple[list[str], list[str]]:
    """Normalize, filter, deduplicate, and deterministically split captions."""
    normalized = {
        re.sub(r"\s+", " ", caption).strip()
        for caption in captions
        if isinstance(caption, str) and caption.strip()
    }
    filtered = sorted(caption for caption in normalized if caption.count(" ") >= min_spaces)
    random.Random(seed).shuffle(filtered)

    if len(filtered) <= test_size:
        raise ValueError(
            f"Need more than test_size={test_size} eligible captions, but only found {len(filtered)}."
        )

    test_captions = filtered[:test_size]
    train_captions = filtered[test_size:]
    if train_size is not None:
        if train_size <= 0:
            raise ValueError(f"train_size must be positive when set, got {train_size}.")
        train_captions = train_captions[:train_size]
    return train_captions, test_captions


def build_split(captions: list[str], split: str) -> datasets.Dataset:
    """Build the parquet schema consumed by the SD3 diffusion trainer."""
    rows = []
    for index, caption in enumerate(captions):
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": caption}],
                "ability": "preference_alignment",
                # The latent DRM scorer reuses rollout prompt embeddings and
                # ignores ground_truth, but RewardLoop expects this field.
                "reward_model": {"style": "model", "ground_truth": caption},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "raw_prompt": caption,
                },
            }
        )
    return datasets.Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name or local dataset path.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing upstream train.txt and test.txt; preserves the provided split instead of re-splitting.",
    )
    parser.add_argument("--source-split", default="train", help="Source split containing the caption column.")
    parser.add_argument("--caption-column", default="caption", help="Name of the source caption column.")
    parser.add_argument("--output-dir", default="~/data/pickscore_sfw/sd3", help="Local parquet output directory.")
    parser.add_argument("--hdfs-dir", default=None, help="Optional HDFS destination for the completed output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--test-size", type=int, default=1024, help="Number of held-out validation prompts.")
    parser.add_argument(
        "--min-spaces",
        type=int,
        default=5,
        help="Minimum spaces in a caption, matching the upstream Flow-GRPO SFW filter.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Optional deterministic cap after filtering and splitting.",
    )
    parser.add_argument("--num-proc", type=int, default=16, help="Dataset loading worker count.")
    args = parser.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser()
        train_captions = read_prompt_file(input_dir / "train.txt")
        test_captions = read_prompt_file(input_dir / "test.txt")
        if args.train_size is not None:
            if args.train_size <= 0:
                raise ValueError(f"train_size must be positive when set, got {args.train_size}.")
            train_captions = train_captions[: args.train_size]
        source_metadata = {
            "source": str(input_dir),
            "split_strategy": "provided train.txt/test.txt",
        }
    else:
        source = datasets.load_dataset(args.dataset, split=args.source_split, num_proc=args.num_proc)
        if args.caption_column not in source.column_names:
            raise KeyError(
                f"Caption column {args.caption_column!r} is absent; available columns: {source.column_names}."
            )
        train_captions, test_captions = split_captions(
            source[args.caption_column],
            seed=args.seed,
            test_size=args.test_size,
            min_spaces=args.min_spaces,
            train_size=args.train_size,
        )
        source_metadata = {
            "source": args.dataset,
            "source_split": args.source_split,
            "caption_column": args.caption_column,
            "split_strategy": "normalized, deduplicated, sorted, and seeded shuffle",
            "seed": args.seed,
            "min_spaces": args.min_spaces,
        }
    train_dataset = build_split(train_captions, "train")
    test_dataset = build_split(test_captions, "test")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.to_parquet(output_dir / "train.parquet")
    test_dataset.to_parquet(output_dir / "test.parquet")

    metadata = {
        **source_metadata,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "data_source": DATA_SOURCE,
        "reward": "DiNa-LRM latent HTTP scorer (configured by the trainer)",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(train_dataset)} train / {len(test_dataset)} test prompts to {output_dir}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=str(output_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
