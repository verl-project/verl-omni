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
Preprocess a video-edit dataset for Lance video editing FlowGRPO training.

Mirrors ``lance_image_edit.py`` — same prompt tokenization, so the text side is
identical to every other Lance recipe — and swaps the reference image for a
reference video.

The reference travels out of band: each row carries it in the ``videos`` column
and binds it to the sample with a literal ``<video>`` marker in the message text.
The dataset layer replaces that marker with the video content item and asserts
that the marker count matches the column, so the marker is consumed rather than
reaching the model as text.  ``LancePipeline._forward_video_edit`` then takes the
reference from ``multi_modal_data["video"]``, rebuilds its chat-template segments
around the user instruction, and prefills the reference through the ViT and the
VAE before denoising the edited video.

Unlike an image, the reference video is referenced by path rather than embedded
as bytes: the pipeline decodes it with decord (falling back to OpenCV) so it can
reuse upstream Lance's own bucket resize and frame sampler, and it reads the
source frame rate from the file.  The path therefore has to be readable from the
rollout workers, which is the usual requirement for video training data.

The rollout's target shape is the reference's own bucketed shape, so no resize
knob is needed here; ``actor_rollout_ref.rollout.pipeline.num_frames`` and the
height/width knobs apply to the text-to-video and image-to-video recipes.

Input layout::

    <input_dir>/train.jsonl     {"video": "0001.mp4", "instruction": "make the fox white"}
    <input_dir>/test.jsonl
    <input_dir>/videos/0001.mp4

Usage::

    python examples/flowgrpo_trainer/data_process/lance_video_edit.py \
        --model_path ~/models/bytedance-research/Lance \
        --input_dir ~/data/lance_video_edit \
        --output_dir ~/data/lance_video_edit/lance
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from lance_pickscore import prepare_training_tokenizer, tokenize_lance_prompt
from transformers import AutoTokenizer
from verl.utils.hdfs_io import copy, makedirs

from verl_omni.pipelines.lance_flow_grpo.lance_model import resolve_checkpoint_dir

#: Marks where the reference video goes in the message text.  The dataset layer
#: substitutes it; see ``verl_omni/utils/dataset/rl_dataset.py``.
VIDEO_MARKER = "<video>"


def convert_split(
    input_dir: str,
    split: str,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    max_samples: int,
) -> pd.DataFrame:
    jsonl_path = os.path.join(input_dir, f"{split}.jsonl")
    video_dir = os.path.join(input_dir, "videos")
    rows = []
    with open(jsonl_path, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            instruction = str(example["instruction"]).strip()
            video_path = os.path.abspath(os.path.join(video_dir, str(example["video"])))
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"reference video not found: {video_path}")

            rows.append(
                {
                    "data_source": "flow_grpo/lance_video_edit",
                    "prompt": [{"role": "user", "content": f"{VIDEO_MARKER}{instruction}"}],
                    "negative_prompt": [{"role": "user", "content": " "}],
                    "prompt_token_ids": np.array(
                        tokenize_lance_prompt(tokenizer, instruction, max_length=max_prompt_length), dtype=np.int64
                    ),
                    "ability": "video_edit",
                    "videos": [video_path],
                    "reward_model": {"style": "model", "ground_truth": instruction},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "video": str(example["video"]),
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True, help="Lance bundle root or Lance_3B_Video.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--hdfs_dir", type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = resolve_checkpoint_dir(os.path.expanduser(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

    train = convert_split(
        os.path.expanduser(args.input_dir),
        "train",
        tokenizer,
        args.max_prompt_length,
        args.train_size,
    )
    test = convert_split(
        os.path.expanduser(args.input_dir),
        "test",
        tokenizer,
        args.max_prompt_length,
        args.test_size,
    )

    output_dir = os.path.expanduser(args.output_dir)
    makedirs(output_dir, exist_ok=True)
    prepare_training_tokenizer(tokenizer, output_dir)
    train.to_parquet(os.path.join(output_dir, "train.parquet"), row_group_size=200)
    test.to_parquet(os.path.join(output_dir, "test.parquet"), row_group_size=200)
    print(f"Wrote {len(train)} training and {len(test)} validation samples to {output_dir}")

    if args.hdfs_dir is not None:
        copy(src=output_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
