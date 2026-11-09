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
Preprocess an image-edit dataset for Lance image editing FlowGRPO training.

Mirrors ``lance_pickscore.py`` (same prompt tokenization, so the text side is
identical to the text-to-video and text-to-image recipes) and adds the reference
image the image-edit node edits.

The image travels out of band: each row carries it in the ``images`` column and
binds it to the sample with a literal ``<image>`` marker in the message text.
The dataset layer replaces that marker with the image content item and asserts
that the marker count matches the column, so the marker is consumed rather than
reaching the model as text.  ``LancePipeline._forward_image_edit`` then takes the
reference from ``multi_modal_data["image"]``, rebuilds its chat-template
segments around the user instruction, and prefills the reference through the ViT
and the VAE.

The reference is resized to the rollout's target image shape, which is the shape
the pipeline's latent grid is built for.

Input layout::

    <input_dir>/train.jsonl     {"image": "0001.png", "instruction": "make the fox white"}
    <input_dir>/test.jsonl
    <input_dir>/images/0001.png

Usage::

    python examples/flowgrpo_trainer/data_process/lance_image_edit.py \
        --model_path ~/models/bytedance-research/Lance \
        --input_dir ~/data/lance_image_edit \
        --output_dir ~/data/lance_image_edit/lance \
        --image_size 1024
"""

import argparse
import io
import json
import os

import numpy as np
import pandas as pd
from lance_pickscore import prepare_training_tokenizer, tokenize_lance_prompt
from PIL import Image
from transformers import AutoTokenizer
from verl.utils.hdfs_io import copy, makedirs

from verl_omni.pipelines.lance_flow_grpo.lance_model import resolve_checkpoint_dir

#: Marks where the reference frame goes in the message text.  The dataset layer
#: substitutes it; see ``verl_omni/utils/dataset/rl_dataset.py``.
IMAGE_MARKER = "<image>"


def load_reference_image(image_path: str, image_size: int) -> bytes:
    """Read a reference image as PNG bytes at the rollout's target shape."""
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def convert_split(
    input_dir: str,
    split: str,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    image_size: int,
    max_samples: int,
) -> pd.DataFrame:
    jsonl_path = os.path.join(input_dir, f"{split}.jsonl")
    image_dir = os.path.join(input_dir, "images")
    rows = []
    with open(jsonl_path, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            instruction = str(example["instruction"]).strip()
            image_path = os.path.join(image_dir, str(example["image"]))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"reference frame not found: {image_path}")

            rows.append(
                {
                    "data_source": "flow_grpo/lance_image_edit",
                    "prompt": [{"role": "user", "content": f"{IMAGE_MARKER}{instruction}"}],
                    "negative_prompt": [{"role": "user", "content": " "}],
                    "prompt_token_ids": np.array(
                        tokenize_lance_prompt(tokenizer, instruction, max_length=max_prompt_length), dtype=np.int64
                    ),
                    "ability": "image_edit",
                    "images": [{"bytes": load_reference_image(image_path, image_size)}],
                    "reward_model": {"style": "model", "ground_truth": instruction},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "image": str(example["image"]),
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
    parser.add_argument("--image_size", type=int, default=1024, help="Rollout target image edge; same.")
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
        args.image_size,
        args.train_size,
    )
    test = convert_split(
        os.path.expanduser(args.input_dir),
        "test",
        tokenizer,
        args.max_prompt_length,
        args.image_size,
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
