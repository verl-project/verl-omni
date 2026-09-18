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
"""Build a tiny local PickScore (CLIP) checkpoint with random weights (offline).

PickScore is a CLIPModel fine-tune. The smoke path only needs a loadable
``CLIPModel`` + image processor/tokenizer so ``pickscore_reward.py`` can score images.

Usage:
    python tests/special_e2e/build_pickscore_tiny_random.py \\
        --output-dir ~/models/tiny-random/PickScore
"""

from __future__ import annotations

import argparse
import os

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, PreTrainedTokenizerFast

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/PickScore")

_CLIP_VOCAB_WORDS = (
    "<|startoftext|>",
    "<|endoftext|>",
    "a",
    "red",
    "circle",
    "on",
    "white",
    "background",
    "blue",
    "square",
    "black",
    "green",
    "triangle",
    "next",
    "to",
    "an",
    "orange",
    "rectangle",
    "the",
    " ",
)


def _build_tiny_clip_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=512,
        special_tokens=["<|startoftext|>", "<|endoftext|>"],
    )
    corpus = [
        " ".join(_CLIP_VOCAB_WORDS),
        "a red circle on a white background",
        "a blue square on a black background",
        "a green triangle next to an orange rectangle",
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=77,
    )


def ensure_tiny_pickscore_checkpoint(
    output_dir: str,
    *,
    hidden_size: int = 32,
    seed: int = 42,
    skip_if_exists: bool = True,
) -> str:
    """Build and save a tiny CLIP PickScore checkpoint if missing."""
    output_dir = os.path.expanduser(output_dir)
    marker = os.path.join(output_dir, "config.json")
    if skip_if_exists and os.path.isfile(marker):
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    config = CLIPConfig(
        projection_dim=hidden_size,
        text_config={
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "vocab_size": 512,
            "max_position_embeddings": 77,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "image_size": 64,
            "patch_size": 8,
            "num_channels": 3,
        },
    )
    model = CLIPModel(config)
    tokenizer = _build_tiny_clip_tokenizer()
    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"height": 64, "width": 64},
        do_center_crop=True,
        crop_size={"height": 64, "width": 64},
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny PickScore CLIP checkpoint offline.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_tiny_pickscore_checkpoint(
        args.output_dir,
        hidden_size=args.hidden_size,
        seed=args.seed,
        skip_if_exists=not args.force,
    )
    print(f"Tiny PickScore checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
