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
"""Build a tiny local SD3 checkpoint with random weights (fully offline, no Hub access).

Component layout follows diffusers' SD3 pipeline fast tests:
https://github.com/huggingface/diffusers/blob/main/tests/pipelines/stable_diffusion_3/test_pipeline_stable_diffusion_3.py
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    PreTrainedTokenizerFast,
    T5Config,
    T5EncoderModel,
)

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/stable-diffusion-3-tiny-random")

# Vocabulary shared by offline CLIP tokenizers (covers smoke-test prompts).
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


def _build_tiny_clip_tokenizer() -> CLIPTokenizerFast:
    """Build a minimal CLIP tokenizer in memory (no Hub download)."""
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
    return CLIPTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=77,
    )


def _build_tiny_t5_tokenizer(*, vocab_size: int = 1000, model_max_length: int = 512) -> PreTrainedTokenizerFast:
    """Build a minimal T5-style tokenizer from an in-memory WordLevel vocab."""
    vocab: dict[str, int] = {"<pad>": 0, "</s>": 1}
    for idx in range(2, vocab_size):
        vocab[f"tok{idx}"] = idx
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="</s>",
        model_max_length=model_max_length,
    )


def get_dummy_sd3_components(*, hidden_size: int = 8, seed: int = 42) -> dict[str, Any]:
    """Instantiate tiny SD3 pipeline components without any Hugging Face downloads."""
    torch.manual_seed(seed)

    clip_text_encoder_config = CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        layer_norm_eps=1e-5,
        num_attention_heads=2,
        num_hidden_layers=2,
        pad_token_id=1,
        vocab_size=1000,
        hidden_act="gelu",
        projection_dim=hidden_size,
        max_position_embeddings=77,
    )

    torch.manual_seed(seed)
    text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)
    torch.manual_seed(seed + 1)
    text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

    t5_config = T5Config(
        vocab_size=1000,
        d_model=hidden_size,
        d_ff=hidden_size * 2,
        d_kv=max(hidden_size // 2, 1),
        num_layers=2,
        num_heads=2,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        feed_forward_proj="gated-gelu",
    )
    torch.manual_seed(seed + 2)
    text_encoder_3 = T5EncoderModel(t5_config)

    tokenizer = _build_tiny_clip_tokenizer()
    tokenizer_2 = _build_tiny_clip_tokenizer()
    tokenizer_3 = _build_tiny_t5_tokenizer()

    torch.manual_seed(seed + 3)
    transformer = SD3Transformer2DModel(
        sample_size=32,
        patch_size=2,
        in_channels=16,
        num_layers=2,
        attention_head_dim=max(hidden_size // 2, 1),
        num_attention_heads=2,
        caption_projection_dim=hidden_size,
        joint_attention_dim=hidden_size,
        pooled_projection_dim=hidden_size * 2,
        out_channels=16,
    )

    torch.manual_seed(seed + 4)
    vae_channels = hidden_size * 2
    vae = AutoencoderKL(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        # Keep every down/up block at the same width to avoid GroupNorm channel mismatches.
        block_out_channels=(vae_channels, vae_channels, vae_channels, vae_channels),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=4,
        use_quant_conv=True,
        use_post_quant_conv=True,
        shift_factor=0.0609,
        scaling_factor=1.5035,
    )

    scheduler = FlowMatchEulerDiscreteScheduler()

    return {
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "text_encoder_3": text_encoder_3,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "tokenizer_3": tokenizer_3,
        "transformer": transformer,
        "vae": vae,
        "image_encoder": None,
        "feature_extractor": None,
    }


def build_tiny_sd3_pipeline(
    *,
    hidden_size: int = 8,
    seed: int = 42,
    dtype: torch.dtype = torch.float16,
) -> StableDiffusion3Pipeline:
    """Create a tiny SD3 pipeline with random weights."""
    components = get_dummy_sd3_components(hidden_size=hidden_size, seed=seed)
    pipeline = StableDiffusion3Pipeline(**components)
    return pipeline.to(dtype=dtype)


def ensure_tiny_sd3_checkpoint(
    output_dir: str,
    *,
    hidden_size: int = 8,
    seed: int = 42,
    dtype: torch.dtype = torch.float16,
    skip_if_exists: bool = True,
) -> str:
    """Build and save a tiny SD3 checkpoint locally if it does not already exist."""
    output_dir = os.path.expanduser(output_dir)
    model_index = os.path.join(output_dir, "model_index.json")
    if skip_if_exists and os.path.isfile(model_index):
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    pipeline = build_tiny_sd3_pipeline(hidden_size=hidden_size, seed=seed, dtype=dtype)
    pipeline.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny SD3 pipeline offline (random weights) and save it locally.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the saved pipeline (default: ~/models/tiny-random/stable-diffusion-3-tiny-random)",
    )
    parser.add_argument("--hidden-size", type=int, default=8, help="Hidden size for shrunk SD3 components")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when output-dir already contains model_index.json",
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    output_dir = ensure_tiny_sd3_checkpoint(
        args.output_dir,
        hidden_size=args.hidden_size,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny SD3 checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
