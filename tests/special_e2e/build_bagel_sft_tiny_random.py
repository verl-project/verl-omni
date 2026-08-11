#!/usr/bin/env python3
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
"""Build a tiny local BAGEL SFT checkpoint with random weights.

The real BAGEL checkpoint is too large for a smoke test.  This script creates
the minimum artifact layout that ``omni_sft_model`` expects:
``config.json``, tokenizer files, and ``ema.safetensors`` with BAGEL-style
checkpoint key prefixes.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/BAGEL-SFT")

_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
]


@dataclass(frozen=True)
class TinyBagelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 128
    latent_patch_size: int = 2
    max_latent_size: int = 2
    latent_channel: int = 2
    vae_downsample: int = 1
    start_of_image_id: int = 3
    end_of_image_id: int = 4

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size**2 * self.latent_channel


def _build_tokenizer(*, vocab_size: int = 256) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=_SPECIAL_TOKENS)
    corpus = [
        "A tiny BAGEL supervised fine-tuning smoke test.",
        "The answer is a short synthetic response.",
        "<|im_start|>user\nDescribe the image.<|im_end|>\n<|im_start|>assistant\nA small square.<|im_end|>",
        " ".join(str(i) for i in range(128)),
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        additional_special_tokens=["<|vision_start|>", "<|vision_end|>"],
        model_max_length=128,
    )


def _tiny_config(vocab_size: int) -> TinyBagelConfig:
    return TinyBagelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=max(vocab_size, 151654),
        max_position_embeddings=128,
        latent_patch_size=2,
        max_latent_size=2,
        latent_channel=2,
        vae_downsample=1,
        start_of_image_id=3,
        end_of_image_id=4,
    )


def _randn(shape: tuple[int, ...], *, scale: float = 0.02) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32) * scale


def _zeros(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def _linear(state: dict[str, torch.Tensor], prefix: str, out_features: int, in_features: int, *, bias: bool) -> None:
    state[f"{prefix}.weight"] = _randn((out_features, in_features))
    if bias:
        state[f"{prefix}.bias"] = _zeros((out_features,))


def _norm(state: dict[str, torch.Tensor], prefix: str, hidden_size: int) -> None:
    state[f"{prefix}.weight"] = torch.ones((hidden_size,), dtype=torch.float32)


def _raw_checkpoint_state(config: TinyBagelConfig) -> dict[str, torch.Tensor]:
    """Create BAGEL-style checkpoint tensors without importing ``verl_omni``.

    Importing ``verl_omni`` also imports vLLM-Omni, which can require CUDA
    libraries even for this offline checkpoint-build step.
    """

    raw_state: dict[str, torch.Tensor] = {}
    raw_state["language_model.model.embed_tokens.weight"] = _randn((config.vocab_size, config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        base = f"language_model.model.layers.{layer_idx}"
        attn = f"{base}.self_attn"
        _linear(raw_state, f"{attn}.q_proj", config.hidden_size, config.hidden_size, bias=True)
        _linear(raw_state, f"{attn}.k_proj", config.kv_dim, config.hidden_size, bias=True)
        _linear(raw_state, f"{attn}.v_proj", config.kv_dim, config.hidden_size, bias=True)
        _linear(raw_state, f"{attn}.o_proj", config.hidden_size, config.hidden_size, bias=False)
        _linear(raw_state, f"{attn}.q_proj_moe_gen", config.hidden_size, config.hidden_size, bias=True)
        _linear(raw_state, f"{attn}.k_proj_moe_gen", config.kv_dim, config.hidden_size, bias=True)
        _linear(raw_state, f"{attn}.v_proj_moe_gen", config.kv_dim, config.hidden_size, bias=True)
        _linear(raw_state, f"{attn}.o_proj_moe_gen", config.hidden_size, config.hidden_size, bias=False)
        _norm(raw_state, f"{attn}.q_norm", config.head_dim)
        _norm(raw_state, f"{attn}.k_norm", config.head_dim)
        _norm(raw_state, f"{attn}.q_norm_moe_gen", config.head_dim)
        _norm(raw_state, f"{attn}.k_norm_moe_gen", config.head_dim)

        _linear(raw_state, f"{base}.mlp.gate_proj", config.intermediate_size, config.hidden_size, bias=False)
        _linear(raw_state, f"{base}.mlp.up_proj", config.intermediate_size, config.hidden_size, bias=False)
        _linear(raw_state, f"{base}.mlp.down_proj", config.hidden_size, config.intermediate_size, bias=False)
        _linear(raw_state, f"{base}.mlp_moe_gen.gate_proj", config.intermediate_size, config.hidden_size, bias=False)
        _linear(raw_state, f"{base}.mlp_moe_gen.up_proj", config.intermediate_size, config.hidden_size, bias=False)
        _linear(raw_state, f"{base}.mlp_moe_gen.down_proj", config.hidden_size, config.intermediate_size, bias=False)
        _norm(raw_state, f"{base}.input_layernorm", config.hidden_size)
        _norm(raw_state, f"{base}.input_layernorm_moe_gen", config.hidden_size)
        _norm(raw_state, f"{base}.post_attention_layernorm", config.hidden_size)
        _norm(raw_state, f"{base}.post_attention_layernorm_moe_gen", config.hidden_size)

    _norm(raw_state, "language_model.model.norm", config.hidden_size)
    _norm(raw_state, "language_model.model.norm_moe_gen", config.hidden_size)
    _linear(raw_state, "time_embedder.mlp.0", config.hidden_size, 256, bias=True)
    _linear(raw_state, "time_embedder.mlp.2", config.hidden_size, config.hidden_size, bias=True)
    _linear(raw_state, "vae2llm", config.hidden_size, config.patch_latent_dim, bias=True)
    _linear(raw_state, "llm2vae", config.patch_latent_dim, config.hidden_size, bias=True)
    raw_state["latent_pos_embed.pos_embed"] = _randn(
        (config.max_latent_size * config.max_latent_size, config.hidden_size)
    )
    return raw_state


def build(output_dir: str, *, seed: int = 42) -> str:
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    torch.manual_seed(seed)

    tokenizer = _build_tokenizer()
    config = _tiny_config(vocab_size=len(tokenizer))

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "architectures": ["OmniBagelForConditionalGeneration"],
                "model_type": "bagel_sft_tiny_random",
                "tie_word_embeddings": False,
                "latent_patch_size": config.latent_patch_size,
                "max_latent_size": config.max_latent_size,
                "vae_config": {
                    "z_channels": config.latent_channel,
                    "downsample": config.vae_downsample,
                },
                "llm_config": {
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "num_attention_heads": config.num_attention_heads,
                    "num_key_value_heads": config.num_key_value_heads,
                    "vocab_size": config.vocab_size,
                    "rms_norm_eps": config.rms_norm_eps,
                    "rope_theta": config.rope_theta,
                    "max_position_embeddings": config.max_position_embeddings,
                    "tie_word_embeddings": False,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    save_file(_raw_checkpoint_state(config), os.path.join(output_dir, "ema.safetensors"))
    return output_dir


def ensure_tiny_bagel_sft_checkpoint(output_dir: str, *, seed: int = 42, skip_if_exists: bool = True) -> str:
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if skip_if_exists and os.path.isfile(os.path.join(output_dir, "ema.safetensors")):
        return output_dir
    return build(output_dir, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny BAGEL SFT checkpoint offline.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Rebuild even when ema.safetensors exists")
    args = parser.parse_args()

    output_dir = ensure_tiny_bagel_sft_checkpoint(args.output_dir, seed=args.seed, skip_if_exists=not args.force)
    print(f"Tiny BAGEL SFT checkpoint ready at {output_dir}", flush=True)


if __name__ == "__main__":
    main()
