"""Build a tiny random Wan checkpoint for offline smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/.cache/models/tiny-random/Wan2.2-TI2V-5B-Diffusers")
_METADATA_FILE = "tiny_checkpoint_metadata.json"
_FORMAT_VERSION = 1


def _build_tokenizer() -> T5TokenizerFast:
    return T5TokenizerFast(
        vocab=[
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
            ("▁a", -1.0),
            ("▁video", -1.0),
            ("▁test", -1.0),
            ("▁red", -1.0),
            ("▁blue", -1.0),
            ("▁circle", -1.0),
            ("▁square", -1.0),
        ],
        extra_ids=0,
        model_max_length=256,
    )


def _write_model_index(output_dir: str) -> None:
    model_index = {
        "_class_name": "WanPipeline",
        "_diffusers_version": "0.35.0.dev0",
        "boundary_ratio": None,
        "expand_timesteps": True,
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": [None, None],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }
    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)


def _checkpoint_is_current(output_dir: str) -> bool:
    try:
        with open(os.path.join(output_dir, _METADATA_FILE), encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == {"format_version": _FORMAT_VERSION}


def build(output_dir: str, *, seed: int = 42, dtype: torch.dtype = torch.bfloat16) -> str:
    output_dir = os.path.expanduser(output_dir)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    tokenizer = _build_tokenizer()
    text_encoder = UMT5EncoderModel(
        UMT5Config(
            vocab_size=len(tokenizer),
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.0,
        )
    )
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=256,
    )
    vae = AutoencoderKLWan(
        base_dim=32,
        decoder_base_dim=32,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        in_channels=3,
        out_channels=3,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
    )

    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    text_encoder.to(dtype).save_pretrained(os.path.join(output_dir, "text_encoder"))
    transformer.to(dtype).save_pretrained(os.path.join(output_dir, "transformer"))
    vae.to(dtype).save_pretrained(os.path.join(output_dir, "vae"))
    UniPCMultistepScheduler().save_pretrained(os.path.join(output_dir, "scheduler"))
    _write_model_index(output_dir)
    with open(os.path.join(output_dir, _METADATA_FILE), "w", encoding="utf-8") as f:
        json.dump({"format_version": _FORMAT_VERSION}, f, indent=2, sort_keys=True)
    return output_dir


def ensure_tiny_wan_checkpoint(
    output_dir: str,
    *,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    output_dir = os.path.expanduser(output_dir)
    if _checkpoint_is_current(output_dir):
        return output_dir
    return build(output_dir, seed=seed, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny random Wan checkpoint offline.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--verify-load", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    output_dir = ensure_tiny_wan_checkpoint(args.output_dir, seed=args.seed, dtype=dtype)
    if args.verify_load:
        WanPipeline.from_pretrained(output_dir, dtype=dtype)
    print(output_dir)


if __name__ == "__main__":
    main()
