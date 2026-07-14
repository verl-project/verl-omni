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
"""Build a tiny Qwen-Image-Edit-Plus checkpoint for smoke tests.

The tokenizer and processor come from the source checkpoint because image
placeholder expansion depends on their special-token IDs and geometry.

Usage:
    python tests/special_e2e/build_qwen_image_edit_plus_tiny_random.py \
        --output-dir ~/models/tiny-random/qwen-image-edit-plus
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/qwen-image-edit-plus")
DEFAULT_SOURCE_MODEL = "Qwen/Qwen-Image-Edit-2511"

# VAE latent statistics (16 channels). Copied from the real checkpoint so the
# packing math (transformer in_channels == z_dim * patch_size**2 == 64) and the
# latent normalization stay self-consistent.
_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
]

# Qwen2.5-VL special-token ids (must match the copied tokenizer/processor).
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653
_VISION_TOKEN_ID = 151654
_BOS_TOKEN_ID = 151643
_EOS_TOKEN_ID = 151645
_VOCAB_SIZE = 152064


def _mrope_section(head_dim: int) -> list[int]:
    """Split ``head_dim // 2`` into a 3-way (temporal, height, width) M-RoPE section.

    The M-RoPE rotary embedding requires ``sum(mrope_section) == head_dim // 2``.
    """
    half = head_dim // 2
    if half < 3:
        raise ValueError(f"head_dim // 2 must be >= 3 for a 3-way M-RoPE split, got {half}")
    t = half // 2
    h = (half - t) // 2
    w = half - t - h
    return [t, h, w]


def get_dummy_components(*, hidden_size: int = 16, seed: int = 42) -> dict[str, Any]:
    """Instantiate tiny Qwen-Image-Edit diffusion components (random weights)."""
    torch.manual_seed(seed)
    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=hidden_size,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
        zero_cond_t=True,
    )

    torch.manual_seed(seed + 1)
    vae = AutoencoderKLQwenImage(
        base_dim=16,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=_LATENTS_MEAN,
        latents_std=_LATENTS_STD,
    )

    torch.manual_seed(seed + 2)
    text_num_heads = 2
    text_head_dim = hidden_size // text_num_heads
    text_encoder_config = Qwen2_5_VLConfig(
        vocab_size=_VOCAB_SIZE,
        tie_word_embeddings=True,
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
        vision_token_id=_VISION_TOKEN_ID,
        bos_token_id=_BOS_TOKEN_ID,
        eos_token_id=_EOS_TOKEN_ID,
        text_config=dict(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=text_num_heads,
            num_key_value_heads=1,
            intermediate_size=hidden_size * 2,
            vocab_size=_VOCAB_SIZE,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            rope_scaling={"rope_type": "default", "mrope_section": _mrope_section(text_head_dim)},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            num_heads=2,
            out_hidden_size=hidden_size,  # projector target: must match text hidden_size
            intermediate_size=32,
            patch_size=14,
            spatial_patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_chans=3,
            fullatt_block_indexes=[1],  # must be < depth
            window_size=112,
            hidden_act="silu",
        ),
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration(text_encoder_config)

    return {"transformer": transformer, "vae": vae, "text_encoder": text_encoder}


def _resolve_source_dir(source_model: str) -> str:
    """Resolve ``source_model`` to a local directory (offline; never downloads weights)."""
    local = os.path.expanduser(source_model)
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download

    return snapshot_download(
        source_model,
        local_files_only=True,
        allow_patterns=["tokenizer/*", "processor/*", "scheduler/*"],
    )


def _copy_pretrained_assets(source_model: str, output_dir: str) -> None:
    """Re-serialize tokenizer, processor and scheduler from a cached source checkpoint.

    Everything is loaded from the local HF cache (no Hub access); the source
    checkpoint only needs its tokenizer/processor/scheduler files present -- the
    multi-GB weight shards are never loaded here.
    """
    src = _resolve_source_dir(source_model)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    processor = AutoProcessor.from_pretrained(os.path.join(src, "processor"))
    processor.save_pretrained(os.path.join(output_dir, "processor"))

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(src, "scheduler"))
    scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))


def _write_model_index(output_dir: str) -> None:
    """Write the diffusers ``model_index.json`` describing every pipeline component."""
    model_index = {
        "_class_name": "QwenImageEditPlusPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "processor": ["transformers", "Qwen2VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage"],
    }
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)


def build(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    hidden_size: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Construct and save a tiny random-weight Qwen-Image-Edit-Plus checkpoint."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    components = get_dummy_components(hidden_size=hidden_size, seed=seed)
    components["transformer"].to(dtype).save_pretrained(os.path.join(output_dir, "transformer"))
    components["vae"].to(dtype).save_pretrained(os.path.join(output_dir, "vae"))
    components["text_encoder"].to(dtype).save_pretrained(os.path.join(output_dir, "text_encoder"))

    _copy_pretrained_assets(source_model, output_dir)
    _write_model_index(output_dir)
    return output_dir


def ensure_tiny_qwen_image_edit_checkpoint(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    hidden_size: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    skip_if_exists: bool = True,
) -> str:
    """Build the tiny checkpoint only if it is not already present."""
    output_dir = os.path.expanduser(output_dir)
    if skip_if_exists and os.path.isfile(os.path.join(output_dir, "model_index.json")):
        return output_dir
    return build(output_dir, source_model=source_model, hidden_size=hidden_size, seed=seed, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny Qwen-Image-Edit-Plus checkpoint offline (random weights).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Cached checkpoint to copy tokenizer/processor/scheduler from (local_files_only).",
    )
    parser.add_argument("--hidden-size", type=int, default=16, help="Shared context/hidden size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when output-dir already contains model_index.json",
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if args.force and os.path.isdir(os.path.expanduser(args.output_dir)):
        shutil.rmtree(os.path.expanduser(args.output_dir))
    output_dir = ensure_tiny_qwen_image_edit_checkpoint(
        args.output_dir,
        source_model=args.source_model,
        hidden_size=args.hidden_size,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny Qwen-Image-Edit-Plus checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
