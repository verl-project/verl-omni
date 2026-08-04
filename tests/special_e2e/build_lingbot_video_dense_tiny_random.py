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
"""Build a tiny LingBot-Video Dense T2V checkpoint for smoke tests.

Random weights, tiny dims, offline: the FlowGRPO smoke test needs a checkpoint
that loads through both LingBot transformer implementations (the pip
``lingbot_video`` package used by FSDP training and vLLM-Omni's in-tree copy
used by rollout) without downloading multi-GB weights.

The processor and scheduler are re-serialized from a cached source checkpoint
because the structured-caption tokenization and the FlowMatch sigma schedule
depend on their exact special-token ids and config keys.  The multi-GB weight
shards of the source checkpoint are never loaded here.

Usage:
    python tests/special_e2e/build_lingbot_video_dense_tiny_random.py \
        --output-dir ~/models/tiny-random/lingbot-video-dense \
        --source-model ~/models/lingbot-video-dense-1.3b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch
from diffusers import AutoencoderKLWan
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/lingbot-video-dense")
DEFAULT_SOURCE_MODEL = os.path.expanduser("~/models/lingbot-video-dense-1.3b")

# Wan VAE latent statistics (16 channels).  Copied verbatim from the real Dense
# checkpoint so the latent normalization in the rollout adapter stays
# self-consistent (transformer in_channels == vae z_dim == 16).
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

# Qwen3VL text-encoder special-token ids (must match the copied processor).
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653
_BOS_TOKEN_ID = 151643
_EOS_TOKEN_ID = 151645
# The full vocab is kept (only the hidden size shrinks) so every id the copied
# tokenizer can emit -- including the ~151k special tokens -- stays in range.
_VOCAB_SIZE = 151936

# Tiny transformer geometry.  head_dim == hidden_size // num_attention_heads ==
# sum(axes_dims) == 32; every axis dim is even for the rotary embedding.
_DIT_HIDDEN_SIZE = 64
_DIT_NUM_HEADS = 2
_DIT_DEPTH = 2
_DIT_INTERMEDIATE_SIZE = 128
_DIT_AXES_DIMS = (8, 12, 12)
# RoPE position tables only need to cover the tiny latent grid (a few tokens per
# axis after patchifying), but we keep comfortable headroom.
_DIT_AXES_LENS = (256, 128, 128)


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


def get_dummy_components(*, text_dim: int = 32, seed: int = 42) -> dict[str, Any]:
    """Instantiate tiny LingBot Dense components with random weights.

    ``text_dim`` is the cross-model contract: the Qwen3VL last-hidden dimension
    is fed to the transformer as ``encoder_hidden_states`` with no projection,
    so the text-encoder hidden size and the transformer ``text_dim`` must match.
    """
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

    torch.manual_seed(seed)
    transformer = LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=16,
        out_channels=16,
        hidden_size=_DIT_HIDDEN_SIZE,
        num_attention_heads=_DIT_NUM_HEADS,
        depth=_DIT_DEPTH,
        intermediate_size=_DIT_INTERMEDIATE_SIZE,
        text_dim=text_dim,
        freq_dim=256,
        norm_eps=1e-6,
        rope_theta=256.0,
        axes_dims=_DIT_AXES_DIMS,
        axes_lens=_DIT_AXES_LENS,
        qkv_bias=False,
        out_bias=True,
        patch_embed_bias=True,
        timestep_mlp_bias=True,
        num_experts=0,  # dense-only: the FlowGRPO adapters reject num_experts != 0
    )

    torch.manual_seed(seed + 1)
    vae = AutoencoderKLWan(
        base_dim=8,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=_LATENTS_MEAN,
        latents_std=_LATENTS_STD,
    )

    torch.manual_seed(seed + 2)
    text_head_dim = 16
    text_encoder_config = Qwen3VLConfig(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
        tie_word_embeddings=True,
        text_config=dict(
            vocab_size=_VOCAB_SIZE,
            hidden_size=text_dim,
            intermediate_size=text_dim * 2,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=text_head_dim,
            rms_norm_eps=1e-6,
            max_position_embeddings=8192,
            rope_theta=5000000.0,
            rope_scaling={
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": _mrope_section(text_head_dim),
            },
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=4,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=text_dim,  # projector target: must match the text hidden size
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=256,
            deepstack_visual_indexes=[1, 2, 3],  # must be < depth
        ),
    )
    text_encoder = Qwen3VLForConditionalGeneration(text_encoder_config)

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
        allow_patterns=["processor/*", "scheduler/*"],
    )


def _copy_pretrained_assets(source_model: str, output_dir: str) -> None:
    """Re-serialize the processor and scheduler from a cached source checkpoint.

    Both are loaded from the local filesystem / HF cache (no Hub access).  The
    processor bundles the tokenizer for LingBot (there is no separate tokenizer
    subfolder), and the scheduler is re-serialized with the class the FlowGRPO
    adapters actually load it as, so the tiny config stays self-consistent.
    """
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    src = _resolve_source_dir(source_model)

    processor = AutoProcessor.from_pretrained(os.path.join(src, "processor"))
    processor.save_pretrained(os.path.join(output_dir, "processor"))

    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(src, subfolder="scheduler")
    scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))


def _write_model_index(output_dir: str) -> None:
    """Write the diffusers ``model_index.json`` describing every pipeline component.

    ``_class_name`` is what ``DiffusionModelConfig`` reads to auto-detect the
    architecture and dispatch to the LingBot FlowGRPO adapters; the component
    entries mirror the real Dense checkpoint.  The adapters load each component
    by subfolder directly, so the custom module references are never imported.
    """
    model_index = {
        "_class_name": "LingBotVideoPipeline",
        "_diffusers_version": "0.37.1",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["lingbot_video.scheduling_flow_unipc", "FlowUniPCMultistepScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["lingbot_video.transformer_lingbot_video", "LingBotVideoTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)


def build(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    text_dim: int = 32,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Construct and save a tiny random-weight LingBot Dense checkpoint."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    components = get_dummy_components(text_dim=text_dim, seed=seed)
    components["transformer"].to(dtype).save_pretrained(os.path.join(output_dir, "transformer"))
    # The Wan VAE runs in fp32 in the rollout adapter; keep its saved weights fp32.
    components["vae"].to(torch.float32).save_pretrained(os.path.join(output_dir, "vae"))
    components["text_encoder"].to(dtype).save_pretrained(os.path.join(output_dir, "text_encoder"))

    _copy_pretrained_assets(source_model, output_dir)
    _write_model_index(output_dir)
    return output_dir


def ensure_tiny_lingbot_video_checkpoint(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    text_dim: int = 32,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    skip_if_exists: bool = True,
) -> str:
    """Build the tiny checkpoint only if it is not already present."""
    output_dir = os.path.expanduser(output_dir)
    if skip_if_exists and os.path.isfile(os.path.join(output_dir, "model_index.json")):
        return output_dir
    return build(output_dir, source_model=source_model, text_dim=text_dim, seed=seed, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny LingBot-Video Dense checkpoint offline (random weights).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Cached Dense checkpoint to copy processor/scheduler from (local_files_only).",
    )
    parser.add_argument(
        "--text-dim",
        type=int,
        default=32,
        help="Shared Qwen3VL hidden size == transformer text_dim (cross-attention context).",
    )
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
    output_dir = ensure_tiny_lingbot_video_checkpoint(
        args.output_dir,
        source_model=args.source_model,
        text_dim=args.text_dim,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny LingBot-Video Dense checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
