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
"""Build a tiny Qwen-Image-Edit-Plus checkpoint for smoke tests fully offline.

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
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
)

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/qwen-image-edit-plus")
_CHECKPOINT_METADATA_FILE = "tiny_checkpoint_metadata.json"

_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'text' %}"
    "{{ content['text'] }}"
    "{% elif content['type'] == 'image' %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% elif content['type'] == 'video' %}"
    "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

_MM_EXTRA_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "vision_eos_token": "<|vision_end|>",
}

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


def _build_tiny_chatml_tokenizer(*, vocab_size: int = 2048) -> Qwen2TokenizerFast:
    """Build a tiny Qwen tokenizer with the multimodal tokens used by the processor."""
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_pad|>",
        *_MM_EXTRA_SPECIAL_TOKENS.values(),
    ]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(
        [
            "Describe the key features of the input image and follow the user's edit instruction.",
            "Picture 1: a red circle on a white background.",
            "Change the red circle to a blue square.",
            "<|im_start|>user\nEdit this image.<|im_end|>\n<|im_start|>assistant\n",
            " ".join(str(i) for i in range(256)),
        ],
        trainer=trainer,
    )
    return Qwen2TokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        model_max_length=2048,
        chat_template=_CHATML_TEMPLATE,
        extra_special_tokens=_MM_EXTRA_SPECIAL_TOKENS,
    )


def _build_tiny_processor(tokenizer: Qwen2TokenizerFast) -> Qwen2VLProcessor:
    """Build the Qwen2-VL image processor without reading a Hub checkpoint."""
    try:
        from transformers import Qwen2VLVideoProcessor
    except ImportError:
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    return Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=56 * 56,
            max_pixels=1024 * 1024,
        ),
        video_processor=Qwen2VLVideoProcessor(
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
        ),
        tokenizer=tokenizer,
        chat_template=_CHATML_TEMPLATE,
    )


def get_dummy_components(*, tokenizer: Qwen2TokenizerFast, hidden_size: int = 16, seed: int = 42) -> dict[str, Any]:
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
        vocab_size=len(tokenizer),
        tie_word_embeddings=True,
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        vision_token_id=tokenizer.convert_tokens_to_ids("<|vision_pad|>"),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        text_config=dict(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=text_num_heads,
            num_key_value_heads=1,
            intermediate_size=hidden_size * 2,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
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


def _write_checkpoint_metadata(output_dir: str) -> None:
    """Record builder-owned assets so stale smoke checkpoints are rebuilt."""
    metadata = {"format_version": 1, "chat_template": _CHATML_TEMPLATE}
    with open(os.path.join(output_dir, _CHECKPOINT_METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _checkpoint_is_current(output_dir: str) -> bool:
    if not os.path.isfile(os.path.join(output_dir, "model_index.json")):
        return False
    try:
        with open(os.path.join(output_dir, _CHECKPOINT_METADATA_FILE)) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == {"format_version": 1, "chat_template": _CHATML_TEMPLATE}


def build(
    output_dir: str,
    *,
    hidden_size: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Construct and save a tiny random-weight Qwen-Image-Edit-Plus checkpoint."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = _build_tiny_chatml_tokenizer()
    components = get_dummy_components(tokenizer=tokenizer, hidden_size=hidden_size, seed=seed)
    components["transformer"].to(dtype).save_pretrained(os.path.join(output_dir, "transformer"))
    components["vae"].to(dtype).save_pretrained(os.path.join(output_dir, "vae"))
    components["text_encoder"].to(dtype).save_pretrained(os.path.join(output_dir, "text_encoder"))

    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    _build_tiny_processor(tokenizer).save_pretrained(os.path.join(output_dir, "processor"))
    FlowMatchEulerDiscreteScheduler().save_pretrained(os.path.join(output_dir, "scheduler"))
    _write_model_index(output_dir)
    _write_checkpoint_metadata(output_dir)
    return output_dir


def ensure_tiny_qwen_image_edit_checkpoint(
    output_dir: str,
    *,
    hidden_size: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    skip_if_exists: bool = True,
) -> str:
    """Build the tiny checkpoint only if it is not already present."""
    output_dir = os.path.expanduser(output_dir)
    if skip_if_exists and _checkpoint_is_current(output_dir):
        return output_dir
    return build(output_dir, hidden_size=hidden_size, seed=seed, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny Qwen-Image-Edit-Plus checkpoint offline (random weights).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
        hidden_size=args.hidden_size,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny Qwen-Image-Edit-Plus checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
