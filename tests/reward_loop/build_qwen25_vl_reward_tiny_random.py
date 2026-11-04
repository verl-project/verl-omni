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
"""Build an offline Qwen2.5-VL checkpoint for the standalone NPU reward smoke."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
)

from tests.special_e2e.build_qwen_image_tiny_random import _build_tiny_chatml_tokenizer

_METADATA_FILE = "tiny_reward_checkpoint_metadata.json"
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
)


def ensure_tiny_qwen25_vl_reward_checkpoint(output_dir: str | Path, *, seed: int = 42) -> Path:
    """Create or reuse this builder's complete checkpoint without overwriting assets."""
    output_dir = Path(output_dir).expanduser()
    metadata = {"builder": "qwen25_vl_reward_npu", "format_version": 1, "seed": seed}
    if output_dir.is_symlink():
        raise ValueError(f"Tiny reward checkpoint must not be a symbolic link: {output_dir}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        marker = output_dir / _METADATA_FILE
        if (
            not output_dir.is_dir()
            or any(path.is_symlink() for path in output_dir.rglob("*"))
            or not marker.is_file()
            or json.loads(marker.read_text()) != metadata
            or not all((output_dir / name).is_file() for name in _REQUIRED_FILES)
        ):
            raise ValueError(f"Use an empty output directory or this builder's complete checkpoint: {output_dir}")
        return output_dir

    tokenizer = _build_tiny_chatml_tokenizer()
    config = Qwen2_5_VLConfig(
        image_token_id=tokenizer.image_token_id,
        video_token_id=tokenizer.video_token_id,
        vision_start_token_id=tokenizer.vision_bos_token_id,
        vision_end_token_id=tokenizer.vision_eos_token_id,
        tie_word_embeddings=True,
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "mrope_section": [32, 16, 16],
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 2,
            "out_hidden_size": 256,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "fullatt_block_indexes": [1],
            "window_size": 112,
            "hidden_act": "silu",
        },
    )
    processor = Qwen2_5_VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=56 * 56,
            max_pixels=112 * 112,
        ),
        video_processor=Qwen2VLVideoProcessor(patch_size=14, merge_size=2, temporal_patch_size=2),
        chat_template=tokenizer.chat_template,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen2_5_VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir
