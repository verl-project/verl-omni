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
"""离线构造 Qwen-Image 2.1 tiny checkpoint，供 NFT 在线训练 smoke 使用。

运行：python -m tests.special_e2e.build_qwen_image_21_tiny_random --output-dir DIR
保留 64-channel、16× VAE、RGBA 和 Qwen3-VL 图像 slot 合约，所有权重随机初始化。
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from transformers import (
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from tests.special_e2e.build_qwen_image_tiny_random import _build_tiny_chatml_tokenizer


def build_transformer(seed=12071):
    """在线 smoke 与 FSDP 数值测试共用同一个 tiny DiT。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return QwenImage21Transformer2DModel(
            in_channels=64,
            out_channels=64,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=128,
            context_in_dim=32,
            axes_dims_rope=(16, 56, 56),
        ).float()


def build(output_dir: Path):
    """保存可独立加载的组件；拒绝覆盖已有 checkpoint。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = _build_tiny_chatml_tokenizer()
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=256 * 256,
            max_pixels=4096 * 4096,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(output_dir / "processor")
    build_transformer().save_pretrained(output_dir / "transformer")
    torch.manual_seed(42)
    vae = AutoencoderKLQwenImage21(
        base_dim=16,
        decoder_base_dim=16,
        z_dim=64,
        dim_mult=[1, 2, 4, 4, 4],
        num_res_blocks=1,
        in_channels=4,
        out_channels=4,
        is_residual=True,
        temperal_downsample=[False, True, True, True],
        scale_factor_spatial=16,
        scale_factor_temporal=8,
        latents_mean=[0.0] * 64,
        latents_std=[1.0] * 64,
    )
    vae.save_pretrained(output_dir / "vae")
    config = Qwen3VLConfig(
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "max_position_embeddings": 32768,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000.0,
                "mrope_interleaved": True,
                "mrope_section": [4, 2, 2],
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0, 1],
            "num_position_embeddings": 256,
        },
    )
    # 上游按 checkpoint dtype 加载 encoder；其 embeddings 必须匹配 BF16 rollout DiT。
    Qwen3VLForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(output_dir / "text_encoder")
    FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000).save_pretrained(output_dir / "scheduler")
    (output_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "processor": ["transformers", "Qwen3VLProcessor"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage21"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args().output_dir)
