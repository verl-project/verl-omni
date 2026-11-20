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
"""Qwen-Image 2.1 的 NFT 目标区域预测。"""

import json
import shutil
from pathlib import Path

from diffusers import FlowMatchEulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_diffusion_nft.diffusers_training_adapter import QwenImageDiffusionNFT

from .common import align_prompt_tags, condition_target_latents, image_shapes, joint_image_mask


@DiffusionModelBase.register("QwenImage21Pipeline", algorithm="diffusion_nft")
class QwenImage21DiffusionNFT(QwenImageDiffusionNFT):
    """复用 NFT forward noising，按 2.1 布局计算目标尾部预测。"""

    # The timestep embedding uses these precomputed FP32 frequencies before the BF16 projections.
    preserve_buffer_names = ("time_text_embed.time_proj.freqs",)

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """复用现有 processor 准备接口，从 encoder 配置补齐 AutoProcessor 的类型信息。"""
        processor_dir = Path(model_path) / "processor"
        if not processor_dir.is_dir():
            raise FileNotFoundError(f"Qwen-Image 2.1 processor directory not found: {processor_dir}")
        config_path = processor_dir / "config.json"
        if not config_path.is_file():
            encoder_config = Path(model_path) / "text_encoder" / "config.json"
            config = json.loads(encoder_config.read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3_vl":
                raise ValueError("Qwen-Image 2.1 requires a Qwen3-VL text encoder configuration.")
            shutil.copyfile(encoder_config, config_path)
        return str(processor_dir)

    @classmethod
    def build_scheduler(cls, model_config):
        """NFT 使用 rollout 的 timestep；这里只读取 checkpoint 的 flow 时间标度。"""
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_config.local_path, subfolder="scheduler")
        if scheduler.config.num_train_timesteps != 1000:
            raise ValueError("Qwen-Image 2.1 requires a 1000-step training time scale.")
        return scheduler

    @classmethod
    def preserve_fp32_modules(cls) -> bool:
        """由 FSDP 混合精度管理参数，避免单个 wrap unit 混合参数 dtype。"""
        return False

    @classmethod
    def validate_engine_config(cls, engine_config) -> None:
        """首版限制为已验证的 FSDP2 与无 sequence parallel 路径。"""
        if engine_config.strategy != "fsdp2" or engine_config.ulysses_sequence_parallel_size != 1:
            raise ValueError("Qwen-Image 2.1 NFT requires FSDP2 with ulysses_sequence_parallel_size=1.")

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    ):
        shapes = image_shapes(micro_batch["condition_image_shapes"], micro_batch["target_image_shape"])
        conditions = micro_batch["condition_image_rows"]
        hidden_states = condition_target_latents(latents, conditions, shapes)
        if hidden_states.shape[-1] != module.config.in_channels:
            raise ValueError("Latent channel count does not match the transformer configuration.")
        common = {
            "hidden_states": hidden_states,
            "timestep": timesteps / 1000.0,
            "img_shapes": shapes,
            "kv_cache": None,
            "kv_cache_mode": None,
            "return_dict": False,
            "_target_seq_len": latents.shape[1],
        }
        positive = {
            **common,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "img_mask": joint_image_mask(
                align_prompt_tags(micro_batch["prompt_token_tags"], prompt_embeds_mask),
                prompt_embeds_mask,
                shapes,
                latents.shape[1],
            ),
        }
        negative = None
        if model_config.pipeline.true_cfg_scale > 1:
            if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
                raise ValueError("Qwen-Image 2.1 CFG requires negative prompt embeddings and masks.")
            negative = {
                **common,
                "encoder_hidden_states": negative_prompt_embeds,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "img_mask": joint_image_mask(
                    align_prompt_tags(micro_batch["negative_prompt_token_tags"], negative_prompt_embeds_mask),
                    negative_prompt_embeds_mask,
                    shapes,
                    latents.shape[1],
                ),
            }
        return positive, negative

    @classmethod
    def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
        """先裁剪目标尾部，再进行无 norm rescale 的 CFG。"""

        def predict(inputs):
            inputs = dict(inputs)
            target_tokens = inputs.pop("_target_seq_len")
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                prediction = module(**inputs)[0]
            if prediction.shape[1] < target_tokens:
                raise ValueError("Transformer output is shorter than the target latent sequence.")
            return prediction[:, -target_tokens:]

        prediction = predict(model_inputs)
        scale = model_config.pipeline.true_cfg_scale
        if scale > 1:
            if negative_model_inputs is None:
                raise ValueError("CFG requires negative model inputs.")
            negative = predict(negative_model_inputs)
            prediction = negative + scale * (prediction - negative)
        return prediction
