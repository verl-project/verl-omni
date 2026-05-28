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
"""Build a tiny local SD3 checkpoint from config only (random weights, no weight download)."""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

import diffusers
import torch
import transformers

DEFAULT_PIPELINE_CONFIG_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/stable-diffusion-3-tiny-random")


def get_original_model_configs(
    pipeline_cls: type[diffusers.DiffusionPipeline],
    pipeline_id: str,
) -> dict[str, Any]:
    """Load component configs from a diffusers pipeline repo (configs only, not weights)."""
    pipeline_config: dict[str, list[str]] = pipeline_cls.load_config(pipeline_id)
    model_configs: dict[str, Any] = {}

    for subfolder, import_strings in pipeline_config.items():
        if subfolder.startswith("_"):
            continue
        module = importlib.import_module(".".join(import_strings[:-1]))
        cls = getattr(module, import_strings[-1])
        if issubclass(cls, transformers.PreTrainedModel):
            config_class: type[transformers.PretrainedConfig] = cls.config_class
            model_configs[subfolder] = config_class.from_pretrained(pipeline_id, subfolder=subfolder)
        elif issubclass(cls, diffusers.ModelMixin) and issubclass(cls, diffusers.ConfigMixin):
            model_configs[subfolder] = cls.load_config(pipeline_id, subfolder=subfolder)

    return model_configs


def shrink_sd3_model_configs(model_configs: dict[str, Any], *, hidden_size: int = 8) -> None:
    """Shrink SD3 component configs in place for fast smoke tests."""
    text_encoder = model_configs["text_encoder"]
    text_encoder.hidden_size = hidden_size
    text_encoder.intermediate_size = hidden_size * 2
    text_encoder.num_attention_heads = 2
    text_encoder.num_hidden_layers = 2
    text_encoder.projection_dim = hidden_size

    text_encoder_2 = model_configs["text_encoder_2"]
    text_encoder_2.hidden_size = hidden_size
    text_encoder_2.intermediate_size = hidden_size * 2
    text_encoder_2.num_attention_heads = 2
    text_encoder_2.num_hidden_layers = 2
    text_encoder_2.projection_dim = hidden_size

    text_encoder_3 = model_configs["text_encoder_3"]
    text_encoder_3.d_model = hidden_size
    text_encoder_3.d_ff = hidden_size * 2
    text_encoder_3.d_kv = hidden_size // 2
    text_encoder_3.num_heads = 2
    text_encoder_3.num_layers = 2

    transformer = model_configs["transformer"]
    transformer["num_layers"] = 2
    transformer["num_attention_heads"] = 2
    transformer["attention_head_dim"] = hidden_size // 2
    transformer["pooled_projection_dim"] = hidden_size * 2
    transformer["joint_attention_dim"] = hidden_size
    transformer["caption_projection_dim"] = hidden_size

    vae = model_configs["vae"]
    vae["layers_per_block"] = 1
    vae["block_out_channels"] = [hidden_size] * 4
    vae["norm_num_groups"] = 2
    vae["latent_channels"] = 16


def load_pipeline_from_configs(
    pipeline_cls: type[diffusers.DiffusionPipeline],
    pipeline_id: str,
    model_configs: dict[str, Any],
) -> diffusers.DiffusionPipeline:
    """Instantiate a pipeline from shrunk configs with randomly initialized weights."""
    pipeline_config: dict[str, list[str]] = pipeline_cls.load_config(pipeline_id)
    components: dict[str, Any] = {}

    for subfolder, import_strings in pipeline_config.items():
        if subfolder.startswith("_"):
            continue
        module = importlib.import_module(".".join(import_strings[:-1]))
        cls = getattr(module, import_strings[-1])
        if issubclass(cls, transformers.PreTrainedModel):
            components[subfolder] = cls(model_configs[subfolder])
        elif issubclass(cls, transformers.PreTrainedTokenizerBase):
            components[subfolder] = cls.from_pretrained(pipeline_id, subfolder=subfolder)
        elif issubclass(cls, diffusers.ModelMixin) and issubclass(cls, diffusers.ConfigMixin):
            components[subfolder] = cls.from_config(model_configs[subfolder])
        elif issubclass(cls, diffusers.SchedulerMixin) and issubclass(cls, diffusers.ConfigMixin):
            components[subfolder] = cls.from_pretrained(pipeline_id, subfolder=subfolder)
        else:
            raise ValueError(f"Unsupported SD3 pipeline component {subfolder}: {import_strings}")

    return pipeline_cls(**components)


def build_tiny_sd3_pipeline(
    *,
    pipeline_config_id: str = DEFAULT_PIPELINE_CONFIG_ID,
    hidden_size: int = 8,
    seed: int = 42,
    dtype: torch.dtype = torch.float16,
) -> diffusers.StableDiffusion3Pipeline:
    """Create a tiny SD3 pipeline with random weights."""
    torch.manual_seed(seed)
    pipeline_cls = diffusers.StableDiffusion3Pipeline
    model_configs = get_original_model_configs(pipeline_cls, pipeline_config_id)
    shrink_sd3_model_configs(model_configs, hidden_size=hidden_size)
    pipeline = load_pipeline_from_configs(pipeline_cls, pipeline_config_id, model_configs)
    return pipeline.to(dtype=dtype)


def ensure_tiny_sd3_checkpoint(
    output_dir: str,
    *,
    pipeline_config_id: str = DEFAULT_PIPELINE_CONFIG_ID,
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
    pipeline = build_tiny_sd3_pipeline(
        pipeline_config_id=pipeline_config_id,
        hidden_size=hidden_size,
        seed=seed,
        dtype=dtype,
    )
    pipeline.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny SD3 pipeline from config (random weights) and save it locally.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the saved pipeline (default: ~/models/tiny-random/stable-diffusion-3-tiny-random)",
    )
    parser.add_argument(
        "--pipeline-config-id",
        default=DEFAULT_PIPELINE_CONFIG_ID,
        help="Diffusers pipeline id used only to fetch component configs/tokenizers (no weights)",
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
        pipeline_config_id=args.pipeline_config_id,
        hidden_size=args.hidden_size,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny SD3 checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
