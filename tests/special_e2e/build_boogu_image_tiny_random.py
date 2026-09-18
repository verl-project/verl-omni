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
"""Build a tiny-random Boogu-Image checkpoint for e2e smoke tests.

Layout mirrors ``Boogu/Boogu-Image-0.1-Base``:

    model_index.json
    transformer/   (tiny canonical BooguImageTransformer2DModel + shim)
    mllm/          (tiny Qwen3VLForConditionalGeneration)
    processor/     (copied verbatim from the source checkpoint)
    scheduler/     (copied verbatim -- preserves Boogu's time-shift keys,
                    which a diffusers ConfigMixin round-trip would drop)
    vae/           (tiny AutoencoderKL, 16 latent channels)

Requires the ``boogu-image`` package (canonical transformer class) and a
locally cached source checkpoint for the processor/scheduler files.
"""

import argparse
import json
import os
import shutil
from typing import Any

import torch

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/Boogu-Image")
DEFAULT_SOURCE_MODEL = "Boogu/Boogu-Image-0.1-Base"

# The checkpoint's transformer_boogu.py shim, reproduced verbatim so the tiny
# checkpoint loads through the same AutoModel + trust_remote_code path as the
# real one.
_TRANSFORMER_SHIM = '''\
"""Compatibility entry point for the Boogu transformer (tiny-random copy)."""

from boogu.models.transformers.transformer_boogu import (
    BooguImageTransformer2DModel,
    PromptEmbedding,
)

__all__ = ["BooguImageTransformer2DModel", "PromptEmbedding"]
'''


def get_dummy_components(*, hidden_size: int = 16, seed: int = 42) -> dict[str, Any]:
    """Instantiate tiny Boogu-Image components (random weights).

    Constraint: ``hidden_size // num_attention_heads == sum(axes_dim_rope)``.
    """
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from diffusers import AutoencoderKL
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(seed)
    transformer = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=None,
        hidden_size=hidden_size,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=16,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        axes_dim_rope=(4, 2, 2),
        # axis 0 must cover the longest packed sequence: on the Edit path that
        # is instruction + noise-image + reference-image tokens (~1024 at 256px);
        # undersizing it makes the Edit rope gather go out of bounds. axes 1/2
        # are the spatial grid (16x16 at 256px). Real ckpt: (2048, 1664, 1664).
        axes_lens=(4096, 256, 256),
        timestep_scale=1000.0,
        instruction_feature_configs={
            "instruction_feat_dim": hidden_size,
            "num_instruction_feature_layers": 1,
            "reduce_type": "mean",
        },
        prompt_tuning_configs={
            "hidden_size": hidden_size,
            "num_layers": 0,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "multiple_of": 16,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "num_trainable_prompt_tokens": 0,
            "use_causal_mask": True,
            "use_prompt_tuning": False,
        },
    )

    torch.manual_seed(seed + 1)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=(8, 8, 8, 8),  # 4 blocks -> vae_scale_factor 8
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1,
        norm_num_groups=4,
        sample_size=32,
        scaling_factor=0.3611,
        shift_factor=0.1159,
    )

    torch.manual_seed(seed + 2)
    # Special-token ids follow the Qwen tokenizer family shipped in the source
    # checkpoint's processor directory; keep them aligned with that tokenizer.
    mllm_config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=hidden_size * 2,
            vocab_size=152064,
            rms_norm_eps=1e-6,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            num_heads=2,
            out_hidden_size=hidden_size,
            intermediate_size=32,
        ),
    )
    mllm = Qwen3VLForConditionalGeneration(mllm_config)

    return {"transformer": transformer, "vae": vae, "mllm": mllm}


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
    """Copy processor and scheduler directories verbatim from the source checkpoint.

    The scheduler config carries Boogu-specific keys (``do_shift``,
    ``time_shift_version``, ``seq_len``, ...) that a diffusers ConfigMixin
    load/save round-trip would silently drop, so plain file copies are used.
    """
    src = _resolve_source_dir(source_model)
    for subdir in ("processor", "scheduler"):
        dst = os.path.join(output_dir, subdir)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(src, subdir), dst)


def _write_model_index(output_dir: str) -> None:
    model_index = {
        "_class_name": "BooguImagePipeline",
        "_diffusers_version": "0.35.2",
        "mllm": ["transformers", "Qwen3VLForConditionalGeneration"],
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["scheduling_flow_match_euler_discrete_time_shifting", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["transformer_boogu", "BooguImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
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
    os.makedirs(output_dir, exist_ok=True)
    components = get_dummy_components(hidden_size=hidden_size, seed=seed)

    components["transformer"].to(dtype).save_pretrained(os.path.join(output_dir, "transformer"))
    with open(os.path.join(output_dir, "transformer", "transformer_boogu.py"), "w") as f:
        f.write(_TRANSFORMER_SHIM)

    components["vae"].to(dtype).save_pretrained(os.path.join(output_dir, "vae"))
    components["mllm"].to(dtype).save_pretrained(os.path.join(output_dir, "mllm"))

    _copy_pretrained_assets(source_model, output_dir)
    _write_model_index(output_dir)
    return output_dir


def ensure_tiny_boogu_image_checkpoint(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    hidden_size: int = 16,
    seed: int = 42,
    skip_if_exists: bool = True,
) -> str:
    if skip_if_exists and os.path.isfile(os.path.join(output_dir, "model_index.json")):
        return output_dir
    return build(output_dir, source_model=source_model, hidden_size=hidden_size, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Cached checkpoint to copy processor/scheduler from (local_files_only).",
    )
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when output-dir already contains model_index.json",
    )
    args = parser.parse_args()

    path = ensure_tiny_boogu_image_checkpoint(
        args.output_dir,
        source_model=args.source_model,
        hidden_size=args.hidden_size,
        seed=args.seed,
        skip_if_exists=not args.force,
    )
    print(f"tiny Boogu-Image checkpoint ready at {path}")


if __name__ == "__main__":
    main()
