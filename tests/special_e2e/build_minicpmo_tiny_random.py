#!/usr/bin/env python
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
"""Build a tiny random MiniCPM-o 4.5 thinker checkpoint for e2e smoke tests.

MiniCPM-o 4.5 has no transformers-native implementation: every load goes through
the checkpoint's remote code, and the verl-omni training path additionally
requires the checkpoint's real processor/tokenizer (the prompt-parity module is
vocabulary-coupled to it). So unlike ``build_qwen3_omni_tiny_random.py`` this
builder starts from the source checkpoint's small files — remote code, configs,
tokenizer (~12 MB, never weights) — shrinks ``config.json``, and
random-initializes weights through the remote ``MiniCPMO`` class with verl-omni's
transformers-5 compatibility patches.

Config invariants the shrink must keep (both the HF remote code and vLLM-Omni's
vendored MiniCPMOConfig read these formulas):

* ``hidden_size % 128 == 0`` — the resampler uses ``num_heads = hidden_size // 128``
* ``num_attention_heads * head_dim == hidden_size`` — Qwen3 text stack
* audio ``encoder_ffn_dim % 4 == 0`` — the audio projector input is ``ffn // 4``
* audio ``d_model % encoder_attention_heads == 0``
* vision ``hidden_size`` / ``patch_size`` / ``image_size`` stay at the source
  values — SigLIP's learned position table and 2D position ids are sized by them
* ``version == "4.5"`` — vLLM-Omni's pipeline routing predicate for MiniCPMO
  checkpoints; also keep ``stream_input`` and the slice/query/audio-pool keys

Usage:
    python tests/special_e2e/build_minicpmo_tiny_random.py \
        --output-dir ~/models/tiny-random/MiniCPM-o-4_5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/MiniCPM-o-4_5")
DEFAULT_SOURCE_MODEL = "openbmb/MiniCPM-o-4_5"
DEFAULT_SOURCE_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"

# Remote code + config + tokenizer files; weights are never fetched.
_SOURCE_FILES = (
    "configuration_minicpmo.py",
    "modeling_minicpmo.py",
    "modeling_navit_siglip.py",
    "processing_minicpmo.py",
    "tokenization_minicpmo_fast.py",
    "utils.py",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

# Tiny overrides applied on top of the source config.json. Sub-dicts merge into
# the source section; every touched key must exist there (an upstream rename
# must fail the build, not silently keep the production size).
_TINY_OVERRIDES: dict[str, dict | bool] = {
    # Qwen3 text stack (top-level keys).
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_window_layers": 2,
    # Whisper encoder-only tower.
    "audio_config": {"d_model": 256, "encoder_layers": 2, "encoder_attention_heads": 4, "encoder_ffn_dim": 1024},
    # SigLIP tower: shrink depth/MLP only (see module docstring for the rest).
    "vision_config": {"num_hidden_layers": 2, "intermediate_size": 1152},
    # The smoke trains the thinker only; no TTS stage is built or saved.
    "init_tts": False,
}


def _shrink_config(source_config: dict) -> dict:
    """Return a deep copy of ``source_config`` with the tiny overrides applied."""
    config = json.loads(json.dumps(source_config))
    for key, value in _TINY_OVERRIDES.items():
        if key not in config:
            raise KeyError(f"Source MiniCPM-o config has no {key!r}; the checkpoint layout changed upstream.")
        if isinstance(value, dict):
            for sub_key in value:
                if sub_key not in config[key]:
                    raise KeyError(
                        f"Source config[{key!r}] has no {sub_key!r}; the checkpoint layout changed upstream."
                    )
            config[key].update(value)
        else:
            config[key] = value
    _validate_tiny_config(config)
    return config


def _validate_tiny_config(config: dict) -> None:
    """Fail fast on the cross-field invariants the tiny config must satisfy."""
    hidden_size = config["hidden_size"]
    if hidden_size % 128:
        raise ValueError(f"hidden_size {hidden_size} must be a multiple of 128 (resampler num_heads formula).")
    if config["num_attention_heads"] * config["head_dim"] != hidden_size:
        raise ValueError("num_attention_heads * head_dim must equal hidden_size (Qwen3 text stack).")
    audio = config["audio_config"]
    if audio["encoder_ffn_dim"] % 4:
        raise ValueError("audio encoder_ffn_dim must be a multiple of 4 (projector input is ffn // 4).")
    if audio["d_model"] % audio["encoder_attention_heads"]:
        raise ValueError("audio d_model must be divisible by encoder_attention_heads.")
    vision = config["vision_config"]
    if vision["image_size"] % vision["patch_size"]:
        raise ValueError("vision image_size must be divisible by patch_size.")


def _source_dir(source_model: str) -> str:
    """Resolve ``source_model`` to a local directory holding the source files.

    Uses a local path or the Hugging Face cache when possible and otherwise
    downloads only the pinned small files (never weights).
    """
    local = os.path.expanduser(source_model)
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns = list(_SOURCE_FILES)
    kwargs = {"allow_patterns": patterns}
    if source_model == DEFAULT_SOURCE_MODEL:
        kwargs["revision"] = DEFAULT_SOURCE_REVISION
    try:
        return snapshot_download(source_model, local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        return snapshot_download(source_model, **kwargs)


def _copy_source_files(source_dir: str, output_dir: str) -> None:
    """Copy the pinned source files into ``output_dir`` (idempotent)."""
    for name in _SOURCE_FILES:
        source = os.path.join(source_dir, name)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"{name} is missing from the MiniCPM-o source at {source_dir}.")
        shutil.copy2(source, os.path.join(output_dir, name))


def _build_model(output_dir: str, *, seed: int, dtype: torch.dtype) -> None:
    """Random-initialize and save the tiny MiniCPMO weights through its remote code."""
    from transformers import AutoConfig
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    from verl_omni.models.transformers.minicpm_o import (
        patch_minicpm_auto_model_init,
        patch_minicpm_siglip_flash_attn_support,
    )

    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    # Same training-path invariants the thinker adapter sets before from_pretrained.
    config.init_tts = False
    config.use_cache = False
    config.stream_input = False
    patch_minicpm_auto_model_init(output_dir, config=config)
    patch_minicpm_siglip_flash_attn_support(output_dir, config=config)
    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], output_dir, trust_remote_code=True)

    torch.manual_seed(seed)
    model = model_cls(config).to(dtype)
    model.save_pretrained(output_dir)


def _finalize_sidecar_files(source_dir: str, output_dir: str, shrunk_config: dict) -> None:
    """Restore the shrunk config.json and source generation_config after ``save_pretrained``.

    ``save_pretrained`` re-serializes config.json from the config object, which
    drops the private per-tower keys the source checkpoint ships; the smoke runs
    against the exact JSON this builder wrote, so it is restored verbatim.
    """
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(shrunk_config, handle, indent=2, sort_keys=True)
    shutil.copy2(os.path.join(source_dir, "generation_config.json"), os.path.join(output_dir, "generation_config.json"))


def _validate_checkpoint(output_dir: str) -> None:
    """Load the finished checkpoint through the same entry points training uses."""
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    if config.vocab_size < 100_000:
        raise ValueError("The tiny checkpoint must keep the source tokenizer vocabulary.")
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"Tokenizer size {len(tokenizer)} != config vocab_size {config.vocab_size}.")
    if getattr(tokenizer, "audio_start_id", None) is None:
        raise ValueError("The MiniCPM-o tokenizer wrapper attributes are missing; remote code did not load.")
    processor = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
    if type(processor).__name__ != "MiniCPMOProcessor":
        raise ValueError(f"Expected the remote MiniCPMOProcessor, got {type(processor).__name__}.")


def build(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Construct and save the tiny MiniCPM-o 4.5 thinker checkpoint at ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    source_dir = _source_dir(source_model)
    _copy_source_files(source_dir, output_dir)

    with open(os.path.join(output_dir, "config.json"), encoding="utf-8") as handle:
        source_config = json.load(handle)
    shrunk_config = _shrink_config(source_config)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(shrunk_config, handle, indent=2, sort_keys=True)

    _build_model(output_dir, seed=seed, dtype=dtype)
    _finalize_sidecar_files(source_dir, output_dir, shrunk_config)
    _validate_checkpoint(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny MiniCPM-o 4.5 thinker checkpoint (random weights).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-model", default=DEFAULT_SOURCE_MODEL, help="Hub repo id or a local MiniCPM-o 4.5 checkout."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--force", action="store_true", help="Rebuild even when output-dir already contains weights.")
    args = parser.parse_args()

    if not args.force and os.path.isfile(os.path.join(args.output_dir, "model.safetensors")):
        print(f"Tiny MiniCPM-o checkpoint already present at {args.output_dir}")
        return
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    output_dir = build(args.output_dir, source_model=args.source_model, seed=args.seed, dtype=dtype)
    print(f"Tiny MiniCPM-o checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
