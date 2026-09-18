#!/usr/bin/env python3
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
"""Validate and merge the published MLX Hindi SFT LoRA into Qwen3-TTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXPECTED_RANK = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_base(
    base_model: str | Path,
    expected_files: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify every runtime asset from the documented Base revision."""
    base_model = Path(base_model).expanduser().resolve()
    if expected_files is None:
        expected_files = {
            "config.json": "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
            "generation_config.json": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
            "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
            "model.safetensors": "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",
            "preprocessor_config.json": "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119",
            "speech_tokenizer/config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
            "speech_tokenizer/configuration.json": "6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd",
            "speech_tokenizer/model.safetensors": ("836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"),
            "speech_tokenizer/preprocessor_config.json": (
                "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb"
            ),
            "tokenizer_config.json": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
            "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        }

    root_weights = sorted(
        path.name
        for pattern in ("model*.safetensors", "model*.safetensors.index.json", "pytorch_model*.bin")
        for path in base_model.glob(pattern)
    )
    if root_weights != ["model.safetensors"]:
        raise ValueError(f"Expected only the pinned unsharded model.safetensors, found: {root_weights}")

    actual_files = {}
    for relative_path, expected_sha256 in expected_files.items():
        path = base_model / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Pinned Base asset not found: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Pinned Base asset SHA-256 mismatch for {relative_path}: "
                f"expected {expected_sha256}, found {actual_sha256}."
            )
        actual_files[relative_path] = actual_sha256
    return actual_files


def checkpoint_dtype(model_dir: str | Path) -> torch.dtype:
    dtype_names = set()
    for path in Path(model_dir).glob("model*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            dtype_names.update(handle.get_slice(key).get_dtype() for key in handle.keys())
    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    if len(dtype_names) != 1 or next(iter(dtype_names), None) not in dtype_map:
        raise ValueError(f"Expected one supported checkpoint dtype, found: {sorted(dtype_names)}")
    return dtype_map[dtype_names.pop()]


def expected_modules() -> set[str]:
    modules = set()
    for prefix, layers in (("talker.model.layers", 28), ("talker.code_predictor.model.layers", 5)):
        for layer in range(layers):
            for target in TARGET_MODULES:
                block = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
                modules.add(f"{prefix}.{layer}.{block}.{target}")
    return modules


def load_mlx_adapter(
    source: str | Path,
    expected_sha256: str | None = "bd06b9474b128f12863e6ad167fd055fae0ab5f48e5d18df56089707781358df",
) -> dict[str, torch.Tensor]:
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MLX adapter not found: {source}")
    actual_sha256 = _sha256(source)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"MLX adapter SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}.")

    with safe_open(source, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    modules = expected_modules()
    expected = {f"{module}.lora_{side}" for module in modules for side in ("a", "b")}
    missing, unexpected = sorted(expected - tensors.keys()), sorted(tensors.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            "Published MLX adapter topology mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)."
        )

    state = {}
    for module in sorted(modules):
        lora_a, lora_b = tensors[f"{module}.lora_a"], tensors[f"{module}.lora_b"]
        if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != EXPECTED_RANK or lora_b.shape[1] != EXPECTED_RANK:
            raise ValueError(
                f"{module}: expected rank-{EXPECTED_RANK} LoRA matrices, got {lora_a.shape} and {lora_b.shape}."
            )
        if not lora_a.is_floating_point() or not lora_b.is_floating_point():
            raise TypeError(f"{module}: LoRA tensors must be floating point.")
        if not torch.isfinite(lora_a).all() or not torch.isfinite(lora_b).all():
            raise ValueError(f"{module}: LoRA tensors contain NaN or infinity.")
        prefix = f"base_model.model.{module}"
        state[f"{prefix}.lora_A.weight"] = lora_a.contiguous()
        state[f"{prefix}.lora_B.weight"] = lora_b.contiguous()
    return state


def merge_adapter(base_model: str | Path, source: str | Path, output_dir: str | Path) -> dict:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    base_model = Path(base_model).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"Use a local Qwen3-TTS checkpoint directory, got: {base_model}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    base_files_sha256 = validate_pinned_base(base_model)
    state = load_mlx_adapter(source)
    output_dtype = checkpoint_dtype(base_model)
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    model = Qwen3TTSForConditionalGeneration.from_pretrained(base_model, dtype=torch.float32, local_files_only=True)
    model.speech_tokenizer = None
    lora = get_peft_model(
        model,
        LoraConfig(r=EXPECTED_RANK, lora_alpha=16, target_modules=list(TARGET_MODULES), bias="none"),
    )
    expected = get_peft_model_state_dict(lora)
    if expected.keys() != state.keys():
        missing, unexpected = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
        raise ValueError(f"PEFT topology mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}.")
    incompatible = set_peft_model_state_dict(lora, state)
    missing_lora = [key for key in incompatible.missing_keys if ".lora_" in key]
    if missing_lora or incompatible.unexpected_keys:
        raise ValueError(
            f"PEFT load mismatch: missing={missing_lora[:5]}, unexpected={incompatible.unexpected_keys[:5]}."
        )
    merged = lora.merge_and_unload(safe_merge=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_model, output_dir)
    for pattern in ("model*.safetensors", "model*.safetensors.index.json", "pytorch_model*.bin"):
        for path in output_dir.glob(pattern):
            path.unlink()
    merged_state = merged.state_dict()
    residual_lora = [key for key in merged_state if ".lora_" in key]
    if residual_lora:
        raise ValueError(f"Merged checkpoint still contains LoRA tensors: {residual_lora[:5]}")
    save_file(
        {key: value.detach().to(output_dtype).contiguous() for key, value in merged_state.items()},
        output_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    manifest = {
        "base_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "base_revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "base_files_sha256": base_files_sha256,
        "adapter": "akashicmarga/qwen3-tts-hindi-lora",
        "adapter_revision": "a8718cac15b5a40bd4926b47d0854162ff32010a",
        "adapter_sha256": _sha256(source),
        "adapter_tensors": len(state),
        "checkpoint_dtype": str(output_dtype).removeprefix("torch."),
        "merged_tensors": len(merged_state),
    }
    (output_dir / "hindi_sft_merge.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local Qwen3-TTS base checkpoint")
    parser.add_argument("--source", required=True, help="Published adapters.safetensors")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(merge_adapter(args.base_model, args.source, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
