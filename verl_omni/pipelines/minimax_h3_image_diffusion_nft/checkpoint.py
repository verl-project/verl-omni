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
"""Convert a native MiniMax H3 image transformer to the Diffusers layout the Actor trains, and verify it.

The image release ships one native transformer (``MiniMaxH3DiTModel``: fused QKV grouped per head as
``[q, k, v]``, ``mlp.fc1`` as ``[gate, up]``) plus per-step fused output heads. The Actor uses Diffusers'
``MiniMaxH3Transformer3DModel``. Every tensor is renamed, split or half-swapped without arithmetic, so the
conversion is exactly invertible and ``verify`` compares bit patterns.

Diffusers' class has no audio switch, so the two audio projections the audio-free release lacks are
written as zeros; the image Actor feeds zero audio rows and never reads them.

This module needs only ``torch`` and ``safetensors`` and can be run by path::

    python verl_omni/pipelines/minimax_h3_image_diffusion_nft/checkpoint.py convert \
        --native-root $MODEL --output $OUT/transformer
    python verl_omni/pipelines/minimax_h3_image_diffusion_nft/checkpoint.py verify \
        --native-root $MODEL --diffusers-transformer $OUT/transformer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

NATIVE_CLASS = "MiniMaxH3DiTModel"
DIFFUSERS_CLASS = "MiniMaxH3Transformer3DModel"
NATIVE_INDEX = "model.safetensors.index.json"
DIFFUSERS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
PDD_FILES = ("pdd_fused_heads.safetensors", "pdd_config.json")
MANIFEST_NAME = "conversion_manifest.json"
ROPE_BUFFER = "rope.inv_freq"

_CONFIG_RENAMES = {
    "token_refiner_num_layers": "num_refiner_layers",
    "ffn_hidden_size": "ffn_dim",
    "latents_dim": "in_channels",
    "audio_latents_dim": "audio_in_channels",
    "timestep_input_dim": "freq_dim",
    "time_embed_hidden_size": "time_embed_hidden_dim",
    "rope_inv_freq_len": "rope_freq_dim",
}
_CONFIG_SHARED = (
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "attention_head_dim",
    "patch_size",
    "text_dim",
    "time_embed_dim",
    "norm_eps",
    "qk_norm_eps",
    "final_norm_eps",
)
_TOPLEVEL_RENAMES = (
    ("video_patch_proj", "proj_in"),
    ("condition_proj", "context_embedder"),
    ("time_embedder.proj_in", "time_embedder.linear_1"),
    ("time_embedder.proj_out", "time_embedder.linear_2"),
    ("final_layer.norm", "norm_out.norm"),
    ("final_layer.adaln_proj.linear", "norm_out.linear"),
    ("final_layer.video_out", "proj_out"),
)
_AUDIO_NATIVE_PREFIXES = ("audio_patch_proj.", "final_layer.audio_out.")
_QKV_SUFFIX = ".attn.qkv_proj.weight"
_FC1_SUFFIX = ".mlp.fc1.weight"


def diffusers_config(native_config: dict) -> dict:
    """Translate an audio-free native transformer config to ``MiniMaxH3Transformer3DModel``'s."""
    if native_config.get("_class_name") != NATIVE_CLASS:
        raise ValueError(f"Expected a {NATIVE_CLASS} config, got {native_config.get('_class_name')!r}.")
    if native_config.get("audio_enabled", True) is not False:
        raise ValueError("The image converter handles audio-free releases only (audio_enabled: false).")
    hidden = int(native_config["hidden_size"])
    if native_config.get("adaln_out_features") != 18 * hidden or native_config.get("final_adaln_out_features") != (
        2 * hidden
    ):
        raise ValueError("Native AdaLN widths must be 18 * hidden_size and 2 * hidden_size.")
    config = {name: native_config[name] for name in _CONFIG_SHARED}
    config.update({target: native_config[source] for source, target in _CONFIG_RENAMES.items()})
    config["patch_size"] = list(config["patch_size"])
    config["rope_theta"] = float(native_config.get("rope_theta", 10000.0))
    config["_class_name"] = DIFFUSERS_CLASS
    return config


def synthesized_audio_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Zero tensors written for the audio projections Diffusers' class always owns."""
    hidden, audio = int(config["hidden_size"]), int(config["audio_in_channels"])
    return {
        "audio_proj_in.weight": (hidden, audio),
        "audio_proj_in.bias": (hidden,),
        "audio_proj_out.weight": (audio, hidden),
        "audio_proj_out.bias": (audio,),
    }


def diffusers_name(name: str) -> str:
    """Rename one native parameter that keeps its tensor unchanged."""
    for source, target in _TOPLEVEL_RENAMES:
        if name == source or name.startswith(source + "."):
            return target + name[len(source) :]
    name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
    name = name.replace(".attn.q_norm.", ".attn.norm_q.")
    name = name.replace(".attn.k_norm.", ".attn.norm_k.")
    name = name.replace(".attn.out_proj.", ".attn.to_out.0.")
    name = name.replace(".mlp.fc2.", ".ff.net.2.")
    if name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks.") :]
    return name


def _block_base(native_name: str, suffix: str) -> str:
    return diffusers_name(native_name[: -len(suffix)] + ".")[:-1]


def split_grouped_qkv(weight: torch.Tensor, heads: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    """Native rows are ``[q_h, k_h, v_h]`` per head; return the full ``q``, ``k`` and ``v`` matrices."""
    if weight.shape[0] != heads * 3 * head_dim:
        raise ValueError(f"Grouped QKV weight has {weight.shape[0]} rows, expected {heads * 3 * head_dim}.")
    grouped = weight.reshape(heads, 3, head_dim, *weight.shape[1:])
    return tuple(grouped[:, part].reshape(heads * head_dim, *weight.shape[1:]).contiguous() for part in range(3))


def join_grouped_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Inverse of :func:`split_grouped_qkv`."""
    rest = q.shape[1:]
    parts = [part.reshape(heads, head_dim, *rest) for part in (q, k, v)]
    return torch.stack(parts, dim=1).reshape(heads * 3 * head_dim, *rest).contiguous()


def swap_halves(weight: torch.Tensor) -> torch.Tensor:
    """Native ``fc1`` is ``[gate, up]``; Diffusers ``ff.net.0.proj`` is ``[up, gate]`` (self-inverse)."""
    if weight.shape[0] % 2:
        raise ValueError(f"fc1 rows must split evenly, got {tuple(weight.shape)}.")
    first, second = weight.chunk(2, dim=0)
    return torch.cat([second, first], dim=0).contiguous()


def convert_tensor(name: str, tensor: torch.Tensor, heads: int, head_dim: int) -> dict[str, torch.Tensor]:
    """Convert one native tensor to its Diffusers tensor(s); ``rope.inv_freq`` yields nothing."""
    if name == ROPE_BUFFER:
        return {}
    if name.startswith(_AUDIO_NATIVE_PREFIXES):
        raise ValueError(f"Audio tensor {name} in an audio-free release.")
    if name.endswith(_QKV_SUFFIX):
        base = _block_base(name, _QKV_SUFFIX)
        q, k, v = split_grouped_qkv(tensor, heads, head_dim)
        return {f"{base}.attn.to_q.weight": q, f"{base}.attn.to_k.weight": k, f"{base}.attn.to_v.weight": v}
    if name.endswith(_FC1_SUFFIX):
        return {f"{_block_base(name, _FC1_SUFFIX)}.ff.net.0.proj.weight": swap_halves(tensor)}
    return {diffusers_name(name): tensor}


def native_sources(name: str) -> tuple[str, ...]:
    """Diffusers tensor names one native tensor is rebuilt from, in the order :func:`restore_tensor` takes."""
    if name == ROPE_BUFFER:
        return ()
    if name.endswith(_QKV_SUFFIX):
        base = _block_base(name, _QKV_SUFFIX)
        return tuple(f"{base}.attn.to_{part}.weight" for part in "qkv")
    if name.endswith(_FC1_SUFFIX):
        return (f"{_block_base(name, _FC1_SUFFIX)}.ff.net.0.proj.weight",)
    return (diffusers_name(name),)


def restore_tensor(name: str, sources: tuple[torch.Tensor, ...], heads: int, head_dim: int) -> torch.Tensor:
    """Rebuild one native tensor from its Diffusers tensor(s)."""
    if name.endswith(_QKV_SUFFIX):
        return join_grouped_qkv(*sources, heads, head_dim)
    if name.endswith(_FC1_SUFFIX):
        return swap_halves(sources[0])
    return sources[0]


def expected_rope_inv_freq(config: dict) -> torch.Tensor:
    """The buffer Diffusers computes from its config instead of loading."""
    dim = int(config["rope_freq_dim"])
    theta = float(config.get("rope_theta", 10000.0))
    return 1.0 / (theta ** (torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim)))


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Same shape, dtype and bit pattern (NaN payloads and signed zeros included)."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state_dict(state: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    """In-memory conversion of a whole native state dict (tests and tiny checkpoints)."""
    heads, head_dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        for target, value in convert_tensor(name, tensor, heads, head_dim).items():
            if target in converted:
                raise ValueError(f"Duplicate Diffusers tensor {target}.")
            converted[target] = value
    for name, shape in synthesized_audio_shapes(config).items():
        converted[name] = torch.zeros(shape, dtype=torch.float32)
    return converted


def restore_state_dict(
    state: dict[str, torch.Tensor], native_names: list[str], config: dict
) -> dict[str, torch.Tensor]:
    """In-memory inverse of :func:`convert_state_dict` for the given native tensor names."""
    heads, head_dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    restored: dict[str, torch.Tensor] = {}
    for name in native_names:
        if name == ROPE_BUFFER:
            restored[name] = expected_rope_inv_freq(config)
            continue
        restored[name] = restore_tensor(name, tuple(state[source] for source in native_sources(name)), heads, head_dim)
    return restored


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _native_weight_map(transformer: Path) -> dict[str, str]:
    index = transformer / NATIVE_INDEX
    if index.is_file():
        return dict(_read_json(index)["weight_map"])
    from safetensors import safe_open

    single = transformer / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No {NATIVE_INDEX} or model.safetensors under {transformer}.")
    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return {name: single.name for name in handle.keys()}


def convert(native_root: Path, output: Path) -> dict:
    """Stream-convert ``<native_root>/transformer`` shard by shard and copy the PDD files beside it."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.time()
    transformer = native_root / "transformer"
    missing = [filename for filename in PDD_FILES if not (native_root / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{native_root} lacks {missing}.")
    native_config = _read_json(transformer / "config.json")
    config = diffusers_config(native_config)
    heads, head_dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    weight_map = _native_weight_map(transformer)
    if any(name.startswith(_AUDIO_NATIVE_PREFIXES) for name in weight_map):
        raise ValueError("Audio tensors found in an audio-free release.")
    if ROPE_BUFFER not in weight_map:
        raise ValueError(f"Native checkpoint lacks {ROPE_BUFFER}.")
    shards = sorted(set(weight_map.values()))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("diffusion_pytorch_model*")):
        raise FileExistsError(f"{output} already holds a converted transformer.")

    out_map: dict[str, str] = {}
    counts = {"native": 0, "diffusers": 0, "values": 0}
    for position, shard in enumerate(shards, 1):
        out_name = f"diffusion_pytorch_model-{position:05d}-of-{len(shards):05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(transformer / shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                counts["native"] += 1
                counts["values"] += tensor.numel()
                if name == ROPE_BUFFER and not bits_equal(tensor, expected_rope_inv_freq(config)):
                    raise ValueError(f"{ROPE_BUFFER} differs from the buffer Diffusers computes from its config.")
                tensors.update(convert_tensor(name, tensor, heads, head_dim))
        if position == 1:
            for name, shape in synthesized_audio_shapes(config).items():
                tensors[name] = torch.zeros(shape, dtype=torch.float32)
        duplicated = set(tensors) & set(out_map)
        if duplicated:
            raise ValueError(f"Duplicate Diffusers tensors: {sorted(duplicated)[:4]}")
        save_file(tensors, str(output / out_name), metadata={"format": "pt"})
        out_map.update({name: out_name for name in tensors})
        counts["diffusers"] += len(tensors)
        print(f"converted shard {position}/{len(shards)}: {shard} -> {out_name} ({len(tensors)} tensors)", flush=True)
        del tensors

    _write_json(output / "config.json", config)
    _write_json(output / DIFFUSERS_INDEX, {"metadata": {"format": "pt"}, "weight_map": out_map})
    pdd = {}
    for filename in PDD_FILES:
        shutil.copyfile(native_root / filename, output / filename)
        pdd[filename] = file_sha256(output / filename)
        if pdd[filename] != file_sha256(native_root / filename):
            raise ValueError(f"Copied {filename} differs from its source.")
    manifest = {
        "source": str(native_root),
        "native_tensors": counts["native"],
        "diffusers_tensors": counts["diffusers"],
        "native_values": counts["values"],
        "dropped": [ROPE_BUFFER],
        "synthesized_zero": sorted(synthesized_audio_shapes(config)),
        "pdd_sha256": pdd,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    print(f"CONVERTED: {json.dumps(manifest, sort_keys=True)}", flush=True)
    return manifest


def verify(native_root: Path, diffusers_transformer: Path, report: Path | None = None) -> dict:
    """Rebuild every native tensor from the converted checkpoint and compare bit patterns and SHA-256."""
    from contextlib import ExitStack

    from safetensors import safe_open

    started = time.time()
    transformer = native_root / "transformer"
    config = _read_json(diffusers_transformer / "config.json")
    if config != diffusers_config(_read_json(transformer / "config.json")):
        raise AssertionError("Converted config.json does not match the native config.")
    heads, head_dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    native_map = _native_weight_map(transformer)
    converted_map = dict(_read_json(diffusers_transformer / DIFFUSERS_INDEX)["weight_map"])
    synthesized = synthesized_audio_shapes(config)
    consumed: set[str] = set()
    rows = []
    values = 0

    with ExitStack() as stack:
        native_handles = {
            name: stack.enter_context(safe_open(str(transformer / name), framework="pt", device="cpu"))
            for name in sorted(set(native_map.values()))
        }
        converted_handles = {
            name: stack.enter_context(safe_open(str(diffusers_transformer / name), framework="pt", device="cpu"))
            for name in sorted(set(converted_map.values()))
        }
        on_disk = {name for handle in converted_handles.values() for name in handle.keys()}
        if on_disk != set(converted_map):
            raise AssertionError("Converted index does not list exactly the tensors on disk.")
        for position, name in enumerate(sorted(native_map), 1):
            native = native_handles[native_map[name]].get_tensor(name)
            if name == ROPE_BUFFER:
                restored = expected_rope_inv_freq(config)
                sources: tuple[str, ...] = ()
            else:
                sources = native_sources(name)
                missing = [source for source in sources if source not in converted_map]
                if missing:
                    raise AssertionError(f"No converted tensor for {name}: {missing}")
                restored = restore_tensor(
                    name,
                    tuple(converted_handles[converted_map[source]].get_tensor(source) for source in sources),
                    heads,
                    head_dim,
                )
                consumed.update(sources)
            if not bits_equal(native, restored):
                raise AssertionError(
                    f"Mismatch for {name}: native={tuple(native.shape)}/{native.dtype}, "
                    f"restored={tuple(restored.shape)}/{restored.dtype}"
                )
            native_hash, restored_hash = tensor_sha256(native), tensor_sha256(restored)
            if native_hash != restored_hash:
                raise AssertionError(f"SHA-256 mismatch for {name}.")
            values += native.numel()
            rows.append(
                {
                    "native": name,
                    "diffusers": list(sources),
                    "shape": list(native.shape),
                    "dtype": str(native.dtype).removeprefix("torch."),
                    "sha256": native_hash,
                }
            )
            del native, restored
            if position % 100 == 0:
                print(f"verified {position}/{len(native_map)} native tensors", flush=True)
        for name, shape in synthesized.items():
            tensor = converted_handles[converted_map[name]].get_tensor(name)
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or bool(tensor.count_nonzero()):
                raise AssertionError(f"Synthesized tensor {name} must be FP32 zeros of shape {shape}.")
    unconsumed = set(converted_map) - consumed - set(synthesized)
    if unconsumed:
        raise AssertionError(f"Converted tensors without a native source: {sorted(unconsumed)[:8]}")
    pdd = {}
    for filename in PDD_FILES:
        pdd[filename] = file_sha256(diffusers_transformer / filename)
        if pdd[filename] != file_sha256(native_root / filename):
            raise AssertionError(f"{filename} differs from the native release.")
    summary = {
        "native_tensors": len(native_map),
        "diffusers_tensors": len(converted_map),
        "synthesized_zero": sorted(synthesized),
        "native_values": values,
        "pdd_sha256": pdd,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if report is not None:
        _write_json(report, {"summary": summary, "tensors": rows})
    print(
        f"PASS: {len(native_map)} native tensors bit-identical to {len(converted_map) - len(synthesized)} converted "
        f"tensors (+{len(synthesized)} synthesized zeros); values={values:,}; "
        f"elapsed={summary['elapsed_seconds']}s",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("--native-root", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--native-root", type=Path, required=True)
    verify_parser.add_argument("--diffusers-transformer", type=Path, required=True)
    verify_parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "convert":
        convert(args.native_root, args.output)
    else:
        verify(args.native_root, args.diffusers_transformer, args.report)


if __name__ == "__main__":
    main()
