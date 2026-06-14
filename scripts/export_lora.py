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
"""Export a verl-omni FSDP LoRA checkpoint to a standard PEFT adapter directory.

The output directory contains ``adapter_model.safetensors`` + ``adapter_config.json``
and is loadable via ``diffusers.DiffusionPipeline.load_lora_weights()`` or
``peft.PeftModel.from_pretrained()``.

Example (Qwen-Image-Edit-Plus FlowGRPO LoRA training)::

    python scripts/export_lora.py \\
        --checkpoint_dir outputs/run_qwen_image_edit_lora_pickscore/checkpoints/global_step_30/actor \\
        --output_dir ./exported_lora \\
        --target_modules to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,\\
to_add_out,img_mlp.net.0.proj,img_mlp.net.2,txt_mlp.net.0.proj,txt_mlp.net.2

Then use it with diffusers::

    from diffusers import QwenImageEditPipeline
    pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511")
    pipe.load_lora_weights("./exported_lora")
    pipe.fuse_lora()

``--target_modules`` should be passed explicitly whenever any trained module name
contains a dot (e.g. ``to_out.0``, ``img_mlp.net.0.proj``); the auto-inferred set
is computed from key paths via ``split('.')[-3]`` and is unreliable for such names.

Security note: this script calls ``torch.load(weights_only=False)`` because verl's
checkpoint format includes ``DTensor`` and metadata objects that the safe loader
rejects. Only export checkpoints you produced or trust.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import torch

# DTensor lives at different paths across torch versions.
try:
    from torch.distributed.tensor import DTensor
except ImportError:  # torch < 2.5
    from torch.distributed._tensor import DTensor


# --- argparse ----------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the LoRA export utility."""
    parser = argparse.ArgumentParser(
        description="Export a verl-omni FSDP LoRA checkpoint to a PEFT adapter dir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the trainer 'actor/' directory containing fsdp_config.json, "
        "lora_train_meta.json, and model_world_size_{N}_rank_{i}.pt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory; will hold adapter_model.safetensors and adapter_config.json.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="Comma-separated PEFT target_modules. Required when any trained module "
        "name contains a dot (e.g. 'to_out.0', 'img_mlp.net.0.proj'). When omitted, "
        "the script attempts to infer the set from key paths and prints a warning.",
    )
    parser.add_argument(
        "--rank_files_pattern",
        type=str,
        default="model_world_size_{world_size}_rank_{rank}.pt",
        help="str.format template for per-rank checkpoint file names. "
        "Available keys: {world_size}, {rank}.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat metadata mismatches and post-write validation issues as errors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty --output_dir.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="keep",
        choices=["keep", "bf16", "fp16", "fp32"],
        help="Cast LoRA tensors before saving. Default 'keep' preserves on-disk dtype.",
    )
    return parser.parse_args()


# --- IO helpers --------------------------------------------------------------


def _load_fsdp_config(ckpt_dir: Path) -> dict[str, Any]:
    """Load ``fsdp_config.json`` from the checkpoint directory."""
    cfg_path = ckpt_dir / "fsdp_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"fsdp_config.json not found under {ckpt_dir}. "
            "Make sure --checkpoint_dir points to a trainer 'actor/' directory."
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "world_size" not in cfg:
        raise ValueError(f"world_size missing from {cfg_path}")
    return cfg


def _load_lora_meta(ckpt_dir: Path, strict: bool) -> Optional[dict[str, Any]]:
    """Load ``lora_train_meta.json`` if present; warn or error when missing."""
    meta_path = ckpt_dir / "lora_train_meta.json"
    if not meta_path.exists():
        msg = (
            f"lora_train_meta.json not found under {ckpt_dir}. "
            "Falling back to inferring rank from tensor shapes; lora_alpha defaults to 0."
        )
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg, stacklevel=2)
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _load_rank(path: Path) -> dict[str, torch.Tensor]:
    """Load a single FSDP rank state dict (full pickle, includes DTensors)."""
    if not path.exists():
        raise FileNotFoundError(f"Rank file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_all_ranks(
    ckpt_dir: Path, world_size: int, pattern: str
) -> list[dict[str, torch.Tensor]]:
    """Load all per-rank state dicts in parallel."""
    paths = [
        ckpt_dir / pattern.format(world_size=world_size, rank=r) for r in range(world_size)
    ]
    out: list[Optional[dict]] = [None] * world_size
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as ex:
        futures = {ex.submit(_load_rank, p): i for i, p in enumerate(paths)}
        for fut in futures:
            out[futures[fut]] = fut.result()
    # mypy / sanity: all populated
    return [sd for sd in out if sd is not None]


# --- LoRA filtering, merging, renaming --------------------------------------


def _is_lora_key(k: str) -> bool:
    """Return True if a state-dict key belongs to a LoRA adapter."""
    return "lora_" in k


def _filter_lora(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a new dict containing only the LoRA-related keys from ``sd``."""
    return {k: v for k, v in sd.items() if _is_lora_key(k)}


def _has_dtensor(sd: dict[str, torch.Tensor]) -> bool:
    """Return True if any value in ``sd`` is a ``DTensor``."""
    return any(isinstance(v, DTensor) for v in sd.values())


def _merge_dtensor_shards(
    rank_sds: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Merge per-rank LoRA state dicts containing ``DTensor`` shards.

    Mirrors ``verl.model_merger.fsdp_model_merger.FSDPModelMerger._load_and_merge_state_dicts``
    but operates on the LoRA-only subset.
    """
    if not rank_sds:
        return {}

    keys = set(rank_sds[0].keys())
    for sd in rank_sds[1:]:
        if set(sd.keys()) != keys:
            raise ValueError(
                "Inconsistent LoRA keys across ranks; cannot merge. "
                f"Rank 0 has {len(keys)} keys; another rank has {len(sd.keys())}."
            )

    merged: dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        per_rank = [sd[k] for sd in rank_sds]
        first = per_rank[0]

        if isinstance(first, DTensor):
            placements = tuple(first.placements)
            if len(placements) != 1:
                raise NotImplementedError(
                    f"Multi-dim placements are not supported (key={k}, placements={placements}). "
                    "Only ('fsdp',) and ('ddp', 'fsdp') meshes are supported."
                )
            placement = placements[0]
            local_shards = [t._local_tensor for t in per_rank]  # type: ignore[union-attr]
            if placement.is_replicate():
                merged[k] = local_shards[0]
            elif placement.is_shard():
                merged[k] = torch.cat(local_shards, dim=placement.dim).contiguous()
            elif placement.is_partial():
                raise NotImplementedError(
                    f"Partial placement is not supported (key={k})."
                )
            else:
                raise NotImplementedError(f"Unsupported placement {placement} for key={k}")
        else:
            # Plain tensors: assume identical replication across ranks.
            merged[k] = first
    return merged


def _normalize_lora_keys(
    sd: dict[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], set[str]]:
    """Strip PEFT/FSDP wrappers from key names; return (renamed_sd, inferred_target_modules).

    Renames ``base_model.model.<...>.lora_A.default.weight`` ->
    ``base_model.model.<...>.lora_A.weight`` and removes any leftover
    ``_fsdp_wrapped_module.`` prefix. The third-to-last segment is collected as a
    naive ``target_modules`` candidate.
    """
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    inferred: set[str] = set()
    for k, v in sd.items():
        new_k = k.replace(".default.weight", ".weight").replace("_fsdp_wrapped_module.", "")
        parts = new_k.split(".")
        if len(parts) >= 3:
            inferred.add(parts[-3])
        out[new_k] = v
    return out, inferred


def _resolve_target_modules(cli: Optional[str], inferred: set[str]) -> list[str]:
    """Resolve final target_modules list from the CLI override or inferred set."""
    if cli:
        return [m.strip() for m in cli.split(",") if m.strip()]
    if any("." in m or m.isdigit() for m in inferred):
        warnings.warn(
            "Auto-inferred target_modules may be incorrect: detected dotted or "
            "numeric segments. Pass --target_modules explicitly when training used "
            "modules like 'to_out.0' or 'img_mlp.net.0.proj'.",
            stacklevel=2,
        )
    else:
        warnings.warn(
            "target_modules was inferred from key names via split('.')[-3]. "
            "If your trained modules contain dots, pass --target_modules explicitly.",
            stacklevel=2,
        )
    return sorted(inferred)


# --- output writers ----------------------------------------------------------


def _build_peft_config(
    rank: int,
    alpha: int,
    task_type: Optional[str],
    target_modules: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable PEFT ``adapter_config.json`` dict."""
    import peft

    cfg = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        task_type=task_type,
    ).to_dict()
    # PEFT enums must be serialized as their string values.
    if hasattr(cfg.get("task_type"), "value"):
        cfg["task_type"] = cfg["task_type"].value
    if hasattr(cfg.get("peft_type"), "value"):
        cfg["peft_type"] = cfg["peft_type"].value
    cfg["target_modules"] = list(cfg["target_modules"])
    return cfg


def _write_outputs(
    out_dir: Path,
    sd: OrderedDict[str, torch.Tensor],
    peft_cfg: dict[str, Any],
    overwrite: bool,
) -> None:
    """Write ``adapter_model.safetensors`` + ``adapter_config.json`` to ``out_dir``."""
    from safetensors.torch import save_file

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"--output_dir {out_dir} is not empty. Pass --overwrite to proceed."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = out_dir / "adapter_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(peft_cfg, f, ensure_ascii=False, indent=4)

    weights_path = out_dir / "adapter_model.safetensors"
    # safetensors requires contiguous tensors.
    contiguous = OrderedDict((k, v.contiguous()) for k, v in sd.items())
    save_file(contiguous, str(weights_path))


def _strict_validate(out_dir: Path, expected_keys: list[str]) -> None:
    """Re-load the written outputs and verify they round-trip cleanly."""
    from peft import PeftConfig
    from safetensors.torch import load_file

    cfg_path = out_dir / "adapter_config.json"
    weights_path = out_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        raise RuntimeError(
            f"Strict mode: missing output file under {out_dir}. "
            f"Expected both adapter_config.json and adapter_model.safetensors."
        )

    PeftConfig.from_pretrained(str(out_dir))  # raises on malformed JSON

    loaded = load_file(str(weights_path))
    if set(loaded.keys()) != set(expected_keys):
        missing = set(expected_keys) - set(loaded.keys())
        extra = set(loaded.keys()) - set(expected_keys)
        raise RuntimeError(
            f"Strict mode: written safetensors keys differ from in-memory state. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


# --- main --------------------------------------------------------------------


def _resolve_dtype(name: str) -> Optional[torch.dtype]:
    """Map the ``--dtype`` CLI value to a torch dtype, or None for 'keep'."""
    return {
        "keep": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _resolve_rank(meta: Optional[dict[str, Any]], any_lora_tensor: torch.Tensor, strict: bool) -> int:
    """Resolve LoRA rank from metadata or fall back to tensor shape inference."""
    inferred = min(any_lora_tensor.shape[0], any_lora_tensor.shape[1])
    if meta is None or "r" not in meta:
        return inferred
    meta_r = int(meta["r"])
    if meta_r != inferred:
        msg = f"LoRA rank mismatch: metadata={meta_r} vs inferred={inferred}; using metadata."
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return meta_r


def _resolve_alpha(meta: Optional[dict[str, Any]], strict: bool) -> int:
    """Resolve LoRA alpha from metadata; warn (or error in --strict) when missing/zero."""
    if meta is None or "lora_alpha" not in meta or not meta["lora_alpha"]:
        msg = "lora_alpha is missing or zero; falling back to 0. Verify lora_train_meta.json."
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        return 0
    return int(meta["lora_alpha"])


def main() -> None:
    """Export a verl-omni FSDP LoRA checkpoint into a standard PEFT adapter directory."""
    args = _parse_args()
    ckpt = Path(args.checkpoint_dir).resolve()
    out = Path(args.output_dir).resolve()

    if not ckpt.is_dir():
        raise NotADirectoryError(f"--checkpoint_dir is not a directory: {ckpt}")

    fsdp_cfg = _load_fsdp_config(ckpt)
    world_size = int(fsdp_cfg["world_size"])
    fsdp_version = int(fsdp_cfg.get("FSDP_version", 1))

    meta = _load_lora_meta(ckpt, strict=args.strict)

    rank0_path = ckpt / args.rank_files_pattern.format(world_size=world_size, rank=0)
    rank0 = _load_rank(rank0_path)
    rank0_lora = _filter_lora(rank0)
    needs_merge = fsdp_version >= 2 or _has_dtensor(rank0_lora)

    if needs_merge:
        print(
            f"[export_lora] Detected sharded LoRA tensors (fsdp_version={fsdp_version}, "
            f"has_dtensor={_has_dtensor(rank0_lora)}); loading all {world_size} ranks."
        )
        all_ranks = _load_all_ranks(ckpt, world_size, args.rank_files_pattern)
        all_ranks_lora = [_filter_lora(sd) for sd in all_ranks]
        merged = _merge_dtensor_shards(all_ranks_lora)
    else:
        print(
            f"[export_lora] Replicated FSDP1 detected (fsdp_version={fsdp_version}); "
            "using rank 0 only."
        )
        merged = rank0_lora

    if not merged:
        raise RuntimeError(
            f"No LoRA keys (containing 'lora_') were found under {ckpt}. "
            "Was this checkpoint trained with LoRA?"
        )

    renamed, inferred_modules = _normalize_lora_keys(merged)
    target_modules = _resolve_target_modules(args.target_modules, inferred_modules)

    cast_dtype = _resolve_dtype(args.dtype)
    if cast_dtype is not None:
        for k in list(renamed.keys()):
            renamed[k] = renamed[k].to(cast_dtype)

    sample_tensor = next(iter(renamed.values()))
    rank_value = _resolve_rank(meta, sample_tensor, strict=args.strict)
    alpha_value = _resolve_alpha(meta, strict=args.strict)
    task_type = (meta or {}).get("task_type")

    peft_cfg = _build_peft_config(
        rank=rank_value,
        alpha=alpha_value,
        task_type=task_type,
        target_modules=target_modules,
    )

    _write_outputs(out, renamed, peft_cfg, overwrite=args.overwrite)
    print(f"[export_lora] Wrote {len(renamed)} LoRA tensors to {out}")
    print(f"[export_lora] r={rank_value}, lora_alpha={alpha_value}, task_type={task_type!r}")
    print(f"[export_lora] target_modules = {target_modules}")

    if args.strict:
        _strict_validate(out, list(renamed.keys()))
        print("[export_lora] Strict validation passed.")


if __name__ == "__main__":
    main()
