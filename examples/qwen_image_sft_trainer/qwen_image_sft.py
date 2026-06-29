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
"""Qwen-Image SFT trainer for CoRT-style t2i/edit atomic entries.

This is a pragmatic FSDP training entrypoint modelled after verl's SFT loop:
dataset -> distributed sampler -> training batches -> periodic validation and
checkpointing.  The objective is diffusion teacher forcing, so unlike
``verl.trainer.sft_trainer`` this script trains the Qwen-Image transformer with
flow-matching MSE instead of language-model CE.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageFile, PngImagePlugin
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 200_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2**20)

LOGGER = logging.getLogger("qwen_image_sft")

ENTRY_TYPE_ALIASES = {
    "gen_t2i": "t2i",
    "gen_edit": "edit",
    "image_edit": "edit",
    "understanding": "und",
}

DEFAULT_LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


@dataclass
class AtomicEntry:
    entry_type: str
    prompt: str
    target_image: Any | None = None
    source_image: Any | None = None
    reflection: str | None = None
    sample_id: str = ""
    turn_index: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image SFT on CoRT-style atomic entries")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen-Image")
    parser.add_argument(
        "--pipeline_class",
        choices=("auto", "qwen_image", "qwen_image_edit"),
        default="auto",
        help="Use QwenImageEditPipeline for image-conditioned edit entries.",
    )
    parser.add_argument("--train_files", nargs="+", required=True, help="JSONL/JSON/parquet files with data rows.")
    parser.add_argument("--val_files", nargs="*", default=[])
    parser.add_argument(
        "--cort_intermediate_dirs",
        nargs="*",
        default=[],
        help="Optional CoRT intermediate roots used to resolve img0/img1/img2 for meta JSONL rows.",
    )
    parser.add_argument(
        "--train_entry_types",
        default="t2i,edit",
        help="Comma separated trainable entry types. `und` is parsed but skipped by Qwen-Image loss.",
    )
    parser.add_argument(
        "--cort_t2i_target",
        choices=("final", "img0"),
        default="final",
        help="For CoRT k-turn rows, train t2i on the final target image or on img0 like the chain planner.",
    )
    parser.add_argument(
        "--edit_prompt_mode",
        choices=("fix", "prompt_fix", "reflection", "prompt_reflection"),
        default="prompt_fix",
    )
    parser.add_argument("--edit_as_t2i", action="store_true", help="Train edit rows as plain t2i rows.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--total_training_steps", type=int, default=1000)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--test_freq", type=int, default=0)
    parser.add_argument("--output_dir", default="checkpoints/qwen_image_sft")
    parser.add_argument("--resume_from", default=None, help="Path to a checkpoint directory saved by this script.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--fsdp", action="store_true", help="Wrap transformer in FSDP. Implied when world_size > 1.")
    parser.add_argument("--fsdp_use_orig_params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", nargs="*", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return parser.parse_args()


def setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s [rank %(rank)s] %(message)s",
    )
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)


def npu_is_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device_type(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if npu_is_available():
            return "npu"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by --device cuda, but torch.cuda.is_available() is false.")
    if requested == "npu" and not npu_is_available():
        raise RuntimeError("NPU was requested by --device npu, but torch_npu is unavailable or no NPU is visible.")
    return requested


def set_accelerator_device(device_type: str, local_rank: int) -> torch.device:
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if device_type == "npu":
        torch.npu.set_device(local_rank)
        return torch.device("npu", local_rank)
    return torch.device("cpu")


def setup_distributed(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    device_type = resolve_device_type(args.device)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        device = set_accelerator_device(device_type, args.local_rank)
        backend = {"cuda": "nccl", "npu": "hccl"}.get(device_type, "gloo")
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = set_accelerator_device(device_type, args.local_rank)
    setup_logging(rank)
    return rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def str_to_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def extract_image_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        raw = value.get("bytes")
        return raw if isinstance(raw, bytes) else None
    return None


def as_image_ref(value: Any, base_dir: Path | None = None) -> Any | None:
    if value is None:
        return None
    if extract_image_bytes(value) is not None:
        return value
    if isinstance(value, str | os.PathLike):
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return str(path)
    return value


def load_rgb_image(ref: Any) -> Image.Image:
    raw = extract_image_bytes(ref)
    if raw is not None:
        image = Image.open(io.BytesIO(raw))
    elif isinstance(ref, Image.Image):
        image = ref
    elif isinstance(ref, str | os.PathLike):
        image = Image.open(ref)
    else:
        raise TypeError(f"Unsupported image reference type: {type(ref)!r}")

    if image.mode == "RGBA" or image.info.get("transparency") is not None:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def image_to_tensor(image: Image.Image, height: int, width: int) -> torch.Tensor:
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    data = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
    return data.div(127.5).sub(1.0)


def collate_atomic_entries(batch: list[AtomicEntry]) -> dict[str, Any]:
    return {
        "entry_type": [entry.entry_type for entry in batch],
        "prompt": [entry.prompt for entry in batch],
        "target_image": [entry.target_image for entry in batch],
        "source_image": [entry.source_image for entry in batch],
        "reflection": [entry.reflection for entry in batch],
        "sample_id": [entry.sample_id for entry in batch],
        "turn_index": [entry.turn_index for entry in batch],
    }


def read_data_rows(path: str) -> Iterable[dict[str, Any]]:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("data", [])
        for row in rows:
            yield row
        return
    if suffix == ".parquet":
        import pandas as pd

        table = pd.read_parquet(data_path)
        for row in table.to_dict(orient="records"):
            yield row
        return
    raise ValueError(f"Unsupported data file: {path}")


def format_reflection(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        problem = value.get("problem", "")
        fix = value.get("fix", "")
        return f"<problem>{problem}</problem>\n<fix>{fix}</fix>"
    return str(value)


def extract_fix(reflection: str | None) -> str:
    if not reflection:
        return ""
    match = re.search(r"<fix>(.*?)</fix>", reflection, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return reflection.strip()


def build_edit_prompt(prompt: str, reflection: str | None, mode: str) -> str:
    fix = extract_fix(reflection)
    reflection = reflection or ""
    if mode == "fix":
        return fix or prompt
    if mode == "prompt_fix":
        return f"Original instruction: {prompt}\nEdit instruction: {fix or reflection or prompt}"
    if mode == "reflection":
        return reflection or fix or prompt
    if mode == "prompt_reflection":
        return f"Original instruction: {prompt}\nReflection:\n{reflection or fix or prompt}"
    raise ValueError(f"Unsupported edit_prompt_mode: {mode}")


class CoRTAtomicDataset(Dataset):
    """Expands CoRT k-turn rows or atomic JSONL rows into trainable entries."""

    def __init__(
        self,
        files: list[str],
        *,
        cort_intermediate_dirs: list[str],
        train_entry_types: set[str],
        cort_t2i_target: str,
        edit_prompt_mode: str,
        edit_as_t2i: bool,
        max_samples: int | None = None,
    ):
        self.cort_intermediate_dirs = [Path(p) for p in cort_intermediate_dirs]
        self.train_entry_types = train_entry_types
        self.cort_t2i_target = cort_t2i_target
        self.edit_prompt_mode = edit_prompt_mode
        self.edit_as_t2i = edit_as_t2i
        self._sample_dir_cache: dict[str, Path | None] = {}
        entries: list[AtomicEntry] = []

        for file_path in files:
            base_dir = Path(file_path).parent
            for row in read_data_rows(file_path):
                entries.extend(self._expand_row(row, base_dir=base_dir))
                if max_samples is not None and len(entries) >= max_samples:
                    entries = entries[:max_samples]
                    break
            if max_samples is not None and len(entries) >= max_samples:
                break

        self.entries = [entry for entry in entries if entry.entry_type in self.train_entry_types]
        skipped = len(entries) - len(self.entries)
        LOGGER.info("Loaded %d trainable atomic entries (%d skipped by entry type)", len(self.entries), skipped)
        if "und" in train_entry_types:
            LOGGER.warning("`und` entries were requested, but Qwen-Image SFT has no text CE head; they are skipped.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> AtomicEntry:
        return self.entries[index]

    def _expand_row(self, row: dict[str, Any], *, base_dir: Path | None) -> list[AtomicEntry]:
        entry_type = row.get("entry_type") or row.get("task") or row.get("task_type") or row.get("type")
        entry_type = normalize_entry_type(entry_type)
        if entry_type in {"t2i", "edit", "und"}:
            return self._normalize_atomic_row(row, entry_type, base_dir=base_dir)
        return self._expand_cort_sample(row, base_dir=base_dir)

    def _normalize_atomic_row(
        self,
        row: dict[str, Any],
        entry_type: str,
        *,
        base_dir: Path | None,
    ) -> list[AtomicEntry]:
        prompt = row.get("prompt") or row.get("instruction") or row.get("text") or ""
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        reflection = format_reflection(row.get("reflection") or row.get("response"))

        if entry_type == "t2i":
            target = first_present(row, ("target_image", "image", "img", "image_path", "target", "img0"))
            return [
                AtomicEntry(
                    entry_type="t2i",
                    prompt=prompt,
                    target_image=as_image_ref(target, base_dir=base_dir),
                    sample_id=sample_id,
                )
            ]

        if entry_type == "edit":
            source = first_present(row, ("source_image", "input_image", "source", "image", "img0"))
            target = first_present(row, ("target_image", "output_image", "edited_image", "target", "img1"))
            prompt = build_edit_prompt(prompt, reflection, self.edit_prompt_mode)
            normalized_type = "t2i" if self.edit_as_t2i else "edit"
            return [
                AtomicEntry(
                    entry_type=normalized_type,
                    prompt=prompt,
                    source_image=as_image_ref(source, base_dir=base_dir),
                    target_image=as_image_ref(target, base_dir=base_dir),
                    reflection=reflection,
                    sample_id=sample_id,
                )
            ]

        image = first_present(row, ("image", "source_image", "img", "img0"))
        return [
            AtomicEntry(
                entry_type="und",
                prompt=prompt,
                source_image=as_image_ref(image, base_dir=base_dir),
                reflection=reflection,
                sample_id=sample_id,
            )
        ]

    def _expand_cort_sample(self, row: dict[str, Any], *, base_dir: Path | None) -> list[AtomicEntry]:
        prompt = row.get("prompt", "")
        sample_id = str(row.get("sample_id") or row.get("source_id") or "")
        num_turns = row.get("num_turns")
        if num_turns is None:
            return []
        num_turns = int(num_turns)
        if num_turns < 0:
            return []

        sample_dir = self._find_sample_dir(row)
        image_refs = {
            index: self._resolve_cort_image(row, index=index, sample_dir=sample_dir, base_dir=base_dir)
            for index in range(num_turns + 1)
        }
        if any(image_refs[index] is None for index in range(num_turns + 1)):
            LOGGER.warning("Skipping %s because at least one CoRT image is missing", sample_id)
            return []

        entries: list[AtomicEntry] = []
        final_index = self._final_image_index(row, num_turns)
        t2i_index = 0 if self.cort_t2i_target == "img0" else final_index
        entries.append(
            AtomicEntry(
                entry_type="t2i",
                prompt=prompt,
                target_image=image_refs[t2i_index],
                sample_id=sample_id,
                turn_index=0,
            )
        )

        for turn in range(1, num_turns + 1):
            reflection = format_reflection(row.get(f"reflection{turn}"))
            source_image = image_refs[turn - 1]
            target_image = image_refs[turn]
            entries.append(
                AtomicEntry(
                    entry_type="und",
                    prompt=prompt,
                    source_image=source_image,
                    reflection=reflection,
                    sample_id=sample_id,
                    turn_index=turn,
                )
            )
            edit_prompt = build_edit_prompt(prompt, reflection, self.edit_prompt_mode)
            entries.append(
                AtomicEntry(
                    entry_type="t2i" if self.edit_as_t2i else "edit",
                    prompt=edit_prompt,
                    source_image=source_image,
                    target_image=target_image,
                    reflection=reflection,
                    sample_id=sample_id,
                    turn_index=turn,
                )
            )
        return entries

    def _final_image_index(self, row: dict[str, Any], num_turns: int) -> int:
        gt_img = row.get("gt_img")
        if isinstance(gt_img, str) and gt_img.startswith("img"):
            try:
                return min(int(gt_img[3:]), num_turns)
            except ValueError:
                pass
        return num_turns

    def _resolve_cort_image(
        self,
        row: dict[str, Any],
        *,
        index: int,
        sample_dir: Path | None,
        base_dir: Path | None,
    ) -> Any | None:
        key = f"img{index}"
        value = row.get(key)
        if value is not None:
            return as_image_ref(value, base_dir=base_dir)
        if sample_dir is not None:
            path = sample_dir / f"{key}.png"
            if path.exists():
                return str(path)
        path_key = f"{key}_path"
        if row.get(path_key):
            return as_image_ref(row[path_key], base_dir=base_dir)
        return None

    def _find_sample_dir(self, row: dict[str, Any]) -> Path | None:
        if not self.cort_intermediate_dirs:
            return None
        sample_id = str(row.get("sample_id") or row.get("source_id") or "")
        source = row.get("source")
        generator_model = row.get("generator_model")
        cache_key = "|".join(str(x or "") for x in (sample_id, source, generator_model))
        if cache_key in self._sample_dir_cache:
            return self._sample_dir_cache[cache_key]

        candidates: list[Path] = []
        for root in self.cort_intermediate_dirs:
            if source and generator_model:
                candidates.append(root / str(source) / str(generator_model) / sample_id)
            if source:
                candidates.append(root / str(source) / sample_id)
            if generator_model:
                candidates.append(root / str(generator_model) / sample_id)
            candidates.append(root / sample_id)

        for candidate in candidates:
            if (candidate / "meta.json").exists():
                self._sample_dir_cache[cache_key] = candidate
                return candidate

        for root in self.cort_intermediate_dirs:
            matches = list(root.glob(f"**/{sample_id}/meta.json"))
            if matches:
                sample_dir = matches[0].parent
                self._sample_dir_cache[cache_key] = sample_dir
                return sample_dir

        self._sample_dir_cache[cache_key] = None
        return None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def normalize_entry_type(entry_type: Any) -> str | None:
    if entry_type is None:
        return None
    value = str(entry_type).strip().lower()
    return ENTRY_TYPE_ALIASES.get(value, value)


def should_use_edit_pipeline(args: argparse.Namespace, train_entry_types: set[str] | None = None) -> bool:
    if train_entry_types is None:
        train_entry_types = {
            normalized for item in args.train_entry_types.split(",") if (normalized := normalize_entry_type(item))
        }
    return args.pipeline_class == "qwen_image_edit" or (
        args.pipeline_class == "auto" and (not args.edit_as_t2i and "edit" in train_entry_types)
    )


def validate_pipeline_entry_types(args: argparse.Namespace, train_entry_types: set[str]) -> None:
    if "edit" in train_entry_types and args.pipeline_class == "qwen_image" and not args.edit_as_t2i:
        raise ValueError("Edit entries require --pipeline_class qwen_image_edit or --edit_as_t2i.")
    if should_use_edit_pipeline(args, train_entry_types) and "t2i" in train_entry_types:
        raise ValueError(
            "QwenImageEditPipeline requires image-conditioned entries and cannot train plain t2i rows. "
            "Use --train_entry_types edit with Qwen-Image-Edit, or use Qwen/Qwen-Image with "
            "--pipeline_class qwen_image for t2i rows."
        )


def load_pipeline(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    if should_use_edit_pipeline(args):
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    else:
        from diffusers import QwenImagePipeline

        pipe = QwenImagePipeline.from_pretrained(args.model_name_or_path, torch_dtype=dtype)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.vae.to(device=device, dtype=dtype)
    pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.transformer.to(device=device, dtype=dtype)

    if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()
    return pipe


def maybe_apply_lora(pipe, args: argparse.Namespace):
    if args.lora_rank <= 0:
        return
    from peft import LoraConfig, get_peft_model

    alpha = args.lora_alpha if args.lora_alpha > 0 else args.lora_rank
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_target_modules),
    )
    pipe.transformer = get_peft_model(pipe.transformer, config)
    pipe.transformer.print_trainable_parameters()


def maybe_wrap_fsdp(pipe, args: argparse.Namespace, world_size: int, device: torch.device, dtype: torch.dtype) -> bool:
    use_fsdp = args.fsdp or world_size > 1
    if not use_fsdp:
        return False

    mp_policy = None
    if dtype in (torch.bfloat16, torch.float16):
        mp_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    pipe.transformer = FSDP(
        pipe.transformer,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device if device.type in ("cuda", "npu") else None,
        use_orig_params=args.fsdp_use_orig_params,
    )
    return True


def unwrap_transformer(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, FSDP) else module


def load_images(image_refs: list[Any]) -> list[Image.Image]:
    return [load_rgb_image(ref) for ref in image_refs]


def resize_images(images: list[Image.Image], height: int, width: int) -> list[Image.Image]:
    return [image.resize((width, height), Image.Resampling.LANCZOS) for image in images]


def images_to_tensor(images: list[Image.Image], height: int, width: int, device: torch.device, dtype: torch.dtype):
    tensors = [image_to_tensor(image, height=height, width=width) for image in images]
    return torch.stack(tensors, dim=0).to(device=device, dtype=dtype)


def preprocess_images(image_refs: list[Any], height: int, width: int, device: torch.device, dtype: torch.dtype):
    return images_to_tensor(load_images(image_refs), height=height, width=width, device=device, dtype=dtype)


def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "argmax"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


@torch.no_grad()
def encode_target_latents(pipe, pixel_values: torch.Tensor, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
    image_latents = retrieve_latents(pipe.vae.encode(pixel_values), sample_mode="argmax")
    z_dim = int(getattr(pipe.vae.config, "z_dim", image_latents.shape[1]))
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std
    latent_h, latent_w = image_latents.shape[-2:]
    packed = pipe._pack_latents(image_latents, batch_size, z_dim, latent_h, latent_w)
    return packed.to(dtype=dtype)


def build_img_shapes(batch_size: int, height: int, width: int, vae_scale_factor: int, include_source: bool):
    latent_height = height // vae_scale_factor // 2
    latent_width = width // vae_scale_factor // 2
    target_shape = (1, latent_height, latent_width)
    if include_source:
        return [[target_shape, target_shape]] * batch_size
    return [[target_shape]] * batch_size


def _pad_prompt_embeds(
    prompt_embeds_list: list[torch.Tensor],
    prompt_mask_list: list[torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    max_seq_len = max(embeds.shape[1] for embeds in prompt_embeds_list)
    padded_embeds = []
    padded_masks = []
    for embeds, mask in zip(prompt_embeds_list, prompt_mask_list, strict=True):
        pad_len = max_seq_len - embeds.shape[1]
        if pad_len > 0:
            embeds = F.pad(embeds, (0, 0, 0, pad_len))
        padded_embeds.append(embeds)

        if mask is None:
            mask = torch.ones((embeds.shape[0], embeds.shape[1] - pad_len), device=embeds.device, dtype=torch.long)
        if pad_len > 0:
            mask = F.pad(mask, (0, pad_len))
        padded_masks.append(mask)

    prompt_embeds = torch.cat(padded_embeds, dim=0)
    prompt_embeds_mask = torch.cat(padded_masks, dim=0)
    return prompt_embeds, prompt_embeds_mask


def encode_prompts(pipe, prompts: list[str], source_images: list[Image.Image] | None, args: argparse.Namespace, device):
    if source_images is not None:
        prompt_embeds_list = []
        prompt_mask_list = []
        for prompt, image in zip(prompts, source_images, strict=True):
            prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                prompt=prompt,
                image=image,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )
            prompt_embeds_list.append(prompt_embeds)
            prompt_mask_list.append(prompt_embeds_mask)
        return _pad_prompt_embeds(prompt_embeds_list, prompt_mask_list)

    kwargs = {
        "prompt": prompts,
        "device": device,
        "num_images_per_prompt": 1,
        "max_sequence_length": args.max_sequence_length,
    }
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(**kwargs)
    return prompt_embeds, prompt_embeds_mask


def transformer_forward(
    pipe,
    hidden_states: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    img_shapes,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    transformer = pipe.transformer
    guidance = None
    config = getattr(unwrap_transformer(transformer), "config", None)
    if getattr(config, "guidance_embeds", False):
        guidance = torch.full(
            [hidden_states.shape[0]], guidance_scale, device=hidden_states.device, dtype=torch.float32
        )

    output = transformer(
        hidden_states=hidden_states,
        timestep=timesteps / 1000.0,
        guidance=guidance,
        encoder_hidden_states_mask=prompt_embeds_mask,
        encoder_hidden_states=prompt_embeds,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    return output[:, : hidden_states.shape[1]]


def compute_group_loss(
    pipe,
    batch: dict[str, Any],
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    entry_type: str,
) -> torch.Tensor:
    prompts = [batch["prompt"][i] for i in indices]
    target_refs = [batch["target_image"][i] for i in indices]
    target_pixels = preprocess_images(target_refs, args.height, args.width, device, dtype)
    batch_size = len(indices)
    x0 = encode_target_latents(pipe, target_pixels, batch_size=batch_size, dtype=dtype)

    source_pixels = None
    source_latents = None
    source_images = None
    if entry_type == "edit":
        source_refs = [batch["source_image"][i] for i in indices]
        source_images = resize_images(load_images(source_refs), args.height, args.width)
        source_pixels = images_to_tensor(source_images, args.height, args.width, device, dtype)
        source_latents = encode_target_latents(pipe, source_pixels, batch_size=batch_size, dtype=dtype)

    prompt_embeds, prompt_embeds_mask = encode_prompts(pipe, prompts, source_images, args, device)

    noise = torch.randn_like(x0, dtype=torch.float32).to(dtype)
    t = torch.rand((batch_size,), device=device, dtype=torch.float32).clamp_(min=1e-5)
    t_view = t.view(batch_size, *([1] * (x0.ndim - 1))).to(dtype)
    xt = (1.0 - t_view) * x0 + t_view * noise
    hidden_states = xt if source_latents is None else torch.cat([xt, source_latents], dim=1)
    img_shapes = build_img_shapes(
        batch_size=batch_size,
        height=args.height,
        width=args.width,
        vae_scale_factor=pipe.vae_scale_factor,
        include_source=source_latents is not None,
    )

    pred = transformer_forward(
        pipe=pipe,
        hidden_states=hidden_states,
        timesteps=t * 1000.0,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
    )
    pred = pred[:, : x0.shape[1]]
    target = noise - x0
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def compute_batch_loss(pipe, batch: dict[str, Any], args: argparse.Namespace, device, dtype) -> torch.Tensor:
    losses = []
    weights = []
    for entry_type in ("t2i", "edit"):
        indices = [i for i, value in enumerate(batch["entry_type"]) if value == entry_type]
        if not indices:
            continue
        if entry_type == "edit" and not hasattr(pipe, "processor"):
            if args.edit_as_t2i:
                continue
            raise RuntimeError("Edit entries require QwenImageEditPipeline. Use --pipeline_class qwen_image_edit.")
        group_loss = compute_group_loss(pipe, batch, indices, args, device, dtype, entry_type=entry_type)
        losses.append(group_loss * len(indices))
        weights.append(len(indices))
    if not losses:
        raise RuntimeError("Batch has no trainable t2i/edit entries.")
    return torch.stack(losses).sum() / float(sum(weights))


def create_optimizer(pipe, args: argparse.Namespace):
    params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable transformer parameters found.")
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def create_scheduler(optimizer, args: argparse.Namespace):
    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.total_training_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(pipe, optimizer, scheduler, args: argparse.Namespace, step: int, rank: int, use_fsdp: bool) -> None:
    ckpt_dir = Path(args.output_dir) / f"global_step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transformer = pipe.transformer
    state_dict = None
    if use_fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = transformer.state_dict()
    else:
        state_dict = transformer.state_dict()

    if rank == 0:
        unwrapped = unwrap_transformer(transformer)
        unwrapped.save_pretrained(ckpt_dir / "transformer", state_dict=state_dict, safe_serialization=True)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            ckpt_dir / "trainer_state.pt",
        )
        with (ckpt_dir / "training_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        LOGGER.info("Saved checkpoint to %s", ckpt_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_training_state(path: str, optimizer, scheduler, rank: int) -> int:
    state_path = Path(path) / "trainer_state.pt"
    if not state_path.exists():
        LOGGER.warning("No trainer_state.pt found under %s; model resume must be handled via model_name_or_path.", path)
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = int(state.get("step", 0))
    if rank == 0:
        LOGGER.info("Loaded trainer state from %s at step %d", state_path, step)
    return step


@torch.no_grad()
def validate(pipe, dataloader, args: argparse.Namespace, device, dtype, max_batches: int = 8) -> float:
    pipe.transformer.eval()
    losses = []
    for idx, batch in enumerate(dataloader):
        if idx >= max_batches:
            break
        losses.append(compute_batch_loss(pipe, batch, args, device, dtype).detach())
    pipe.transformer.train()
    if not losses:
        return float("nan")
    mean_loss = torch.stack(losses).mean()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
    return mean_loss.item()


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed(args)
    seed_everything(args.seed, rank)
    dtype = str_to_dtype(args.dtype)
    train_entry_types = set()
    for item in args.train_entry_types.split(","):
        normalized = normalize_entry_type(item)
        if normalized:
            train_entry_types.add(normalized)
    args.train_entry_types = ",".join(sorted(train_entry_types))

    validate_pipeline_entry_types(args, train_entry_types)

    train_dataset = CoRTAtomicDataset(
        args.train_files,
        cort_intermediate_dirs=args.cort_intermediate_dirs,
        train_entry_types=train_entry_types,
        cort_t2i_target=args.cort_t2i_target,
        edit_prompt_mode=args.edit_prompt_mode,
        edit_as_t2i=args.edit_as_t2i,
    )
    val_dataset = (
        CoRTAtomicDataset(
            args.val_files,
            cort_intermediate_dirs=args.cort_intermediate_dirs,
            train_entry_types=train_entry_types,
            cort_t2i_target=args.cort_t2i_target,
            edit_prompt_mode=args.edit_prompt_mode,
            edit_as_t2i=args.edit_as_t2i,
        )
        if args.val_files
        else None
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No trainable entries loaded. Check data paths and --train_entry_types.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_atomic_entries,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.train_batch_size,
            sampler=val_sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_atomic_entries,
            drop_last=False,
        )

    pipe = load_pipeline(args, dtype=dtype, device=device)
    maybe_apply_lora(pipe, args)
    use_fsdp = maybe_wrap_fsdp(pipe, args, world_size=world_size, device=device, dtype=dtype)
    pipe.transformer.train()

    optimizer = create_optimizer(pipe, args)
    scheduler = create_scheduler(optimizer, args)
    global_step = load_training_state(args.resume_from, optimizer, scheduler, rank) if args.resume_from else 0

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.output_dir) / "training_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        LOGGER.info("Training with args: %s", json.dumps(vars(args), indent=2, sort_keys=True))

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.total_epochs):
        train_sampler.set_epoch(epoch)
        iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch + 1}/{args.total_epochs}")
        for batch in iterator:
            grad_norm = torch.tensor(0.0, device=device)
            loss = compute_batch_loss(pipe, batch, args, device, dtype)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if use_fsdp:
                        grad_norm = pipe.transformer.clip_grad_norm_(args.max_grad_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if rank == 0 and global_step % args.log_freq == 0:
                lr = scheduler.get_last_lr()[0]
                iterator.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr:.2e}")
                LOGGER.info(
                    "step=%d loss=%.6f lr=%.6e grad_norm=%.4f",
                    global_step,
                    loss.detach().item(),
                    lr,
                    float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                )

            if args.test_freq > 0 and val_loader is not None and global_step % args.test_freq == 0:
                val_loss = validate(pipe, val_loader, args, device, dtype)
                if rank == 0:
                    LOGGER.info("step=%d val/loss=%.6f", global_step, val_loss)

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_checkpoint(pipe, optimizer, scheduler, args, global_step, rank, use_fsdp)

            if global_step >= args.total_training_steps:
                save_checkpoint(pipe, optimizer, scheduler, args, global_step, rank, use_fsdp)
                cleanup_distributed()
                return

    save_checkpoint(pipe, optimizer, scheduler, args, global_step, rank, use_fsdp)
    cleanup_distributed()


if __name__ == "__main__":
    main()
