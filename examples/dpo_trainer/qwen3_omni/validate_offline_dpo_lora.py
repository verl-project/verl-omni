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

"""Validate Qwen3-Omni offline DPO LoRA with held-out preference accuracy.

The metric is the fraction of held-out preference pairs where the policy assigns
higher log-probability to the chosen answer than to the rejected answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_omni.pipelines.model_base import OmniModelBase  # noqa: E402
from verl_omni.pipelines.utils import compute_omni_preference_logps, prepare_omni_preference_inputs  # noqa: E402
from verl_omni.utils.dataset.offline_mllm_dpo_dataset import (  # noqa: E402
    OfflineMLLMDPODataset,
    offline_mllm_dpo_collate_fn,
)

logger = logging.getLogger("qwen3_omni_dpo_validation")


@dataclass
class EvalModelConfig:
    """Small config object compatible with Qwen3-Omni adapter helpers."""

    path: str
    architecture: str
    processor: Any
    model_stage: str = "thinker"
    trust_remote_code: bool = True
    external_lib: str | None = None
    lora_rank: int = 1
    lora_adapter_path: str | None = None

    def get_processor(self):
        return self.processor

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class RunningStats:
    total: int = 0
    correct: int = 0
    ties: int = 0
    margin_sum: float = 0.0

    def update(self, margin: float) -> None:
        self.total += 1
        self.margin_sum += margin
        if margin > 0:
            self.correct += 1
        elif margin == 0:
            self.ties += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mean_margin(self) -> float:
        return self.margin_sum / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "correct": self.correct,
            "ties": self.ties,
            "accuracy": self.accuracy,
            "mean_margin": self.mean_margin,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base Qwen3-Omni model path or HF repo id.")
    parser.add_argument(
        "--adapter-path",
        required=True,
        help=(
            "PEFT LoRA adapter path or a verl FSDP checkpoint directory. If an FSDP checkpoint directory is "
            "passed, the script exports <adapter-path>/lora_adapter before validation."
        ),
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        required=True,
        help="Held-out Omni-Preference parquet/json/jsonl files.",
    )
    parser.add_argument("--output-jsonl", default=None, help="Optional per-sample logprob result JSONL path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Validation batch size within each modality.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit samples for smoke tests.")
    parser.add_argument("--device", default="cuda", help="Device to run validation on, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--device-map",
        default=None,
        help='Optional transformers device_map, e.g. "auto". When set, --device is used only for fallback logging.',
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. Use auto to let transformers choose.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional HF attention implementation override, e.g. flash_attention_2 or sdpa.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-lib", default=None, help="Optional external lib that registers the omni adapter.")
    parser.add_argument(
        "--average-log-prob",
        action="store_true",
        help="Use mean token logprob instead of sum logprob.",
    )
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N batches.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument("--source-name-key", default="data_source")
    parser.add_argument(
        "--mm-configs",
        default=None,
        help='JSON string for multimodal transform kwargs, e.g. \'{"fps": 2.0, "max_frames": 16}\'.',
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def get_model_input_device(model) -> torch.device:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def resolve_adapter_path(adapter_path: str, base_model_name_or_path: str | None = None) -> str:
    path = Path(os.path.expanduser(adapter_path))
    logger.info("Resolving LoRA adapter path from %s", path)
    candidates = [path, path / "lora_adapter", path / "huggingface"]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            logger.info("Found PEFT adapter at %s", candidate)
            return str(candidate)

    fsdp_files = [
        path / "fsdp_config.json",
        path / "lora_train_meta.json",
    ]
    if all(fsdp_file.is_file() for fsdp_file in fsdp_files):
        from verl_omni.utils.fsdp_utils import export_fsdp_lora_adapter

        logger.info("Detected raw FSDP LoRA checkpoint at %s; exporting PEFT adapter", path)
        result = export_fsdp_lora_adapter(input_dir=path, base_model_name_or_path=base_model_name_or_path)
        logger.info("Exported PEFT adapter to %s", result["output_dir"])
        return result["output_dir"]

    raise FileNotFoundError(
        "Could not find adapter_config.json in "
        f"{path}, {path / 'lora_adapter'}, or {path / 'huggingface'}. "
        "If this is a raw FSDP LoRA checkpoint, ensure fsdp_config.json and lora_train_meta.json exist."
    )


def load_qwen3_omni_lora_model(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModelForMultimodalLM
    from verl.utils.import_utils import import_external_libs

    logger.info("Stage 1/4: resolving adapter checkpoint")
    adapter_path = resolve_adapter_path(args.adapter_path, base_model_name_or_path=args.model_path)
    if args.external_lib is not None:
        logger.info("Importing external library: %s", args.external_lib)
        import_external_libs(args.external_lib)
    config_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.attn_implementation is not None:
        config_kwargs["attn_implementation"] = args.attn_implementation
    logger.info("Stage 2/4: loading HF config from %s", args.model_path)
    hf_config = AutoConfig.from_pretrained(args.model_path, **config_kwargs)
    if not hasattr(hf_config, "tie_word_embeddings"):
        hf_config.tie_word_embeddings = False
    architecture = hf_config.architectures[0]
    logger.info("Detected architecture=%s", architecture)

    model_config = EvalModelConfig(
        path=args.model_path,
        architecture=architecture,
        processor=None,
        trust_remote_code=args.trust_remote_code,
        external_lib=args.external_lib,
        lora_adapter_path=adapter_path,
    )
    logger.info("Stage 3/4: loading tokenizer and processor")
    tokenizer, processor = OmniModelBase.load_tokenizer_and_processor(args.model_path, model_config)
    model_config.processor = processor

    torch_dtype = resolve_dtype(args.dtype)
    logger.info(
        "Stage 4/4: loading base model with dtype=%s device=%s device_map=%s attn=%s",
        args.dtype,
        args.device,
        args.device_map or "none",
        args.attn_implementation or "default",
    )
    model_kwargs = {
        "config": hf_config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForMultimodalLM.from_pretrained(args.model_path, **model_kwargs)
    adapter_cls = OmniModelBase.get_class(model_config)
    model = adapter_cls.configure_model(model, model_config)

    logger.info("Loading LoRA adapter weights from %s", adapter_path)
    if hasattr(model, "load_lora_adapter"):
        model.load_lora_adapter(adapter_path)
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    if args.device_map is None:
        logger.info("Moving model to %s and switching to eval mode", args.device)
        model.to(args.device)
    else:
        logger.info("Using transformers device_map=%s; skipping model.to(%s)", args.device_map, args.device)
    model.eval()
    input_device = get_model_input_device(model)
    logger.info("Model and adapter are ready; input tensors will be moved to %s", input_device)
    return model, tokenizer, processor, model_config, adapter_path, input_device


def build_dataset(args: argparse.Namespace, processor) -> OfflineMLLMDPODataset:
    mm_configs = json.loads(args.mm_configs) if args.mm_configs else {}
    logger.info("Building held-out dataset from %s", args.data_files)
    if mm_configs:
        logger.info("Using multimodal transform configs: %s", mm_configs)
    data_config = OmegaConf.create(
        {
            "prompt_key": args.prompt_key,
            "chosen_key": args.chosen_key,
            "rejected_key": args.rejected_key,
            "source_name_key": args.source_name_key,
            "base_transform": "qwen3_omni_moe",
            "data_source": "offline_mllm_dpo",
            "mm_configs": mm_configs,
        }
    )
    dataset = OfflineMLLMDPODataset(
        data_files=args.data_files,
        tokenizer=None,
        processor=processor,
        config=data_config,
        max_samples=args.max_samples,
    )
    logger.info("Dataset ready with %d sample(s)", len(dataset))
    return dataset


def iter_modality_batches(dataset: OfflineMLLMDPODataset, batch_size: int):
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    indices_by_modality: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        indices_by_modality[dataset.get_modality(index)].append(index)

    for modality in sorted(indices_by_modality):
        indices = indices_by_modality[modality]
        for start in range(0, len(indices), batch_size):
            yield modality, indices[start : start + batch_size]


def tensor_batch_only(batch: dict[str, Any], device: str | torch.device, average_log_prob: bool) -> dict[str, Any]:
    model_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    model_batch["average_log_prob"] = average_log_prob
    return model_batch


def safe_json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def write_results(output_jsonl: str | None, rows: list[dict[str, Any]]) -> None:
    if output_jsonl is None:
        return
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def modality_counts(dataset: OfflineMLLMDPODataset) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for index in range(len(dataset)):
        counts[dataset.get_modality(index)] += 1
    return dict(sorted(counts.items()))


def count_batches(counts: dict[str, int], batch_size: int) -> int:
    return sum((count + batch_size - 1) // batch_size for count in counts.values())


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    started_at = time.perf_counter()
    logger.info("Starting Qwen3-Omni offline DPO LoRA validation")
    logger.info(
        "Config: model=%s adapter=%s batch_size=%d max_samples=%d device=%s device_map=%s "
        "average_log_prob=%s output_jsonl=%s",
        args.model_path,
        args.adapter_path,
        args.batch_size,
        args.max_samples,
        args.device,
        args.device_map or "none",
        args.average_log_prob,
        args.output_jsonl,
    )
    model, _tokenizer, processor, model_config, adapter_path, input_device = load_qwen3_omni_lora_model(args)
    dataset = build_dataset(args, processor)
    counts = modality_counts(dataset)
    total_batches = count_batches(counts, args.batch_size)
    logger.info("Validation modality counts: %s", counts)
    logger.info("Validation will run %d batch(es)", total_batches)

    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        if output_path.exists():
            logger.info("Removing existing output JSONL: %s", output_path)
            output_path.unlink()
        logger.info("Per-sample results will be written to %s", output_path)

    stats = RunningStats()
    stats_by_modality: dict[str, RunningStats] = defaultdict(RunningStats)
    batch_count = 0
    current_modality = None
    logger.info("Loaded adapter: %s", adapter_path)
    logger.info("Evaluating %d held-out preference pair(s)", len(dataset))

    with torch.inference_mode():
        for modality, indices in iter_modality_batches(dataset, args.batch_size):
            if modality != current_modality:
                current_modality = modality
                logger.info("Starting modality=%s with %d sample(s)", modality, counts.get(modality, 0))
            logger.debug(
                "Preparing batch %d/%d modality=%s indices=%s", batch_count + 1, total_batches, modality, indices
            )
            features = [dataset[index] for index in indices]
            batch = offline_mllm_dpo_collate_fn(features)
            model_batch = tensor_batch_only(batch, input_device, args.average_log_prob)
            logger.debug("Building preference model inputs for batch %d/%d", batch_count + 1, total_batches)
            model_inputs, labels, segment_ranges = prepare_omni_preference_inputs(
                model_config,
                model_batch,
                dtype=next(model.parameters()).dtype,
            )
            logger.debug("Running forward pass for batch %d/%d", batch_count + 1, total_batches)
            outputs = model(**model_inputs, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            chosen_logps, rejected_logps = compute_omni_preference_logps(
                model_config,
                logits,
                labels,
                segment_ranges,
                average_log_prob=args.average_log_prob,
            )

            result_rows = []
            for offset, index in enumerate(indices):
                chosen = float(chosen_logps[offset].detach().cpu())
                rejected = float(rejected_logps[offset].detach().cpu())
                margin = chosen - rejected
                stats.update(margin)
                stats_by_modality[modality].update(margin)
                result_rows.append(
                    {
                        "index": int(index),
                        "uid": safe_json_value(batch.get("uid", [None])[offset]),
                        "modality": modality,
                        "chosen_logp": chosen,
                        "rejected_logp": rejected,
                        "margin": margin,
                        "correct": margin > 0,
                    }
                )
            write_results(args.output_jsonl, result_rows)

            batch_count += 1
            if args.log_every > 0 and batch_count % args.log_every == 0:
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "Progress: batches=%d/%d samples=%d/%d accuracy=%.4f mean_margin=%.4f elapsed=%.1fs",
                    batch_count,
                    total_batches,
                    stats.total,
                    len(dataset),
                    stats.accuracy,
                    stats.mean_margin,
                    elapsed,
                )

    summary = {
        "overall": stats.to_dict(),
        "by_modality": {modality: modality_stats.to_dict() for modality, modality_stats in stats_by_modality.items()},
        "average_log_prob": args.average_log_prob,
    }
    logger.info("Validation finished in %.1fs", time.perf_counter() - started_at)
    logger.info("Final accuracy=%.4f mean_margin=%.4f", stats.accuracy, stats.mean_margin)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
