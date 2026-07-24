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

Metrics:
    raw_policy.accuracy:
        Pairwise win rate for the LoRA policy. It is the fraction of held-out
        preference pairs where policy_logp(chosen) > policy_logp(rejected).
    raw_policy.mean_margin:
        Mean policy log-probability gap,
        policy_logp(chosen) - policy_logp(rejected).
    reference_raw.accuracy:
        Pairwise win rate for the reference/base model with LoRA disabled.
        This provides the pre-DPO baseline on the same preference pairs.
    reference_raw.mean_margin:
        Mean reference log-probability gap,
        reference_logp(chosen) - reference_logp(rejected).
    dpo_reward.accuracy:
        Pairwise win rate after reference adjustment. It is the fraction of
        pairs where the DPO reward for chosen is higher than rejected.
    dpo_reward.mean_margin:
        Mean DPO reward gap,
        (policy_logp(chosen) - reference_logp(chosen))
        - (policy_logp(rejected) - reference_logp(rejected)).

By default, log-probabilities are summed over answer tokens. Pass
``--average-log-prob`` to compare mean token log-probabilities instead, which is
often easier to compare across answers with different lengths.

Example:

    python examples/dpo_trainer/qwen3_omni/validate_offline_dpo_lora.py \
        --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
        --adapter-path /path/to/checkpoints/global_step_50/actor \
        --data-files \
            /path/to/omni-preference/image/test.parquet \
            /path/to/omni-preference/video/test.parquet \
            /path/to/omni-preference/audio/test.parquet \
        --batch-size 1 \
        --dtype bfloat16 \
        --attn-implementation flash_attention_2 \
        --output-jsonl outputs/qwen3_omni_dpo_lora_eval.jsonl

If OOM, please try to reduce the batch size or set `--device-map "auto"` to split the model across multiple GPUs.
"""

from __future__ import annotations

import argparse
import importlib
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
from verl_omni.utils.peft_utils import peft_adapters_disabled, set_peft_adapter  # noqa: E402

logger = logging.getLogger("qwen3_omni_dpo_validation")

DEFAULT_EXTERNAL_LIB = "verl_omni.models.transformers.qwen3_omni_thinker_experts"
DEFAULT_EXCLUDE_MODULES = ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"
DEFAULT_MM_CONFIGS = {
    "scale_factor": 28,
    "image_min_pixels": 3136,
    "image_max_pixels": 602112,
    "video_min_pixels": 3136,
    "video_max_pixels": 602112,
    "max_ratio": 200,
    "min_frames": 2,
    "max_frames": 4,
    "frame_factor": 1,
    "sample_rate": 16000,
    "fps": 2.0,
    "use_audio_in_video": False,
}


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


@dataclass
class PreferenceScores:
    chosen_logps: torch.Tensor
    rejected_logps: torch.Tensor
    label_token_counts: torch.Tensor


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
    parser.add_argument(
        "--external-lib",
        default=DEFAULT_EXTERNAL_LIB,
        help=(
            "Optional external lib that registers the omni adapter. Defaults to the Qwen3-Omni expert unfuse "
            "module used by the training script."
        ),
    )
    parser.add_argument(
        "--average-log-prob",
        action="store_true",
        help="Use mean token logprob instead of sum logprob.",
    )
    parser.add_argument(
        "--exclude-modules",
        default=DEFAULT_EXCLUDE_MODULES,
        help=(
            "Regex/list passed to PEFT LoraConfig.exclude_modules before injection. Defaults to the thinker-only "
            "training script exclusion pattern; pass an empty string to use the adapter config unchanged."
        ),
    )
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N batches.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument("--source-name-key", default="data_source")
    parser.add_argument(
        "--mm-configs",
        default=None,
        help=(
            "JSON string for multimodal transform kwargs. If omitted, validation uses the same Qwen3-Omni "
            "defaults as the LoRA DPO training script; pass '{}' to intentionally use processor defaults."
        ),
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip base/reference log-probs and report only raw policy chosen-vs-rejected accuracy.",
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
        force=True,
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


def iter_external_libs(external_lib: str | None) -> list[str]:
    if not external_lib:
        return []
    return [module.strip() for module in external_lib.split(",") if module.strip()]


def maybe_unfuse_qwen3_omni_experts(model, external_lib: str | None) -> None:
    """Mirror the training-time Qwen3-Omni expert structure before loading LoRA."""
    for module_name in iter_external_libs(external_lib):
        module = importlib.import_module(module_name)
        unfuse_fn = getattr(module, "unfuse_qwen3_omni_thinker_experts", None)
        if callable(unfuse_fn):
            converted = unfuse_fn(model)
            logger.info("External lib %s unfused %d thinker expert module(s)", module_name, converted)


def score_preference_batch(
    *,
    model,
    model_config,
    model_batch: dict[str, Any],
    input_device: torch.device,
    average_log_prob: bool,
) -> PreferenceScores:
    model_inputs, labels, segment_ranges = prepare_omni_preference_inputs(
        model_config,
        model_batch,
        dtype=next(model.parameters()).dtype,
    )
    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    chosen_logps, rejected_logps = compute_omni_preference_logps(
        model_config,
        logits,
        labels,
        segment_ranges,
        average_log_prob=average_log_prob,
    )
    label_token_counts = (labels != -100).sum(dim=-1).detach().to(input_device)
    return PreferenceScores(
        chosen_logps=chosen_logps,
        rejected_logps=rejected_logps,
        label_token_counts=label_token_counts,
    )


def load_lora_adapter_weights(
    model,
    adapter_path: str,
    adapter_name: str = "default",
    exclude_modules: str | list[str] | None = None,
) -> None:
    """Load LoRA weights without wrapping the model in ``PeftModel``."""
    load_lora_adapter = getattr(model, "load_lora_adapter", None)
    if callable(load_lora_adapter):
        if exclude_modules:
            logger.warning(
                "Native load_lora_adapter() does not receive validation exclude_modules=%r.", exclude_modules
            )
        load_lora_adapter(adapter_path, adapter_name=adapter_name)
        return

    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
    from safetensors.torch import load_file as safetensors_load_file

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"LoRA adapter config not found at {adapter_config_path}")
    if not os.path.isfile(adapter_weights_path):
        raise FileNotFoundError(f"LoRA adapter weights not found at {adapter_weights_path}")

    if hasattr(LoraConfig, "from_dict"):
        with open(adapter_config_path) as f:
            lora_config = LoraConfig.from_dict(json.load(f))
    else:
        lora_config = LoraConfig.from_pretrained(adapter_path)

    if exclude_modules:
        if hasattr(lora_config, "exclude_modules"):
            logger.info("Applying LoRA exclude_modules before injection: %s", exclude_modules)
            lora_config.exclude_modules = exclude_modules
        else:
            logger.warning("PEFT LoraConfig does not expose exclude_modules; using adapter config unchanged.")

    inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)
    adapter_state_dict = safetensors_load_file(adapter_weights_path)
    current_state = get_peft_model_state_dict(model, adapter_name=adapter_name)

    adapter_state_by_key = dict(adapter_state_dict)
    for key, tensor in adapter_state_dict.items():
        if key.startswith("base_model.model."):
            adapter_state_by_key.setdefault(key.removeprefix("base_model.model."), tensor)

    loadable_keys = {
        key
        for key, tensor in current_state.items()
        if key in adapter_state_by_key and tensor.shape == adapter_state_by_key[key].shape
    }
    missing_load = set(current_state) - loadable_keys
    unexpected_load = set(adapter_state_dict) - {
        key if key in adapter_state_dict else f"base_model.model.{key}" for key in loadable_keys
    }
    if not loadable_keys:
        raise RuntimeError(
            "No matching LoRA keys found between adapter checkpoint and injected model. "
            f"Missing={len(missing_load)}, unexpected={len(unexpected_load)}."
        )

    if missing_load:
        logger.warning(
            "LoRA adapter %r: %d keys in model but not in checkpoint. They will keep their initial values.",
            adapter_name,
            len(missing_load),
        )
    if unexpected_load:
        logger.warning(
            "LoRA adapter %r: %d keys in checkpoint but not in model. They will be ignored.",
            adapter_name,
            len(unexpected_load),
        )

    with torch.no_grad():
        for key in loadable_keys:
            current_state[key].copy_(adapter_state_by_key[key])


def set_lora_adapter(model, adapter_name: str = "default") -> None:
    """Activate an adapter for both wrapped PEFT models and manually injected modules."""
    activated = set_peft_adapter(model, adapter_name)
    if activated == 0:
        logger.warning("LoRA adapter %r: no set_adapter hooks found after injection.", adapter_name)


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
    adapter_cls = OmniModelBase.get_class_by_name(
        model_config.architecture,
        model_config.model_stage,
        model_config.external_lib,
    )
    tokenizer = adapter_cls.configure_tokenizer(args.model_path, model_config)
    processor = adapter_cls.configure_processor(args.model_path, model_config)
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
    maybe_unfuse_qwen3_omni_experts(model, args.external_lib)

    logger.info("Loading LoRA adapter weights from %s", adapter_path)
    load_lora_adapter_weights(model, adapter_path, exclude_modules=args.exclude_modules or None)

    set_lora_adapter(model, "default")
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
    mm_configs = json.loads(args.mm_configs) if args.mm_configs is not None else dict(DEFAULT_MM_CONFIGS)
    logger.info("Building held-out dataset from %s", args.data_files)
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

    raw_stats = RunningStats()
    raw_stats_by_modality: dict[str, RunningStats] = defaultdict(RunningStats)
    reference_raw_stats = RunningStats()
    reference_raw_stats_by_modality: dict[str, RunningStats] = defaultdict(RunningStats)
    dpo_stats = RunningStats()
    dpo_stats_by_modality: dict[str, RunningStats] = defaultdict(RunningStats)
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
            policy_scores = score_preference_batch(
                model=model,
                model_config=model_config,
                model_batch=model_batch,
                input_device=input_device,
                average_log_prob=args.average_log_prob,
            )
            reference_scores = None
            if not args.skip_reference:
                with peft_adapters_disabled(model):
                    reference_scores = score_preference_batch(
                        model=model,
                        model_config=model_config,
                        model_batch=model_batch,
                        input_device=input_device,
                        average_log_prob=args.average_log_prob,
                    )

            result_rows = []
            for offset, index in enumerate(indices):
                chosen = float(policy_scores.chosen_logps[offset].detach().cpu())
                rejected = float(policy_scores.rejected_logps[offset].detach().cpu())
                raw_margin = chosen - rejected
                raw_stats.update(raw_margin)
                raw_stats_by_modality[modality].update(raw_margin)
                row = {
                    "index": int(index),
                    "uid": safe_json_value(batch.get("uid", [None])[offset]),
                    "modality": modality,
                    "policy_chosen_logp": chosen,
                    "policy_rejected_logp": rejected,
                    "raw_policy_margin": raw_margin,
                    "raw_policy_correct": raw_margin > 0,
                    "chosen_label_tokens": int(policy_scores.label_token_counts[offset * 2].detach().cpu()),
                    "rejected_label_tokens": int(policy_scores.label_token_counts[offset * 2 + 1].detach().cpu()),
                }
                if reference_scores is not None:
                    ref_chosen = float(reference_scores.chosen_logps[offset].detach().cpu())
                    ref_rejected = float(reference_scores.rejected_logps[offset].detach().cpu())
                    reference_raw_margin = ref_chosen - ref_rejected
                    reference_raw_stats.update(reference_raw_margin)
                    reference_raw_stats_by_modality[modality].update(reference_raw_margin)
                    chosen_reward = chosen - ref_chosen
                    rejected_reward = rejected - ref_rejected
                    dpo_margin = chosen_reward - rejected_reward
                    dpo_stats.update(dpo_margin)
                    dpo_stats_by_modality[modality].update(dpo_margin)
                    row.update(
                        {
                            "reference_chosen_logp": ref_chosen,
                            "reference_rejected_logp": ref_rejected,
                            "reference_raw_margin": reference_raw_margin,
                            "reference_raw_correct": reference_raw_margin > 0,
                            "chosen_reward": chosen_reward,
                            "rejected_reward": rejected_reward,
                            "dpo_margin": dpo_margin,
                            "dpo_correct": dpo_margin > 0,
                        }
                    )
                result_rows.append(row)
            write_results(args.output_jsonl, result_rows)

            del features, batch, model_batch, policy_scores, reference_scores, result_rows
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

            batch_count += 1
            if args.log_every > 0 and batch_count % args.log_every == 0:
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "Progress: batches=%d/%d samples=%d/%d raw_accuracy=%.4f raw_margin=%.4f "
                    "reference_raw_accuracy=%.4f reference_raw_margin=%.4f "
                    "dpo_accuracy=%.4f dpo_margin=%.4f elapsed=%.1fs",
                    batch_count,
                    total_batches,
                    raw_stats.total,
                    len(dataset),
                    raw_stats.accuracy,
                    raw_stats.mean_margin,
                    reference_raw_stats.accuracy,
                    reference_raw_stats.mean_margin,
                    dpo_stats.accuracy,
                    dpo_stats.mean_margin,
                    elapsed,
                )

    summary = {
        "raw_policy": {
            "overall": raw_stats.to_dict(),
            "by_modality": {
                modality: modality_stats.to_dict() for modality, modality_stats in raw_stats_by_modality.items()
            },
        },
        "reference_raw": None
        if args.skip_reference
        else {
            "overall": reference_raw_stats.to_dict(),
            "by_modality": {
                modality: modality_stats.to_dict()
                for modality, modality_stats in reference_raw_stats_by_modality.items()
            },
        },
        "dpo_reward": None
        if args.skip_reference
        else {
            "overall": dpo_stats.to_dict(),
            "by_modality": {
                modality: modality_stats.to_dict() for modality, modality_stats in dpo_stats_by_modality.items()
            },
        },
        "average_log_prob": args.average_log_prob,
        "skip_reference": args.skip_reference,
    }
    logger.info("Validation finished in %.1fs", time.perf_counter() - started_at)
    logger.info(
        "Final raw_accuracy=%.4f raw_margin=%.4f reference_raw_accuracy=%.4f "
        "reference_raw_margin=%.4f dpo_accuracy=%.4f dpo_margin=%.4f",
        raw_stats.accuracy,
        raw_stats.mean_margin,
        reference_raw_stats.accuracy,
        reference_raw_stats.mean_margin,
        dpo_stats.accuracy,
        dpo_stats.mean_margin,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
