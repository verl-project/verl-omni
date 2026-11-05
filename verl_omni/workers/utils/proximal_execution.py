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

"""Structural prerequisites for the opt-in single-update Megatron path.

These checks do not prove numerical equivalence of a new model or kernel.
Backend qualification still requires matched inference/training GPU tests.
The descriptor is recomputed at preparation and update, never a cached flag.
"""

import math
import os

import torch


def _zero(value, name):
    if value is None:
        return
    if type(value) not in (int, float, bool) or not math.isfinite(value) or value != 0:
        raise ValueError(f"Proximal reuse requires disabled {name}")


def _dropout_config(config, prefix, found):
    if isinstance(config, dict):
        for key, value in config.items():
            name = f"{prefix}.{key}"
            if isinstance(key, str) and any(
                token in key.lower() for token in ("dropout", "drop_path", "layerdrop", "jitter")
            ):
                _zero(value, name)
                found[name] = value
            elif isinstance(value, dict | list | tuple):
                _dropout_config(value, name, found)
    elif isinstance(config, list | tuple):
        for index, value in enumerate(config):
            _dropout_config(value, f"{prefix}.{index}", found)


def megatron_proximal_execution(engine):
    """Inspect initialized modules without moving tensors or changing train mode."""
    cfg, model, tf = engine.engine_config, engine.model_config, engine.tf_config
    if tf is None or not engine.module:
        raise ValueError("Model execution must already be initialized")
    if cfg.use_dynamic_bsz or cfg.micro_batch_size_per_gpu != cfg.infer_micro_batch_size_per_gpu:
        raise ValueError("Old/actor microbatch partitions differ")
    if not cfg.micro_batch_size_per_gpu or cfg.micro_batch_size_per_gpu < 0:
        raise ValueError("Static microbatch size is required")
    if cfg.forward_only or cfg.use_fused_kernels or cfg.dynamic_context_parallel:
        raise ValueError("Unqualified forward-only, fused-output or dynamic-CP execution")
    if cfg.pipeline_model_parallel_size != 1 or cfg.context_parallel_size != 1:
        raise ValueError("Pipeline/context parallel execution requires separate qualification")
    if getattr(cfg, "virtual_pipeline_model_parallel_size", None) is not None:
        raise ValueError("Virtual pipeline execution requires separate qualification")
    if getattr(cfg, "expert_model_parallel_size", None) != 1:
        raise ValueError("Expert parallel execution requires separate qualification")
    expert_tp = getattr(cfg, "expert_tensor_parallel_size", None)
    if expert_tp is not None and expert_tp != cfg.tensor_model_parallel_size:
        raise ValueError("Independent expert tensor parallel execution requires separate qualification")
    if engine.enable_routing_replay or engine._qat_enabled or engine.peft_cls is not None:
        raise ValueError("Replay, QAT or adapters require separate qualification")
    if engine.is_value_model or model.mtp.enable or model.lora.get("rank", 0) > 0:
        raise ValueError("Auxiliary heads or adapters require independent old inference")
    if model.lora_adapter_path is not None or model.lora_rank > 0:
        raise ValueError("Loaded adapters require independent old inference")
    checked = {
        "expert_model_parallel_size": cfg.expert_model_parallel_size,
        "expert_tensor_parallel_size": expert_tp,
        "tensor_model_parallel_size": cfg.tensor_model_parallel_size,
        "virtual_pipeline_model_parallel_size": cfg.virtual_pipeline_model_parallel_size,
    }
    for name in ("attention_dropout", "hidden_dropout"):
        if not hasattr(tf, name):
            raise ValueError(f"Missing actual TransformerConfig.{name}")
        checked[name] = getattr(tf, name)
        _zero(checked[name], name)
    for name in ("moe_input_jitter_eps", "moe_router_enable_expert_bias", "mtp_num_layers", "fp8", "fp4"):
        value = getattr(tf, name, None)
        _zero(value, name)
        checked[name] = value
    balancing = getattr(tf, "moe_router_load_balancing_type", "none")
    if balancing not in (None, "none"):
        raise ValueError("Training-only MoE balancing is not qualified")
    checked["moe_router_load_balancing_type"] = balancing
    if getattr(tf, "num_moe_experts", None):
        # Record the native router setting; do not require a private sorting patch.
        # Matching structure alone is not a numerical qualification.
        if getattr(tf, "moe_router_fusion", False):
            raise ValueError("Fused MoE routing requires separate qualification")
        if getattr(tf, "moe_router_score_function", None) != "softmax":
            raise ValueError("Non-softmax MoE routing requires separate qualification")
        checked["moe_router_score_function"] = "softmax"
        checked["moe_router_force_sorted_topk"] = getattr(tf, "moe_router_force_sorted_topk", False)
        checked["moe_router_fusion"] = False
    _dropout_config(model.hf_config.to_dict(), "hf", checked)
    kinds = []
    for chunk_index, chunk in enumerate(engine.module):
        if not isinstance(chunk, torch.nn.Module):
            raise ValueError("Unknown initialized module tree")
        for name, module in chunk.named_modules():
            path = f"module.{chunk_index}.{name}"
            kinds.append((path, f"{type(module).__module__}.{type(module).__qualname__}"))
            if any(
                cls.__module__.startswith("transformer_engine.") and cls.__name__ == "DotProductAttention"
                for cls in type(module).__mro__
            ):
                # This is TE's native setting, not a replacement attention
                # kernel. The default backward was not repeatable in the
                # paired model test; do not admit it from structural checks.
                deterministic = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
                if deterministic != "0":
                    raise ValueError("TE attention requires deterministic execution qualification")
                checked[f"{path}.NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = deterministic
            if (
                type(module).__module__ == "megatron.core.transformer.moe.router"
                and type(module).__name__ == "TopKRouter"
            ):
                router_config = module.config
                if getattr(router_config, "moe_router_fusion", False):
                    raise ValueError("Actual fused MoE router requires separate qualification")
                if getattr(router_config, "moe_router_score_function", None) != "softmax":
                    raise ValueError("Actual non-softmax MoE router requires separate qualification")
                # TopKRouter caches this attribute during construction and
                # uses it directly in routing, independently of its config.
                if getattr(module, "score_function", None) != "softmax":
                    raise ValueError("Actual runtime MoE score function requires separate qualification")
                checked[f"{path}.runtime_score_function"] = "softmax"
                checked[f"{path}.moe_router_score_function"] = "softmax"
                checked[f"{path}.moe_router_force_sorted_topk"] = getattr(
                    router_config, "moe_router_force_sorted_topk", False
                )
                checked[f"{path}.moe_router_fusion"] = False
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                raise ValueError(f"Train/eval running statistics are not qualified: {path}")
            if isinstance(module, torch.nn.modules.dropout._DropoutNd | torch.nn.RReLU):
                _zero(getattr(module, "p", 1), path)
            for attribute in ("dropout", "attention_dropout", "hidden_dropout", "drop_path_rate", "drop_prob"):
                value = getattr(module, attribute, None)
                if isinstance(value, torch.nn.Module):
                    continue  # The registered child itself is checked above.
                if value is not None:
                    _zero(value, f"{path}.{attribute}")
                    checked[f"{path}.{attribute}"] = value
            child_config = getattr(module, "config", None)
            if child_config is not None and hasattr(child_config, "to_dict"):
                _dropout_config(child_config.to_dict(), f"{path}.config", checked)
    return {"model_type": model.model_type, "checked_controls": checked, "module_types": kinds}
