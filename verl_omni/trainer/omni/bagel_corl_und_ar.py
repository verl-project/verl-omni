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
"""UND AR (Thinker) replica config for ``bagel_corl_sync``: deploy-config readers and the entry-config clone."""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig, OmegaConf, open_dict

from verl_omni.trainer.omni.bagel_corl_config import actor_merges_lora

logger = logging.getLogger(__name__)

#: Bagel publishes weights only (``model_type: bagel``, no ``auto_map``); the Thinker is an LLM and
#: needs the language sub-config that ships alongside in the snapshot.
UND_HF_SUBCONFIG = "llm_config.json"

#: ``load_format`` for the UND AR replica. ``dummy`` here means the Thinker decodes from randomly
#: initialised weights for the whole run: the replica is colocated and the naive checkpoint
#: backend never publishes to it.
UND_AR_LOAD_FORMAT = "auto"

#: Stage of the ``bagel_think`` deploy that holds the policy LLM. Stage 1 is the DiT, never trained by UND.
UND_AR_POLICY_STAGE_ID = 0

#: A merged LoRA target and the untied output head, both verified after publication.
UND_AR_SYNC_CHECK_PARAM = "language_model.model.layers.0.self_attn.o_proj.weight"
UND_AR_SYNC_CHECK_PARAMS = (UND_AR_SYNC_CHECK_PARAM, "language_model.lm_head.weight")


def und_deploy_config_path(config: DictConfig) -> str | None:
    """``agent.und_deploy_config`` or the ``engine_kwargs.vllm_omni.und_deploy_config`` fallback."""
    rollout = config.actor_rollout_ref.rollout
    path = (rollout.get("agent") or {}).get("und_deploy_config")
    if path:
        return str(path)
    ek = rollout.get("engine_kwargs") or {}
    vo = (ek.get("vllm_omni") or {}) if hasattr(ek, "get") else {}
    path = vo.get("und_deploy_config") if hasattr(vo, "get") else None
    return str(path) if path else None


def und_ar_stage_ids(deploy_config: str) -> tuple[int, ...]:
    """Stage ids declared by the UND AR deploy config (``bagel_think``: 0 + 1)."""
    if not os.path.isfile(deploy_config):
        raise ValueError(
            f"bagel_corl_sync cannot read agent.und_deploy_config={deploy_config!r} to derive the "
            "UND AR stage ids. Without them the per-stage tensor_parallel_size pin cannot be "
            "emitted, and every stage would inherit the replica-width value."
        )
    stages = OmegaConf.load(deploy_config).get("stages") or []
    stage_ids = tuple(int(stage["stage_id"]) for stage in stages if "stage_id" in stage)
    if not stage_ids:
        raise ValueError(f"bagel_corl_sync: {deploy_config!r} declares no stages to pin")
    return stage_ids


def und_stage_devices(deploy_config: str) -> dict[int, list[int]]:
    """Logical device indices each stage of the UND AR deploy config declares, in order."""
    stages = OmegaConf.load(deploy_config).get("stages") or []
    devices: dict[int, list[int]] = {}
    for stage in stages:
        if "stage_id" not in stage:
            continue
        runtime = stage.get("runtime") or {}
        raw = runtime.get("devices", stage.get("devices"))
        if raw is None:
            continue
        indices = [int(token) for token in str(raw).replace(",", " ").split() if token.strip()]
        if indices:
            devices[int(stage["stage_id"])] = indices
    return devices


def und_stage_device_indices(deploy_config: str) -> dict[int, int]:
    """Highest logical device index each stage of the UND AR deploy config asks for."""
    return {stage_id: max(indices) for stage_id, indices in und_stage_devices(deploy_config).items()}


def und_stage_max_model_len(deploy_config: str) -> int | None:
    """Widest ``max_model_len`` the UND AR deploy config declares across its stages."""
    if not os.path.isfile(deploy_config):
        return None
    stages = OmegaConf.load(deploy_config).get("stages") or []
    declared = [int(stage["max_model_len"]) for stage in stages if stage.get("max_model_len") is not None]
    return max(declared) if declared else None


def resolve_und_hf_config_path(model, agent, *, subconfig: str = UND_HF_SUBCONFIG) -> str | None:
    """HF config path for the UND AR replica (RFC §4.7): explicit knob, else the snapshot's LLM sub-config."""
    explicit = agent.get("und_hf_config_path")
    if explicit:
        return str(explicit)
    path = model.get("path")
    if not path:
        return None
    try:
        from verl_omni.utils.fs import resolve_model_local_dir

        local_dir = resolve_model_local_dir(str(path))
    except Exception as exc:  # noqa: BLE001 - optional derivation only
        logger.warning("bagel_corl_sync cannot locate model snapshot %s: %s", path, exc)
        return None
    candidate = os.path.join(local_dir, subconfig)
    return candidate if os.path.isfile(candidate) else None


def _und_model_node(model, agent, hf_config_path: str | None) -> dict:
    """``OmniModelConfig`` node for the Thinker: merged full weights (LoRA rank 0 when the actor merges)."""
    return {
        "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
        "path": model.get("path"),
        "tokenizer_path": model.get("tokenizer_path") or model.get("path"),
        "model_type": "omni_model",
        "architecture": model.get("architecture") or "OmniBagelForConditionalGeneration",
        "trust_remote_code": bool(model.get("trust_remote_code", True)),
        "composite_mode": "bagel_corl",
        "lora_rank": 0 if actor_merges_lora(model) else int(model.get("lora_rank") or 0),
        "lora_alpha": int(model.get("lora_alpha") or 0),
        "hf_config_path": hf_config_path,
        "load_tokenizer": False,
    }


def _und_rollout_node(
    rollout, agent, *, und_deploy, max_prompt, max_resp, max_model_len, load_format, stage_id
) -> dict:
    """``RolloutConfig`` node for a standalone ``vllm_omni`` AR replica driven by ``und_deploy``.

    ``enable_prefix_caching`` is off because a top-level engine arg is copied into every stage
    and the DiT stage rejects it; ``max_num_batched_tokens`` covers one ~8625-token image.
    """
    und_n_gpus = int(agent.get("und_n_gpus") or 1)
    ckpt = rollout.get("checkpoint_engine")
    ckpt_engine = (OmegaConf.to_container(ckpt, resolve=True) if ckpt is not None else None) or {"backend": "naive"}
    return {
        "_target_": "verl.workers.config.RolloutConfig",
        "name": "vllm_omni",
        "load_format": load_format,
        "logprobs_mode": rollout.get("logprobs_mode", "processed_logprobs"),
        "tensor_model_parallel_size": und_n_gpus,
        "data_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "n_gpus_per_node": und_n_gpus,
        "nnodes": 1,
        "prompt_length": max_prompt,
        "response_length": max_resp,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": int(agent.get("und_max_num_batched_tokens") or 16384),
        "gpu_memory_utilization": float(agent.get("und_gpu_memory_utilization") or 0.40),
        "enforce_eager": True,
        "free_cache_engine": True,
        "enable_sleep_mode": True,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "checkpoint_engine": ckpt_engine,
        "disaggregation": {"enabled": False},
        "prometheus": {"_target_": "verl.workers.config.PrometheusConfig", "enable": False},
        "engine_kwargs": {
            "vllm_omni": {
                "output_mode": "ar",
                "deploy_config": und_deploy,
                "stage_overrides": {str(sid): {"tensor_parallel_size": 1} for sid in und_ar_stage_ids(und_deploy)},
                "weight_sync_stage_ids": [stage_id],
            }
        },
        "agent": OmegaConf.to_container(agent, resolve=True) or {},
    }


def build_und_ar_entrypoint_config(
    config: DictConfig,
    *,
    load_format: str = UND_AR_LOAD_FORMAT,
    policy_stage_id: int = UND_AR_POLICY_STAGE_ID,
    hf_config_path: str | None = None,
    max_model_len_from_deploy=und_stage_max_model_len,
) -> DictConfig:
    """Clone the entry config for a standalone UND AR ``LLMServerManager`` (Omni model, ``output_mode=ar``)."""
    und_deploy = und_deploy_config_path(config)
    if not und_deploy:
        raise ValueError("bagel_corl_sync dual-role requires agent.und_deploy_config")
    model = config.actor_rollout_ref.model
    rollout = config.actor_rollout_ref.rollout
    agent = rollout.get("agent") or {}
    if hf_config_path is None:
        hf_config_path = resolve_und_hf_config_path(model, agent)
    max_prompt = int(config.data.get("max_prompt_length") or rollout.get("prompt_length") or 1024)
    max_resp = int(config.data.get("max_response_length") or rollout.get("response_length") or max_prompt)
    max_model_len = int(
        agent.get("und_max_model_len") or max_model_len_from_deploy(und_deploy) or (max_prompt + max_resp)
    )
    if max_model_len < max_prompt + max_resp:
        raise ValueError(
            f"bagel_corl_sync: UND AR max_model_len={max_model_len} is smaller than the episode "
            f"context the loop has to serve (max_prompt_length={max_prompt} + "
            f"max_response_length={max_resp} = {max_prompt + max_resp}). The loop starts a turn while "
            "len(prompt_ids)+len(response_ids) is still below that budget, and the AR strategy rejects "
            "a prompt that leaves no room for generation ('Prompt length (...) meets or exceeds the "
            "model's maximum context length'). Lower data.max_prompt_length/data.max_response_length, "
            "or raise agent.und_max_model_len / the deploy config's max_model_len."
        )
    und_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    with open_dict(und_cfg):
        und_cfg.actor_rollout_ref.model = OmegaConf.create(_und_model_node(model, agent, hf_config_path))
        und_cfg.actor_rollout_ref.rollout = OmegaConf.create(
            _und_rollout_node(
                rollout,
                agent,
                und_deploy=und_deploy,
                max_prompt=max_prompt,
                max_resp=max_resp,
                max_model_len=max_model_len,
                load_format=load_format,
                stage_id=policy_stage_id,
            )
        )
    return und_cfg


def validate_und_ar_pool(und_rollout, und_n_gpus: int, deploy_config: str, actor_world_size: int | None = None) -> None:
    """Fail loud when the AR replica's pool and its deploy config cannot agree."""
    declared_world_size = (
        int(und_rollout.get("tensor_model_parallel_size") or 1)
        * int(und_rollout.get("data_parallel_size") or 1)
        * int(und_rollout.get("pipeline_model_parallel_size") or 1)
    )
    if declared_world_size != und_n_gpus:
        raise ValueError(
            f"bagel_corl_sync: the UND AR replica declares world_size={declared_world_size} "
            f"(TP*DP*PP) but agent.und_n_gpus={und_n_gpus} workers are handed to it. "
            "The replica's width declaration and the actor-pool slice must match."
        )
    if actor_world_size is not None and actor_world_size % und_n_gpus:
        raise ValueError(
            f"bagel_corl_sync: agent.und_n_gpus={und_n_gpus} does not divide the actor pool "
            f"(n_gpus_per_node={actor_world_size}). The UND AR slice is taken with "
            "split_resource_pool, which requires an exact split, so this would otherwise die "
            "as 'split_size must be a divisor of world_size'. Pick a divisor of the pool "
            "(on a 4-card pool: UND_N_GPUS in {1, 2, 4})."
        )
    widest_stage_device = 1 + max(und_stage_device_indices(deploy_config).values(), default=0)
    if und_n_gpus < widest_stage_device:
        raise ValueError(
            f"bagel_corl_sync: agent.und_n_gpus={und_n_gpus} cannot cover the stage devices in "
            f"{deploy_config!r} (needs >= {widest_stage_device}). Stage devices are logical "
            "indices into the replica's CUDA_VISIBLE_DEVICES; either raise UND_N_GPUS or put "
            "every stage back on logical device 0 (1-GPU smoke test)."
        )
