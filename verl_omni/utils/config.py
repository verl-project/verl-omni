# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail-fast validation shared by VeRL-Omni trainer entrypoints."""

from __future__ import annotations

from typing import Any


def _select(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for part in path.split("."):
        if value is None:
            return default
        if hasattr(value, "get"):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
    return default if value is None else value


def _validate_delta_sharded(config: Any) -> None:
    """Fail-fast gates for the ``delta_sharded`` checkpoint engine backend (RFC #38).

    The delta path syncs full weights to standalone rollout replicas and diffs each
    rank's shard against the last export, so it is only defined for full-weight
    separate_async runs; anything else raises here instead of degrading silently.
    """
    backend = _select(config, "actor_rollout_ref.rollout.checkpoint_engine.backend")
    if backend != "delta_sharded":
        return

    mode = _select(config, "trainer.v1.trainer_mode")
    valid_modes = ("separate_async", "omni_separate_async")
    if mode not in valid_modes:
        raise ValueError(
            f"actor_rollout_ref.rollout.checkpoint_engine.backend='delta_sharded' requires "
            f"trainer.v1.trainer_mode in {list(valid_modes)} (got {mode!r})."
        )
    if mode == "separate_async" and not _select(config, "trainer.use_v1", False):
        raise ValueError(
            "checkpoint_engine.backend='delta_sharded' requires the v1 diffusion trainer "
            "(trainer.use_v1=true); the legacy trainer does not drive the delta sync flow."
        )

    model_key = "actor_rollout_ref.model"
    lora_rank = _select(config, f"{model_key}.lora.rank", 0) or 0
    legacy_lora_rank = _select(config, f"{model_key}.lora_rank", 0) or 0
    lora_adapter_path = _select(config, f"{model_key}.lora_adapter_path")
    if lora_rank > 0 or legacy_lora_rank > 0 or lora_adapter_path is not None:
        raise ValueError(
            "checkpoint_engine.backend='delta_sharded' requires full-weight training; LoRA is "
            "refused in either merge mode because the shard export's names and values differ "
            "from the adapter (merge=false) and merged (merge=true) full exports. "
            f"Got {model_key}.lora.rank={lora_rank}, {model_key}.lora_rank={legacy_lora_rank}, "
            f"{model_key}.lora_adapter_path={lora_adapter_path}."
        )

    for qat_path in ("actor_rollout_ref.actor.fsdp_config.qat.enable", "actor_rollout_ref.actor.megatron.qat.enable"):
        if _select(config, qat_path, False):
            raise ValueError(
                f"checkpoint_engine.backend='delta_sharded' does not support QAT ({qat_path}=true): "
                "the quantized, fused full export does not share coordinates with the shard export."
            )


def validate_config(config: Any) -> None:
    """Validate configuration values that otherwise trigger silent fallbacks."""
    if _select(config, "actor_rollout_ref.actor.enable_timestep_staging", False):
        sp_size = _select(config, "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 1)
        if sp_size != 1:
            raise ValueError("Timestep staging requires ulysses_sequence_parallel_size=1.")

    _validate_delta_sharded(config)

    resume_mode = _select(config, "trainer.resume_mode")
    valid_resume_modes = ("disable", "auto", "resume_path")
    if resume_mode not in valid_resume_modes:
        raise ValueError(f"Unknown trainer.resume_mode={resume_mode!r}. Available options: {list(valid_resume_modes)}.")
    if resume_mode == "resume_path" and not _select(config, "trainer.resume_from_path"):
        raise ValueError("trainer.resume_from_path must be set when trainer.resume_mode='resume_path'.")

    total_steps = _select(config, "trainer.total_training_steps")
    if total_steps is not None:
        try:
            total_steps = int(total_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.") from exc
        if total_steps <= 0:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.")
