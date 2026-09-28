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


def validate_config(config: Any) -> None:
    """Validate configuration values that otherwise trigger silent fallbacks."""
    if _select(config, "actor_rollout_ref.actor.enable_timestep_staging", False):
        sp_size = _select(config, "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 1)
        if sp_size != 1:
            raise ValueError("Timestep staging requires ulysses_sequence_parallel_size=1.")

    if _select(config, "actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation", False):
        strategy = _select(config, "actor_rollout_ref.actor.strategy")
        if strategy not in ("fsdp", "fsdp2"):
            raise ValueError(
                "actor.use_no_sync_for_gradient_accumulation=true requires actor.strategy "
                f"fsdp or fsdp2, got {strategy!r}."
            )

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

    _validate_dynamic_resource_scheduling(config)


def _validate_dynamic_resource_scheduling(config: Any) -> None:
    """Refuse the fully_async_policy scheduler on v1/v0 entrypoints.

    ``async_training.use_dynamic_resource_scheduling`` drives verl's
    ``DynamicResourceController`` (verl#6556) on the MessageQueue
    ``fully_async_policy`` stack. ``main_omni`` / ``main_diffusion`` /
    ``main_diffusion_v1`` never construct that controller, so the flag would
    be a silent no-op. The v1 analog is ``hybrid_rollout.enable_switch``.
    """
    if not _select(config, "async_training.use_dynamic_resource_scheduling", False):
        return
    raise ValueError(
        "async_training.use_dynamic_resource_scheduling=true is the fully_async_policy "
        "DynamicResourceController (verl#6556) and is not wired on main_omni / "
        "main_diffusion / main_diffusion_v1. For v1 separate_async, set "
        "trainer.v1.separate_async.hybrid_rollout.enable_switch=true instead."
    )
