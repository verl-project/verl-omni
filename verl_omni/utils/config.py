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
