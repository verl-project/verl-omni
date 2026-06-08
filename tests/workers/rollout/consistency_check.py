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
"""Shared train–rollout consistency check helpers for GPU tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import ray
import torch
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.testing.train_rollout_consistency import (
    TrainRolloutConsistencyMetrics,
    compute_train_rollout_consistency_metrics,
    format_metrics_report,
    log_metrics,
)
from verl_omni.workers.config import FSDPDiffusionActorConfig
from verl_omni.workers.rollout.replica import DiffusionOutput

from .hybrid_consistency_session import HybridConsistencySession, run_hybrid_session
from .train_rollout_consistency_helpers import (
    QwenImageConsistencyProfile,
    build_rollout_server,
    build_train_batch_from_rollout,
    create_training_worker,
    default_prompts,
    package_versions,
    require_qwen_image_model,
    rollout_sampling_params,
)

_LOGPROB_ABS_MAX_TOL = 0.01
_RATIO_MEAN_TOL = 0.02
_PG_CLIPFRAC_MAX = 0.02


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def format_metrics_hint(metrics: TrainRolloutConsistencyMetrics) -> str:
    return format_metrics_report(metrics, label="consistency_check")


def assert_consistency_metrics(metrics: TrainRolloutConsistencyMetrics) -> None:
    assert metrics.logprob_abs_max < _LOGPROB_ABS_MAX_TOL, format_metrics_hint(metrics)
    assert abs(metrics.ratio_mean - 1.0) < _RATIO_MEAN_TOL, format_metrics_hint(metrics)
    # pg_clipfrac is quantized by SDE window size (multiples of 1/num_steps); with only a
    # few steps, rely on logprob/ratio checks when logprobs are already within tolerance.
    if metrics.num_steps <= 4:
        assert metrics.frac_steps_over_tol < _PG_CLIPFRAC_MAX, format_metrics_hint(metrics)
    else:
        assert metrics.pg_clipfrac < _PG_CLIPFRAC_MAX, format_metrics_hint(metrics)


def _normalize_rollout_log_probs(output: DiffusionOutput) -> torch.Tensor:
    assert output.log_probs is not None, "rollout log_probs missing; ensure logprobs=True"
    rollout_log_probs = output.log_probs
    if isinstance(rollout_log_probs, torch.Tensor):
        rollout_log_probs = rollout_log_probs.detach().cpu().float()
    else:
        rollout_log_probs = torch.tensor(rollout_log_probs, dtype=torch.float32)
    if rollout_log_probs.dim() == 1:
        rollout_log_probs = rollout_log_probs.unsqueeze(0)
    return rollout_log_probs


def _validate_rollout_extra(extra: dict[str, Any]) -> None:
    for key in ("all_latents", "all_timesteps", "prompt_embeds", "prompt_embeds_mask"):
        assert extra.get(key) is not None, f"rollout extra_fields missing {key}"


def _build_metadata(profile: QwenImageConsistencyProfile, model_path: str, **extra: Any) -> dict[str, Any]:
    return {
        **package_versions(),
        "git_commit": git_commit(),
        "model_variant": profile.variant,
        "recipe": profile.recipe,
        "dtype": "bfloat16",
        "noise_level": 1.2,
        "sde_window_size": profile.sde_window_size,
        "sde_window_range": list(profile.sde_window_range),
        "true_cfg_scale": 4.0,
        "model_path": model_path,
        "fsdp_param_offload": profile.fsdp_param_offload,
        "lora_rank": profile.lora_rank,
        "layered_summon": profile.layered_summon,
        **extra,
    }


def _log_and_return_metrics(
    *,
    profile: QwenImageConsistencyProfile,
    rollout_log_probs: torch.Tensor,
    train_log_probs: torch.Tensor,
    actor_config: FSDPDiffusionActorConfig,
    tmp_path: Path,
    metadata: dict[str, Any],
    label_suffix: str = "",
) -> TrainRolloutConsistencyMetrics:
    metrics = compute_train_rollout_consistency_metrics(
        rollout_log_probs,
        train_log_probs,
        actor_config=actor_config,
    )
    label = (
        f"qwen-image-{profile.variant}-{profile.recipe}{label_suffix} vllm-omni {metadata.get('vllm_omni', 'unknown')}"
    )
    json_path = tmp_path / f"train_rollout_consistency_metrics{label_suffix.replace(' ', '_')}.json"
    log_metrics(metrics, label=label, json_path=json_path, metadata=metadata)
    return metrics


def run_standalone_consistency_check(
    profile: QwenImageConsistencyProfile,
    tmp_path: Path,
    *,
    metadata_extra: dict[str, Any] | None = None,
) -> TrainRolloutConsistencyMetrics:
    if profile.uses_hybrid_lora_sync:
        pytest.skip("Standalone path does not support LoRA layered_summon; use hybrid consistency check.")

    model_path = require_qwen_image_model(profile)
    prompt_ids, negative_prompt_ids = default_prompts(model_path)

    server = build_rollout_server(profile)
    try:
        output: DiffusionOutput = ray.get(
            server.generate.remote(
                prompt_ids=prompt_ids,
                negative_prompt_ids=negative_prompt_ids,
                sampling_params=rollout_sampling_params(profile=profile, seed=42),
                request_id=f"consistency_{uuid4().hex[:8]}",
            ),
            timeout=profile.generate_timeout_s,
        )
    finally:
        ray.kill(server)
        torch.cuda.empty_cache()

    extra = output.extra_fields
    _validate_rollout_extra(extra)
    rollout_log_probs = _normalize_rollout_log_probs(output)

    train_batch = build_train_batch_from_rollout(extra)
    worker_group, actor_config = create_training_worker(profile)
    try:
        infer_output = worker_group.infer_batch(train_batch)
        output_dict = infer_output.get()
    finally:
        del worker_group
        torch.cuda.empty_cache()

    assert "log_probs" in output_dict, f"keys={list(output_dict.keys())}"
    train_log_probs = output_dict["log_probs"].detach().cpu().float()
    metadata = _build_metadata(profile, model_path, **(metadata_extra or {}))
    metrics = _log_and_return_metrics(
        profile=profile,
        rollout_log_probs=rollout_log_probs,
        train_log_probs=train_log_probs,
        actor_config=actor_config,
        tmp_path=tmp_path,
        metadata=metadata,
    )
    assert_consistency_metrics(metrics)
    return metrics


async def run_hybrid_consistency_check(
    profile: QwenImageConsistencyProfile,
    tmp_path: Path,
    *,
    metadata_extra: dict[str, Any] | None = None,
    label_suffix: str = "",
    sync_before_check: bool = True,
    session: HybridConsistencySession | None = None,
) -> TrainRolloutConsistencyMetrics:
    if not profile.uses_hybrid_lora_sync:
        raise ValueError("Hybrid consistency check requires LoRA with layered_summon enabled.")

    model_path = require_qwen_image_model(profile)
    prompt_ids, negative_prompt_ids = default_prompts(model_path)

    owns_session = session is None
    session = session or await HybridConsistencySession.create(profile)
    try:
        if sync_before_check:
            await session.sync_weights(global_steps=0)

        output = session.generate(
            prompt_ids=prompt_ids,
            negative_prompt_ids=negative_prompt_ids,
            seed=42,
        )
        extra = output.extra_fields
        _validate_rollout_extra(extra)
        rollout_log_probs = _normalize_rollout_log_probs(output)
        train_log_probs = await session.recompute_log_probs(extra)

        actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(session.config.actor_rollout_ref.actor)
        metadata = _build_metadata(
            profile,
            model_path,
            hybrid_engine=True,
            **(metadata_extra or {}),
        )
        metrics = _log_and_return_metrics(
            profile=profile,
            rollout_log_probs=rollout_log_probs,
            train_log_probs=train_log_probs,
            actor_config=actor_config,
            tmp_path=tmp_path,
            metadata=metadata,
            label_suffix=label_suffix,
        )
        assert_consistency_metrics(metrics)
        return metrics
    finally:
        if owns_session:
            await session.close()


def run_hybrid_consistency_check_sync(
    profile: QwenImageConsistencyProfile,
    tmp_path: Path,
    **kwargs: Any,
) -> TrainRolloutConsistencyMetrics:
    return run_hybrid_session(run_hybrid_consistency_check(profile, tmp_path, **kwargs))
