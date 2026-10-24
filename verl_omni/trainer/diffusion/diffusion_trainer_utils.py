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
"""Shared helpers for diffusion Ray trainers."""

import math
from typing import Any, Optional

from verl import DataProto
from verl.trainer.distillation import is_distillation_enabled

# ``rollout_media.ARTIFACT_PREFIX``, spelled literally to keep this core
# trainer module free of the pipeline package's import graph.
MEDIA_ARTIFACT_FIELD_PREFIX = "media_artifact__"


def _to_diffusion_worker_tensordict(batch: DataProto):
    """Project a driver batch for actor/ref workers without copying tensor storage.

    Named media artifacts are reward/dump-only consumers' data; compute workers never
    read them, so shipping them would slice and serialize decoded previews per microbatch.
    """
    worker_batch = batch.to_tensordict()
    worker_batch.pop("responses", None)
    for key in [key for key in worker_batch.keys() if key.startswith(MEDIA_ARTIFACT_FIELD_PREFIX)]:
        worker_batch.pop(key)
    return worker_batch


OLD_POLICY_DECAY_SCHEDULES = {
    "copy": (0, 0.0, 0.0),
    "linear_to_0_3": (0, 0.001, 0.3),
    "linear_to_0_5": (0, 0.001, 0.5),
    "linear_from_0_25_to_0_5": (0, 0.001, 0.5),
    "delayed_linear_to_0_999": (75, 0.0075, 0.999),
}


def old_policy_decay(step: int, schedule: str) -> float:
    """Return the LoRA EMA decay for a named forward-process policy schedule.

    The decay is used as ``old <- decay * old + (1 - decay) * current`` when refreshing
    the rollout adapter. The schedules mirror the reference DiffusionNFT ``return_decay``
    helper: ``copy`` hard-copies the current adapter, ``linear_to_0_5`` ramps from 0 to
    0.5 (``linear_to_0_3``, the DGPO reference EMA, stops at 0.3), and
    ``delayed_linear_to_0_999`` waits 75 steps before ramping to 0.999.
    CRD's ``linear_from_0_25_to_0_5`` starts at 0.25 and ramps by 0.001 per step.
    """
    if schedule in OLD_POLICY_DECAY_SCHEDULES:
        warmup_steps, ramp_rate, max_decay = OLD_POLICY_DECAY_SCHEDULES[schedule]
    else:
        raise ValueError(f"Unsupported old_policy_decay_schedule: {schedule}")
    initial_decay = 0.25 if schedule == "linear_from_0_25_to_0_5" else 0.0
    return 0.0 if step < warmup_steps else min(initial_decay + (step - warmup_steps) * ramp_rate, max_decay)


def track_nonfinite_grad_streak(streak: int, grad_norm: Optional[float]) -> int:
    """Update the consecutive-non-finite-gradient streak, for visibility only (see #388 item A3)."""
    if grad_norm is None or math.isfinite(grad_norm):
        return 0
    return streak + 1


def skip_old_log_prob(config) -> bool:
    """Whether the policy-gradient update reads none of the recomputed old log-probs, so the stage can be skipped."""
    from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
    from verl_omni.trainer.diffusion.rollout_correction import rollout_correction_enabled

    if config.algorithm.get("trainer_type", "policy_gradient") != "policy_gradient":
        return False
    loss_fn = get_diffusion_loss_fn(config.actor_rollout_ref.actor.diffusion_loss.get("loss_mode", "flow_grpo"))
    reads_log_probs = "log_probs" in loss_fn.required_model_output_keys or bool(
        {"old_log_probs", "old_prev_sample_mean"} & set(loss_fn.required_data_keys)
    )
    return not reads_log_probs and not rollout_correction_enabled(config.algorithm.get("rollout_correction", None))


def validate_distillation_config(config) -> None:
    """Cross-check the distillation switch against the losses that consume teacher outputs."""
    actor = config.actor_rollout_ref.actor
    distill_active = actor.diffusion_loss.get("loss_mode", "flow_grpo") == "distill_kl" or actor.use_distill_loss
    enabled = is_distillation_enabled(config.get("distillation"))
    if enabled and not distill_active:
        raise ValueError(
            "distillation.enabled=true but no distillation loss is active; set "
            "actor.diffusion_loss.loss_mode=distill_kl or actor.use_distill_loss=true."
        )
    if distill_active and not enabled:
        raise ValueError(
            "A distillation loss is active but no teacher is configured; set distillation.enabled=true "
            "and distillation.teacher_models.teacher_model.model_path."
        )
    if enabled:
        student_guidance_scale = config.actor_rollout_ref.rollout.pipeline.get("guidance_scale") or 0.0
        inherited_guidance_scale = config.actor_rollout_ref.model.pipeline.get("guidance_scale") or 0.0
        for name, teacher in config.distillation.teacher_models.items():
            teacher_guidance_scale = teacher.get("guidance_scale")
            if teacher_guidance_scale is None:
                teacher_guidance_scale = inherited_guidance_scale
            if teacher_guidance_scale > 1.0 and student_guidance_scale <= 1.0:
                raise ValueError(
                    f"distillation.teacher_models.{name} runs with guidance_scale={teacher_guidance_scale} (its own "
                    "value or the inherited actor_rollout_ref.model.pipeline.guidance_scale), which needs negative "
                    "prompt embeds that the rollout only produces with "
                    "actor_rollout_ref.rollout.pipeline.guidance_scale > 1."
                )
    if enabled and actor.use_distill_loss and actor.distill_loss_mode != "distill_kl":
        raise NotImplementedError(
            f"The teacher runtime produces teacher_prev_sample_mean, which only distill_kl consumes, "
            f"but got distill_loss_mode={actor.distill_loss_mode!r} (distill_fm_mse has no producer here)."
        )
    if enabled and config.algorithm.trainer_type != "policy_gradient":
        raise NotImplementedError("Diffusion distillation requires algorithm.trainer_type=policy_gradient.")
    if enabled and config.distillation.get("scheduler", "inline") == "one_step_off":
        v1 = config.trainer.get("use_v1", False) and config.trainer.v1.trainer_mode == "separate_async"
        if not v1 or config.distillation.nnodes <= 0:
            raise ValueError(
                "distillation.scheduler=one_step_off requires the v1 separate_async trainer and standalone "
                "teachers (distillation.nnodes > 0); colocated teachers share the actor GPUs and have "
                "nothing to overlap with."
            )
        if config.trainer.v1.separate_async.get("sync_compatible", False):
            raise ValueError(
                "distillation.scheduler=one_step_off contradicts "
                "trainer.v1.separate_async.sync_compatible=true, which emulates synchronous training."
            )
        if config.trainer.v1.separate_async.get("num_warmup_batches", 1) < 2:
            raise ValueError(
                "distillation.scheduler=one_step_off requires trainer.v1.separate_async.num_warmup_batches >= 2: "
                "the teacher pipeline consumes one batch of generation lead at start-up, and with less lead every "
                "sample waits on its own batch's generation."
            )


class NoOpCheckpointManager:
    """Checkpoint-engine facade used when training does not start rollout replicas."""

    def update_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sleep_replicas(self) -> None:
        return None
