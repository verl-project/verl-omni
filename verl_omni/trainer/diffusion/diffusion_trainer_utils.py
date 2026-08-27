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

from typing import Any

from verl import DataProto
from verl.trainer.distillation import is_distillation_enabled


def _to_diffusion_worker_tensordict(batch: DataProto):
    """Project a driver batch for actor/ref workers without copying tensor storage."""
    worker_batch = batch.to_tensordict()
    worker_batch.pop("responses", None)
    return worker_batch


OLD_POLICY_DECAY_SCHEDULES = {
    "copy": (0, 0.0, 0.0),
    "linear_to_0_5": (0, 0.001, 0.5),
    "delayed_linear_to_0_999": (75, 0.0075, 0.999),
}


def old_policy_decay(step: int, schedule: str) -> float:
    """Return the old-policy LoRA EMA decay for a named DiffusionNFT schedule.

    The decay is used as ``old <- decay * old + (1 - decay) * current`` when refreshing
    the rollout adapter. The schedules mirror the reference DiffusionNFT ``return_decay``
    helper: ``copy`` hard-copies the current adapter, ``linear_to_0_5`` ramps from 0 to
    0.5, and ``delayed_linear_to_0_999`` waits 75 steps before ramping to 0.999.
    """
    if schedule in OLD_POLICY_DECAY_SCHEDULES:
        warmup_steps, ramp_rate, max_decay = OLD_POLICY_DECAY_SCHEDULES[schedule]
    else:
        raise ValueError(f"Unsupported old_policy_decay_schedule: {schedule}")
    return 0.0 if step < warmup_steps else min((step - warmup_steps) * ramp_rate, max_decay)


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
    if enabled and actor.use_distill_loss and actor.distill_loss_mode != "distill_kl":
        raise NotImplementedError(
            f"The teacher runtime produces teacher_prev_sample_mean, which only distill_kl consumes, "
            f"but got distill_loss_mode={actor.distill_loss_mode!r} (distill_fm_mse has no producer here)."
        )
    if enabled and config.algorithm.trainer_type != "policy_gradient":
        raise NotImplementedError("Diffusion distillation requires algorithm.trainer_type=policy_gradient.")


class NoOpCheckpointManager:
    """Checkpoint-engine facade used when training does not start rollout replicas."""

    def update_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sleep_replicas(self) -> None:
        return None
