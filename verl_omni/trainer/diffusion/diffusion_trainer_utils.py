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


class NoOpCheckpointManager:
    """Checkpoint-engine facade used when training does not start rollout replicas."""

    def update_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sleep_replicas(self) -> None:
        return None
