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


def old_policy_decay(step: int, decay_type: int) -> float:
    """Old-policy LoRA EMA decay schedules."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unsupported old_policy_decay_type: {decay_type}")
    return 0.0 if step < flat else min((step - flat) * uprate, uphold)


class NoOpCheckpointManager:
    """Checkpoint-engine facade used when training does not start rollout replicas."""

    def update_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sleep_replicas(self) -> None:
        return None
