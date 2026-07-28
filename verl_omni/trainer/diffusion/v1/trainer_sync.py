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
"""Synchronous v1 policy-gradient diffusion trainer.

Mirrors upstream ``PPOTrainerSync`` hook semantics, adapted to verl-omni
diffusion rollout:

1. Trainer and rollout are colocated.
2. Partial rollout is disabled (no overproduction / abort).
3. After sampling, rollout replicas are slept to free weight memory.
4. At step end, rollout weights are updated from the actor.

Diffusion-specific compute (reward, old/ref log-prob, Flow-GRPO advantage,
actor update, metrics, dumping) lives in ``PolicyGradientDiffusionTrainerV1``;
this subclass only defines the mode lifecycle hooks.
"""

import logging
import os

from verl.utils.debug import marked_timer

from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    register_diffusion_trainer,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_diffusion_trainer("sync")
class PolicyGradientDiffusionTrainerV1Sync(PolicyGradientDiffusionTrainerV1):
    """Synchronous policy-gradient diffusion trainer (v1).

    Hook behavior:

    - ``on_init_end``: update rollout weights at the current ``global_steps``.
    - ``on_sample_end``: sleep/offload rollout weight memory (discard weights
      and any KV cache; pure diffusion has no KV cache so this is a no-op for
      cache but still frees rollout weights).
    - ``on_step_end``: update rollout weights after the actor update.
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights from the freshly-trained actor
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # sleep all replicas to discard weights and (no-op) KV cache
        self.checkpoint_manager.sleep_replicas()
