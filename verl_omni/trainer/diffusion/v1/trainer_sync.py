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

from verl import DataProto
from verl.trainer.ppo.utils import Role
from verl.utils.debug import marked_timer

from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    register_diffusion_trainer,
)
from verl_omni.workers.config.reward import (
    NativeRewardModelConfig,
    get_reward_model_entries,
    parse_reward_model_config,
    streaming_reward_enabled,
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

    _confirm_validation_reward_release = False

    def _init_online_rollout_stack(self, actor_rollout_resource_pool):
        if (
            Role.RewardModel in self.mapping
            and self.resource_pool_manager.get_resource_pool(Role.RewardModel) is actor_rollout_resource_pool
            and not streaming_reward_enabled(self.config)
        ):
            models = []
            for name, entry in get_reward_model_entries(self.config).items():
                model = parse_reward_model_config(name, entry)
                if isinstance(model, NativeRewardModelConfig) and model.placement.is_cpu:
                    continue
                if not model.resolved_offload:
                    raise ValueError(f"Named reward model {name!r} sharing the rollout pool requires offload=true")
                models.append(model)
            # Engine-backed lifecycle and sharded placement have separate contracts.
            self._confirm_validation_reward_release = bool(models) and all(
                isinstance(model, NativeRewardModelConfig) and not model.placement.replica_groups for model in models
            )
            if self._confirm_validation_reward_release and not self.config.actor_rollout_ref.rollout.free_cache_engine:
                raise ValueError("Shared named reward validation requires rollout.free_cache_engine=true")
        super()._init_online_rollout_stack(actor_rollout_resource_pool)

    def _score_validation_batch(self, data: DataProto) -> DataProto:
        if not self._confirm_validation_reward_release:
            return super()._score_validation_batch(data)
        try:
            self.checkpoint_manager.sleep_replicas()
        except BaseException as exc:
            exc.add_note("Validation rollout sleep failed")
            raise

        released = False

        def record_release():
            nonlocal released
            released = True

        scoring_error = None
        try:
            rewards = self._compute_reward_colocate(data, on_release=record_release)
        except BaseException as exc:
            if not released:
                exc.add_note("Validation reward release unconfirmed")
                raise
            scoring_error = exc

        try:
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.wait_for_weight_update()
        except BaseException as exc:
            exc.add_note("Validation rollout weight restore failed")
            if scoring_error is not None:
                raise BaseExceptionGroup(
                    "Validation scoring and rollout weight restore failed", [scoring_error, exc]
                ) from None
            raise
        if scoring_error is not None:
            raise scoring_error
        return data.union(rewards)

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
        if self.curr_step_profile:
            self._stop_rollout_profiling()
