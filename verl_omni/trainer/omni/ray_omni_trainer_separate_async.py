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
"""Omni separate-async trainer: a thin ``PPOTrainerSeparateAsync`` subclass (RFC #320)."""

import ray
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker


@register_trainer("omni_separate_async")
class OmniPPOTrainerSeparateAsync(PPOTrainerSeparateAsync):
    """``PPOTrainerSeparateAsync`` with omni tokenizer/processor wiring and LoRA-aware weight sync."""

    def __init__(self, config):
        super().__init__(config)
        # PPOTrainer reads v1.{trainer_mode}.parameter_sync_step (absent -> 1), but the
        # parent syncs on v1.separate_async; use the validated key, and mirror it onto
        # the ReplayBuffer, which was built from v1.{trainer_mode} before this runs.
        self.parameter_sync_step = config.trainer.v1.separate_async.get("parameter_sync_step", 1)
        self.replay_buffer.parameter_sync_step = self.parameter_sync_step

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _init_resource_pool_mgr(self):
        # The omni worker's LoRA-aware weight sync must win over upstream's (which
        # resends base weights under lora.merge=False); Detach adds CPU save/restore.
        super()._init_resource_pool_mgr()
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        self.role_worker_mapping[actor_role] = ray.remote(OmniDetachActorWorker)

    def _setup(self):
        super()._setup()
        # LoRA-aware manager: pushes the actor's peft_config so replicas apply
        # adapter deltas via add_lora.
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = OmniCheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )
