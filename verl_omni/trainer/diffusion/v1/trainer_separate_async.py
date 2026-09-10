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
"""Separate-async v1 policy-gradient diffusion trainer.

Mirrors upstream ``verl.trainer.ppo.v1.trainer_separate_async.PPOTrainerSeparateAsync``
hook semantics, adapted to verl-omni diffusion rollout:

1. Trainer and rollout are separate. The colocated rollout replicas may switch
   to rollout mode when idle; a standalone rollout manager handles the bulk of
   generation traffic on dedicated GPUs.
2. Partial rollout is enabled: the trainer overproduces rollout work, the
   replay buffer samples complete prompt groups, and unfinished requests are
   aborted when switching to trainer mode. Aborted diffusion samples are
   retried as whole samples by ``DiffusionWholeSampleRetryLLMServerClient``.
3. Weight synchronization from actor to standalone rollout uses a non-naive
   checkpoint backend (nccl/nixl/mooncake/...); the colocated rollout still uses
   the naive in-place backend.

Diffusion-specific compute (reward, old/ref log-prob, Flow-GRPO advantage,
actor update, metrics, dumping) lives in ``PolicyGradientDiffusionTrainerV1``;
this subclass only defines the mode lifecycle hooks and the standalone rollout
wiring.
"""

import logging
import os
from enum import Enum

import ray
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import LLMServerManager

from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    register_diffusion_trainer,
)
from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager
from verl_omni.workers.detach_actor_worker import DiffusionDetachActorWorker
from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HybridEngineMode(Enum):
    TRAINER = "trainer"
    ROLLOUT = "rollout"


@register_diffusion_trainer("separate_async")
class PolicyGradientDiffusionTrainerV1SeparateAsync(PolicyGradientDiffusionTrainerV1):
    """Asynchronous policy-gradient diffusion trainer (v1) with separate rollout.

    Hook behavior:

    - ``on_init_end``: update weights on both the standalone and colocated
      checkpoint managers.
    - ``on_train_begin``: top up the configured warmup depth after restoring in-flight prompts.
    - ``on_validate_begin``: switch to rollout mode if currently training.
    - ``on_sample_begin``: switch to rollout mode if training and the switch
      strategy says so.
    - ``on_sample_end``: switch to trainer mode (abort + sleep colocated
      replicas, remove them from the standalone load balancer). When
      ``sync_compatible`` is True, also pause the standalone rollout.
    - ``on_step_end``: after ``parameter_sync_step`` local actor updates, push
      actor weights into the standalone rollout replicas. Colocated replicas
      stay slept during training (they share GPUs with the actor) and are only
      synced at init and when switching to rollout mode. When
      ``sync_compatible`` is True, resume standalone generation after weight
      sync.
    """

    def __init__(self, config: DictConfig):
        train_batch_size = config.data.train_batch_size
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        parameter_sync_step = config.trainer.v1.separate_async.parameter_sync_step
        assert parameter_sync_step > 0, f"parameter_sync_step must be positive, got {parameter_sync_step}"
        assert train_batch_size == parameter_sync_step * ppo_mini_batch_size, (
            "train_batch_size must equal parameter_sync_step * ppo_mini_batch_size "
            f"in separate async training, but got train_batch_size={train_batch_size}, "
            f"parameter_sync_step={parameter_sync_step}, ppo_mini_batch_size={ppo_mini_batch_size}"
        )
        assert config.actor_rollout_ref.rollout.nnodes > 0, (
            "actor_rollout_ref.rollout.nnodes must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.n_gpus_per_node > 0, (
            "actor_rollout_ref.rollout.n_gpus_per_node must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive", (
            "please use nccl/nixl/mooncake/... backend for separate async training"
        )
        separate_async_config = config.trainer.v1.separate_async
        assert (
            not separate_async_config.get("sync_compatible", False)
            or separate_async_config.get("num_warmup_batches", 0) == 0
        ), "sync_compatible=True requires num_warmup_batches=0"

        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(DiffusionDetachActorWorker)

    def _compute_old_log_prob(self, data: DataProto) -> DataProto:
        """Compute every local update's proximal log-probs with cycle-start weights."""
        if self.parameter_sync_step == 1:
            return super()._compute_old_log_prob(data)

        if self.local_trigger_step == 0:
            try:
                self.actor_rollout_wg.save_model_to_cpu(0)
                return super()._compute_old_log_prob(data)
            except BaseException:
                self.actor_rollout_wg.clear_cpu_model(0)
                raise

        snapshot_id = self.local_trigger_step
        snapshot_saved = False
        try:
            self.actor_rollout_wg.save_model_to_cpu(snapshot_id)
            snapshot_saved = True
            self.actor_rollout_wg.restore_model_from_cpu(0)
            return super()._compute_old_log_prob(data)
        finally:
            try:
                if snapshot_saved:
                    self.actor_rollout_wg.restore_model_from_cpu(snapshot_id)
            finally:
                self.actor_rollout_wg.clear_cpu_model(snapshot_id)
                if self.local_trigger_step == self.parameter_sync_step - 1:
                    self.actor_rollout_wg.clear_cpu_model(0)

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """Run one parameter-sync cycle and always release its base snapshot."""
        try:
            return super().step(metrics, timing_raw)
        finally:
            if self.parameter_sync_step > 1:
                self.actor_rollout_wg.clear_cpu_model(0)

    def _should_prefetch_local_batches(self) -> bool:
        """Collect the full cycle before pausing sync-compatible rollout."""
        return self.sync_compatible and self.parameter_sync_step > 1

    def _init_online_rollout_stack(self, actor_rollout_resource_pool):
        """Build colocated rollout stack (naive ckpt) + standalone rollout stack.

        Overridden so the colocated checkpoint manager uses the naive in-place
        backend (actor and colocated rollout share GPUs), while a second
        standalone ``LLMServerManager`` / ``CheckpointEngineManager`` pair uses
        the configured non-naive backend for trainer -> standalone weight sync.
        """
        from verl_omni.reward_loop import OmniRewardLoopManager

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = OmniRewardLoopManager(config=self.config, rm_resource_pool=resource_pool)
        self.enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # Colocated rollout replicas (share GPUs with the actor).
        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        colocated_ckpt_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        colocated_ckpt_config.backend = "naive"
        self.checkpoint_manager = CheckpointEngineManager(
            config=colocated_ckpt_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

    def _setup(self):
        super()._setup()

        # Standalone rollout replicas on dedicated GPUs. start_rank skips the
        # colocated replica ranks to avoid Ray named-actor collisions.
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
        self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, start_rank=hybrid_num_replicas
        )

        standalone_ckpt_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = OmniCheckpointEngineManager(
            config=standalone_ckpt_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        self.current_mode = HybridEngineMode.TRAINER
        self._colocated_slept = False

        self.sync_compatible = self.config.trainer.v1.separate_async.get("sync_compatible", False)
        self._standalone_paused = False
        if self.sync_compatible:
            logger.warning(
                "separate_async sync_compatible=True: standalone rollout will pause "
                "generation during actor training (sync-mode parity)."
            )

    def get_llm_client(self):
        """Get the diffusion whole-sample-retry client backed by the standalone rollout."""
        return self.standalone_server_manager.get_client(client_cls=DiffusionWholeSampleRetryLLMServerClient)

    def on_init_end(self):
        # Push actor weights into both standalone and colocated rollout replicas.
        logger.warning(
            "LORA_SYNC_PROOF separate_async on_init_end: sending weights to BOTH "
            "standalone and colocated checkpoint managers (global_steps=%s)",
            self.global_steps,
        )
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches
        train_batch_size = self.config.data.train_batch_size
        target_warmup_prompts = num_warmup_batches * train_batch_size
        queue_state = tq.kv_list("train") or {}
        restored_inflight_prompts = sum(
            tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
            for tag in queue_state.get("train", {}).values()
        )
        prompts_to_add = max(0, target_warmup_prompts - restored_inflight_prompts)
        full_batches, remaining_prompts = divmod(prompts_to_add, train_batch_size)
        for _ in range(full_batches):
            self._add_batch_to_generate()
        if remaining_prompts:
            self._add_prompts_to_generate(remaining_prompts)
        logger.info(
            "Added %d warmup prompts to the agent loop manager; restored %d in-flight prompts",
            prompts_to_add,
            restored_inflight_prompts,
        )

    def on_validate_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER and self._colocated_slept:
            logger.info("Switching hybrid engine to rollout mode for validation")
            self.switch_to_rollout()
        else:
            logger.info(
                "Skipping colocated rollout switch for validation "
                "(colocated replicas never slept; using standalone replicas only)"
            )
        if self.sync_compatible and self._standalone_paused:
            # Validation uses the standalone rollout via get_llm_client(), so
            # make sure it is resumed if it was paused for actor training.
            self._resume_standalone_generation()

    def on_validate_end(self):
        if self.current_mode == HybridEngineMode.ROLLOUT:
            logger.info("Switching hybrid engine back to trainer mode after validation")
            self.switch_to_trainer()

    def on_sample_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER and self.should_switch_to_rollout():
            logger.info("Switching hybrid engine to rollout mode for generation")
            self.switch_to_rollout()

    def on_sample_end(self):
        if self.current_mode == HybridEngineMode.ROLLOUT:
            logger.info("Switching hybrid engine to trainer mode for training")
            self.switch_to_trainer()
        if self.sync_compatible and not self._standalone_paused:
            self._pause_standalone_generation()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            logger.warning(
                "LORA_SYNC_PROOF separate_async on_step_end: sending weights to "
                "STANDALONE checkpoint manager (global_steps=%s)",
                self.global_steps,
            )
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
            if self.sync_compatible and self._standalone_paused:
                # Sync-compatible mode: resume standalone generation after the
                # actor update + weight sync so the next generate phase uses
                # fresh weights, exactly like sync mode waking colocated replicas.
                self._resume_standalone_generation()

    def _pause_standalone_generation(self):
        """Stop the standalone rollout from accepting/serving new requests.

        Aborts in-flight requests and removes standalone servers from the global
        load balancer so no new requests are routed while the actor is training.
        Does NOT sleep the replicas (they are on dedicated GPUs, so freeing their
        weight memory is not needed for the actor).
        """
        logger.info("sync_compatible: pausing standalone rollout generation for actor training")
        self.standalone_checkpoint_manager.abort_replicas()
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        ray.get(global_load_balancer.remove_servers.remote(self.standalone_server_manager.server_addresses))
        self._standalone_paused = True

    def _resume_standalone_generation(self):
        """Resume standalone rollout generation after weight sync.

        Re-adds standalone servers to the global load balancer and resumes
        generation on all replicas (clears the abort state so new requests can
        be served with the freshly-synced weights).
        """
        logger.info("sync_compatible: resuming standalone rollout generation after weight sync")
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        servers = dict(
            zip(
                self.standalone_server_manager.server_addresses,
                self.standalone_server_manager.server_handles,
                strict=True,
            )
        )
        ray.get(global_load_balancer.add_servers.remote(servers))
        self.standalone_checkpoint_manager.resume_generation_replicas()
        self._standalone_paused = False

    def switch_to_rollout(self):
        # Wake colocated replicas (their weights were offloaded by sleep),
        # sync fresh actor weights, and let them serve generation again.
        self.checkpoint_manager.wake_up_replicas()
        self._colocated_slept = False
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()
        self.add_replicas_to_balancer()
        self.current_mode = HybridEngineMode.ROLLOUT

    def switch_to_trainer(self):
        # Remove colocated replicas from the balancer and free their memory for training.
        self.remove_replicas_from_balancer()
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()
        self._colocated_slept = True
        self.current_mode = HybridEngineMode.TRAINER

    def add_replicas_to_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        servers = dict(
            zip(self.llm_server_manager.server_addresses, self.llm_server_manager.server_handles, strict=True)
        )
        ray.get(global_load_balancer.add_servers.remote(servers))

    def remove_replicas_from_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))

    def should_switch_to_rollout(self):
        # TODO: implement a switch strategy based on replay buffer state / switch overhead.
        return False
