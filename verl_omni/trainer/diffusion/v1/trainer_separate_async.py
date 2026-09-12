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
4. Hybrid rollout switching (``separate_async.hybrid_rollout.enable_switch``)
   lends the colocated replicas to generation at the end of a step when the
   replay buffer is short for the next one, and reclaims them once enough
   prompt groups are sampleable. The switch policy is ported method by method
   from the upstream trainer. Unlike upstream, the lend decision is taken
   after the standalone weight sync, so a batch that lands during the sync
   is counted, and colocated replicas join the load balancer only after the
   naive sync has resumed them, because the vLLM-Omni engine rejects
   requests while any sleeping tag is set.

Diffusion-specific compute (reward, old/ref log-prob, Flow-GRPO advantage,
actor update, metrics, dumping) lives in ``PolicyGradientDiffusionTrainerV1``;
this subclass only defines the mode lifecycle hooks and the standalone rollout
wiring.
"""

import logging
import os
import time
from collections import deque
from enum import Enum

import ray
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.config import HybridRolloutSwitchConfig
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

    - ``_setup``: colocated replicas start in rollout mode and registered with
      the standalone load balancer, so warmup and validation use both pools.
    - ``on_init_end``: update weights on both the standalone and colocated
      checkpoint managers.
    - ``on_train_begin``: enqueue ``num_warmup_batches`` prompt batches.
    - ``on_step_begin``: reclaim the colocated replicas (abort + sleep, remove
      from the balancer) when switching is disabled or the replay buffer already
      holds the switch threshold; otherwise ``prepare_step`` submits this step's
      prompts and waits for the threshold before reclaiming.
    - ``on_validate_begin``: switch to rollout mode if currently training; the
      next ``on_step_begin`` reclaims.
    - ``on_sample_begin`` / ``on_sample_end``: record replay-buffer shortfall
      and wait time. When ``sync_compatible`` is True, ``on_sample_end`` also
      pauses the standalone rollout.
    - ``on_step_end``: after ``parameter_sync_step`` local actor updates, push
      actor weights into the standalone rollout replicas. With switching
      enabled, lend the colocated replicas to the next step's generation when
      the estimated benefit exceeds the recent switch cost. When
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
        self.hybrid_rollout_config: HybridRolloutSwitchConfig = omega_conf_to_dataclass(
            self.config.trainer.v1.separate_async.hybrid_rollout
        )
        if self.hybrid_rollout_config.enable_switch:
            assert not separate_async_config.get("sync_compatible", False), (
                "trainer.v1.separate_async.hybrid_rollout.enable_switch requires sync_compatible=false"
            )
            rollout_cfg = self.config.get("actor_rollout_ref", {}).get("rollout", {})
            disaggregation_cfg = rollout_cfg.get("disaggregation", {})
            if bool(disaggregation_cfg.get("enabled", False)):
                raise ValueError(
                    "trainer.v1.separate_async.hybrid_rollout.enable_switch does not support rollout disaggregation"
                )
            required_methods = ("wait_for_sampleable", "get_sampleable_count")
            if any(not hasattr(self.replay_buffer, method) for method in required_methods):
                raise TypeError(
                    f"{type(self.replay_buffer).__name__} must implement {required_methods} when "
                    "trainer.v1.separate_async.hybrid_rollout.enable_switch=True"
                )
            self._init_hybrid_rollout_state()

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(DiffusionDetachActorWorker)

    def _init_hybrid_rollout_state(self) -> None:
        config = self.hybrid_rollout_config
        self._switch_threshold_ratio = config.switch_threshold_ratio
        self._idle_steps = 0
        self._calm_steps = 0
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self._sample_start = time.perf_counter()
        self._step_threshold = 0
        self._wait_seconds = 0.0
        self._wait_samples = 0
        self._to_rollout_costs: deque[float] = deque(maxlen=config.switch_cost_window_size)
        self._to_trainer_costs: deque[float] = deque(maxlen=config.switch_cost_window_size)
        rollout_cfg = self.config.get("actor_rollout_ref", {}).get("rollout", {})
        trainer_cfg = self.config.trainer
        hybrid_gpus = trainer_cfg.nnodes * trainer_cfg.n_gpus_per_node
        standalone_gpus = rollout_cfg.nnodes * rollout_cfg.n_gpus_per_node
        self._scaling_factor = (hybrid_gpus + standalone_gpus) / standalone_gpus

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

        self.sync_compatible = self.config.trainer.v1.separate_async.get("sync_compatible", False)
        self._standalone_paused = False
        if self.sync_compatible:
            logger.warning(
                "separate_async sync_compatible=True: standalone rollout will pause "
                "generation during actor training (sync-mode parity)."
            )

        # hybrid engine is in rollout mode after initialization
        self.current_mode = HybridEngineMode.ROLLOUT
        self.add_replicas_to_balancer()

    def get_llm_client(self):
        """Get the diffusion whole-sample-retry client backed by the standalone rollout."""
        return self.standalone_server_manager.get_client(client_cls=DiffusionWholeSampleRetryLLMServerClient)

    def on_init_end(self):
        # Push actor weights into both standalone and colocated rollout replicas.
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_validate_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER:
            logger.info("Switching hybrid engine to rollout mode for validation")
            self.switch_to_rollout()
        if self.sync_compatible and self._standalone_paused:
            # Validation uses the standalone rollout via get_llm_client(), so
            # make sure it is resumed if it was paused for actor training.
            self._resume_standalone_generation()

    def on_step_begin(self):
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self._step_threshold = 0
        if self.hybrid_rollout_config.enable_switch:
            self.timing_raw["switch_wait"] = 0.0
        if self.current_mode != HybridEngineMode.ROLLOUT:
            return
        if not self.hybrid_rollout_config.enable_switch:
            self._timed_switch_to_trainer()
            return

        self._step_threshold = self._switch_threshold()
        sampleable_count = self.replay_buffer.get_sampleable_count(self.global_steps, "train")
        if sampleable_count >= self._step_threshold:
            self._timed_switch_to_trainer()

    def _timed_switch_to_trainer(self) -> None:
        """Switch Hybrid back to training and record the full remove/abort/sleep cost."""
        switch_start = time.perf_counter()
        with marked_timer("switch_to_trainer", self.timing_raw, color="cyan"):
            self.switch_to_trainer()
        if self.hybrid_rollout_config.enable_switch:
            self._to_trainer_costs.append(time.perf_counter() - switch_start)

    def on_sample_begin(self):
        if self.hybrid_rollout_config.enable_switch:
            sampleable = self.replay_buffer.get_sampleable_count(self.global_steps, "train")
            mini_batch_size = self.config.data.train_batch_size // self.parameter_sync_step
            self._step_wait_samples += max(0, mini_batch_size - sampleable)
        self._sample_start = time.perf_counter()

    def on_sample_end(self):
        self._step_sample_wait_seconds += time.perf_counter() - self._sample_start
        if self.sync_compatible and not self._standalone_paused:
            self._pause_standalone_generation()

    def prepare_step(self) -> dict:
        metrics = super().prepare_step()
        metrics.update(self._wait_for_sampleable_and_switch())
        return metrics

    def _wait_for_sampleable_and_switch(self) -> dict:
        if self.current_mode != HybridEngineMode.ROLLOUT:
            return {}

        logger.info(f"Lending hybrid engine to generation until {self._step_threshold} groups are sampleable")
        with marked_timer("switch_wait", self.timing_raw, color="yellow"):
            _, eviction_metrics = self.replay_buffer.wait_for_sampleable(
                self.global_steps, "train", self._step_threshold
            )
        self._timed_switch_to_trainer()
        return eviction_metrics

    def _switch_threshold(self) -> int:
        """Sampleable prompts before switching to trainer, floored at one mini-batch."""
        train_batch_size = self.config.data.train_batch_size
        mini_batch_size = train_batch_size // self.parameter_sync_step
        target = round(self._switch_threshold_ratio * train_batch_size)
        return min(max(target, mini_batch_size), train_batch_size)

    def _step_had_idle(self) -> bool:
        """Whether waiting for sampleable prompts."""
        poll_interval = getattr(self.replay_buffer, "poll_interval", 2.0)
        return self._step_sample_wait_seconds > poll_interval

    def _adapt_switch_threshold(self, had_idle: bool) -> None:
        config = self.hybrid_rollout_config
        if had_idle:
            self._calm_steps = 0
            self._idle_steps = min(self._idle_steps + 1, config.switch_threshold_release_steps)
            if self._idle_steps < config.switch_threshold_release_steps:
                return
            self._switch_threshold_ratio = min(1.0, self._switch_threshold_ratio + config.switch_threshold_step_up)
            return

        self._idle_steps = 0
        self._calm_steps = min(self._calm_steps + 1, config.switch_threshold_release_steps)
        if self._calm_steps < config.switch_threshold_release_steps:
            return
        min_ratio = 1.0 / self.parameter_sync_step
        self._switch_threshold_ratio = max(min_ratio, self._switch_threshold_ratio - config.switch_threshold_step_down)

    def _effective_switch_cost(self) -> float | None:
        if not self._to_rollout_costs or not self._to_trainer_costs:
            return None
        return sum(self._to_rollout_costs) / len(self._to_rollout_costs) + sum(self._to_trainer_costs) / len(
            self._to_trainer_costs
        )

    def on_step_end(self):
        config = self.hybrid_rollout_config
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = dict(
                self.standalone_checkpoint_manager.update_weights(self.global_steps) or {}
            )
            if self.sync_compatible and self._standalone_paused:
                # Sync-compatible mode: resume standalone generation after the
                # actor update + weight sync so the next generate phase uses
                # fresh weights, exactly like sync mode waking colocated replicas.
                self._resume_standalone_generation()

        if not config.enable_switch:
            return

        # Decide after the standalone sync: the batch that was in flight at step
        # end has landed by now, so the inventory count is not a phantom gap.
        ratio_used = self._switch_threshold_ratio
        had_idle = self._step_had_idle()
        if self._step_wait_samples > 0 and self._step_sample_wait_seconds > 0:
            self._wait_seconds += self._step_sample_wait_seconds
            self._wait_samples += self._step_wait_samples
        if config.adaptive_switch_threshold:
            self._adapt_switch_threshold(had_idle)

        decision_threshold = self._switch_threshold()
        sampleable_count = self.replay_buffer.get_sampleable_count(self.global_steps + 1, "train")
        remaining = max(0, decision_threshold - sampleable_count)
        per_sample_time = self._wait_seconds / self._wait_samples if self._wait_samples > 0 else None
        effective_switch_cost = self._effective_switch_cost()
        benefit = (
            remaining * per_sample_time * (1.0 - 1.0 / self._scaling_factor) if per_sample_time is not None else None
        )
        should_switch = (
            self.global_steps < self.total_training_steps
            and remaining > 0
            and (benefit is None or effective_switch_cost is None or benefit > effective_switch_cost)
        )
        self._pending_sync_metrics.update(
            {
                "separate_async/switch/threshold_ratio": ratio_used,
                "separate_async/switch/wait_samples": float(self._step_wait_samples),
                "separate_async/switch/idle": float(had_idle),
                "separate_async/decision/sampleable_count": float(sampleable_count),
                "separate_async/decision/remaining": float(remaining),
                "separate_async/decision/should_switch_to_rollout": float(should_switch),
            }
        )
        if per_sample_time is not None:
            self._pending_sync_metrics["separate_async/decision/per_sample_time_seconds"] = per_sample_time
        if effective_switch_cost is not None:
            self._pending_sync_metrics["separate_async/decision/effective_switch_cost_seconds"] = effective_switch_cost

        if should_switch:
            switch_start = time.perf_counter()
            with marked_timer("switch_to_rollout", self.timing_raw, color="cyan"):
                logger.info("Switching hybrid engine to rollout mode for the next step")
                self.switch_to_rollout()
                self.clear_sticky_cache()
            self._to_rollout_costs.append(time.perf_counter() - switch_start)

    def _get_n_gpus_for_throughput(self) -> int:
        """Include standalone rollout GPUs in the throughput denominator."""
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        rollout_gpus = (
            self.config.actor_rollout_ref.rollout.n_gpus_per_node * self.config.actor_rollout_ref.rollout.nnodes
        )
        return trainer_gpus + rollout_gpus

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
        """Install committed weights and make Hybrid replicas available for generation.

        The naive update resumes the slept replicas' weights and cache before
        installing the actor weights, so no frontend wake-up is needed. The
        replicas are registered with the balancer only after that, because the
        engine rejects requests while any sleeping tag is set.
        """
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()
        self.add_replicas_to_balancer()
        self.current_mode = HybridEngineMode.ROLLOUT

    def switch_to_trainer(self):
        """Stop routing to Hybrid, abort partial requests, and return its GPU memory to training."""
        self.remove_replicas_from_balancer()
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()
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

    def clear_sticky_cache(self) -> dict:
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        return ray.get(global_load_balancer.clear_sticky_cache.remote())
