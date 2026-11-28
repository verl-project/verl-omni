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
"""Colocated asynchronous FlowGRPO with whole-sample diffusion retries."""

import time

import ray
import transfer_queue as tq
from verl.utils.debug import marked_timer

from verl_omni.trainer.diffusion.v1.trainer_base import register_diffusion_trainer
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.workers.config.reward import reward_pool_is_separate
from verl_omni.workers.rollout.diffusion_llm_server import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DiffusionWholeSampleRetryLLMServerClient,
)


@register_diffusion_trainer("colocate_async")
class PolicyGradientDiffusionTrainerV1ColocateAsync(PolicyGradientDiffusionTrainerV1Sync):
    """Collect complete groups, reclaim rollout GPUs, train, then commit weights.

    Pending requests wait outside the rollout servers while the actor owns the
    GPUs. Aborted requests restart from the original prompt and noise seed.
    Actor PPO mini-batches run within one trainer step and one weight sync.
    """

    def __init__(self, config):
        rollout = config.actor_rollout_ref.rollout
        if rollout.nnodes != 0 or rollout.checkpoint_engine.backend != "naive":
            raise ValueError("colocate_async requires colocated rollout (nnodes=0) and checkpoint_engine.backend=naive")
        if not rollout.free_cache_engine and rollout.step_execution:
            raise ValueError("resident colocate_async requires request-level execution (step_execution=false)")
        if config.trainer.v1.colocate_async.get("parameter_sync_step", 1) != 1:
            raise ValueError("colocate_async supports one parameter sync per step (parameter_sync_step=1)")
        warmup = config.trainer.v1.colocate_async.num_warmup_batches
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("colocate_async.num_warmup_batches must be a nonnegative integer")
        if (
            config.algorithm.trainer_type != "policy_gradient"
            or config.actor_rollout_ref.model.algorithm != "flow_grpo"
        ):
            raise ValueError("colocate_async currently supports FlowGRPO policy-gradient training")
        train_batch_size = config.data.train_batch_size
        mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in (train_batch_size, mini_batch_size)
        ):
            raise ValueError("colocate_async requires positive integer train_batch_size and ppo_mini_batch_size")
        if train_batch_size % mini_batch_size:
            raise ValueError("colocate_async requires train_batch_size divisible by ppo_mini_batch_size")
        super().__init__(config)

    def get_llm_client(self):
        """Retry whole samples while the hybrid pool is temporarily unavailable."""
        return self.llm_server_manager.get_client(
            client_cls=DiffusionWholeSampleRetryLLMServerClient,
            only_hybrid=True,
            request_timeout_s=self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S),
        )

    def on_train_begin(self):
        """Top up warmup without duplicating restored prompt groups."""
        # Restored terminal groups count toward warmup as well as in-flight groups.
        items = (tq.kv_list("train") or {}).get("train", {})
        queued = sum(tag.get("is_prompt", False) for tag in items.values())
        target = self.config.trainer.v1.colocate_async.num_warmup_batches * self.config.data.train_batch_size
        if target > queued:
            self._add_prompts_to_generate(target - queued)

    def on_sample_end(self):
        """Close routing and reclaim rollout resources before actor computation."""
        # Remove routing before aborting: retries wait until committed weights are ready.
        manager = self.llm_server_manager
        with marked_timer("colocate_handoff_to_actor", self.timing_raw):
            with marked_timer("colocate_route_close", self.timing_raw):
                ray.get(manager.global_load_balancer.remove_servers.remote(manager.server_addresses))
            with marked_timer("colocate_abort", self.timing_raw):
                self.checkpoint_manager.abort_replicas()
            with marked_timer("colocate_sleep", self.timing_raw):
                super().on_sample_end()

    def on_step_end(self):
        """Reopen routing only after committing weights and resuming generation."""
        with marked_timer("update_weights", self.timing_raw, color="red"):
            with marked_timer("colocate_weight_sync", self.timing_raw):
                self.checkpoint_manager.update_weights(self.global_steps)
            with marked_timer("colocate_resume", self.timing_raw):
                self.checkpoint_manager.resume_generation_replicas()
            manager = self.llm_server_manager
            servers = dict(zip(manager.server_addresses, manager.server_handles, strict=True))
            with marked_timer("colocate_route_open", self.timing_raw):
                ray.get(manager.global_load_balancer.add_servers.remote(servers))

    def _compute_validation_reward(self, data):
        if reward_pool_is_separate(self.config):
            return super()._compute_validation_reward(data)

        manager = self.llm_server_manager
        timeout = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
        ray.get(manager.global_load_balancer.remove_servers.remote(manager.server_addresses))
        ray.get([server.quiesce_generation.remote(timeout) for server in manager.server_handles], timeout=timeout)
        self.checkpoint_manager.sleep_replicas()

        reward = self._compute_reward_colocate(data)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.checkpoint_manager.wake_up_replicas()
        self.checkpoint_manager.resume_generation_replicas()
        servers = dict(zip(manager.server_addresses, manager.server_handles, strict=True))
        ray.get(manager.global_load_balancer.add_servers.remote(servers))
        return reward

    def prepare_step(self) -> dict:
        """Feed normally and observe group counts from the existing replay snapshot."""
        metrics = super().prepare_step()
        replay = self.replay_buffer
        metrics["rollout/queue/sampleable_groups"] = replay.get_sampleable_count(self.global_steps, "train")
        for name in ("pending", "running", "finished", "failure"):
            metrics[f"rollout/queue/{name}_groups"] = len(getattr(replay, f"{name}_keys")["train"])
        return metrics

    def _sample_training_batch(self, batch_size):
        refill = self.replay_buffer.refill_fn
        timeout_s = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
        deadline = time.monotonic() + timeout_s

        def refill_before_deadline(num_prompts):
            if time.monotonic() >= deadline:
                raise TimeoutError("colocate_async sampling exhausted its rollout refill deadline")
            return refill(num_prompts)

        self.replay_buffer.refill_fn = refill_before_deadline
        try:
            batch, metrics = super()._sample_training_batch(batch_size)
        finally:
            self.replay_buffer.refill_fn = refill
        tags = [tag for tag in batch.tags if not tag.get("is_padding", False)]
        metrics["rollout/trained_trajectories/retries"] = sum(tag.get("rollout_retries", 0) for tag in tags)
        metrics["rollout/trained_trajectories/retry_wait_s"] = sum(tag.get("rollout_retry_wait_s", 0) for tag in tags)
        return batch, metrics

    def _compute_metrics(self, batch_meta, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch_meta, metrics, timing_raw, global_steps, epoch)
        metrics["perf/colocate_gpu_handoff_s"] = timing_raw["colocate_handoff_to_actor"] + timing_raw["update_weights"]
        metrics["perf/trained_trajectories_per_s"] = (
            sum(not tag.get("is_padding", False) for tag in batch_meta.tags) / timing_raw["step"]
        )
