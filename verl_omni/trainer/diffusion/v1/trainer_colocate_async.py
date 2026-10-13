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

import ray
import transfer_queue as tq
from verl.utils.debug import marked_timer

from verl_omni.trainer.diffusion.v1.trainer_base import register_diffusion_trainer
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient


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
        if not rollout.free_cache_engine:
            raise ValueError("colocate_async requires rollout.free_cache_engine=true to reclaim rollout GPU memory")
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
        return self.llm_server_manager.get_client(client_cls=DiffusionWholeSampleRetryLLMServerClient, only_hybrid=True)

    def on_train_begin(self):
        # Restored terminal groups count toward warmup as well as in-flight groups.
        items = (tq.kv_list("train") or {}).get("train", {})
        queued = sum(tag.get("is_prompt", False) for tag in items.values())
        target = self.config.trainer.v1.colocate_async.num_warmup_batches * self.config.data.train_batch_size
        if target > queued:
            self._add_prompts_to_generate(target - queued)

    def on_sample_end(self):
        # Remove routing before aborting: retries wait until committed weights are ready.
        manager = self.llm_server_manager
        ray.get(manager.global_load_balancer.remove_servers.remote(manager.server_addresses))
        self.checkpoint_manager.abort_replicas()
        super().on_sample_end()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.resume_generation_replicas()
            manager = self.llm_server_manager
            servers = dict(zip(manager.server_addresses, manager.server_handles, strict=True))
            ray.get(manager.global_load_balancer.add_servers.remote(servers))
