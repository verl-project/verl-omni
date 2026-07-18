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
"""The Qwen3-TTS talker training entrypoint (GSPO and online DPO), analogous to main_diffusion.py.

Both TTS recipes launch through here. For GSPO it behaves exactly like base verl's PPO entry. For
DPO it additionally binds the pairwise tts_dpo_loss on the actor, which base verl cannot select from
config: the DPO loss is sequence-level and needs ref_log_prob and the uid grouping, so it cannot be
a registered policy-loss `loss_mode` like gspo (base verl's ppo_loss drops `data` before dispatching
to the registry). It must be bound on the actor worker, which base verl only does for the critic,
through the public set_loss_fn seam (ray_trainer.py::init_workers). To insert that one call we run
our own trainer, but base verl's TaskRunner is a @ray.remote actor that hard-codes RayPPOTrainer and
cannot be subclassed, so OmniTaskRunner below is a verbatim copy of
verl/trainer/main_ppo_v0.py::TaskRunner (the legacy V0 flow the recipe runs; use_v1 defaults false),
following the existing verl_omni/trainer/main_diffusion.py precedent. The ONLY departure from base is
the marked set_loss_fn block in run() (a no-op unless loss_mode is dpo). Launch:
python3 -m verl_omni.trainer.main_tts.
"""

import os
import socket
from functools import partial

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device


class OmniTaskRunner:
    """Copy of verl/trainer/main_ppo_v0.py::TaskRunner; see the module docstring for why."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import TrainingWorker

        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_resource_pool(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"OmniTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        # --- the only departure from base verl's TaskRunner.run ---
        # Bind the online-DPO loss on the actor through the same public set_loss_fn seam base verl
        # uses for the critic (ray_trainer.py::init_workers). No-op for non-DPO configs.
        actor_cfg = config.actor_rollout_ref.actor
        if actor_cfg.policy_loss.get("loss_mode", None) == "dpo":
            from verl_omni.workers.utils.losses import tts_dpo_loss

            trainer.actor_rollout_wg.set_loss_fn(partial(tts_dpo_loss, config=omega_conf_to_dataclass(actor_cfg)))
        # --- end departure ---

        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    if config.trainer.get("use_v1", False):
        raise ValueError("verl_omni.trainer.main_tts supports the legacy V0 trainer only; set trainer.use_v1=false.")
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(OmniTaskRunner))


if __name__ == "__main__":
    main()
