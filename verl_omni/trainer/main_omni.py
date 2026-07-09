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
"""Entrypoint for omni-model RL training."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs


@hydra.main(config_path="./config", config_name="omni_trainer", version_base=None)
def main(config):
    """Main entry point for omni-model training with Hydra configuration management."""
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_omni(config)


def run_omni(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed omni training."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner = (
            OfflineOmniDPOTaskRunner if config.algorithm.get("sample_source", "online") == "offline" else OmniTaskRunner
        )
        task_runner_class = ray.remote(num_cpus=1)(task_runner)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class OmniTaskRunner:
    """PPO task runner with omni-specific processor and collate hooks."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker using the omni model-engine implementation."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

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
        """Add critic worker when the algorithm requires a critic."""
        if not need_critic(config):
            return

        from verl.trainer.ppo.ray_trainer import Role

        from verl_omni.workers.engine_workers import TrainingWorker

        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
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

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_resource_pool(self, config):
        """Register reward-model resource pool if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        """Register teacher-model resource pool if distillation is enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if is_distillation_enabled(config.get("distillation")):
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Reference policy is fused into ActorRolloutRefWorker."""
        return

    def _get_omni_adapter(self, model_config):
        architecture = model_config.get("architecture", None)
        model_stage = model_config.get("model_stage", None)
        if architecture is None or model_stage is None:
            return None

        external_lib = model_config.get("external_lib", None)
        if external_lib is not None:
            import_external_libs(external_lib)

        from verl_omni.pipelines.model_base import OmniModelBase

        return OmniModelBase.get_class(model_config)

    def _build_tokenizer_and_processor(self, config, local_path):
        model_config = config.actor_rollout_ref.model
        adapter = self._get_omni_adapter(model_config)
        if adapter is not None:
            tokenizer = adapter.configure_tokenizer(local_path, model_config)
            processor = adapter.configure_processor(local_path, model_config)
            return tokenizer, processor

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        return tokenizer, processor

    def run(self, config):
        """Execute the omni training workflow."""
        from pprint import pprint

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        tokenizer, processor = self._build_tokenizer_and_processor(config, local_path)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler, get_collate_fn

        collate_fn = get_collate_fn(config.data)
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
        trainer.fit()


class OfflineOmniDPOTaskRunner(OmniTaskRunner):
    """Task runner for actor-only offline omni DPO."""

    def add_actor_rollout_worker(self, config):
        """Add actor worker without starting rollout or vLLM components."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        actor_cls = ray.remote(ActorRolloutRefWorker)
        self.role_worker_mapping[Role.Actor] = actor_cls
        # RayPPOTrainer's legacy initializer still checks for an actor-rollout role.
        # The offline trainer only spawns Role.Actor.
        self.role_worker_mapping[Role.ActorRollout] = actor_cls
        self.mapping[Role.Actor] = "global_pool"
        return ActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        """Execute actor-only offline DPO training."""
        from pprint import pprint

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        _, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        tokenizer, processor = self._build_tokenizer_and_processor(config, local_path)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl_omni.trainer.omni.ray_omni_dpo_trainer import RayOmniDPOTrainer
        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler, get_collate_fn

        collate_fn = get_collate_fn(config.data)
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

        trainer = RayOmniDPOTrainer(
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
        trainer.fit()


if __name__ == "__main__":
    main()
