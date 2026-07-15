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

"""Shared Ray initialization, launch, and remote task runner for verl-omni trainers."""

from __future__ import annotations

import inspect
import os
import socket
from pprint import pprint
from typing import Any

import ray
from omegaconf import OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import is_cuda_available

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
)
from verl_omni.utils.fs import resolve_model_local_dir

__all__ = [
    "RayTrainerTaskRunner",
    "get_ray_trainer_cls",
    "launch_ray_task_runner",
    "maybe_set_determinism_env",
]


def get_ray_trainer_cls(config):
    """Return the trainer class selected by ``algorithm.trainer_type`` and model type."""
    trainer_type = config.algorithm.trainer_type
    if trainer_type == "policy_gradient":
        return PolicyGradientRayTrainer
    if trainer_type == "direct_preference":
        if config.actor_rollout_ref.model.get("model_type", "language_model") == "omni_model":
            from verl_omni.trainer.omni.ray_omni_trainer import OmniDirectPreferenceRayTrainer

            return OmniDirectPreferenceRayTrainer
        return DirectPreferenceRayTrainer
    raise ValueError(
        f"Unsupported trainer_type {trainer_type!r}. Expected one of: 'policy_gradient', 'direct_preference'."
    )


class RayTrainerTaskRunner:
    """Ray remote class for executing distributed training with the unified model engine."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor (and optional rollout/ref) workers using the unified model engine."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        if config.algorithm.sample_source == "offline":
            if not hasattr(Role, "Actor"):
                raise ValueError("Offline training without rollout requires verl Role.Actor support.")
            role = Role.Actor
        elif need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout

        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

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

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_resource_pool(self, config):
        """Register reward-model GPU pool for online sampling."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.sample_source == "online":
            if config.reward.reward_model.enable:
                if config.reward.reward_model.enable_resource_pool:
                    self.mapping[Role.RewardModel] = "reward_pool"
                else:
                    self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        del config, ref_policy_cls
        return

    def get_trainer_cls(self, config):
        """Return the trainer class for this task runner."""
        return get_ray_trainer_cls(config)

    def before_load_tokenizer(self, config):
        """Hook invoked before tokenizer/processor loading."""
        external_lib = config.actor_rollout_ref.model.get("external_lib", None)
        if external_lib is not None:
            if isinstance(external_lib, str) and external_lib.strip().startswith("["):
                external_lib = OmegaConf.create(external_lib)
            if OmegaConf.is_config(external_lib):
                external_lib = OmegaConf.to_container(external_lib, resolve=True)
            from verl.utils.import_utils import import_external_libs

            import_external_libs(external_lib)

    def run(self, config):
        """Execute the main training workflow."""
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        local_path = resolve_model_local_dir(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        if config.actor_rollout_ref.model.tokenizer_path is None:
            tokenizer_path = os.path.join(local_path, "tokenizer")
            config.actor_rollout_ref.model.tokenizer_path = (
                tokenizer_path if os.path.exists(tokenizer_path) else local_path
            )

        self.before_load_tokenizer(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path, trust_remote_code=trust_remote_code)
        processor_path = os.path.join(local_path, "processor")
        if not os.path.exists(processor_path):
            processor_path = local_path
        processor = hf_processor(processor_path, trust_remote_code=trust_remote_code, use_fast=True)

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

        trainer_cls = self.get_trainer_cls(config)
        trainer = trainer_cls(
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


def maybe_set_determinism_env(config) -> None:
    """Propagate determinism env vars before ``ray.init()`` when configured."""
    rollout_cfg = config.actor_rollout_ref.rollout
    rm_rollout_cfg = config.reward.reward_model.rollout
    if rollout_cfg.full_determinism or (config.reward.reward_model.enable and rm_rollout_cfg.full_determinism):
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_cfg.seed)


def _resolve_ppo_runtime_env(config) -> dict[str, Any]:
    signature = inspect.signature(get_ppo_ray_runtime_env)
    if len(signature.parameters) == 0:
        return get_ppo_ray_runtime_env()
    return get_ppo_ray_runtime_env(config)


def launch_ray_task_runner(
    config,
    task_runner_class,
    *,
    enable_transfer_queue_env: bool = False,
    propagate_determinism: bool = False,
) -> None:
    """Initialize Ray (if needed) and run ``task_runner_class.run(config)`` remotely."""
    if propagate_determinism:
        maybe_set_determinism_env(config)

    if not ray.is_initialized():
        default_runtime_env = _resolve_ppo_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if enable_transfer_queue_env and OmegaConf.select(config, "transfer_queue.enable", default=False):
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and OmegaConf.select(config, "global_profiler.steps") is not None
        and len(OmegaConf.select(config, "global_profiler.steps")) > 0
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
