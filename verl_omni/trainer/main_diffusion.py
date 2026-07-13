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
"""Entrypoint for diffusion model RL training."""

import json
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import auto_set_device, is_cuda_available

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
)
from verl_omni.utils.diffusion_attention import fallback_fa3_if_unavailable, validate_attention_consistency


def _prepare_processor_files(local_path: str, model_config) -> str | None:
    """Run the registered model hook before the driver loads its processor."""
    architecture = model_config.get("architecture")
    if architecture is None:
        with open(os.path.join(local_path, "model_index.json")) as model_index_file:
            architecture = json.load(model_index_file)["_class_name"]

    from verl_omni.pipelines.model_base import DiffusionModelBase

    return DiffusionModelBase.get_class_by_name(
        architecture,
        model_config.algorithm,
        model_config.get("external_lib"),
    ).prepare_processor_files(local_path)


@hydra.main(config_path="./config", config_name="diffusion_trainer", version_base=None)
def main(config):
    """Main entry point for diffusion model training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    OmegaConf.resolve(config)
    fallback_fa3_if_unavailable(config)
    validate_attention_consistency(config)
    run_diffusion(config)


def run_diffusion(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed diffusion training.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed diffusion training including Ray initialization
                settings, model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # When NVIDIA Nsight Systems is selected for the controller, launch the TaskRunner under nsys
    # using the Ray runtime env, mirroring verl/trainer/main_ppo.py.
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

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _get_trainer_cls(config):
    """Return the trainer class selected by ``algorithm.trainer_type``."""
    trainer_type = config.algorithm.trainer_type
    if trainer_type == "policy_gradient":
        return PolicyGradientRayTrainer
    if trainer_type == "direct_preference":
        return DirectPreferenceRayTrainer
    raise ValueError(
        f"Unsupported diffusion trainer_type {trainer_type!r}. Expected one of: 'policy_gradient', 'direct_preference'."
    )


class TaskRunner:
    """Ray remote class for executing distributed diffusion training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

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

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Register reward-model GPU pool for online sampling (used by RewardLoopManager)."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.sample_source == "online":
            if config.reward.reward_model.enable:
                # we do not use reward model workers, so we only register reward model in resource pool
                # without continue to register reward model worker in role mapping
                if config.reward.reward_model.enable_resource_pool:
                    self.mapping[Role.RewardModel] = "reward_pool"
                else:
                    self.mapping[Role.RewardModel] = "global_pool"
        elif config.algorithm.sample_source == "offline":
            return

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        # Ref policy has been fused into ActorRolloutRefWorker in new model engine.
        # we don't need to add a separate ref policy worker group.
        return

    def run(self, config):
        """Execute the main diffusion training workflow.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the diffusion training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl_omni.utils.fs import resolve_model_local_dir

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        self.add_reward_model_resource_pool(config)

        # Add a reference policy worker if KL loss is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Resolve the model path to an on-disk directory (downloads from HDFS or HF Hub
        # if necessary). `use_shm` enables shared-memory copy for faster reloads.
        local_path = resolve_model_local_dir(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        if config.actor_rollout_ref.model.tokenizer_path is None:
            tokenizer_path = os.path.join(local_path, "tokenizer")
            config.actor_rollout_ref.model.tokenizer_path = (
                tokenizer_path if os.path.exists(tokenizer_path) else local_path
            )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        prepared_processor_path = _prepare_processor_files(local_path, config.actor_rollout_ref.model)
        processor_path = os.path.join(local_path, "processor")
        if prepared_processor_path is not None:
            processor_path = prepared_processor_path
        if not os.path.exists(processor_path):
            processor_path = local_path
        processor = hf_processor(processor_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler, get_collate_fn

        collate_fn = get_collate_fn(config.data)

        # Create training and validation datasets.
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

        trainer_cls = _get_trainer_cls(config)
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
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()
