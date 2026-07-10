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
"""Entrypoint for Omni model RL training."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.utils.device import auto_set_device

from verl_omni.trainer.main_diffusion import TaskRunner as DiffusionTaskRunner
from verl_omni.trainer.main_diffusion import run_diffusion
from verl_omni.trainer.omni.ray_omni_trainer import OmniDirectPreferenceRayTrainer


@hydra.main(config_path="./config", config_name="omni_trainer", version_base=None)
def main(config):
    """Main entry point for Omni model training with Hydra configuration management."""
    auto_set_device(config)
    OmegaConf.resolve(config)
    run_omni(config)


def run_omni(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed Omni training."""
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(OmniTaskRunner)
    run_diffusion(config, task_runner_class=task_runner_class)


class OmniTaskRunner(DiffusionTaskRunner):
    """Task runner that reuses the unified worker setup with an Omni trainer."""

    def run(self, config):
        """Execute the main Omni training workflow."""
        from pprint import pprint

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
        from verl.utils.import_utils import import_external_libs

        import_external_libs(config.actor_rollout_ref.model.get("external_lib", None))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor_path = os.path.join(local_path, "processor")
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

        trainer = OmniDirectPreferenceRayTrainer(
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
