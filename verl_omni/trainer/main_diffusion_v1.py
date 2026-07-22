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
"""V1 entrypoint for diffusion model RL training.

Mirrors verl's ``verl.trainer.main_ppo.run_ppo`` / ``TaskRunnerV1`` but selects a
``PolicyGradientDiffusionTrainerV1`` subclass via ``trainer.v1.trainer_mode`` and
wires a ``DiffusionAgentLoopManagerTQ``. TransferQueue is initialized and closed
inside the Ray task runner. The trainer is self-contained (it resolves the
model, tokenizer, processor, datasets, workers, rollout server, reward loop,
and checkpoint engine in ``init``), so this runner stays thin.
"""

import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_class_from_fqn

from verl_omni.utils.diffusion_attention import fallback_fa3_if_unavailable, validate_attention_consistency

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def run_diffusion_v1(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed v1 diffusion training.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed diffusion training including Ray initialization
                settings, model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
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
        task_runner_class = ray.remote(num_cpus=1)(DiffusionTaskRunnerV1)

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


@ray.remote
class DiffusionTaskRunnerV1:
    """V1 TaskRunner for diffusion policy-gradient training.

    The trainer owns all worker/dataset/rollout/reward/checkpoint setup; this
    runner only selects the trainer class, initializes TransferQueue, creates
    the ``DiffusionAgentLoopManagerTQ``, and drives ``trainer.init``/``fit``.
    """

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        """Initialize the diffusion agent loop manager.

        Users can plug a custom manager via
        ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``; otherwise the
        default ``DiffusionAgentLoopManagerTQ`` is used. The only requirements are
        implementing ``generate_sequences`` and putting agent loop outputs into
        TransferQueue.
        """
        from verl_omni.agent_loop import DiffusionAgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = DiffusionAgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        """Run the v1 diffusion training process."""
        import transfer_queue as tq

        from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

        # TransferQueue is required for v1; force-enable it regardless of the yaml default.
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        # initialize transfer queue inside the Ray task runner
        tq.init(config.transfer_queue)
        try:
            trainer_cls = get_diffusion_trainer_cls(config.trainer.v1.trainer_mode)
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            tq.close()


@hydra.main(config_path="./config", config_name="diffusion_trainer", version_base=None)
def main(config):
    """Main entry point for v1 diffusion training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    OmegaConf.resolve(config)
    fallback_fa3_if_unavailable(config)
    validate_attention_consistency(config)

    if config.trainer.get("use_v1", False):
        run_diffusion_v1(config)
    else:
        # Fall back to the legacy (v0) diffusion trainer entrypoint.
        from verl_omni.trainer.main_diffusion import run_diffusion

        run_diffusion(config)


if __name__ == "__main__":
    main()
