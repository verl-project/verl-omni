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
"""Entrypoint for omni (thinker/talker) model RL training.
"""

import hydra
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.utils.device import auto_set_device

import verl_omni.trainer.omni  # noqa: F401


@hydra.main(config_path="./config", config_name="omni_trainer", version_base=None)
def main(config):
    """Omni model training entrypoint.

    Configures device, resolves OmegaConf interpolations, forces V1 trainer
    mode, and delegates to verl's ``run_ppo`` with the V1 task runner.
    """
    auto_set_device(config)
    OmegaConf.resolve(config)

    # Omni models require V1 trainer infrastructure.
    config.trainer.use_v1 = True

    run_ppo(config, task_runner_class=TaskRunnerV1)


if __name__ == "__main__":
    main()
