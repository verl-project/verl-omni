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
"""Generic entrypoint for coupled and decoupled diffusion RL training."""

import hydra
from verl.utils.device import auto_set_device

from verl_omni.trainer.diffusion.main_flowgrpo import run_flowgrpo


@hydra.main(config_path="../config", config_name="diffusion_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_flowgrpo(config)


if __name__ == "__main__":
    main()
