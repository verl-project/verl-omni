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

"""Diffusion-specific algorithm config additions for verl_omni."""

from dataclasses import dataclass

from verl.base_config import BaseConfig

__all__ = ["DiffusionAlgoConfig"]


@dataclass
class DiffusionAlgoConfig(BaseConfig):
    """Diffusion-specific algorithm config."""

    trainer_type: str = "policy_gradient"
    sample_source: str = "online"
    adv_estimator: str = "flow_grpo"
    norm_adv_by_std_in_grpo: bool = True
    bypass_mode: bool = False
    global_std: bool = True
