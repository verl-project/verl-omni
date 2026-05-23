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
"""Config resolution utilities for the reward loop."""

from omegaconf import OmegaConf, open_dict


def resolve_multi_reward_config(config):
    """Resolve multi-reward configuration.

    If ``config.reward.reward_functions`` is non-empty, mutates the config so that
    the upstream RewardLoopWorker instantiates ``MultiVisualRewardManager`` and routes
    through the placeholder function.

    Raises:
        ValueError: If both ``reward_functions`` and ``custom_reward_function.path``
            are set (mutual exclusivity).
    """
    reward_functions = OmegaConf.select(config, "reward.reward_functions", default=None)

    # If reward_functions is empty or None, do nothing (backward compat)
    if not reward_functions:
        return config

    # Mutual exclusivity check
    custom_path = OmegaConf.select(config, "reward.custom_reward_function.path", default=None)
    if custom_path is not None:
        raise ValueError(
            "Cannot use both 'reward.reward_functions' and 'reward.custom_reward_function.path'. "
            "They are mutually exclusive. Remove one of them from your config."
        )

    # Mutate config to wire up MultiVisualRewardManager
    with open_dict(config):
        # Set placeholder so upstream RewardLoopWorker loads something (never called directly)
        config.reward.custom_reward_function.path = "pkg://verl_omni.reward_loop.reward_manager.multi"
        config.reward.custom_reward_function.name = "_multi_reward_placeholder"
        # Point reward_manager to MultiVisualRewardManager
        config.reward.reward_manager.name = "MultiVisualRewardManager"
        config.reward.reward_manager.module.path = "pkg://verl_omni.reward_loop.reward_manager"

    return config
