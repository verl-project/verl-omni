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

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name

from .reward_loop import OmniRewardLoopManager


# TODO: This is a temporary integration for Qwen-Image-Edit on the synchronous V1 trainer.
# Other pipelines should not rely on it until a general accelerator reward-worker lifecycle is available.
class LocalAcceleratorRewardLoopManager(OmniRewardLoopManager):
    """Run a local custom reward function on Ray-assigned accelerators."""

    def __init__(self, config, accelerator_resource_pool):
        if config.reward.reward_model.enable:
            raise ValueError("Local accelerator rewards cannot be combined with reward.reward_model.enable=True")
        if accelerator_resource_pool is None:
            raise ValueError("Local accelerator rewards require an accelerator resource pool")
        self.accelerator_resource_pool = accelerator_resource_pool
        super().__init__(config=config, rm_resource_pool=None)

    def _init_reward_loop_workers(self):
        resource_pool = self.accelerator_resource_pool
        if resource_pool.max_colocate_count < 2:
            raise ValueError(
                "Local accelerator reward workers require resource_pool.max_colocate_count >= 2 "
                "when colocated with ActorRollout"
            )
        placement_groups = resource_pool.get_placement_groups(device_name=get_device_name())
        bundles = [
            (placement_group, bundle_index)
            for bundle_index in range(max(resource_pool.store))
            for placement_group, local_world_size in zip(placement_groups, resource_pool.store, strict=True)
            if bundle_index < local_world_size
        ]

        num_workers = self.config.reward.num_workers
        if num_workers > len(bundles):
            raise ValueError(f"reward.num_workers ({num_workers}) exceeds accelerator pool size ({len(bundles)})")

        accelerator_options = get_platform().ray_resource_options(1 / resource_pool.max_colocate_count)
        self.reward_loop_workers = [
            self.reward_loop_workers_class.options(
                **accelerator_options,
                name=f"reward_loop_worker_{worker_index}",
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                ),
            ).remote(self.config, self.reward_router_address)
            for worker_index, (placement_group, bundle_index) in enumerate(bundles[:num_workers])
        ]


def create_v1_reward_loop_manager(config, rm_resource_pool, accelerator_resource_pool):
    """Select the opt-in accelerator manager for the v1 diffusion trainer."""
    if config.reward.custom_reward_function.get("use_accelerator", False):
        return LocalAcceleratorRewardLoopManager(config, accelerator_resource_pool)
    return OmniRewardLoopManager(config=config, rm_resource_pool=rm_resource_pool)
