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
import asyncio
import copy

import numpy as np
import ray
from omegaconf import open_dict
from tensordict import TensorDict
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from .deployment import (
    EngineRewardExecutor,
    MultiRewardModelManager,
    NativeRewardExecutor,
    accelerator_workers_enabled,
    build_engine_reward_executors,
    build_native_reward_executors,
    has_reward_deployments,
    streaming_reward_enabled,
    validate_reward_deployment_terms,
)


class OmniRewardLoopWorker(RewardLoopWorker):
    """RewardLoopWorker with named engine and native reward executors."""

    def __init__(
        self,
        config,
        reward_router_address=None,
        reward_executor_specs=None,
    ):
        self.reward_executor_specs = reward_executor_specs or {}
        self.engine_reward_executors: dict[str, EngineRewardExecutor] = build_engine_reward_executors(
            self.reward_executor_specs
        )
        self.native_reward_executors: dict[str, NativeRewardExecutor] = build_native_reward_executors(
            self.reward_executor_specs
        )
        self._native_batch_active = False
        super().__init__(config, reward_router_address)

    def _init_reward_fn(self):
        super()._init_reward_fn()
        if hasattr(self.reward_manager, "set_reward_executors"):
            self.reward_manager.set_reward_executors(
                self.engine_reward_executors,
                self.native_reward_executors,
            )

    async def compute_score_batch(self, data):
        self._native_batch_active = True
        try:
            return await super().compute_score_batch(data)
        finally:
            self._native_batch_active = False
            await self._sleep_native_reward_executors()

    async def compute_score(self, data):
        # Streaming agent loops submit one item at a time rather than calling
        # ``compute_score_batch``.  Keep native models bounded to that request
        # in this path; otherwise a worker-local CLIP would stay resident into
        # the subsequent actor backward phase.
        if not self.native_reward_executors or self._native_batch_active:
            return await super().compute_score(data)
        try:
            return await super().compute_score(data)
        finally:
            await self._sleep_native_reward_executors()

    async def close(self):
        await self._sleep_native_reward_executors()

    async def _sleep_native_reward_executors(self) -> None:
        await asyncio.gather(*(executor.sleep() for executor in self.native_reward_executors.values()))


class OmniRewardLoopManager(RewardLoopManager):
    """RewardLoopManager that can start/stop the profiler on the reward-model rollout servers.

    The reward-model servers are the same ``RolloutReplica`` stack as the actor rollout
    servers, whose per-server profiler fan-out already exists (``RolloutReplica.start_profile``);
    upstream ``RewardLoopManager`` just exposes no caller for it. The trainer invokes these
    around the phase where the servers actually score: the generation phase when reward
    computation streams with rollout, or ``compute_rm_score`` in colocate mode. Configured
    via ``reward.reward_model.rollout.profiler``.
    """

    def __init__(self, config, rm_resource_pool=None, accelerator_resource_pool=None):
        self.accelerator_resource_pool = accelerator_resource_pool
        if has_reward_deployments(config):
            validate_reward_deployment_terms(config)
        self.multi_reward_model_manager = MultiRewardModelManager(
            config,
            # The trainer maps Role.RewardModel to global_pool or reward_pool.
            # Each engine deployment receives a sub-pool from this one parent.
            resource_pool=rm_resource_pool,
        )
        use_accelerator_workers = accelerator_workers_enabled(config)
        if self.multi_reward_model_manager.deployments or use_accelerator_workers:
            if use_accelerator_workers and config.reward.reward_model.get("enable", False):
                raise ValueError("Accelerator reward workers cannot be combined with reward.reward_model.enable=True")
            if config.reward.reward_model.get("enable", False):
                raise ValueError(
                    "Use reward.deployments for deployment-backed rewards; "
                    "reward.reward_model.enable cannot be combined with reward.deployments."
                )
            if self.multi_reward_model_manager.deployments and not config.reward.get("reward_functions"):
                raise ValueError("reward.deployments requires non-empty reward.reward_functions")
            self.config = config
            if self.multi_reward_model_manager.deployments:
                with open_dict(config.reward.reward_manager):
                    config.reward.reward_manager.name = "MultiVisualRewardManager"
            self.reward_model_manager = None
            self.reward_router_address = None
            self.reward_loop_workers_class = ray.remote(OmniRewardLoopWorker)
            self.reward_manager_cls = resolve_reward_manager_cls(config)
            self._init_reward_loop_workers()
        else:
            super().__init__(config=config, rm_resource_pool=rm_resource_pool)

    @property
    def reward_loop_worker_handles(self):
        if not streaming_reward_enabled(self.config):
            return None
        return super().reward_loop_worker_handles

    def _init_reward_loop_workers(self):
        self.reward_loop_workers_class = ray.remote(OmniRewardLoopWorker)
        specs = self.multi_reward_model_manager.reward_executor_specs
        self._reward_worker_groups = {}
        self._reward_worker_group_configs = {}

        if self.multi_reward_model_manager.deployments:
            entries = self.config.reward.reward_functions
            native_names = {name for name, spec in specs.items() if spec.backend == "native"}
            terms_by_group = {}
            shared_terms = {}
            for term_name, term in entries.items():
                deployment = term.get("deployment")
                if deployment in native_names:
                    terms_by_group.setdefault(deployment, {})[term_name] = term
                else:
                    shared_terms[term_name] = term

            if shared_terms:
                group_config = self._copy_reward_config(shared_terms)
                workers = self._create_node_affinity_workers(
                    group_config,
                    {name: spec for name, spec in specs.items() if spec.backend == "engine"},
                    "engine_reward_loop_worker",
                )
                self._register_worker_group("shared", workers, group_config)

            for deployment_name, terms in terms_by_group.items():
                placement = self.multi_reward_model_manager.native_device_assignments[deployment_name]
                group_config = self._copy_reward_config(terms)
                with open_dict(group_config.reward):
                    group_config.reward.num_workers = len(placement)
                workers = self._create_native_workers(
                    group_config,
                    {deployment_name: specs[deployment_name]},
                    placement,
                    f"native_reward_loop_worker_{deployment_name}",
                )
                self._register_worker_group(deployment_name, workers, group_config)

            if not self._reward_worker_groups:
                raise ValueError("reward.deployments produced no reward worker groups")
            self.reward_loop_workers = self._flatten_worker_groups()
            return

        if specs:
            self._register_worker_group(
                "shared",
                self._create_node_affinity_workers(self.config, specs, "reward_loop_worker"),
                self.config,
            )
            self.reward_loop_workers = self._reward_worker_groups["shared"]
            return

        use_accelerator_workers = accelerator_workers_enabled(self.config)
        if use_accelerator_workers:
            accelerator_resource_pool = self.accelerator_resource_pool
            if accelerator_resource_pool is None:
                raise ValueError("Accelerator reward workers require an accelerator resource pool")
            from .accelerator_reward_workers import build_accelerator_reward_workers

            self.reward_loop_workers = build_accelerator_reward_workers(
                config=self.config,
                reward_loop_workers_class=self.reward_loop_workers_class,
                accelerator_resource_pool=accelerator_resource_pool,
                reward_router_address=self.reward_router_address,
                reward_executor_specs=specs,
            )
            self._register_worker_group("legacy", self.reward_loop_workers, self.config)
            return
        super()._init_reward_loop_workers()

    def _copy_reward_config(self, reward_functions):
        group_config = copy.deepcopy(self.config)
        with open_dict(group_config.reward):
            group_config.reward.reward_functions = copy.deepcopy(reward_functions)
            group_config.reward.num_workers = self.config.reward.num_workers
        return group_config

    def _register_worker_group(self, name, workers, config):
        self._reward_worker_groups[name] = workers
        self._reward_worker_group_configs[name] = config

    def _flatten_worker_groups(self):
        return [worker for workers in self._reward_worker_groups.values() for worker in workers]

    def _create_node_affinity_workers(self, config, specs, name_prefix):
        node_ids = [
            node["NodeID"]
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("CPU", 0) > 0
        ]
        if not node_ids:
            raise ValueError("No alive Ray node with CPU resources is available for reward workers")
        return [
            self.reward_loop_workers_class.options(
                name=f"{name_prefix}_{index}",
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ids[index % len(node_ids)], soft=True
                ),
            ).remote(config, self.reward_router_address, specs)
            for index in range(config.reward.num_workers)
        ]

    def _create_native_workers(self, config, specs, bundle_indices, name_prefix):
        from .accelerator_reward_workers import build_accelerator_reward_workers

        resource_pool = self.multi_reward_model_manager.native_resource_pool
        if resource_pool is None:
            raise ValueError("Native reward deployments require an allocated native resource pool")
        return build_accelerator_reward_workers(
            config=config,
            reward_loop_workers_class=self.reward_loop_workers_class,
            accelerator_resource_pool=resource_pool,
            reward_router_address=self.reward_router_address,
            reward_executor_specs=specs,
            bundle_indices=list(bundle_indices),
            worker_name_prefix=name_prefix,
        )

    def compute_rm_score(self, data):
        self.multi_reward_model_manager.wake_up()
        try:
            if not getattr(self, "_reward_worker_groups", None) or len(self._reward_worker_groups) <= 1:
                return super().compute_rm_score(data)
            return self._compute_named_deployment_scores(data)
        finally:
            self.multi_reward_model_manager.sleep()

    def _compute_named_deployment_scores(self, data: DataProto) -> DataProto:
        group_outputs = {}
        for group_name, workers in self._reward_worker_groups.items():
            num_workers = len(workers)
            padded_data, pad_size = pad_dataproto_to_divisor(data, num_workers)
            chunks = padded_data.chunk(num_workers)
            outputs = ray.get(
                [worker.compute_score_batch.remote(chunk) for worker, chunk in zip(workers, chunks, strict=True)]
            )
            flattened = [item for sublist in outputs for item in sublist]
            group_outputs[group_name] = flattened[: len(data)] if pad_size else flattened

        merged_scores = []
        merged_infos = []
        for index in range(len(data)):
            total = 0.0
            info = {}
            for outputs in group_outputs.values():
                item = outputs[index]
                total += float(item["reward_score"])
                for key, value in item.get("reward_extra_info", {}).items():
                    if key == "reward/combined":
                        continue
                    if key in info:
                        raise ValueError(f"Duplicate reward extra-info key {key!r} across worker groups")
                    info[key] = value
            info["reward/combined"] = total
            merged_scores.append(total)
            merged_infos.append(info)

        rm_scores = self.reward_manager_cls.assemble_rm_scores(data, merged_scores)
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))
        reward_extra_keys = list(dict.fromkeys(key for info in merged_infos for key in info))
        non_tensor_batch = {
            key: np.array([info.get(key) for info in merged_infos]) for key in reward_extra_keys
        }
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": reward_extra_keys},
        )

    def start_profile(self, **kwargs) -> None:
        """Start profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("start_profile", **kwargs)

    def stop_profile(self) -> None:
        """Stop profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("stop_profile")

    def _run_on_replicas(self, method: str, **kwargs) -> None:
        if self.reward_model_manager is None:
            return
        replicas = self.reward_model_manager.rollout_replicas

        async def run_all():
            await asyncio.gather(*[getattr(replica, method)(**kwargs) for replica in replicas])

        asyncio.run(run_all())
