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
"""Omni teacher model manager for vllm_omni rollout.

Extends verl's MultiTeacherModelManager to use OmniModelConfig for vllm_omni
inference, ensuring proper processor initialization with dedup_pad_tokens.
"""

from verl.experimental.teacher_loop.teacher_model import MultiTeacherModelManager, TeacherModelManager
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.workers.config import OmniModelConfig


class OmniTeacherModelManager(TeacherModelManager):
    """Teacher model manager that uses OmniModelConfig for vllm_omni rollout."""

    def _initialize_llm_servers(self):
        from verl.experimental.teacher_loop.teacher_model import _run_all
        from verl.workers.rollout.replica import get_rollout_replica_class

        teacher_model_config = self.teacher_model_config
        per_replica_world_size = teacher_model_config.per_replica_world_size
        num_replicas = teacher_model_config.num_replicas
        expected_pool_size = num_replicas * per_replica_world_size
        if self.resource_pool.world_size != expected_pool_size:
            raise ValueError(
                f"Teacher {teacher_model_config.key!r} expected sub-pool of size "
                f"{expected_pool_size} (num_replicas={num_replicas} * "
                f"per_replica_world_size={per_replica_world_size}), but got "
                f"{self.resource_pool.world_size}."
            )

        gpus_per_node = self.distillation_config.n_gpus_per_node
        rollout_config = teacher_model_config.inference
        rollout_replica_class = get_rollout_replica_class(rollout_config.name)

        # Use OmniModelConfig for vllm_omni to properly initialize processor with dedup_pad_tokens
        if rollout_config.name == "vllm_omni":
            model_config = OmniModelConfig(path=teacher_model_config.model_path)
        else:
            from verl.workers.config import HFModelConfig

            model_config = HFModelConfig(path=teacher_model_config.model_path)

        name_suffix = (teacher_model_config.key or "").replace("/", "_")
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=gpus_per_node,
                is_teacher_model=True,
                name_suffix=name_suffix,
            )
            for replica_rank in range(num_replicas)
        ]
        split_resource_pools = split_resource_pool(self.resource_pool, split_size=per_replica_world_size)
        assert len(split_resource_pools) == len(self.rollout_replicas)
        self._validate_replica_node_alignment(split_resource_pools, per_replica_world_size, gpus_per_node)
        _run_all(
            [
                server.init_colocated(resource_pool)
                for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
            ]
        )
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]


class OmniMultiTeacherModelManager(MultiTeacherModelManager):
    """Multi-teacher manager using OmniTeacherModelManager for vllm_omni rollout."""

    def __init__(self, config, resource_pool):
        self.config = config
        # Materializes teachers as OmniDistillationTeacherModelConfig via the
        # distillation _target_ override in omni_trainer.yaml, so vllm_omni is
        # accepted by _validate_topk_logprobs.
        self.distillation_config = omega_conf_to_dataclass(config.distillation)

        self.resource_pool = resource_pool
        self.teacher_model_managers = {}
        self.server_addresses = {}
        self.server_handles = {}
        self.load_balancer_handle = {}

        self._initialize_teacher_model_managers()

    def _initialize_teacher_model_managers(self):
        teacher_models = self.distillation_config.teacher_models
        split_sizes = [teacher.world_size for teacher in teacher_models.values()]
        split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)

        for (key, teacher_model_config), teacher_pool in zip(teacher_models.items(), split_pools, strict=True):
            manager = OmniTeacherModelManager(
                distillation_config=self.distillation_config,
                teacher_model_config=teacher_model_config,
                resource_pool=teacher_pool,
            )
            self.teacher_model_managers[key] = manager
            self.server_addresses[key] = manager.server_addresses
            self.server_handles[key] = manager.server_handles
            self.load_balancer_handle[key] = manager.load_balancer_handle
