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
"""Ray-entrypoint-compatible shell around the pure distillation control plane.

PR 1 deliberately stops before model allocation. The shell accepts the same
constructor protocol and lifecycle calls as the existing diffusion trainers so
``algorithm.trainer_type=distillation`` reaches an explicit PR 2 boundary rather
than failing with a Python signature error.
"""

from __future__ import annotations

from typing import Any, Optional

from verl_omni.trainer.diffusion.diffusion_trainer_utils import validate_distillation_config
from verl_omni.trainer.diffusion.distillation.contracts import DistillationPlan
from verl_omni.trainer.diffusion.distillation.control_plane import DistillationTrainerControlPlane
from verl_omni.trainer.diffusion.distillation.recipes import build_plan_from_config

__all__ = ["DistillationRayTrainer"]


class DistillationRayTrainer:
    """Production-compatible driver shell for a validated distillation plan."""

    def __init__(
        self,
        config=None,
        tokenizer=None,
        role_worker_mapping=None,
        resource_pool_manager=None,
        ray_worker_group_cls=None,
        processor=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
        *,
        plan: Optional[DistillationPlan] = None,
        capabilities: Optional[frozenset[str]] = None,
        executor: Optional[Any] = None,
        batch_provider: Optional[Any] = None,
        hooks: Optional[Any] = None,
    ) -> None:
        if isinstance(config, DistillationPlan) and plan is None:
            plan = config
            config = None
        if config is not None:
            validate_distillation_config(config)
        if plan is not None and config is not None and capabilities is not None:
            raise ValueError("Pass either an explicit plan or config+capabilities, not both.")
        if plan is None and config is not None and capabilities is not None:
            plan = build_plan_from_config(config, capabilities)

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.train_sampler = train_sampler
        self.device_name = device_name
        self.plan = plan
        self.capabilities = capabilities
        self.executor = executor
        self.batch_provider = batch_provider
        self.hooks = hooks
        self._control_plane: Optional[DistillationTrainerControlPlane] = None

    def init_workers(self) -> None:
        """Validate the PR 1 boundary before PR 2 supplies role-group workers."""
        if self.executor is None or self.batch_provider is None:
            raise NotImplementedError(
                "The multi-role distillation workers and architecture capability binding land in PR 2. "
                "PR 1 accepts the production trainer interface but does not allocate model workers."
            )
        if self.plan is None:
            raise ValueError("A validated DistillationPlan is required when an executor is bound.")

    def build_control_plane(self) -> DistillationTrainerControlPlane:
        """Construct the pure control plane from a plan and bound collaborators."""
        self.init_workers()
        assert self.plan is not None
        self._control_plane = DistillationTrainerControlPlane(
            plan=self.plan,
            executor=self.executor,
            batch_provider=self.batch_provider,
            hooks=self.hooks,
        )
        return self._control_plane

    @property
    def control_plane(self) -> DistillationTrainerControlPlane:
        if self._control_plane is None:
            return self.build_control_plane()
        return self._control_plane

    def fit(self, num_cycles: int = 0) -> None:
        """Drive the injected CPU control plane; production data plane arrives in PR 2."""
        self.control_plane.run(num_cycles)
