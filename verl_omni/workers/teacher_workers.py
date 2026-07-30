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
"""Frozen teacher worker for diffusion online policy distillation (RFC #293).

A thin, forward-only wrapper over the same ``TrainingWorker`` + engine stack the
actor and reference use -- the replay computation is identical, only the
checkpoint differs.

It is a separate worker class for a semantic reason rather than a mechanical
one. Reuse would be easy: ``str(Role.TeacherModel)`` is already ``"teacher"``
and the spawn loop is role-generic. But ``ActorRolloutRefWorker`` constructs
optimizers, manages checkpoint save/load and participates in rollout weight
sync -- machinery a frozen teacher must not inherit or be able to trigger by
misconfiguration. A separate class makes "frozen" structural instead of a
convention.
"""

import hashlib
from typing import Optional

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.workers.config.engine import TrainingWorkerConfig

from verl_omni.workers.config.diffusion import (
    DiffusionTeacherConfig,
    resolve_teacher_engine_config,
    resolve_teacher_model_config,
)
from verl_omni.workers.engine_workers import TrainingWorker

__all__ = ["DiffusionTeacherWorker", "build_teacher_training_config", "build_teacher_response"]

TEACHER_MESH_NAME = "teacher"


def build_teacher_training_config(config: DictConfig, world_size: int) -> tuple[str, TrainingWorkerConfig]:
    """Assemble the teacher's ``TrainingWorkerConfig`` from the actor_rollout_ref subtree.

    ``forward_only`` is re-checked on the constructed engine config rather than
    trusted: it is what skips optimizer construction and selects the no-grad
    path, so getting it wrong would not merely waste memory, it would break the
    frozen guarantee.
    """
    teacher_config = omega_conf_to_dataclass(config.teacher, DiffusionTeacherConfig)
    teacher_key, entry = next(iter(teacher_config.models.items()))

    actor_config = omega_conf_to_dataclass(config.actor)
    model_config = resolve_teacher_model_config(omega_conf_to_dataclass(config.model), entry)
    engine_config = resolve_teacher_engine_config(
        actor_cfg=config.actor,
        rollout_cfg=config.rollout,
        actor_engine=actor_config.engine,
        entry=entry,
        world_size=world_size,
    )

    if not engine_config.forward_only:
        raise ValueError(
            f"Teacher {teacher_key!r}: engine_config.forward_only must be True. It selects the "
            "no-grad path and skips optimizer construction, so a trainable teacher is not a "
            "performance regression but a broken frozen guarantee."
        )

    training_config = TrainingWorkerConfig(
        model_type=model_config.model_type,
        model_config=model_config,
        engine_config=engine_config,
        # No optimizer by construction: forward_only skips _build_optimizer entirely.
        optimizer_config=None,
        checkpoint_config=None,
    )
    return teacher_key, training_config


def build_teacher_response(output: TensorDict, all_timesteps: torch.Tensor, teacher_key: str) -> TensorDict:
    """Namespace, validate and downcast the engine output into the teacher_* contract.

    Exactly one key at MVP. ``teacher_noise_pred`` has a merged consumer but no
    producer on the policy-gradient path, so emitting a placeholder here would be
    worse than its absence.
    """
    prev_sample_mean = tu.get(output, "prev_sample_mean", default=None)
    if prev_sample_mean is None:
        raise ValueError(
            f"Teacher {teacher_key!r}: engine output has no 'prev_sample_mean'; got keys {sorted(output.keys())}."
        )

    expected_prefix = tuple(all_timesteps.shape[:2])
    actual_prefix = tuple(prev_sample_mean.shape[:2])
    if actual_prefix != expected_prefix:
        raise ValueError(
            f"Teacher {teacher_key!r}: prev_sample_mean has [batch, steps] prefix {actual_prefix}, "
            f"but the request carried {expected_prefix}."
        )

    teacher_mean = prev_sample_mean.detach().float().cpu()
    if not torch.isfinite(teacher_mean).all():
        raise ValueError(
            f"Teacher {teacher_key!r}: prev_sample_mean contains non-finite values; refusing to "
            "union a corrupt distillation target into the batch."
        )
    return tu.get_tensordict({"teacher_prev_sample_mean": teacher_mean})


def parameter_checksum(module: torch.nn.Module) -> str:
    """Stable hash of a module's parameters, for the frozen-teacher probes (§6.3)."""
    hasher = hashlib.sha256()
    for name, param in sorted(module.named_parameters(), key=lambda item: item[0]):
        hasher.update(name.encode())
        hasher.update(param.detach().to(torch.float32).cpu().numpy().tobytes())
    return hasher.hexdigest()


class DiffusionTeacherWorker(Worker, DistProfilerExtension):
    """Frozen, forward-only diffusion teacher.

    Builds no optimizer, exposes no checkpoint save/load, and takes no part in
    rollout weight sync. Residency is FSDP's to manage: ``forward_only`` already
    forces CPU offload in both FSDP paths and makes the engine's ``to()`` a
    no-op, so there is no wake/sleep to expose here.
    """

    def __init__(self, config: DictConfig, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.teacher_key: Optional[str] = None
        self.teacher: Optional[TrainingWorker] = None

        # The teacher shares the actor's pool and step, so it reports under the
        # actor's profiler settings; the teacher namespace adds no profiler surface.
        profiler_config = omega_conf_to_dataclass(config.actor.get("profiler", {}), dataclass_type=ProfilerConfig)
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Build the frozen teacher. Must run before the actor/rollout group (§6.1)."""
        self.teacher_key, training_config = build_teacher_training_config(self.config, world_size=self.world_size)
        self.teacher = TrainingWorker(config=training_config)
        self.teacher.reset()
        self.set_dispatch_collect(mesh_name=TEACHER_MESH_NAME, **self.teacher.get_dispatch_collect())

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name=TEACHER_MESH_NAME))
    @DistProfiler.annotate(color="purple", role="teacher_score")
    def compute_teacher_outputs(self, data: TensorDict) -> Optional[TensorDict]:
        """Replay the student's visited states under the teacher's weights."""
        all_timesteps = tu.get(data, "all_timesteps")
        output = self.teacher.infer_batch(data=data)
        if output is None:
            return None
        return build_teacher_response(output, all_timesteps=all_timesteps, teacher_key=self.teacher_key).cpu()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def teacher_param_checksum(self) -> str:
        """Probe for the frozen guarantee: unchanged across optimizer steps (§6.3)."""
        return parameter_checksum(self.teacher.engine.module)
