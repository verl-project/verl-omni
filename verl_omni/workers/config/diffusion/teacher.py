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
"""Config for the frozen diffusion teacher used by online policy distillation.

The namespace is ``actor_rollout_ref.teacher``: the teacher is a sibling of the
reference policy, not verl's top-level ``distillation`` (which selects the token
distillation loss and would swap out ``diffusion_loss``).

Only fields describing *this* checkpoint live here. Everything defining the
replayed trajectory -- scheduler, sde_type/noise_level, guidance, latent layout --
is inherited from the actor, because ``distill_kl`` compares two Gaussians that
must share a transition variance.

``omega_conf_to_dataclass`` must be called with ``DiffusionTeacherConfig`` as the
explicit ``dataclass_type``; the ``_target_`` route leaves ``models`` as plain
dicts, which skips both the entry defaults and the validation below.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig

__all__ = [
    "TeacherCheckpointConfig",
    "TeacherEngineConfig",
    "TeacherModelEntry",
    "TeacherPlacementConfig",
    "DiffusionTeacherConfig",
]

VALID_PLACEMENT_MODES = ("colocated", "standalone")


@dataclass
class TeacherCheckpointConfig(BaseConfig):
    """How one teacher checkpoint sits on disk. ``None`` means "inherit from the actor"."""

    path: str = MISSING
    trust_remote_code: Optional[bool] = None
    use_shm: Optional[bool] = None
    transformer_subfolder: str = "transformer"


@dataclass
class TeacherEngineConfig(BaseConfig):
    """Teacher engine knobs. ``None`` means "inherit from the actor".

    ``strategy`` is deliberately *not* inherited: PR A implements the FSDP engine
    only, and inheriting would resolve to ``veomni`` on a VeOmni actor.
    Offload is absent by design -- a forward-only engine is CPU-offloaded
    unconditionally, so a knob here would be inert.
    """

    strategy: str = "fsdp"
    model_dtype: Optional[str] = None
    micro_batch_size_per_gpu: Optional[int] = None


@dataclass
class TeacherModelEntry(BaseConfig):
    """One teacher identity: a checkpoint plus its engine settings."""

    model: TeacherCheckpointConfig = field(default_factory=TeacherCheckpointConfig)
    engine: TeacherEngineConfig = field(default_factory=TeacherEngineConfig)


@dataclass
class TeacherPlacementConfig(BaseConfig):
    """Where the teacher runs. Deployment axis, orthogonal to teacher identity."""

    mode: str = "colocated"
    n_gpus_per_node: Optional[int] = None
    nnodes: Optional[int] = None


@dataclass
class DiffusionTeacherConfig(BaseConfig):
    """Teacher runtime config. Validation is skipped entirely while disabled."""

    enabled: bool = False
    models: dict[str, TeacherModelEntry] = field(default_factory=dict)
    placement: TeacherPlacementConfig = field(default_factory=TeacherPlacementConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        if not self.models:
            raise ValueError(
                "actor_rollout_ref.teacher.enabled is set but teacher.models is empty; "
                "add an entry such as `+actor_rollout_ref.teacher.models.default.model.path=<ckpt>`."
            )
        if len(self.models) > 1:
            raise NotImplementedError(
                f"The diffusion teacher runtime supports exactly one teacher, got {sorted(self.models)}. "
                "Multi-teacher routing is a follow-up; the config shape already allows it."
            )

        if self.placement.mode not in VALID_PLACEMENT_MODES:
            raise ValueError(
                f"Invalid actor_rollout_ref.teacher.placement.mode: {self.placement.mode!r}. "
                f"Must be one of {list(VALID_PLACEMENT_MODES)}."
            )
        if self.placement.mode == "standalone":
            raise NotImplementedError(
                "standalone teacher placement is introduced by the next runtime PR; use mode=colocated."
            )
        set_resources = [name for name in ("n_gpus_per_node", "nnodes") if getattr(self.placement, name) is not None]
        if set_resources:
            raise ValueError(
                f"actor_rollout_ref.teacher.placement.{set_resources} may only be set under "
                "mode=standalone; under colocated the teacher shares the actor's resource pool "
                "and these values would be silently ignored."
            )
