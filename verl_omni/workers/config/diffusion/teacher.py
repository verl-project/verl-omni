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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.model import MtpConfig

from .model import DiffusionModelConfig

__all__ = [
    "TeacherCheckpointConfig",
    "TeacherEngineConfig",
    "TeacherModelEntry",
    "TeacherPlacementConfig",
    "DiffusionTeacherConfig",
    "resolve_teacher_model_config",
    "resolve_teacher_engine_config",
]

VALID_PLACEMENT_MODES = ("colocated", "standalone")


def _or_actor(teacher_value: Any, actor_value: Any) -> Any:
    """Teacher-owned field defaulting to the actor's; ``None`` means "not set"."""
    return actor_value if teacher_value is None else teacher_value


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


def resolve_teacher_model_config(actor: DiffusionModelConfig, entry: TeacherModelEntry) -> DiffusionModelConfig:
    """Build the teacher's model config as an allow-list over a fresh instance.

    Deliberately not a copy of the actor's config with a few overrides: that is a
    deny-list over a ~20-field dataclass, and everything unlisted leaks. The
    sharpest leak is ``config_path``, which the VeOmni engine resolves as
    ``config_path or weights_path`` -- a non-empty actor value would silently
    define the teacher's architecture from the student's directory.
    """
    fields = {}

    # 1. teacher-owned: how this checkpoint sits on disk
    fields.update(
        path=entry.model.path,
        trust_remote_code=_or_actor(entry.model.trust_remote_code, actor.trust_remote_code),
        use_shm=_or_actor(entry.model.use_shm, actor.use_shm),
        transformer_subfolder=entry.model.transformer_subfolder,
    )

    # 2. inherited: trajectory semantics belong to the replayed rollout, not to the
    #    teacher. `algorithm` is a verl-omni registry key that model_index.json cannot
    #    supply, and the pipeline adapter is chosen by (architecture, algorithm)
    #    jointly, so only half of that pair comes from disk.
    fields.update(
        model_type=actor.model_type,
        algorithm=actor.algorithm,
        pipeline=deepcopy(actor.pipeline),
        algo=deepcopy(actor.algo),
        attn_backend=actor.attn_backend,
        external_lib=actor.external_lib,
        fsdp_layer_prefixes=list(actor.fsdp_layer_prefixes),
    )

    # 3. checkpoint-derived: nothing here on purpose. Leaving architecture /
    #    transformer_config / local_path / tokenizer_path unset makes __post_init__
    #    derive them, and it resolves the local dir *before* reading model_index.json,
    #    so an HF/HDFS path is materialised first rather than read as a local one.

    # 4. forced frozen defaults: fields whose dataclass defaults are wrong for a
    #    teacher (the worker loads transformer + scheduler only, and nothing backprops).
    fields.update(
        load_tokenizer=False,
        tokenizer=None,
        processor=None,
        tokenizer_path=None,
        local_tokenizer_path=None,
        enable_gradient_checkpointing=False,
        lora_rank=0,
        lora_adapter_path=None,
        lora={},
        lora_dtype=None,
        policy_state_adapters=(),
        mtp=MtpConfig(enable=False),
        config_path=None,
        local_path=None,
    )

    return DiffusionModelConfig(**fields)


def resolve_teacher_engine_config(
    actor_cfg,
    rollout_cfg,
    actor_engine: FSDPEngineConfig,
    entry: TeacherModelEntry,
    world_size: int,
) -> FSDPEngineConfig:
    """Build the teacher's engine config. Allow-list again; two fields carry traps.

    ``infer_micro_batch_size_per_gpu`` is read from the *source* configs rather
    than from ``actor_engine``: the diffusion actor branch writes it at worker
    init, so reading it here -- before worker construction -- would yield None,
    and ``TrainingWorker.infer_batch`` requires a finite value.

    ``forward_only`` is a correctness invariant, not a preference: it selects the
    no-grad path, skips optimizer/LR-scheduler construction, and picks the
    wrapping and offload branches. It is also why no offload knobs exist here --
    both FSDP paths already force CPU offload for a forward-only module, so a
    knob would be inert.

    Fields not named below (wrap_policy, mixed_precision, reshard_after_forward,
    use_orig_params, use_torch_compile, ...) take their FSDPEngineConfig defaults
    and are never inherited from the actor.
    """
    micro_bsz = entry.engine.micro_batch_size_per_gpu
    if micro_bsz is None:
        micro_bsz = rollout_cfg.get("log_prob_micro_batch_size_per_gpu")
    if micro_bsz is None:
        micro_bsz = actor_cfg.ppo_micro_batch_size_per_gpu
    if micro_bsz is None:
        raise ValueError(
            "Diffusion teacher requires teacher.engine.micro_batch_size_per_gpu, "
            "rollout.log_prob_micro_batch_size_per_gpu, or actor.ppo_micro_batch_size_per_gpu "
            "(used by prepare_micro_batches)."
        )

    return FSDPEngineConfig(
        strategy=entry.engine.strategy,
        forward_only=True,
        model_dtype=_or_actor(entry.engine.model_dtype, actor_engine.model_dtype),
        infer_micro_batch_size_per_gpu=micro_bsz,
        fsdp_size=world_size,
        ulysses_sequence_parallel_size=actor_engine.ulysses_sequence_parallel_size,
        param_offload=False,
        offload_policy=False,
        optimizer_offload=False,
        grad_offload=False,
    )
