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
"""Omni distillation configs.

Subclass verl's distillation configs to teach the on-policy distillation
path about (a) the ``vllm_omni`` rollout engine, (b) the hidden-state
(nitrobrew) loss modes, which consume teacher hidden states instead of
top-k logprobs and add ``kd_temperature``, and (c) multimodal on-policy
self-distillation (SDPO / OPSD), where the policy acts as its own teacher
without provisioning a separate teacher cluster.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig
from verl.trainer.distillation.losses import DistillationLossSettings
from verl.workers.config import DistillationConfig, DistillationLossConfig, DistillationTeacherModelConfig

__all__ = [
    "DEFAULT_SELF_DISTILLATION_TEMPLATE",
    "HIDDEN_STATE_LOSS_MODES",
    "SELF_DISTILLATION_MODES",
    "SELF_TEACHER_SOURCES",
    "OmniDistillationConfig",
    "OmniDistillationLossConfig",
    "OmniDistillationLossSettings",
    "OmniDistillationTeacherModelConfig",
    "SelfDistillationConfig",
    "is_self_distillation_enabled",
    "validate_self_teacher_rollout_topk",
]

# Loss modes that consume per-position teacher hidden states.
HIDDEN_STATE_LOSS_MODES = ("nitrobrew", "nitrobrew_reverse_kl")

# Supported privileged-context selection modes for on-policy self-distillation.
SELF_DISTILLATION_MODES = ("sdpo", "opsd", "auto")

# Supported self-teacher parameter sources.
SELF_TEACHER_SOURCES = ("policy", "ema")

DEFAULT_SELF_DISTILLATION_TEMPLATE = (
    "\n\n[Additional reference context for verification — answer the original request directly.]\n{context}"
)


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for multimodal on-policy self-distillation (SDPO / OPSD)."""

    enabled: bool = False
    mode: str = "sdpo"
    teacher_source: str = "policy"
    ema_decay: float = 0.99
    fallback_to_pg: bool = True
    reward_threshold: float = 0.5
    include_environment_feedback: bool = True
    environment_feedback_key: str = "feedback"
    reference_solution_key: str = "ground_truth"
    dont_reprompt_on_self_success: bool = True
    preserve_multimodal_inputs: bool = True
    max_demonstration_chars: int = 4096
    max_feedback_chars: int = 2048
    max_reprompt_tokens: Optional[int] = None
    reprompt_template: str = DEFAULT_SELF_DISTILLATION_TEMPLATE

    def __post_init__(self) -> None:
        if self.mode not in SELF_DISTILLATION_MODES:
            raise ValueError(f"self_teacher.mode must be one of {SELF_DISTILLATION_MODES}, got {self.mode!r}.")
        if self.teacher_source not in SELF_TEACHER_SOURCES:
            raise ValueError(
                f"self_teacher.teacher_source must be one of {SELF_TEACHER_SOURCES}, got {self.teacher_source!r}."
            )
        if not (0.0 < float(self.ema_decay) < 1.0):
            raise ValueError(f"self_teacher.ema_decay must be in (0, 1), got {self.ema_decay!r}.")
        if self.max_demonstration_chars <= 0:
            raise ValueError(
                f"self_teacher.max_demonstration_chars must be positive, got {self.max_demonstration_chars!r}."
            )
        if self.max_feedback_chars <= 0:
            raise ValueError(f"self_teacher.max_feedback_chars must be positive, got {self.max_feedback_chars!r}.")
        if self.max_reprompt_tokens is not None and self.max_reprompt_tokens <= 0:
            raise ValueError(
                f"self_teacher.max_reprompt_tokens must be positive when set, got {self.max_reprompt_tokens!r}."
            )
        if "{context}" not in self.reprompt_template:
            raise ValueError("self_teacher.reprompt_template must contain a '{context}' placeholder.")


def is_self_distillation_enabled(distillation_config: Any) -> bool:
    """Return True when both distillation and self-teacher mode are enabled."""
    if distillation_config is None:
        return False
    enabled = (
        distillation_config.get("enabled", False)
        if isinstance(distillation_config, Mapping)
        else getattr(distillation_config, "enabled", False)
    )
    if not enabled:
        return False
    self_teacher = (
        distillation_config.get("self_teacher", None)
        if isinstance(distillation_config, Mapping)
        else getattr(distillation_config, "self_teacher", None)
    )
    if self_teacher is None:
        return False
    st_enabled = (
        self_teacher.get("enabled", False)
        if isinstance(self_teacher, Mapping)
        else getattr(self_teacher, "enabled", False)
    )
    return bool(st_enabled)


def validate_self_teacher_rollout_topk(rollout_config: Any, distillation_config: "OmniDistillationConfig") -> None:
    """Validate and seed ``max_logprobs`` on the student rollout engine for top-k self-distillation."""
    loss_cfg = distillation_config.distillation_loss
    if not loss_cfg.loss_settings.use_topk:
        return
    topk = loss_cfg.topk
    if topk is None or topk <= 0:
        raise ValueError("distillation.distillation_loss.topk must be positive when use_topk is True.")

    engine_kwargs = (
        rollout_config.get("engine_kwargs", {})
        if isinstance(rollout_config, Mapping)
        else getattr(rollout_config, "engine_kwargs", {})
    )
    if engine_kwargs is None:
        return
    rollout_name = (
        rollout_config.get("name", "vllm_omni")
        if isinstance(rollout_config, Mapping)
        else getattr(rollout_config, "name", "vllm_omni")
    )
    engine_key = "vllm_omni" if rollout_name == "vllm_omni" else str(rollout_name)
    backend_kwargs = dict(engine_kwargs.get(engine_key, {}) or {})
    max_logprobs = backend_kwargs.get("max_logprobs")
    if max_logprobs is None:
        backend_kwargs["max_logprobs"] = topk
        try:
            from omegaconf import OmegaConf, open_dict

            if OmegaConf.is_config(engine_kwargs):
                with open_dict(engine_kwargs):
                    engine_kwargs[engine_key] = backend_kwargs
            else:
                engine_kwargs[engine_key] = backend_kwargs
        except ImportError:
            engine_kwargs[engine_key] = backend_kwargs
    elif int(max_logprobs) < int(topk):
        raise ValueError(
            f"rollout engine_kwargs.{engine_key}.max_logprobs ({max_logprobs}) must be >= "
            f"distillation_loss.topk ({topk}) for top-k self-distillation."
        )


@dataclass
class OmniDistillationLossSettings(DistillationLossSettings):
    """Loss settings for the omni path."""

    use_hidden_states: bool = False

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator, self.use_hidden_states]) != 1:
            raise ValueError(
                "Exactly one of use_topk, use_estimator, use_hidden_states must be True, "
                f"but got {self.use_topk}, {self.use_estimator}, {self.use_hidden_states}."
            )


@dataclass
class OmniDistillationTeacherModelConfig(DistillationTeacherModelConfig):
    """Teacher config that also accepts ``inference.name == "vllm_omni"``."""

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        if self.inference.name != "vllm_omni":
            raise ValueError(f"the inference.name should be 'vllm_omni', got {self.inference.name}")

        engine_kwargs = self.inference.engine_kwargs
        omni_engine_kwargs = dict(engine_kwargs.get("vllm_omni", {}))
        max_logprobs = omni_engine_kwargs.get("max_logprobs")
        if max_logprobs is None:
            omni_engine_kwargs["max_logprobs"] = topk
            max_logprobs = topk
        if max_logprobs < topk:
            raise ValueError(
                f"vllm_omni max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                f"({topk}) to enable distillation loss computation."
            )
        engine_kwargs["vllm_omni"] = omni_engine_kwargs


@dataclass
class OmniDistillationLossConfig(DistillationLossConfig):
    """Loss config for the omni path.

    Adds ``kd_temperature`` and teaches the hidden-state (nitrobrew) loss
    modes: those are absent from verl's registry (which only knows top-k /
    estimator modes), so their ``loss_settings`` are synthesized here with
    ``use_hidden_states=True`` instead of being looked up.
    """

    kd_temperature: float = 1.0

    def __post_init__(self):
        if self.loss_mode in HIDDEN_STATE_LOSS_MODES:
            self._mutable_fields.add("loss_settings")
            if self.loss_settings is None:
                self.loss_settings = OmniDistillationLossSettings(names=self.loss_mode, use_hidden_states=True)
            else:
                self.loss_settings = OmniDistillationLossSettings(**self.loss_settings.__dict__)
                if not self.loss_settings.use_hidden_states:
                    raise ValueError("loss_settings.use_hidden_states must be True for hidden-state loss modes.")

            if self.policy_loss_mode != "vanilla":
                raise NotImplementedError(
                    f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                    f"but got {self.policy_loss_mode}."
                )
            if self.use_policy_gradient:
                raise ValueError(
                    "nitrobrew full-vocabulary KL is most effective as a supervised distillation loss "
                    "(use_policy_gradient=False), so the whole-vocab signal is backpropagated directly. "
                    "Set distillation.distillation_loss.use_policy_gradient=false."
                )
            # Hidden modes synthesize loss_settings above; the remaining parent validations are either
            # duplicated above (policy_loss_mode) or top-k/estimator-specific, so super().__post_init__()
            # is not called on this path.
            return

        # Non-hidden modes delegate to verl's registry + validations.
        super().__post_init__()


@dataclass
class OmniDistillationConfig(DistillationConfig):
    """Omni distillation config adding self-teacher (SDPO / OPSD) support.

    When ``self_teacher.enabled`` is False, validation delegates unchanged to
    verl's ``DistillationConfig``. When ``self_teacher.enabled`` is True, no
    dedicated teacher GPU pool (``nnodes`` / ``n_gpus_per_node``) or external
    ``teacher_models`` deployment is provisioned.
    """

    self_teacher: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    def __post_init__(self) -> None:
        if isinstance(self.self_teacher, Mapping):
            st_kwargs = {k: v for k, v in self.self_teacher.items() if k != "_target_"}
            object.__setattr__(self, "self_teacher", SelfDistillationConfig(**st_kwargs))
        if not self.enabled:
            if self.self_teacher.enabled:
                raise ValueError("distillation.enabled must be True when distillation.self_teacher.enabled is True.")
            return

        if self.self_teacher.enabled:
            if self.nnodes != 0 or self.n_gpus_per_node != 0:
                raise ValueError(
                    "Self-distillation reuses the policy pool and does not allocate a separate teacher pool; "
                    f"set distillation.nnodes=0 and distillation.n_gpus_per_node=0 "
                    f"(got nnodes={self.nnodes}, n_gpus_per_node={self.n_gpus_per_node})."
                )
            if self.distillation_loss.loss_mode in HIDDEN_STATE_LOSS_MODES:
                raise ValueError(
                    f"Self-distillation does not support hidden-state loss mode "
                    f"{self.distillation_loss.loss_mode!r}; use a logprob or top-k loss mode such as "
                    "'kl', 'low_var_kl', or 'forward_kl_topk'."
                )
            object.__setattr__(self, "teacher_models", {})
            return

        super().__post_init__()
