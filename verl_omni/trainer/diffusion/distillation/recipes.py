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
"""Named distillation recipes and their composed strategy registries."""

from __future__ import annotations

import abc
from functools import partial

from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    ExportSpec,
    RoleBinding,
    RoleGroupSpec,
    RoleLayoutSpec,
    ScoreTransportSpec,
    UpdatePhaseSpec,
    UpdateSchedule,
    validate_distillation_plan,
)

__all__ = [
    "recipe_registry",
    "DMDRecipe",
    "DMD2Recipe",
    "CausVidRecipe",
    "SelfForcingRecipe",
    "build_plan",
    "build_plan_from_config",
    "DistillationRecipeBase",
    "DistillationRecipeRegistry",
    "ObjectiveBase",
    "ObjectiveRegistry",
    "RolloutStrategyBase",
    "RolloutStrategyRegistry",
    "InitializationBase",
    "InitializationRegistry",
    "DistillationRegistry",
    "objective_registry",
    "rollout_registry",
    "initialization_registry",
    "DMDObjective",
    "DMD2Objective",
    "ODERegressionObjective",
    "OneStepRollout",
    "EulerRollout",
    "ConsistencyRenoiseRollout",
    "TeacherForcedCausalRollout",
    "SelfForcedRollout",
    "BackwardSimulatedRollout",
    "BaseInitialization",
    "ODERegressionInitialization",
]


class DistillationRecipeBase(abc.ABC):
    """Base class for a named distillation recipe."""

    @classmethod
    @abc.abstractmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        """Return a validated immutable :class:`DistillationPlan`."""


class DistillationRegistry:
    """Minimal name-to-class registry with duplicate rejection."""

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str, subclass: type | None = None):
        """Register a class directly or return its registration decorator."""
        if not name:
            raise ValueError("Registry names must not be empty.")
        if subclass is None:
            return partial(self.register, name)
        if name in self._registry:
            raise ValueError(f"Duplicate registration for {name!r}.")
        self._registry[name] = subclass
        return subclass

    def get(self, name: str) -> type:
        """Resolve a registered class by name."""
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(f"No {self.kind} registered for {name!r}. Registered: {sorted(self._registry)}") from None

    @property
    def names(self) -> tuple[str, ...]:
        """Return the registered names in stable order."""
        return tuple(sorted(self._registry))

    @property
    def kind(self) -> str:
        """Identify this registry in resolution errors."""
        return self.__class__.__name__


class DistillationRecipeRegistry(DistillationRegistry):
    """Registry of named recipes that build a :class:`DistillationPlan`."""

    @property
    def kind(self) -> str:
        return "distillation recipe"

    def build(self, name: str, config, capabilities) -> DistillationPlan:
        """Build a plan from the recipe registered under ``name``."""
        return self.get(name).build_plan(config, capabilities)


class ObjectiveRegistry(DistillationRegistry):
    """Registry of composed objective strategies."""

    @property
    def kind(self) -> str:
        return "objective"


class RolloutStrategyRegistry(DistillationRegistry):
    """Registry of student rollout strategies."""

    @property
    def kind(self) -> str:
        return "rollout strategy"


class InitializationRegistry(DistillationRegistry):
    """Registry of initialization strategies."""

    @property
    def kind(self) -> str:
        return "initialization"


objective_registry = ObjectiveRegistry()
rollout_registry = RolloutStrategyRegistry()
initialization_registry = InitializationRegistry()


class ObjectiveBase:
    """Name marker for a composed objective implemented by a phase executor."""

    name: str = ""


@objective_registry.register("dmd")
class DMDObjective(ObjectiveBase):
    """DMD detached normalized score-gradient objective."""

    name = "dmd"


@objective_registry.register("dmd2")
class DMD2Objective(ObjectiveBase):
    """DMD2 two-time-scale distribution-matching objective."""

    name = "dmd2"


@objective_registry.register("ode_regression")
class ODERegressionObjective(ObjectiveBase):
    """ODE regression against precomputed trajectory targets."""

    name = "ode_regression"


class RolloutStrategyBase:
    """Name marker for a student rollout implemented by a phase executor."""

    name: str = ""


@rollout_registry.register("one_step")
class OneStepRollout(RolloutStrategyBase):
    """Single-step student rollout."""

    name = "one_step"


@rollout_registry.register("ode_euler")
class EulerRollout(RolloutStrategyBase):
    """Deterministic Euler backward simulation."""

    name = "ode_euler"


@rollout_registry.register("consistency_renoise")
class ConsistencyRenoiseRollout(RolloutStrategyBase):
    """Consistency re-noising backward simulation."""

    name = "consistency_renoise"


@rollout_registry.register("teacher_forced_causal")
class TeacherForcedCausalRollout(RolloutStrategyBase):
    """Teacher-forced causal rollout used by CausVid."""

    name = "teacher_forced_causal"


@rollout_registry.register("self_forced")
class SelfForcedRollout(RolloutStrategyBase):
    """Self-forced autoregressive rollout."""

    name = "self_forced"


@rollout_registry.register("backward_simulated")
class BackwardSimulatedRollout(RolloutStrategyBase):
    """Inference-time backward-simulated multi-step student input."""

    name = "backward_simulated"


class InitializationBase:
    """Name marker for a recipe initialization strategy."""

    name: str = ""


@initialization_registry.register("base")
class BaseInitialization(InitializationBase):
    """Initialize trainable roles directly from their configured base."""

    name = "base"


@initialization_registry.register("ode_regression")
class ODERegressionInitialization(InitializationBase):
    """Initialize a causal student from a provenance-tracked ODE stage."""

    name = "ode_regression"


recipe_registry = DistillationRecipeRegistry()


def shared_base_layout(model_ref: str, with_discriminator: bool = False) -> RoleLayoutSpec:
    """Build the image-first shared-base named-adapter role layout."""
    group_name = "base"
    bindings = [
        RoleBinding(role="student", group=group_name, adapter="student", trainable=True, optimizer_key="student"),
        RoleBinding(role="teacher_score", group=group_name),
        RoleBinding(
            role="fake_score", group=group_name, adapter="fake_score", trainable=True, optimizer_key="fake_score"
        ),
        RoleBinding(role="student_ema", group=group_name, adapter="student_ema"),
    ]
    if with_discriminator:
        bindings.append(
            RoleBinding(
                role="discriminator",
                group=group_name,
                adapter="discriminator",
                trainable=True,
                optimizer_key="discriminator",
            )
        )
    return RoleLayoutSpec(
        groups=(RoleGroupSpec(name=group_name, model_ref=model_ref),),
        bindings=tuple(bindings),
        score_transport=ScoreTransportSpec(),
    )


def causal_bidirectional_layout(causal_model_ref: str, bidirectional_model_ref: str) -> RoleLayoutSpec:
    """Keep causal student/EMA separate from bidirectional score models."""
    return RoleLayoutSpec(
        groups=(
            RoleGroupSpec(name="causal_base", model_ref=causal_model_ref),
            RoleGroupSpec(name="bidirectional_base", model_ref=bidirectional_model_ref),
        ),
        bindings=(
            RoleBinding(
                role="student",
                group="causal_base",
                adapter="student",
                trainable=True,
                optimizer_key="student",
            ),
            RoleBinding(role="student_ema", group="causal_base", adapter="student_ema"),
            RoleBinding(role="teacher_score", group="bidirectional_base"),
            RoleBinding(
                role="fake_score",
                group="bidirectional_base",
                adapter="fake_score",
                trainable=True,
                optimizer_key="fake_score",
            ),
        ),
        score_transport=ScoreTransportSpec(),
    )


def build_update_schedule(
    fake_repeats: int, fake_warmup_cycles: int = 0, with_discriminator: bool = False
) -> UpdateSchedule:
    """Build one student phase followed by ``fake_repeats`` fake phases."""
    fake_roles = ("fake_score", "discriminator") if with_discriminator else ("fake_score",)
    fake_phase = UpdatePhaseSpec(kind="fake_score", repeats=fake_repeats, trainable_roles=fake_roles)
    warmup_phases = (fake_phase,) if fake_warmup_cycles else ()
    return UpdateSchedule(
        phases=(
            UpdatePhaseSpec(kind="student", trainable_roles=("student",), update_ema=True),
            fake_phase,
        ),
        warmup_phases=warmup_phases,
        warmup_cycles=fake_warmup_cycles,
    )


def get_config_value(config, key: str, default=None):
    """Read ``key`` from a mapping-like or attribute-style config."""
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def get_config_or_default(config, key: str, default):
    """Use the default when a config field is absent or explicitly null."""
    value = get_config_value(config, key, default)
    return default if value is None else value


def require_choice(field: str, value: str, valid_values: set[str]) -> str:
    """Validate a named configuration choice."""
    if value not in valid_values:
        raise ValueError(f"Invalid {field} {value!r}; expected one of {sorted(valid_values)}.")
    return value


def common_plan_kwargs(
    config,
    *,
    default_fake_repeats: int = 5,
    with_discriminator: bool = False,
) -> dict:
    """Build architecture-neutral scheduling and export settings."""
    fake_repeats = get_config_or_default(config, "fake_update_ratio", default_fake_repeats)
    fake_warmup_cycles = get_config_or_default(config, "fake_warmup_cycles", 0)
    export_role = require_choice(
        "export_role", get_config_or_default(config, "export_role", "student_ema"), {"student", "student_ema"}
    )
    return {
        "version": 1,
        "update_schedule": build_update_schedule(fake_repeats, fake_warmup_cycles, with_discriminator),
        "export": ExportSpec(role=export_role, checkpoint_engine_backend="naive"),
    }


@recipe_registry.register("dmd")
class DMDRecipe(DistillationRecipeBase):
    """Original DMD with paired trajectory regression."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        profile = require_choice("DMD profile", get_config_or_default(config, "profile", "paper"), {"paper"})
        data_mode = require_choice(
            "DMD data_mode", get_config_or_default(config, "data_mode", "regression_pairs"), {"regression_pairs"}
        )
        rollout = require_choice(
            "DMD rollout_strategy", get_config_or_default(config, "rollout_strategy", "one_step"), {"one_step"}
        )
        model_ref = get_config_value(config, "model_path", "") or ""
        return DistillationPlan(
            name="dmd",
            role_layout=shared_base_layout(model_ref),
            data_requirements={"mode": data_mode},
            objective={"name": "dmd", "profile": profile},
            rollout={"strategy": rollout},
            initialization={"stage": "base"},
            required_capabilities=frozenset({"distribution_matching"}),
            **common_plan_kwargs(config, default_fake_repeats=1),
        )


@recipe_registry.register("dmd2")
class DMD2Recipe(DistillationRecipeBase):
    """DMD2 distribution matching with optional diffusion-GAN profile."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        profile = require_choice(
            "DMD2 profile",
            get_config_or_default(config, "profile", "distribution_only"),
            {"distribution_only", "paper"},
        )
        adversarial = profile == "paper"
        expected_data_mode = "prompt_and_real_latent" if adversarial else "prompts"
        data_mode = require_choice(
            "DMD2 data_mode", get_config_or_default(config, "data_mode", expected_data_mode), {expected_data_mode}
        )
        rollout = require_choice(
            "DMD2 rollout_strategy",
            get_config_or_default(config, "rollout_strategy", "ode_euler"),
            {"ode_euler", "consistency_renoise", "backward_simulated"},
        )
        model_ref = get_config_value(config, "model_path", "") or ""
        return DistillationPlan(
            name="dmd2",
            role_layout=shared_base_layout(model_ref, with_discriminator=adversarial),
            data_requirements={"mode": data_mode},
            objective={"name": "dmd2", "profile": profile, "adversarial": adversarial},
            rollout={"strategy": rollout},
            initialization={"stage": "base"},
            required_capabilities=frozenset({"distribution_matching"} | ({"adversarial"} if adversarial else set())),
            **common_plan_kwargs(config, with_discriminator=adversarial),
        )


@recipe_registry.register("causvid")
class CausVidRecipe(DistillationRecipeBase):
    """ODE-initialized causal student versus bidirectional score models."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        require_choice(
            "CausVid profile", get_config_or_default(config, "profile", "distribution_only"), {"distribution_only"}
        )
        data_mode = require_choice(
            "CausVid data_mode",
            get_config_or_default(config, "data_mode", "prompt_and_real_latent"),
            {"prompt_and_real_latent"},
        )
        rollout = require_choice(
            "CausVid rollout_strategy",
            get_config_or_default(config, "rollout_strategy", "teacher_forced_causal"),
            {"teacher_forced_causal"},
        )
        model_ref = get_config_value(config, "model_path", "") or ""
        causal_ref = get_config_value(config, "causal_model_path", model_ref) or model_ref
        bidirectional_ref = get_config_value(config, "bidirectional_model_path", model_ref) or model_ref
        return DistillationPlan(
            name="causvid",
            role_layout=causal_bidirectional_layout(causal_ref, bidirectional_ref),
            data_requirements={"mode": data_mode},
            objective={"name": "dmd", "profile": "distribution_only"},
            rollout={"strategy": rollout},
            initialization={"stage": "ode_regression", "requires_provenance": True},
            required_capabilities=frozenset({"distribution_matching", "autoregressive"}),
            **common_plan_kwargs(config),
        )


@recipe_registry.register("self_forcing")
class SelfForcingRecipe(DistillationRecipeBase):
    """DMD objective with self-forced autoregressive rollout."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        require_choice(
            "Self-Forcing profile", get_config_or_default(config, "profile", "distribution_only"), {"distribution_only"}
        )
        data_mode = require_choice(
            "Self-Forcing data_mode", get_config_or_default(config, "data_mode", "prompts"), {"prompts"}
        )
        rollout = require_choice(
            "Self-Forcing rollout_strategy",
            get_config_or_default(config, "rollout_strategy", "self_forced"),
            {"self_forced"},
        )
        model_ref = get_config_value(config, "model_path", "") or ""
        causal_ref = get_config_value(config, "causal_model_path", model_ref) or model_ref
        bidirectional_ref = get_config_value(config, "bidirectional_model_path", model_ref) or model_ref
        return DistillationPlan(
            name="self_forcing",
            role_layout=causal_bidirectional_layout(causal_ref, bidirectional_ref),
            data_requirements={"mode": data_mode},
            objective={"name": "dmd", "profile": "distribution_only"},
            rollout={"strategy": rollout},
            initialization={"stage": "ode_regression", "requires_provenance": True},
            required_capabilities=frozenset({"distribution_matching", "autoregressive"}),
            **common_plan_kwargs(config),
        )


def validate_registered_strategies(plan: DistillationPlan) -> None:
    """Resolve every strategy name before execution."""
    objective_registry.get(plan.objective["name"])
    rollout_registry.get(plan.rollout["strategy"])
    initialization_registry.get(plan.initialization["stage"])


def build_plan(name: str, config=None, capabilities=frozenset()) -> DistillationPlan:
    """Build and fail-closed validate a named recipe plan."""
    plan = recipe_registry.build(name, config, capabilities)
    validate_registered_strategies(plan)
    validate_distillation_plan(plan)
    missing = plan.required_capabilities - frozenset(capabilities)
    if missing:
        raise ValueError(
            f"Recipe {name!r} requires capabilities {sorted(missing)} that the architecture adapter "
            f"does not provide. Declared: {sorted(capabilities)}."
        )
    return plan


def build_plan_from_config(config, capabilities) -> DistillationPlan:
    """Build a plan from the composed trainer config and adapter capabilities."""
    distillation = get_config_value(config, "distillation")
    distribution_matching = get_config_value(distillation, "distribution_matching")
    if distribution_matching is None:
        raise ValueError("config.distillation.distribution_matching is required for the distillation trainer.")

    actor_rollout_ref = get_config_value(config, "actor_rollout_ref")
    model = get_config_value(actor_rollout_ref, "model")
    model_path = get_config_value(model, "path", "") or ""
    recipe_config = {
        "model_path": model_path,
        "fake_warmup_cycles": get_config_value(distribution_matching, "fake_warmup_cycles", 0),
        "export_role": get_config_value(distribution_matching, "export_role", "student_ema"),
    }
    for optional_key in ("profile", "fake_update_ratio", "rollout_strategy", "data_mode"):
        value = get_config_value(distribution_matching, optional_key)
        if value is not None:
            recipe_config[optional_key] = value
    return build_plan(get_config_value(distribution_matching, "recipe", "dmd2"), recipe_config, capabilities)
