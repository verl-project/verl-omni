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
"""CPU tests for distillation contracts, role layouts, and recipes."""

import dataclasses
import pickle

import pytest
import torch

from verl_omni.trainer.diffusion.distillation.contracts import (
    ExportSpec,
    LatentBundle,
    RoleBinding,
    RoleGroupSpec,
    RoleLayoutSpec,
    ScoreTransportSpec,
    TrainerCounters,
    UpdatePhaseSpec,
    UpdateSchedule,
    resolve_export_role,
    validate_role_layout,
)
from verl_omni.trainer.diffusion.distillation.recipes import (
    DistillationRegistry,
    build_plan,
    initialization_registry,
    objective_registry,
    recipe_registry,
    rollout_registry,
)

ALL_CAPS = frozenset({"distribution_matching", "autoregressive", "adversarial"})


class TestLatentBundle:
    def test_rejects_empty_bundle(self):
        with pytest.raises(ValueError, match="at least one modality"):
            LatentBundle({})

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            LatentBundle({"image": [1, 2, 3]})

    def test_single_returns_the_only_tensor(self):
        value = torch.randn(2, 3)
        torch.testing.assert_close(LatentBundle({"image": value}).single, value)

    def test_single_rejects_multimodal_bundle(self):
        bundle = LatentBundle({"video": torch.randn(1), "audio": torch.randn(1)})
        with pytest.raises(ValueError, match="one-modality"):
            _ = bundle.single

    def test_map_applies_to_every_modality(self):
        bundle = LatentBundle({"video": torch.ones(2), "audio": torch.ones(3)})
        doubled = bundle.map(lambda tensor: tensor * 2)
        torch.testing.assert_close(doubled.get("video"), torch.full((2,), 2.0))
        torch.testing.assert_close(doubled.get("audio"), torch.full((3,), 2.0))


class TestImmutability:
    @pytest.mark.parametrize(
        "instance,field_name",
        [
            (RoleGroupSpec(name="base"), "name"),
            (RoleBinding(role="student", group="base"), "role"),
            (ExportSpec(), "role"),
            (UpdatePhaseSpec(kind="student", trainable_roles=("student",)), "kind"),
            (
                UpdateSchedule(phases=(UpdatePhaseSpec(kind="student", trainable_roles=("student",)),)),
                "phases",
            ),
        ],
    )
    def test_plan_pieces_are_frozen(self, instance, field_name):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)

    def test_plan_is_deeply_immutable(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, ALL_CAPS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.name = "other"
        with pytest.raises(TypeError):
            plan.objective["name"] = "other"

    def test_frozen_plan_is_hashable_and_pickleable(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, ALL_CAPS)
        assert isinstance(hash(plan), int)
        restored = pickle.loads(pickle.dumps(plan))
        assert restored.objective == plan.objective
        assert restored.rollout == plan.rollout


class TestRoleLayoutValidation:
    def role_layout(self, **kwargs):
        defaults = dict(
            groups=(RoleGroupSpec(name="base"),),
            bindings=(
                RoleBinding(role="student", group="base", adapter="student", trainable=True, optimizer_key="student"),
                RoleBinding(role="student_ema", group="base", adapter="student_ema"),
            ),
        )
        defaults.update(kwargs)
        return RoleLayoutSpec(**defaults)

    def test_valid_layout_passes(self):
        validate_role_layout(self.role_layout())

    def test_binding_to_unknown_group_raises(self):
        layout = self.role_layout(
            bindings=(
                RoleBinding(
                    role="student", group="missing", adapter="student", trainable=True, optimizer_key="student"
                ),
            )
        )
        with pytest.raises(ValueError, match="unknown group"):
            validate_role_layout(layout)

    def test_trainable_role_without_optimizer_key_raises(self):
        layout = self.role_layout(
            bindings=(RoleBinding(role="student", group="base", adapter="student", trainable=True),)
        )
        with pytest.raises(ValueError, match="optimizer_key"):
            validate_role_layout(layout)

    def test_frozen_role_with_optimizer_key_raises(self):
        layout = self.role_layout(bindings=(RoleBinding(role="teacher_score", group="base", optimizer_key="teacher"),))
        with pytest.raises(ValueError, match="Frozen role"):
            validate_role_layout(layout)

    def test_shared_base_trainable_role_requires_adapter(self):
        layout = self.role_layout(
            bindings=(RoleBinding(role="student", group="base", trainable=True, optimizer_key="student"),)
        )
        with pytest.raises(ValueError, match="must name an adapter"):
            validate_role_layout(layout)

    def test_duplicate_adapter_in_shared_base_raises(self):
        layout = self.role_layout(
            bindings=(
                RoleBinding(role="student", group="base", adapter="dup", trainable=True, optimizer_key="student"),
                RoleBinding(role="fake_score", group="base", adapter="dup", trainable=True, optimizer_key="fake"),
            )
        )
        with pytest.raises(ValueError, match="duplicate adapter names"):
            validate_role_layout(layout)

    def test_duplicate_group_name_raises(self):
        layout = self.role_layout(groups=(RoleGroupSpec(name="base"), RoleGroupSpec(name="base")))
        with pytest.raises(ValueError, match="Duplicate role-group"):
            validate_role_layout(layout)

    def test_duplicate_optimizer_key_raises(self):
        layout = self.role_layout(
            bindings=(
                RoleBinding(role="student", group="base", adapter="student", trainable=True, optimizer_key="same"),
                RoleBinding(role="fake_score", group="base", adapter="fake", trainable=True, optimizer_key="same"),
            )
        )
        with pytest.raises(ValueError, match="Duplicate optimizer_key"):
            validate_role_layout(layout)

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"provider": "bogus"}, "Invalid score provider"),
            ({"tensor_backend": "bogus"}, "Invalid score tensor backend"),
            ({"provider": "colocated", "tensor_backend": "mooncake"}, "colocated"),
            ({"provider": "ray", "tensor_backend": "local"}, "Ray score provider"),
        ],
    )
    def test_invalid_transport_raises(self, kwargs, error):
        with pytest.raises(ValueError, match=error):
            ScoreTransportSpec(**kwargs)


class TestExportRole:
    @pytest.mark.parametrize("role", ["student", "student_ema"])
    def test_exportable_roles(self, role):
        assert resolve_export_role(ExportSpec(role=role)) == role

    @pytest.mark.parametrize("role", ["teacher_score", "fake_score", "discriminator"])
    def test_non_exportable_roles_raise(self, role):
        with pytest.raises(ValueError, match="Export role must be"):
            ExportSpec(role=role)


class TestRegistry:
    def test_all_recipe_and_strategy_names_are_registered(self):
        assert set(recipe_registry.names) == {"dmd", "dmd2", "causvid", "self_forcing"}
        assert set(objective_registry.names) == {"dmd", "dmd2", "ode_regression"}
        assert set(initialization_registry.names) == {"base", "ode_regression"}
        assert set(rollout_registry.names) == {
            "backward_simulated",
            "consistency_renoise",
            "ode_euler",
            "one_step",
            "self_forced",
            "teacher_forced_causal",
        }

    def test_duplicate_registration_raises(self):
        registry = DistillationRegistry()

        assert registry.register("thing")(int) is int
        assert registry.get("thing") is int
        with pytest.raises(ValueError, match="Duplicate registration"):
            registry.register("thing")(str)

    def test_unknown_name_raises_with_registered_list(self):
        with pytest.raises(KeyError, match="Registered"):
            DistillationRegistry().get("nope")


class TestRecipePlans:
    @pytest.mark.parametrize("name", ["dmd", "dmd2", "causvid", "self_forcing"])
    def test_every_recipe_builds_a_validated_plan(self, name):
        config = {"model_path": "/m"}
        if name == "dmd":
            config["profile"] = "paper"
        plan = build_plan(name, config, ALL_CAPS)
        assert plan.name == name
        validate_role_layout(plan.role_layout)

    def test_missing_capability_is_fail_closed(self):
        with pytest.raises(ValueError, match="requires capabilities"):
            build_plan("self_forcing", {"model_path": "/m"}, {"distribution_matching"})

    def test_missing_model_reference_is_fail_closed(self):
        with pytest.raises(ValueError, match="model_ref"):
            build_plan("dmd2", {}, ALL_CAPS)

    def test_export_is_a_top_level_plan_contract(self):
        plan = build_plan("dmd2", {"model_path": "/m", "export_role": "student"}, ALL_CAPS)
        assert plan.export.role == "student"
        assert not hasattr(plan.role_layout, "export")

    def test_missing_required_plan_role_is_rejected(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, ALL_CAPS)
        layout = dataclasses.replace(
            plan.role_layout,
            bindings=tuple(binding for binding in plan.role_layout.bindings if binding.role != "student_ema"),
        )
        with pytest.raises(ValueError, match="missing required role"):
            dataclasses.replace(plan, role_layout=layout)

    def test_dmd2_paper_profile_adds_discriminator_to_layout_and_phase(self):
        plan = build_plan("dmd2", {"profile": "paper", "model_path": "/m"}, ALL_CAPS)
        roles = {binding.role for binding in plan.role_layout.bindings}
        fake_phase = next(phase for phase in plan.update_schedule.phases if phase.kind == "fake_score")
        assert "discriminator" in roles
        assert fake_phase.trainable_roles == ("fake_score", "discriminator")
        assert plan.objective["adversarial"] is True

    def test_dmd2_distribution_only_has_no_discriminator(self):
        plan = build_plan("dmd2", {"profile": "distribution_only", "model_path": "/m"}, ALL_CAPS)
        roles = {binding.role for binding in plan.role_layout.bindings}
        fake_phase = next(phase for phase in plan.update_schedule.phases if phase.kind == "fake_score")
        assert "discriminator" not in roles
        assert fake_phase.trainable_roles == ("fake_score",)

    def test_causal_recipes_use_separate_causal_and_bidirectional_groups(self):
        for name in ("causvid", "self_forcing"):
            plan = build_plan(name, {"model_path": "/m"}, ALL_CAPS)
            groups = {group.name for group in plan.role_layout.groups}
            role_groups = {binding.role: binding.group for binding in plan.role_layout.bindings}
            assert groups == {"causal_base", "bidirectional_base"}
            assert role_groups["student"] == role_groups["student_ema"] == "causal_base"
            assert role_groups["teacher_score"] == role_groups["fake_score"] == "bidirectional_base"

    def test_colocated_independent_materializes_one_group_per_role(self):
        plan = build_plan(
            "dmd2",
            {"model_path": "/m", "role_storage": "colocated_independent"},
            ALL_CAPS,
        )
        assert {group.storage for group in plan.role_layout.groups} == {"independent_module"}
        assert len(plan.role_layout.groups) == len(plan.role_layout.bindings) == 4
        assert all(binding.group == f"{binding.role}_model" for binding in plan.role_layout.bindings)

    @pytest.mark.parametrize(
        "name,config,error",
        [
            ("dmd2", {"profile": "typo", "model_path": "/m"}, "profile"),
            ("dmd2", {"rollout_strategy": "typo", "model_path": "/m"}, "rollout_strategy"),
            ("dmd2", {"data_mode": "regression_pairs", "model_path": "/m"}, "data_mode"),
            ("dmd2", {"fake_update_ratio": 0, "model_path": "/m"}, "greater than zero"),
            ("dmd2", {"fake_update_ratio": -2, "model_path": "/m"}, "greater than zero"),
            ("dmd2", {"fake_update_ratio": 1.5, "model_path": "/m"}, "integer"),
            ("dmd2", {"fake_update_ratio": True, "model_path": "/m"}, "integer"),
            ("dmd2", {"fake_warmup_cycles": 1.5, "model_path": "/m"}, "integer"),
            ("dmd2", {"role_storage": "remote", "model_path": "/m"}, "role_storage"),
        ],
    )
    def test_invalid_recipe_values_fail_closed(self, name, config, error):
        with pytest.raises(ValueError, match=error):
            build_plan(name, config, ALL_CAPS)


class TestUpdateSchedule:
    def student_phase(self):
        return UpdatePhaseSpec(kind="student", trainable_roles=("student",))

    def fake_phase(self, repeats=1):
        return UpdatePhaseSpec(kind="fake_score", repeats=repeats, trainable_roles=("fake_score",))

    def test_normal_cycle_is_student_then_fake(self):
        schedule = UpdateSchedule(phases=(self.student_phase(), self.fake_phase(repeats=2)))
        cycle = schedule.next_cycle(TrainerCounters())
        assert cycle.requires_student_update is True
        assert cycle.is_warmup is False
        assert [request.kind for request in cycle.requests] == ["student", "fake_score", "fake_score"]

    def test_warmup_transitions_to_normal_cycles(self):
        schedule = UpdateSchedule(
            phases=(self.student_phase(), self.fake_phase()),
            warmup_phases=(self.fake_phase(repeats=2),),
            warmup_cycles=2,
        )
        counters = TrainerCounters(completed_cycles=0)
        assert schedule.next_cycle(counters).is_warmup is True
        counters.completed_cycles = 1
        assert schedule.next_cycle(counters).is_warmup is True
        counters.completed_cycles = 2
        assert schedule.next_cycle(counters).is_warmup is False

    def test_empty_schedule_raises(self):
        with pytest.raises(ValueError, match="normal-cycle phases"):
            UpdateSchedule(phases=())

    def test_two_student_phases_raise(self):
        with pytest.raises(ValueError, match="exactly one student"):
            UpdateSchedule(phases=(self.student_phase(), self.student_phase()))

    def test_repeated_student_phase_raises_before_any_execution(self):
        with pytest.raises(ValueError, match="repeats=1"):
            UpdateSchedule(
                phases=(UpdatePhaseSpec(kind="student", repeats=2, trainable_roles=("student",)), self.fake_phase())
            )

    @pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
    def test_nonpositive_repeats_raise(self, repeats):
        with pytest.raises(ValueError, match="greater than zero"):
            self.fake_phase(repeats)

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"kind": "bogus", "trainable_roles": ("fake_score",)}, "Invalid phase kind"),
            ({"kind": "fake_score", "batch_policy": "bogus", "trainable_roles": ("fake_score",)}, "batch_policy"),
            ({"kind": "fake_score"}, "at least one trainable role"),
            ({"kind": "fake_score", "trainable_roles": ("fake_score", "fake_score")}, "duplicate"),
            ({"kind": "student", "trainable_roles": ("student", "fake_score")}, "exactly the 'student'"),
            ({"kind": "fake_score", "trainable_roles": ("student",)}, "must not train the student"),
            ({"kind": "fake_score", "trainable_roles": ("fake_score",), "update_ema": True}, "EMA"),
        ],
    )
    def test_invalid_phase_contracts_raise(self, kwargs, error):
        with pytest.raises(ValueError, match=error):
            UpdatePhaseSpec(**kwargs)

    def test_warmup_requires_fake_only_phases(self):
        with pytest.raises(ValueError, match="must not contain a student"):
            UpdateSchedule(
                phases=(self.student_phase(), self.fake_phase()),
                warmup_phases=(self.student_phase(),),
                warmup_cycles=1,
            )

    def test_warmup_cannot_reuse_a_missing_student_batch(self):
        with pytest.raises(ValueError, match="cannot reuse a student batch"):
            UpdateSchedule(
                phases=(self.student_phase(), self.fake_phase()),
                warmup_phases=(
                    UpdatePhaseSpec(
                        kind="fake_score",
                        batch_policy="reuse_student",
                        trainable_roles=("fake_score",),
                    ),
                ),
                warmup_cycles=1,
            )

    def test_warmup_cycles_require_warmup_phases(self):
        with pytest.raises(ValueError, match="requires at least one warmup phase"):
            UpdateSchedule(phases=(self.student_phase(), self.fake_phase()), warmup_cycles=1)

    def test_warmup_phases_require_positive_cycle_count(self):
        with pytest.raises(ValueError, match="require warmup_cycles"):
            UpdateSchedule(
                phases=(self.student_phase(), self.fake_phase()),
                warmup_phases=(self.fake_phase(),),
            )


class TestTrainerCounters:
    def test_counters_start_at_zero(self):
        counters = TrainerCounters()
        assert counters.global_step == 0
        assert counters.optimizer_steps == {}
        assert counters.completed_cycles == 0

    def test_record_step_accumulates_per_role(self):
        counters = TrainerCounters()
        counters.record_step("student")
        counters.record_step("student")
        counters.record_step("fake_score")
        assert counters.optimizer_steps == {"student": 2, "fake_score": 1}
