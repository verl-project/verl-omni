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
"""CPU tests for the generic distillation trainer control plane."""

import ast
from pathlib import Path

import pytest

from verl_omni.trainer.diffusion.distillation.contracts import PhaseResult, UpdatePhaseSpec, UpdateSchedule
from verl_omni.trainer.diffusion.distillation.control_plane import (
    DistillationTrainerControlPlane,
    FakeBatchProvider,
    FakeDistillationHooks,
    FakePhaseExecutor,
)
from verl_omni.trainer.diffusion.distillation.recipes import build_plan

CAPS = frozenset({"distribution_matching", "autoregressive", "adversarial"})


def make_plan(fake_repeats: int = 2, name: str = "dmd2", **config):
    return build_plan(name, {"fake_update_ratio": fake_repeats, "model_path": "/m", **config}, CAPS)


def make_control_plane(plan=None, executor=None, hooks=None, batches: int = 1000):
    plan = plan if plan is not None else make_plan()
    executor = executor if executor is not None else FakePhaseExecutor()
    hooks = hooks if hooks is not None else FakeDistillationHooks()
    return (
        DistillationTrainerControlPlane(plan, executor, FakeBatchProvider(num_batches=batches), hooks),
        executor,
        hooks,
    )


class TwoStepExecutor(FakePhaseExecutor):
    def execute_phase(self, request, batch):
        if request.kind == "student":
            return PhaseResult(optimizer_steps={"student": 2})
        return super().execute_phase(request, batch)


class WrongRoleExecutor(FakePhaseExecutor):
    def execute_phase(self, request, batch):
        if request.kind == "fake_score":
            return PhaseResult(optimizer_steps={"fake_score": 1, "discriminator": 1})
        return super().execute_phase(request, batch)


class NoStepExecutor(FakePhaseExecutor):
    def execute_phase(self, request, batch):
        return PhaseResult()


class TestPhaseExpansion:
    def test_normal_cycle_is_student_then_k_fake(self):
        control_plane, executor, _ = make_control_plane(make_plan(fake_repeats=3))
        control_plane.run_cycle()
        assert [request.kind for request in executor.executed] == [
            "student",
            "fake_score",
            "fake_score",
            "fake_score",
        ]

    def test_deterministic_ordering_across_cycles(self):
        control_plane, executor, _ = make_control_plane(make_plan(fake_repeats=2))
        control_plane.run(3)
        assert [request.kind for request in executor.executed] == ["student", "fake_score", "fake_score"] * 3

    def test_repeat_index_is_per_phase(self):
        control_plane, executor, _ = make_control_plane(make_plan(fake_repeats=3))
        control_plane.run_cycle()
        fake_repeats = [request.repeat_index for request in executor.executed if request.kind == "fake_score"]
        assert fake_repeats == [0, 1, 2]

    def test_warmup_transitions_to_normal_cycle(self):
        control_plane, executor, hooks = make_control_plane(make_plan(fake_repeats=2, fake_warmup_cycles=2))
        first = control_plane.run_cycle()
        second = control_plane.run_cycle()
        third = control_plane.run_cycle()
        assert first.is_warmup is True
        assert second.is_warmup is True
        assert third.is_warmup is False
        assert [request.kind for request in executor.executed] == [
            "fake_score",
            "fake_score",
            "fake_score",
            "fake_score",
            "student",
            "fake_score",
            "fake_score",
        ]
        assert [call["global_step"] for call in hooks.calls] == [1]


class TestCounters:
    def test_global_step_and_completed_cycles_advance(self):
        control_plane, _, _ = make_control_plane(make_plan(fake_repeats=2))
        control_plane.run(4)
        assert control_plane.counters.global_step == 4
        assert control_plane.counters.completed_cycles == 4

    def test_distribution_only_counts_only_bound_trainable_roles(self):
        control_plane, _, _ = make_control_plane(make_plan(fake_repeats=3))
        control_plane.run(2)
        assert control_plane.counters.optimizer_steps == {"student": 2, "fake_score": 6}

    def test_paper_profile_counts_discriminator_steps(self):
        control_plane, _, _ = make_control_plane(make_plan(fake_repeats=2, profile="paper"))
        control_plane.run_cycle()
        assert control_plane.counters.optimizer_steps == {"student": 1, "fake_score": 2, "discriminator": 2}

    def test_phase_request_carries_current_global_step_and_roles(self):
        control_plane, executor, _ = make_control_plane(make_plan(fake_repeats=1))
        control_plane.run(2)
        student_requests = [request for request in executor.executed if request.kind == "student"]
        assert [request.global_step for request in student_requests] == [0, 1]
        assert all(request.trainable_roles == ("student",) for request in student_requests)


class TestPhaseInvariants:
    def test_skipped_student_phase_rolls_back_and_marks_driver_failed(self):
        control_plane, _, _ = make_control_plane(make_plan(), executor=FakePhaseExecutor(skip_student=True))
        with pytest.raises(ValueError, match="no student optimizer step"):
            control_plane.run_cycle()
        assert control_plane.counters.global_step == 0
        assert control_plane.counters.optimizer_steps == {}
        with pytest.raises(RuntimeError, match="cannot be retried"):
            control_plane.run_cycle()

    def test_failed_fake_phase_rolls_back_and_cannot_retry(self):
        control_plane, _, _ = make_control_plane(make_plan(), executor=FakePhaseExecutor(fail_on="fake_score"))
        with pytest.raises(RuntimeError, match="failed on phase fake_score"):
            control_plane.run_cycle()
        assert control_plane.counters.global_step == 0
        assert control_plane.counters.optimizer_steps == {}
        with pytest.raises(RuntimeError, match="cannot be retried"):
            control_plane.run_cycle()

    def test_completed_role_must_report_exactly_one_step(self):
        control_plane, _, _ = make_control_plane(make_plan(), executor=TwoStepExecutor())
        with pytest.raises(ValueError, match="exactly one optimizer step"):
            control_plane.run_cycle()

    def test_unexpected_optimizer_role_is_rejected(self):
        control_plane, _, _ = make_control_plane(make_plan(), executor=WrongRoleExecutor())
        with pytest.raises(ValueError, match="must report optimizer steps for exactly"):
            control_plane.run_cycle()
        assert control_plane.counters.optimizer_steps == {}

    def test_zero_progress_warmup_cycle_raises(self):
        plan = make_plan(fake_repeats=1, fake_warmup_cycles=1)
        control_plane, _, _ = make_control_plane(plan, executor=NoStepExecutor())
        with pytest.raises(ValueError, match="must report optimizer steps"):
            control_plane.run_cycle()

    def test_two_student_phases_are_rejected_before_execution(self):
        student = UpdatePhaseSpec(kind="student", trainable_roles=("student",))
        with pytest.raises(ValueError, match="exactly one student"):
            UpdateSchedule(phases=(student, student))


class TestHookScheduling:
    def test_hook_observes_incremented_global_step(self):
        control_plane, _, hooks = make_control_plane(make_plan(fake_repeats=1))
        control_plane.run(3)
        assert [call["global_step"] for call in hooks.calls] == [1, 2, 3]

    def test_hook_receives_executor_and_metrics(self):
        control_plane, executor, hooks = make_control_plane(make_plan(fake_repeats=1))
        control_plane.run_cycle()
        assert hooks.calls[0]["executor"] is executor
        assert "student" in hooks.calls[0]["metrics"]

    def test_hook_not_called_during_warmup(self):
        control_plane, _, hooks = make_control_plane(make_plan(fake_warmup_cycles=1))
        control_plane.run_cycle()
        assert hooks.calls == []


class TestControlPlanePurity:
    def test_core_modules_have_no_direct_model_or_ray_imports(self):
        import verl_omni.trainer.diffusion.distillation.contracts as contracts_mod
        import verl_omni.trainer.diffusion.distillation.control_plane as control_plane_mod
        import verl_omni.trainer.diffusion.distillation.equations as equations_mod

        forbidden_roots = {"diffusers", "ray", "transformers", "vllm"}
        for module in (contracts_mod, control_plane_mod, equations_mod):
            tree = ast.parse(Path(module.__file__).read_text())
            imported_roots = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])
            assert imported_roots.isdisjoint(forbidden_roots)

    def test_reset_clears_healthy_driver_state(self):
        control_plane, _, _ = make_control_plane(make_plan())
        control_plane.run(2)
        control_plane.reset()
        assert control_plane.counters.global_step == 0
        assert control_plane.counters.completed_cycles == 0
        assert control_plane.counters.optimizer_steps == {}
        assert control_plane.metrics == {}

    def test_state_dict_round_trip_restores_completed_counters(self):
        control_plane, _, _ = make_control_plane(make_plan(fake_repeats=2))
        control_plane.run(3)
        state = control_plane.state_dict()

        restored, _, _ = make_control_plane(make_plan(fake_repeats=2))
        restored.load_state_dict(state)
        assert restored.counters.global_step == 3
        assert restored.counters.completed_cycles == 3
        assert restored.counters.optimizer_steps == {"student": 3, "fake_score": 6}

    def test_invalid_checkpoint_state_is_rejected(self):
        control_plane, _, _ = make_control_plane(make_plan())
        with pytest.raises(ValueError, match="exactly"):
            control_plane.load_state_dict({"global_step": 1})
        with pytest.raises(ValueError, match="non-negative integer"):
            control_plane.load_state_dict({"global_step": -1, "optimizer_steps": {}, "completed_cycles": 0})
        with pytest.raises(ValueError, match="unknown optimizer roles"):
            control_plane.load_state_dict({"global_step": 0, "optimizer_steps": {"unknown": 1}, "completed_cycles": 1})
        with pytest.raises(ValueError, match="must equal global_step"):
            control_plane.load_state_dict({"global_step": 2, "optimizer_steps": {"student": 1}, "completed_cycles": 2})

    def test_failed_driver_cannot_be_checkpointed_or_reset(self):
        control_plane, _, _ = make_control_plane(make_plan(), executor=FakePhaseExecutor(fail_on="student"))
        with pytest.raises(RuntimeError):
            control_plane.run_cycle()
        with pytest.raises(RuntimeError, match="Cannot checkpoint"):
            control_plane.state_dict()
        with pytest.raises(RuntimeError, match="cannot be reset"):
            control_plane.reset()
