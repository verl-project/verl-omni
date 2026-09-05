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
"""CPU tests for the generic multi-role distillation data plane."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProtoFuture
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.distillation.contracts import PhaseRequest
from verl_omni.trainer.diffusion.distillation.recipes import build_plan
from verl_omni.workers.diffusion_distillation_worker import (
    DiffusionDistillationWorkerGroup,
    DistillationPhaseComputation,
    DistillationRoleRuntime,
    resolve_profiler_configs,
)
from verl_omni.workers.engine.fsdp.distillation_impl import DistillationRoleGroupEngine

_CAPABILITIES = frozenset({"distribution_matching"})


def test_distillation_worker_instantiates_nested_profiler_tool_config():
    config = OmegaConf.create(
        {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": "torch",
            "enable": False,
            "all_ranks": False,
            "ranks": [],
            "save_path": "outputs/profile",
            "tool_config": {
                "torch": {
                    "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                    "contents": [],
                    "discrete": False,
                    "name": "torch",
                }
            },
        }
    )

    profiler_config, tool_config = resolve_profiler_configs(config)

    assert profiler_config.tool == "torch"
    assert tool_config.name == "torch"
    assert tool_config.contents == []


class ToyRoleEngine:
    def __init__(self, roles, initial=None):
        values = initial or {}
        self.parameters = {
            role: torch.nn.Parameter(torch.tensor(float(values.get(role, index + 1))))
            for index, role in enumerate(roles)
        }
        self.optimizers = {
            role: torch.optim.SGD([parameter], lr=0.1)
            for role, parameter in self.parameters.items()
            if role in {"student", "fake_score"}
        }
        self.scheduler = object()
        self.model_config = object()
        self.active_role = None

    @contextmanager
    def use_role(self, role):
        previous = self.active_role
        self.active_role = role
        try:
            yield self.parameters[role]
        finally:
            self.active_role = previous

    def optimizer_zero_grad(self, role=None):
        optimizers = self.optimizers.values() if role is None else (self.optimizers[role],)
        for optimizer in optimizers:
            optimizer.zero_grad()

    def backward_role(self, role, loss, retain_graph=False):
        assert self.active_role is None
        loss.backward(retain_graph=retain_graph)

    def optimizer_step(self, role=None):
        parameter = self.parameters[role]
        grad_norm = float(parameter.grad.detach().abs())
        self.optimizers[role].step()
        return True, grad_norm

    def update_role_ema(self, source_role, target_role, decay):
        with torch.no_grad():
            self.parameters[target_role].lerp_(self.parameters[source_role], 1.0 - decay)

    def update_module_ema_from(self, source, decay):
        source_parameter = next(iter(source.parameters.values()))
        target_parameter = next(iter(self.parameters.values()))
        with torch.no_grad():
            target_parameter.lerp_(source_parameter, 1.0 - decay)


class TestRoleRuntime:
    def test_role_group_engines_must_match_the_plan_exactly(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, _CAPABILITIES)
        engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema"))
        with pytest.raises(ValueError, match="missing=.*base"):
            DistillationRoleRuntime(plan, {}, ema_decay=0.9, ema_start_step=0)
        with pytest.raises(ValueError, match="extra=.*other"):
            DistillationRoleRuntime(plan, {"base": engine, "other": engine}, ema_decay=0.9, ema_start_step=0)

    def test_invalid_micro_batch_configuration_fails_closed(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, _CAPABILITIES)
        engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema"))
        with pytest.raises(ValueError, match="micro_batch_sizes"):
            DistillationRoleRuntime(
                plan,
                {"base": engine},
                ema_decay=0.9,
                ema_start_step=0,
                micro_batch_sizes={"student": 1, "fake_score": 0},
            )

    def test_student_and_fake_optimizers_are_isolated_and_ema_updates(self):
        plan = build_plan(
            "dmd2",
            {"model_path": "/m", "fake_update_ratio": 1},
            _CAPABILITIES,
        )
        engine = ToyRoleEngine(
            ("student", "teacher_score", "fake_score", "student_ema"),
            {"student": 1.0, "teacher_score": 7.0, "fake_score": 2.0, "student_ema": -5.0},
        )
        runtime = DistillationRoleRuntime(plan, {"base": engine}, ema_decay=0.5, ema_start_step=0)
        assert engine.parameters["student_ema"].item() == pytest.approx(1.0)

        fake_before = engine.parameters["fake_score"].detach().clone()
        student_request = PhaseRequest(
            kind="student",
            global_step=0,
            repeat_index=0,
            batch_policy="fresh",
            trainable_roles=("student",),
            update_ema=True,
        )
        student_loss = (engine.parameters["student"] - 0.0).square()
        steps, metrics = runtime.backward_and_step(
            student_request,
            DistillationPhaseComputation(losses={"student": student_loss}, metrics={}),
        )
        assert steps == {"student": 1}
        assert metrics["student/loss"] == pytest.approx(1.0)
        assert engine.parameters["fake_score"].detach().equal(fake_before)
        assert engine.parameters["student"].item() == pytest.approx(0.8)
        assert engine.parameters["student_ema"].item() == pytest.approx(0.9)

        student_before = engine.parameters["student"].detach().clone()
        fake_request = PhaseRequest(
            kind="fake_score",
            global_step=1,
            repeat_index=0,
            batch_policy="fresh",
            trainable_roles=("fake_score",),
        )
        fake_loss = (engine.parameters["fake_score"] - 0.0).square()
        steps, _ = runtime.backward_and_step(
            fake_request,
            DistillationPhaseComputation(losses={"fake_score": fake_loss}, metrics={}),
        )
        assert steps == {"fake_score": 1}
        assert engine.parameters["student"].detach().equal(student_before)
        assert engine.parameters["fake_score"].item() == pytest.approx(1.6)
        assert engine.parameters["teacher_score"].grad is None

    def test_independent_module_ema_is_initialized_and_updated(self):
        plan = build_plan(
            "dmd2",
            {"model_path": "/m", "role_storage": "colocated_independent"},
            _CAPABILITIES,
        )
        engines = {}
        for binding in plan.role_layout.bindings:
            engines[binding.group] = ToyRoleEngine(
                (binding.role,), {binding.role: 3.0 if binding.role == "student" else -2.0}
            )
        runtime = DistillationRoleRuntime(plan, engines, ema_decay=0.25, ema_start_step=0)
        student = runtime.engine_for_role("student").parameters["student"]
        ema = runtime.engine_for_role("student_ema").parameters["student_ema"]
        assert ema.item() == pytest.approx(student.item())
        with torch.no_grad():
            student.fill_(7.0)
        runtime.update_ema()
        assert ema.item() == pytest.approx(6.0)

    def test_gradient_accumulation_matches_full_batch_mean(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, _CAPABILITIES)
        accumulated_engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema"), {"student": 1.0})
        full_engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema"), {"student": 1.0})
        accumulated = DistillationRoleRuntime(plan, {"base": accumulated_engine}, ema_decay=0.9, ema_start_step=0)
        full = DistillationRoleRuntime(plan, {"base": full_engine}, ema_decay=0.9, ema_start_step=0)
        request = PhaseRequest("student", 0, 0, "fresh", ("student",), False)

        accumulated.zero_grad(("student",))
        for target in (torch.tensor(0.0), torch.tensor(2.0)):
            parameter = accumulated_engine.parameters["student"]
            accumulated.backward_micro_batch(
                request,
                DistillationPhaseComputation(losses={"student": (parameter - target).square()}, metrics={}),
                weight=0.5,
            )
        accumulated.step_phase(request)

        full.zero_grad(("student",))
        parameter = full_engine.parameters["student"]
        full_loss = torch.stack(((parameter - 0.0).square(), (parameter - 2.0).square())).mean()
        full.backward_and_step(
            request,
            DistillationPhaseComputation(losses={"student": full_loss}, metrics={}),
        )
        assert accumulated_engine.parameters["student"].item() == pytest.approx(
            full_engine.parameters["student"].item()
        )

    def test_export_uses_the_plan_selected_semantic_role(self):
        plan = build_plan(
            "dmd2",
            {"model_path": "/m", "export_role": "student_ema"},
            _CAPABILITIES,
        )
        engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema"))
        engine.iter_export_tensors = lambda role, base_sync_done: (role, base_sync_done)
        runtime = DistillationRoleRuntime(plan, {"base": engine}, ema_decay=0.9, ema_start_step=0)
        assert runtime.export_tensors(base_sync_done=True) == ("student_ema", True)

    def test_phase_losses_must_match_requested_roles(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, _CAPABILITIES)
        engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema"))
        runtime = DistillationRoleRuntime(plan, {"base": engine}, ema_decay=0.9, ema_start_step=0)
        request = PhaseRequest("student", 0, 0, "fresh", ("student",), True)
        with pytest.raises(ValueError, match="must match requested roles"):
            runtime.backward_and_step(
                request,
                DistillationPhaseComputation(
                    losses={"fake_score": engine.parameters["fake_score"].square()}, metrics={}
                ),
            )

    def test_pr2_rejects_multi_optimizer_adversarial_phase(self):
        plan = build_plan("dmd2", {"model_path": "/m", "profile": "paper"}, _CAPABILITIES | {"adversarial"})
        engine = ToyRoleEngine(("student", "teacher_score", "fake_score", "student_ema", "discriminator"))
        runtime = DistillationRoleRuntime(plan, {"base": engine}, ema_decay=0.9, ema_start_step=0)
        request = plan.update_schedule.next_cycle(SimpleNamespace(global_step=0, completed_cycles=0)).requests[-1]
        with pytest.raises(NotImplementedError, match="Multi-role optimizer phases"):
            runtime.backward_and_step(
                request,
                DistillationPhaseComputation(
                    losses={
                        "fake_score": engine.parameters["fake_score"].square(),
                        "discriminator": engine.parameters["discriminator"].square(),
                    },
                    metrics={},
                ),
            )


class ToyPeftModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.active_adapter = "student"
        self.adapters_enabled = True

    def set_adapter(self, name):
        self.active_adapter = name

    def enable_adapters(self):
        self.adapters_enabled = True

    def disable_adapters(self):
        self.adapters_enabled = False


class TestRoleEngineValidation:
    @staticmethod
    def uninitialized_engine():
        engine = object.__new__(DistillationRoleGroupEngine)
        engine.role_group = SimpleNamespace(name="base", storage="shared_base_adapters")
        engine.role_bindings = {
            "student": SimpleNamespace(role="student", group="base", trainable=True),
            "teacher_score": SimpleNamespace(role="teacher_score", group="base", trainable=False),
        }
        engine.optimizer_configs = {"student": object()}
        return engine

    def test_fsdp1_shared_base_requires_orig_params(self):
        engine = self.uninitialized_engine()
        with pytest.raises(ValueError, match="use_orig_params=true"):
            engine.validate_constructor_inputs(SimpleNamespace(strategy="fsdp", use_orig_params=False))

    def test_optimizer_configs_must_match_trainable_roles(self):
        engine = self.uninitialized_engine()
        engine.optimizer_configs = {"fake_score": object()}
        with pytest.raises(ValueError, match="must match trainable roles"):
            engine.validate_constructor_inputs(SimpleNamespace(strategy="fsdp2", use_orig_params=False))

    def test_gradient_leak_into_inactive_role_is_rejected(self):
        engine = self.uninitialized_engine()
        student = torch.nn.Parameter(torch.tensor(1.0))
        fake_score = torch.nn.Parameter(torch.tensor(2.0))
        fake_score.grad = torch.tensor(1.0)
        engine._role_parameters = {"student": (student,), "fake_score": (fake_score,)}
        with pytest.raises(RuntimeError, match="fake_score"):
            engine.assert_gradient_isolation({"student"})


class TestRoleContext:
    @staticmethod
    def make_engine():
        engine = object.__new__(DistillationRoleGroupEngine)
        engine.role_group = SimpleNamespace(name="base", storage="shared_base_adapters")
        engine.role_bindings = {
            "student": SimpleNamespace(adapter="student", trainable=True),
            "student_ema": SimpleNamespace(adapter="student_ema", trainable=False),
            "teacher_score": SimpleNamespace(adapter=None, trainable=False),
        }
        engine.module = ToyPeftModule()
        engine.optimizers = {}
        engine.lr_schedulers = {}
        engine.optimizer_configs = {}
        engine._active_role = "student"
        engine._primary_role = None
        return engine

    def test_frozen_role_is_eval_no_grad_and_context_restores_on_error(self):
        engine = self.make_engine()
        engine.module.train()
        with pytest.raises(RuntimeError, match="boom"):
            with engine.use_role("teacher_score") as module:
                assert not module.training
                assert not torch.is_grad_enabled()
                assert not module.adapters_enabled
                raise RuntimeError("boom")
        assert engine.module.training
        assert engine.module.adapters_enabled
        assert engine.module.active_adapter == "student"
        assert engine._active_role == "student"

    def test_non_student_export_is_rejected(self):
        engine = self.make_engine()
        with pytest.raises(ValueError, match="Only student or student_ema"):
            engine.iter_export_tensors("teacher_score", base_sync_done=False)

    def test_export_uses_the_semantic_roles_adapter(self):
        engine = self.make_engine()
        parameter = torch.tensor(1.0)
        engine.get_per_tensor_param = Mock(return_value=(iter((("weight", parameter),)), {"adapter": "student_ema"}))
        tensors, peft_config = engine.iter_export_tensors("student_ema", base_sync_done=False)
        assert list(tensors) == [("weight", parameter)]
        assert peft_config == {"adapter": "student_ema"}
        engine.get_per_tensor_param.assert_called_once_with(adapter_name="student_ema", base_sync_done=False)


class TestWorkerGroupFacade:
    def test_rank_failure_surfaces_before_lazy_collect_metadata(self, monkeypatch):
        import ray
        from verl.single_controller.base.decorator import MAGIC_ATTR
        from verl.single_controller.ray.base import func_generator

        from verl_omni.workers.diffusion_distillation_worker import DiffusionDistillationWorker

        registration = getattr(DiffusionDistillationWorker.execute_phase, MAGIC_ATTR)
        futures = [object(), object()]

        get_results = Mock(side_effect=ValueError("rank 0: invalid conditioning"))
        collect = Mock(side_effect=AssertionError("Lazy metadata RPCs queued behind a peer collective"))

        monkeypatch.setattr(ray, "get", get_results)
        execute = func_generator(
            object(),
            "execute_phase",
            lambda group, *args, **kwargs: (args, kwargs),
            collect,
            lambda name, *args, **kwargs: futures,
            registration["blocking"],
        )
        with pytest.raises(ValueError, match="rank 0: invalid conditioning"):
            execute(tu.get_tensordict({}, {}))
        get_results.assert_called_once_with(futures)
        collect.assert_not_called()

    def test_tensordict_result_is_converted_to_phase_result(self):
        worker_group = SimpleNamespace(
            execute_phase=Mock(
                return_value=tu.get_tensordict(
                    tensor_dict={},
                    non_tensor_dict={"metrics": {"loss": 1.25}, "optimizer_steps": {"student": 1}},
                )
            )
        )
        facade = DiffusionDistillationWorkerGroup(worker_group)
        request = PhaseRequest("student", 0, 0, "fresh", ("student",), True)
        result = facade.execute_phase(request, tu.get_tensordict({}, {}))
        assert result.metrics == {"loss": 1.25}
        assert result.optimizer_steps == {"student": 1}
        assert tu.get(worker_group.execute_phase.call_args.args[0], "phase_request").kind == "student"

    def test_future_result_is_resolved_before_conversion(self, monkeypatch):
        output = tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {"loss": 2.5}, "optimizer_steps": {"fake_score": 1}},
        )
        monkeypatch.setattr(DataProtoFuture, "get", Mock(return_value=output))
        future = DataProtoFuture(collect_fn=None, futures=[])
        facade = DiffusionDistillationWorkerGroup(SimpleNamespace(execute_phase=Mock(return_value=future)))
        request = PhaseRequest("fake_score", 0, 0, "fresh", ("fake_score",), False)
        result = facade.execute_phase(request, tu.get_tensordict({}, {}))

        assert result.metrics == {"loss": 2.5}
        assert result.optimizer_steps == {"fake_score": 1}
