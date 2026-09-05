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
"""Small GPU tests for role switching on FSDP1 and FSDP2 LoRA modules."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.tensor import DTensor

from verl_omni.trainer.diffusion.distillation.contracts import RoleBinding, RoleGroupSpec
from verl_omni.workers.engine.fsdp.distillation_impl import DistillationRoleGroupEngine


class TinyCheckpointManager:
    def __init__(self, module, optimizer, scheduler):
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_checkpoint(self, local_path, **kwargs):
        os.makedirs(local_path, exist_ok=True)
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            os.path.join(local_path, "primary.pt"),
        )

    def load_checkpoint(self, local_path, **kwargs):
        state = torch.load(os.path.join(local_path, "primary.pt"), weights_only=False)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs)


def wrap_model(strategy):
    model = get_peft_model(
        TinyModel(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
        adapter_name="student",
    ).cuda()
    adapter_config = model.peft_config["student"]
    model.add_adapter("fake_score", adapter_config)
    model.add_adapter("student_ema", adapter_config)
    model.set_adapter("student")
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device())
    from torch.distributed.fsdp import fully_shard

    fully_shard(model)
    return model


def wrap_independent_model(strategy, role):
    model = get_peft_model(
        TinyModel(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
        adapter_name="default",
    ).cuda()
    model.add_adapter(role, model.peft_config["default"])
    model.set_adapter(role)
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device())
    from torch.distributed.fsdp import fully_shard

    fully_shard(model)
    return model


def engine_shell(module):
    engine = object.__new__(DistillationRoleGroupEngine)
    engine.module = module
    engine.role_group = RoleGroupSpec(
        name="base", model_ref="/tiny", storage="shared_base_adapters", placement="colocated"
    )
    engine.role_bindings = {
        "student": RoleBinding("student", "base", "student", True, "student_optim"),
        "teacher_score": RoleBinding("teacher_score", "base", None, False, None),
        "fake_score": RoleBinding("fake_score", "base", "fake_score", True, "fake_score_optim"),
        "student_ema": RoleBinding("student_ema", "base", "student_ema", False, None),
    }
    engine.optimizers = {}
    engine.lr_schedulers = {}
    engine.optimizer_configs = {}
    engine._active_role = "student"
    engine._primary_role = "student"
    role_parameters = {}
    for role in ("student", "fake_score"):
        with engine.use_role(role):
            role_parameters[role] = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    engine._role_parameters = role_parameters
    engine.optimizers = {role: torch.optim.AdamW(parameters, lr=0.1) for role, parameters in role_parameters.items()}
    engine.lr_schedulers = {
        role: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for role, optimizer in engine.optimizers.items()
    }
    engine.optimizer_configs = {role: SimpleNamespace(clip_grad=1.0) for role in engine.optimizers}
    engine.optimizer = engine.optimizers["student"]
    engine.lr_scheduler = engine.lr_schedulers["student"]
    engine.optimizer_config = engine.optimizer_configs["student"]
    engine.rank = dist.get_rank()
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine._uses_fsdp2_cpu_offload_policy = False
    engine.checkpoint_manager = TinyCheckpointManager(engine.module, engine.optimizer, engine.lr_scheduler)
    return engine


def independent_engine_shell(module, role, trainable):
    engine = object.__new__(DistillationRoleGroupEngine)
    engine.module = module
    engine.role_group = SimpleNamespace(name=f"{role}_model", storage="independent_module")
    engine.role_bindings = {
        role: RoleBinding(role, f"{role}_model", role, trainable, f"{role}_optim" if trainable else None)
    }
    engine.optimizers = {}
    engine.lr_schedulers = {}
    engine.optimizer_configs = {}
    engine._active_role = role
    engine._primary_role = role if trainable else None
    return engine


def adapter_snapshot(engine, role):
    binding = engine.role_bindings[role]
    with engine._adapter_state_context(), torch.no_grad():
        parameters = engine._active_adapter_trainable_params(binding.adapter)
        return tuple(
            (parameter.full_tensor() if isinstance(parameter, DTensor) else parameter).detach().cpu().clone()
            for parameter in parameters
        )


def assert_tensors_equal(left, right):
    assert len(left) == len(right)
    for left_tensor, right_tensor in zip(left, right, strict=True):
        torch.testing.assert_close(left_tensor, right_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_distillation_role_switch_preserves_graph_ema_and_state(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP role-isolation tests.")
    if dist.is_initialized():
        pytest.skip("This isolated one-rank FSDP test requires no existing process group.")

    with tempfile.TemporaryDirectory(prefix="distillation_fsdp_role_") as tmp_dir:
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{os.path.join(tmp_dir, 'rendezvous')}",
            rank=0,
            world_size=1,
        )
        try:
            torch.manual_seed(7)
            engine = engine_shell(wrap_model(strategy))
            engine.copy_adapter("student", "student_ema")
            student_before = adapter_snapshot(engine, "student")
            fake_before = adapter_snapshot(engine, "fake_score")
            ema_before = adapter_snapshot(engine, "student_ema")
            assert_tensors_equal(student_before, ema_before)

            inputs = torch.randn(2, 4, device="cuda")
            engine.optimizer_zero_grad("student")
            with engine.use_role("student") as module:
                student_loss = module(inputs).float().square().mean()
            with engine.use_role("teacher_score") as module:
                teacher_before = module(inputs).detach().clone()
                assert not torch.is_grad_enabled()
            with engine.use_role("fake_score", grad_enabled=False) as module:
                fake_output_before = module(inputs).detach().clone()
                assert not torch.is_grad_enabled()
            engine.backward_role("student", student_loss)
            engine.assert_gradient_isolation({"student"})
            engine.optimizers["student"].step()
            engine.update_role_ema("student", "student_ema", decay=0.5)

            student_after = adapter_snapshot(engine, "student")
            fake_after = adapter_snapshot(engine, "fake_score")
            ema_after = adapter_snapshot(engine, "student_ema")
            assert any(
                not torch.equal(before, after) for before, after in zip(student_before, student_after, strict=True)
            )
            assert_tensors_equal(fake_before, fake_after)
            for before, student, ema in zip(ema_before, student_after, ema_after, strict=True):
                torch.testing.assert_close(ema.float(), (before.float() + student.float()) * 0.5)
            with engine.use_role("teacher_score") as module:
                torch.testing.assert_close(module(inputs), teacher_before, rtol=0, atol=0)
            with engine.use_role("fake_score", grad_enabled=False) as module:
                torch.testing.assert_close(module(inputs), fake_output_before, rtol=0, atol=0)

            engine.optimizer_zero_grad("fake_score")
            with engine.use_role("fake_score") as module:
                fake_loss = module(inputs).float().square().mean()
            engine.backward_role("fake_score", fake_loss)
            engine.optimizers["fake_score"].step()
            engine.lr_schedulers["fake_score"].step()
            checkpoint_student = adapter_snapshot(engine, "student")
            checkpoint_fake = adapter_snapshot(engine, "fake_score")
            checkpoint_ema = adapter_snapshot(engine, "student_ema")
            checkpoint_path = os.path.join(tmp_dir, f"{strategy}_checkpoint")
            engine.save_role_group_checkpoint(checkpoint_path, global_step=1)

            with engine.use_role("student") as module:
                second_loss = module(inputs).float().square().mean()
            engine.optimizer_zero_grad("student")
            engine.backward_role("student", second_loss)
            engine.optimizers["student"].step()
            with engine._adapter_state_context(), torch.no_grad():
                for parameter in engine._active_adapter_trainable_params("fake_score"):
                    parameter.fill_(17.0)
            engine.load_role_group_checkpoint(checkpoint_path)
            assert_tensors_equal(adapter_snapshot(engine, "student"), checkpoint_student)
            assert_tensors_equal(adapter_snapshot(engine, "fake_score"), checkpoint_fake)
            assert_tensors_equal(adapter_snapshot(engine, "student_ema"), checkpoint_ema)

            torch.manual_seed(17)
            independent_student = independent_engine_shell(wrap_independent_model(strategy, "student"), "student", True)
            torch.manual_seed(19)
            independent_ema = independent_engine_shell(
                wrap_independent_model(strategy, "student_ema"), "student_ema", False
            )
            with independent_student._adapter_state_context(), torch.no_grad():
                for parameter in independent_student._active_adapter_trainable_params("student"):
                    parameter.fill_(4.0)
            with independent_ema._adapter_state_context(), torch.no_grad():
                for parameter in independent_ema._active_adapter_trainable_params("student_ema"):
                    parameter.fill_(0.0)
            independent_ema.update_module_ema_from(independent_student, decay=0.25)
            independent_values = adapter_snapshot(independent_ema, "student_ema")
            assert independent_values
            assert all(
                torch.allclose(value.float(), torch.full_like(value.float(), 3.0)) for value in independent_values
            )
        finally:
            dist.destroy_process_group()
