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
"""GPU integration tests for dual-adapter LoRA under FSDP/FSDP2."""

import os
import shutil
import tempfile
from copy import deepcopy
from functools import partial

import pytest
import ray
import torch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity

from verl_omni.trainer.diffusion.distillation.contracts import RoleBinding, RoleGroupSpec
from verl_omni.workers.engine.fsdp.distillation_impl import DistillationRoleGroupEngine
from verl_omni.workers.engine_workers import TrainingWorker
from verl_omni.workers.utils.losses import diffusion_loss
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

from ..utils.gpu_test_topology import resolve_requested_num_gpus
from .test_diffusers_fsdp_engine import (
    _create_sp_compatible_model,
    _diffusers_sp_supported,
    create_data_samples,
    create_training_config,
)

_DEFAULT_MODEL_PATH = os.path.expanduser("~/models/tiny-random/Qwen-Image")
_LORA_RTOL = 1e-2
_LORA_ATOL = 1e-2
_FILL_BASE = 7.25
_FILL_STEP = -0.1


def _require_model_path() -> str:
    if not os.path.isdir(_DEFAULT_MODEL_PATH):
        pytest.skip(
            f"Tiny Qwen-Image model not found at {_DEFAULT_MODEL_PATH!r}. "
            "Provide the model or adjust _DEFAULT_MODEL_PATH."
        )
    return _DEFAULT_MODEL_PATH


class DistillationLoRAFSDPTestWorker(Worker):
    """Tiny-model worker exercising the real multi-role FSDP engine."""

    def __init__(self, training_config, model_path):
        Worker.__init__(self)
        initialize_global_process_group_ray(timeout_second=None)
        set_numa_affinity()
        model_config = deepcopy(training_config.model_config)
        engine_config = deepcopy(training_config.engine_config)
        optimizer_config = deepcopy(training_config.optimizer_config)
        checkpoint_config = deepcopy(training_config.checkpoint_config)
        object.__setattr__(model_config, "model_type", "diffusion_distillation_model")
        object.__setattr__(engine_config, "use_orig_params", True)
        group = RoleGroupSpec(
            name="base",
            model_ref=model_path,
            storage="shared_base_adapters",
            placement="colocated",
        )
        bindings = (
            RoleBinding("student", "base", "student", True, "student_optim"),
            RoleBinding("teacher_score", "base", None, False, None),
            RoleBinding("fake_score", "base", "fake_score", True, "fake_score_optim"),
            RoleBinding("student_ema", "base", "student_ema", False, None),
        )
        fake_optimizer_config = deepcopy(optimizer_config)
        object.__setattr__(fake_optimizer_config, "lr", optimizer_config.lr * 0.5)
        self.engine = DistillationRoleGroupEngine(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            role_group=group,
            role_bindings=bindings,
            optimizer_configs={"student": optimizer_config, "fake_score": fake_optimizer_config},
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.initialize()

    def collect_adapter(self, role):
        params, config = self.engine.get_per_tensor_param(adapter_name=role, base_sync_done=True)
        return {name: tensor.detach().cpu() for name, tensor in params}, config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def collect_roles(self):
        student, student_config = self.collect_adapter("student")
        fake, fake_config = self.collect_adapter("fake_score")
        ema, ema_config = self.collect_adapter("student_ema")
        return {
            "student": student,
            "fake_score": fake,
            "student_ema": ema,
            "student_config": student_config,
            "fake_config": fake_config,
            "ema_config": ema_config,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_student_step_across_role_switches(self):
        self.engine.optimizer_zero_grad("student")
        with self.engine.use_role("student"):
            student_parameters = self.engine.parameters_for_role("student")
            loss = sum(parameter.float().square().mean() for parameter in student_parameters if parameter.numel())
        with self.engine.use_role("teacher_score"):
            assert not torch.is_grad_enabled()
        with self.engine.use_role("fake_score", grad_enabled=False):
            assert not torch.is_grad_enabled()
        self.engine.backward_role("student", loss)
        stepped, grad_norm = self.engine.optimizer_step("student")
        self.engine.update_role_ema("student", "student_ema", decay=0.5)
        return {"stepped": stepped, "grad_norm": grad_norm}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_role_group(self, path, step):
        self.engine.save_role_group_checkpoint(path, step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_role_group(self, path):
        self.engine.load_role_group_checkpoint(path)


class LoRAFSDPTestWorker(TrainingWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def report_fsdp_topology(self):
        import torch.distributed as dist
        from verl.utils.fsdp_utils import fsdp_version

        return {
            "fsdp_version": fsdp_version(self.engine.module),
            "world_size": dist.get_world_size(),
            "strategy": self.engine.engine_config.strategy,
            "fsdp_size": self.engine.engine_config.fsdp_size,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def collect_lora_params(self, adapter_name: str = "default"):
        params, _ = self.engine.get_per_tensor_param(
            layered_summon=False,
            base_sync_done=True,
            adapter_name=adapter_name,
        )
        return {name: tensor.detach().cpu() for name, tensor in params}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def fill_lora_adapter(self, adapter_name: str, base: float, step: float):
        with self.engine._adapter_state_context():
            peft_model = getattr(self.engine.module, "_fsdp_wrapped_module", self.engine.module)
            peft_model.set_adapter(adapter_name)
            idx = 0
            for param in peft_model.parameters():
                if param.requires_grad:
                    param.data.fill_(base + idx * step)
                    idx += 1
            return idx

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def copy_default_to_old(self, offset: float):
        with self.engine._adapter_state_context():
            peft_model = getattr(self.engine.module, "_fsdp_wrapped_module", self.engine.module)
            with torch.no_grad():
                peft_model.set_adapter("default")
                source_params = [param.data.clone() for param in peft_model.parameters() if param.requires_grad]
                peft_model.set_adapter("old")
                target_params = [param.data for param in peft_model.parameters() if param.requires_grad]
                for source_param, target_param in zip(source_params, target_params, strict=True):
                    target_param.copy_(source_param + offset)
            return len(source_params)


def _rank0_params(worker_outputs) -> dict[str, torch.Tensor]:
    return worker_outputs[0]


def _lora_params_close(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    rtol: float = _LORA_RTOL,
    atol: float = _LORA_ATOL,
) -> None:
    assert left.keys() == right.keys()
    for name in sorted(left.keys()):
        assert torch.allclose(left[name].float(), right[name].float(), rtol=rtol, atol=atol), name


def _lora_params_differ(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    rtol: float = _LORA_RTOL,
    atol: float = _LORA_ATOL,
) -> None:
    assert left.keys() == right.keys()
    assert any(
        not torch.allclose(left[name].float(), right[name].float(), rtol=rtol, atol=atol) for name in left.keys()
    )


def _assert_ema_blend(
    target: dict[str, torch.Tensor],
    old: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    decay: float,
) -> None:
    assert target.keys() == old.keys() == source.keys()
    for name in sorted(target.keys()):
        expected = old[name].float() * decay + source[name].float() * (1.0 - decay)
        assert torch.allclose(target[name].float(), expected, rtol=_LORA_RTOL, atol=_LORA_ATOL), name


def _resolve_lora_test_device_count(strategy: str) -> int:
    visible_gpus = torch.cuda.device_count()
    device_count = resolve_requested_num_gpus(default_num_gpus=max(2 if strategy == "fsdp2" else 1, visible_gpus))
    if strategy == "fsdp2" and device_count < 2:
        pytest.skip("FSDP2 LoRA adapter tests require at least 2 GPUs to exercise sharded summon/writeback.")
    if device_count > 1 and device_count % 2 != 0:
        pytest.skip(f"Need even GPU count for cp=2/fsdp_size=device_count test, got {device_count}")
    return device_count


def _run_lora_adapter_switch_test(strategy: str) -> None:
    base_model_path = _require_model_path()
    device_count = _resolve_lora_test_device_count(strategy)

    ray.init()
    tmp_dir = tempfile.mkdtemp(prefix="qwen_image_lora_fsdp_")
    try:
        sp_enabled = device_count > 1 and _diffusers_sp_supported()
        if sp_enabled:
            model_path = _create_sp_compatible_model(tmp_dir, base_model_path, num_attention_heads=2)
        else:
            model_path = base_model_path

        training_config, actor_config = create_training_config(
            model_type="diffusion_model",
            strategy=strategy,
            device_count=device_count,
            model=model_path,
            policy_state_adapters=("default", "old"),
        )

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(LoRAFSDPTestWorker), config=training_config)
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.reset()

        topology = wg.report_fsdp_topology()
        assert topology[0]["strategy"] == strategy
        if strategy == "fsdp2":
            assert topology[0]["fsdp_version"] == 2
            assert topology[0]["world_size"] >= 2
            assert topology[0]["fsdp_size"] == device_count

        default_0 = _rank0_params(wg.collect_lora_params("default"))
        old_0 = _rank0_params(wg.collect_lora_params("old"))
        assert default_0
        assert old_0.keys() == default_0.keys()
        _lora_params_differ(default_0, old_0)

        loss_fn = partial(diffusion_loss, config=actor_config)
        wg.set_loss_fn(loss_fn)

        data_td = create_data_samples(device_count, training_config.model_config).to_tensordict()
        data_td = embeds_padding_2_no_padding(data_td)
        ppo_mini_batch_size = 4
        tu.assign_non_tensor(
            data_td,
            global_batch_size=ppo_mini_batch_size * device_count,
            mini_batch_size=ppo_mini_batch_size * device_count,
            epochs=actor_config.ppo_epochs,
            seed=42,
            dataloader_kwargs={"shuffle": actor_config.shuffle},
        )
        output = wg.train_mini_batch(data_td)
        assert "metrics" in output.get()

        filled = wg.fill_lora_adapter("default", base=_FILL_BASE, step=_FILL_STEP)
        assert filled[0] > 0

        default_1 = _rank0_params(wg.collect_lora_params("default"))
        old_1 = _rank0_params(wg.collect_lora_params("old"))
        _lora_params_differ(default_1, default_0, rtol=0, atol=0)
        _lora_params_differ(default_1, old_1)
        assert not torch.allclose(
            next(iter(default_1.values())).float(),
            next(iter(default_0.values())).float(),
            rtol=0,
            atol=0,
        ), f"{strategy} adapter writeback failed: collected default adapter unchanged after fill_lora_adapter"

        copied = wg.copy_default_to_old(offset=50.0)
        assert copied[0] > 0
        old_after_copy = _rank0_params(wg.collect_lora_params("old"))
        default_after_copy = _rank0_params(wg.collect_lora_params("default"))
        for name, old_tensor in old_after_copy.items():
            assert torch.allclose(
                old_tensor.float(),
                (default_after_copy[name].float() + 50.0),
                rtol=_LORA_RTOL,
                atol=_LORA_ATOL,
            ), f"{strategy} adapter copy failed for {name!r}"
    finally:
        ray.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_copy_ema_adapter_test(strategy: str) -> None:
    base_model_path = _require_model_path()
    device_count = _resolve_lora_test_device_count(strategy)

    ray.init()
    tmp_dir = tempfile.mkdtemp(prefix="qwen_image_lora_fsdp_")
    try:
        sp_enabled = device_count > 1 and _diffusers_sp_supported()
        if sp_enabled:
            model_path = _create_sp_compatible_model(tmp_dir, base_model_path, num_attention_heads=2)
        else:
            model_path = base_model_path

        training_config, actor_config = create_training_config(
            model_type="diffusion_model",
            strategy=strategy,
            device_count=device_count,
            model=model_path,
            policy_state_adapters=("default", "old"),
        )

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(LoRAFSDPTestWorker), config=training_config)
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.reset()

        default_0 = _rank0_params(wg.collect_lora_params("default"))
        old_0 = _rank0_params(wg.collect_lora_params("old"))
        assert default_0
        assert old_0.keys() == default_0.keys()
        _lora_params_differ(default_0, old_0)

        wg.copy_adapter(source="default", target="old")
        old_1 = _rank0_params(wg.collect_lora_params("old"))
        _lora_params_close(old_1, default_0)

        loss_fn = partial(diffusion_loss, config=actor_config)
        wg.set_loss_fn(loss_fn)

        data_td = create_data_samples(device_count, training_config.model_config).to_tensordict()
        data_td = embeds_padding_2_no_padding(data_td)
        ppo_mini_batch_size = 4
        tu.assign_non_tensor(
            data_td,
            global_batch_size=ppo_mini_batch_size * device_count,
            mini_batch_size=ppo_mini_batch_size * device_count,
            epochs=actor_config.ppo_epochs,
            seed=42,
            dataloader_kwargs={"shuffle": actor_config.shuffle},
        )
        output = wg.train_mini_batch(data_td)
        assert "metrics" in output.get()

        filled = wg.fill_lora_adapter("default", base=_FILL_BASE, step=_FILL_STEP)
        assert filled[0] > 0

        default_1 = _rank0_params(wg.collect_lora_params("default"))
        old_2 = _rank0_params(wg.collect_lora_params("old"))
        _lora_params_differ(default_1, default_0)
        _lora_params_close(old_2, old_1)

        wg.ema_update_adapter(source="default", target="old", decay=0.9)
        old_3 = _rank0_params(wg.collect_lora_params("old"))
        _assert_ema_blend(old_3, old_2, default_1, decay=0.9)
    finally:
        ray.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_diffusers_fsdp_lora_adapter_switch(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP LoRA adapter tests.")
    _run_lora_adapter_switch_test(strategy)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_diffusers_fsdp_lora_adapter_copy_ema(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP LoRA adapter tests.")
    _run_copy_ema_adapter_test(strategy)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_distillation_role_isolation_ema_and_resume(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for distillation role-group tests.")
    base_model_path = _require_model_path()
    device_count = _resolve_lora_test_device_count(strategy)

    ray.init()
    tmp_dir = tempfile.mkdtemp(prefix="qwen_image_distillation_roles_")
    try:
        sp_enabled = device_count > 1 and _diffusers_sp_supported()
        if sp_enabled:
            model_path = _create_sp_compatible_model(tmp_dir, base_model_path, num_attention_heads=2)
        else:
            model_path = base_model_path
        training_config, _ = create_training_config(
            model_type="diffusion_distillation_model",
            strategy=strategy,
            device_count=device_count,
            model=model_path,
            policy_state_adapters=("default", "student", "fake_score", "student_ema"),
        )
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(DistillationLoRAFSDPTestWorker),
            training_config=training_config,
            model_path=model_path,
        )
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        initial = wg.collect_roles()[0]
        _lora_params_close(initial["student"], initial["student_ema"])
        _lora_params_close(initial["student"], initial["fake_score"])
        assert initial["student_config"] == initial["fake_config"] == initial["ema_config"]

        step_result = wg.run_student_step_across_role_switches()[0]
        assert step_result["stepped"]
        assert step_result["grad_norm"] > 0
        after_step = wg.collect_roles()[0]
        _lora_params_differ(after_step["student"], initial["student"])
        _lora_params_close(after_step["fake_score"], initial["fake_score"])
        _assert_ema_blend(after_step["student_ema"], initial["student_ema"], after_step["student"], decay=0.5)

        checkpoint_path = os.path.join(tmp_dir, "checkpoint")
        wg.save_role_group(checkpoint_path, 1)
        wg.run_student_step_across_role_switches()
        changed = wg.collect_roles()[0]
        _lora_params_differ(changed["student"], after_step["student"])
        wg.load_role_group(checkpoint_path)
        restored = wg.collect_roles()[0]
        _lora_params_close(restored["student"], after_step["student"])
        _lora_params_close(restored["fake_score"], after_step["fake_score"])
        _lora_params_close(restored["student_ema"], after_step["student_ema"])
    finally:
        ray.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)
