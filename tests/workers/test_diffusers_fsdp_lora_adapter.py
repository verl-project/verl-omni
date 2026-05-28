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
from functools import partial

import pytest
import ray
import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu

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


def _require_model_path() -> str:
    if not os.path.isdir(_DEFAULT_MODEL_PATH):
        pytest.skip(
            f"Tiny Qwen-Image model not found at {_DEFAULT_MODEL_PATH!r}. "
            "Provide the model or adjust _DEFAULT_MODEL_PATH."
        )
    return _DEFAULT_MODEL_PATH


class LoRAFSDPTestWorker(TrainingWorker):
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


def _rank0_params(worker_outputs) -> dict[str, torch.Tensor]:
    return worker_outputs[0]


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


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_diffusers_fsdp_lora_adapter_switch(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP LoRA adapter tests.")

    base_model_path = _require_model_path()
    ray.init()
    tmp_dir = tempfile.mkdtemp(prefix="qwen_image_lora_fsdp_")
    try:
        visible_gpus = torch.cuda.device_count()
        device_count = resolve_requested_num_gpus(default_num_gpus=max(1, visible_gpus))
        if device_count > 1 and device_count % 2 != 0:
            pytest.skip(f"Need even GPU count for cp=2/fsdp_size=device_count test, got {device_count}")

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
        )
        training_config.model_config.policy_state_adapters = ("default", "old")

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(LoRAFSDPTestWorker), config=training_config)
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.reset()

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

        filled = wg.fill_lora_adapter("default", base=7.25, step=-0.1)
        assert filled[0] > 0

        default_1 = _rank0_params(wg.collect_lora_params("default"))
        old_1 = _rank0_params(wg.collect_lora_params("old"))
        _lora_params_differ(default_1, default_0)
        _lora_params_differ(default_1, old_1)
    finally:
        ray.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)
