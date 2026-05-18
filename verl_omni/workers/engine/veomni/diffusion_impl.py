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

import gc
import logging
import os
from contextlib import nullcontext
from dataclasses import fields
from typing import Optional

import torch
import torch.distributed
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine.base import EngineRegistry

from verl_omni.workers.config import (
    DiffusionModelConfig,
    VeOmniDiffusionEngineConfig,
    VeOmniDiffusionOptimizerConfig,
)
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@torch.no_grad()
def _offload_veomni_model_to_cpu(model, empty_cache: bool = True):
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None or state._fsdp_param_group is None:
            continue
        state._fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def _load_veomni_model_to_gpu(model):
    model.to(get_device_id())


@torch.no_grad()
def _iter_optimizers(optimizer):
    if optimizer is None:
        return
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        yield from optimizer.optimizers_dict.values()
    else:
        yield optimizer


@torch.no_grad()
def _offload_veomni_optimizer(optimizer):
    for opt in _iter_optimizers(optimizer):
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def _load_veomni_optimizer(optimizer, device):
    for opt in _iter_optimizers(optimizer):
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)


@EngineRegistry.register(model_type="diffusion_model", backend=["veomni"], device=["cuda"])
class VeOmniDiffusionEngine(DiffusersFSDPEngine):
    """VeOmni-backed diffusion training engine for verl-omni RL loops."""

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        engine_config: VeOmniDiffusionEngineConfig,
        optimizer_config: VeOmniDiffusionOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        if model_config.lora_rank > 0 or model_config.lora_adapter_path is not None:
            raise NotImplementedError(
                "VeOmni diffusion backend does not support LoRA training yet. "
                "Use the existing fsdp/fsdp2 diffusion backend for LoRA runs."
            )
        super().__init__(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )

    def _init_device_mesh(self):
        from veomni.distributed import parallel_state

        if self.engine_config.ulysses_parallel_size != 1:
            raise NotImplementedError("VeOmni Qwen-Image diffusion backend does not support Ulysses SP yet.")

        world_size = torch.distributed.get_world_size()
        dp_size = world_size // self.engine_config.ulysses_parallel_size
        fsdp_size = self.engine_config.fsdp_size
        if fsdp_size < 0 or fsdp_size >= dp_size:
            dp_replicate_size = 1
            dp_shard_size = dp_size
        else:
            if dp_size % fsdp_size != 0:
                raise ValueError(f"Data parallel size ({dp_size}) must be divisible by fsdp_size ({fsdp_size}).")
            dp_replicate_size = dp_size // fsdp_size
            dp_shard_size = fsdp_size

        parallel_state.init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            extra_parallel_sizes=(self.engine_config.expert_parallel_size,),
            ulysses_size=self.engine_config.ulysses_parallel_size,
            dp_mode="fsdp2",
        )

        ps = parallel_state.get_parallel_state()
        self.device_mesh = ps.device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_parallel_size
        self.use_ulysses_sp = ps.sp_enabled

    def _build_ops_config(self):
        from veomni.arguments import OpsImplementationConfig

        ops_fields = {field.name for field in fields(OpsImplementationConfig)}
        ops_kwargs = {
            name: getattr(self.engine_config, name) for name in ops_fields if hasattr(self.engine_config, name)
        }
        return OpsImplementationConfig(**ops_kwargs)

    def _build_mixed_precision_config(self):
        from veomni.arguments import MixedPrecisionConfig

        return MixedPrecisionConfig(
            enable=self.engine_config.mixed_precision,
            param_dtype=self.engine_config.mixed_precision_param_dtype,
            reduce_dtype=self.engine_config.mixed_precision_reduce_dtype,
            output_dtype=self.engine_config.mixed_precision_output_dtype,
            cast_forward_inputs=self.engine_config.mixed_precision_cast_forward_inputs,
        )

    def _get_veomni_torch_dtype(self) -> str:
        dtype = PrecisionType.to_dtype(self.engine_config.model_dtype)
        if dtype == torch.float32:
            return "float32"
        if dtype == torch.float16:
            return "float16"
        if dtype == torch.bfloat16:
            return "bfloat16"
        raise ValueError(f"Unsupported VeOmni model dtype: {self.engine_config.model_dtype}")

    def _get_veomni_model_paths(self) -> tuple[str, str]:
        weights_path = os.path.join(self.model_config.local_path, self.model_config.veomni_transformer_subfolder)
        config_path = self.model_config.veomni_config_path or weights_path
        return config_path, weights_path

    def _build_optimizer(self, module):
        from veomni.optim import build_optimizer

        return build_optimizer(
            module,
            lr=self.optimizer_config.lr,
            betas=tuple(self.optimizer_config.betas),
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
            fused=self.optimizer_config.fused,
            optimizer_type=self.optimizer_config.optimizer,
        )

    def _build_lr_scheduler(self, optimizer):
        from veomni.optim import build_lr_scheduler

        return build_lr_scheduler(
            optimizer,
            train_steps=self.optimizer_config.total_training_steps,
            lr=self.optimizer_config.lr,
            lr_min=self.optimizer_config.lr_min,
            lr_decay_style=self.optimizer_config.lr_scheduler_type,
            lr_decay_ratio=self.optimizer_config.lr_decay_ratio,
            lr_warmup_ratio=self.optimizer_config.lr_warmup_steps_ratio,
            lr_start=self.optimizer_config.lr_start,
        )

    def _build_model_optimizer(self):
        from veomni.distributed.offloading import build_activation_offloading_context
        from veomni.distributed.torch_parallelize import build_parallelize_model
        from veomni.models.auto import build_foundation_model

        config_path, weights_path = self._get_veomni_model_paths()
        mixed_precision = self._build_mixed_precision_config()

        module = build_foundation_model(
            config_path=config_path,
            weights_path=weights_path,
            torch_dtype=self._get_veomni_torch_dtype(),
            init_device=self.engine_config.init_device,
            ops_implementation=self._build_ops_config(),
        )

        module = build_parallelize_model(
            module,
            init_device=self.engine_config.init_device,
            weights_path=weights_path,
            enable_reshard_after_forward=self.engine_config.reshard_after_forward,
            mixed_precision=mixed_precision,
            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
            basic_modules=list(set(getattr(module, "_no_split_modules", None) or [])),
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.forward_prefetch,
        )

        scheduler = self._build_scheduler()
        if not self.engine_config.forward_only:
            optimizer = self._build_optimizer(module)
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.engine_config.enable_activation_offload,
            self.model_config.enable_gradient_checkpointing,
            self.engine_config.activation_gpu_limit,
        )

    def initialize(self):
        self._build_model_optimizer()
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

    def get_data_parallel_rank(self):
        from veomni.distributed import parallel_state

        return parallel_state.get_parallel_state().dp_rank

    def get_data_parallel_size(self):
        from veomni.distributed import parallel_state

        return parallel_state.get_parallel_state().dp_size

    def get_data_parallel_group(self):
        from veomni.distributed import parallel_state

        return parallel_state.get_parallel_state().dp_group

    def is_mp_src_rank_with_outputs(self):
        from veomni.distributed import parallel_state

        ps = parallel_state.get_parallel_state()
        return ps.sp_rank == 0 if ps.sp_enabled else True

    def optimizer_step(self):
        assert self.optimizer_config.clip_grad is not None

        if hasattr(self.module, "clip_grad_norm_"):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super(DiffusersFSDPEngine, self).to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                _load_veomni_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                _load_veomni_optimizer(self.optimizer, get_device_id())
            gc.collect()
        elif device == "cpu":
            if model:
                _offload_veomni_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                _offload_veomni_optimizer(self.optimizer)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            _load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            _offload_veomni_model_to_cpu(self.module)
        gc.collect()
        aggressive_empty_cache(force_sync=True)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        if self._is_offload_param:
            _load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            _offload_veomni_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            _offload_veomni_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        if self.model_config.lora_rank > 0 or self.model_config.lora_adapter_path is not None:
            raise NotImplementedError("VeOmni diffusion backend does not support LoRA weight export yet.")

        _load_veomni_model_to_gpu(self.module)
        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self._is_offload_param:
            _offload_veomni_model_to_cpu(self.module)

        device = get_device_id()

        def param_generator():
            for name, param in params.items():
                tensor = param.full_tensor() if isinstance(param, DTensor) else param
                tensor = tensor.to(device, non_blocking=True)
                if tensor.is_floating_point() and tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16, non_blocking=True)
                yield f"transformer.{name}", tensor

        return param_generator(), None

    def disable_adapter(self):
        return nullcontext()
