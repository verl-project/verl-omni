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
from typing import Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import enable_full_determinism, prepare_micro_batches

from verl_omni.pipelines.utils import build_scheduler, forward_and_sample_previous_step, prepare_model_inputs
from verl_omni.workers.config import (
    DiffusionModelConfig,
    VeOmniDiffusionEngineConfig,
    VeOmniDiffusionOptimizerConfig,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
device_name = get_device_name()


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
class VeOmniDiffusionEngine(BaseEngine):
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
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.rank = torch.distributed.get_rank()

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = False

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

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

    def _build_scheduler(self):
        return build_scheduler(self.model_config)

    def _build_model_optimizer(self):
        from veomni.distributed.offloading import build_activation_offloading_context
        from veomni.distributed.torch_parallelize import build_parallelize_model
        from veomni.models.auto import build_foundation_model

        config_path, weights_path = self._get_veomni_model_paths()
        mixed_precision = self._build_mixed_precision_config()

        # Keep this sequence aligned with VeOmni's DiTTrainer._build_model:
        # build the foundation DiT first, then hand it to VeOmni's parallelizer,
        # optimizer, scheduler, and training contexts.
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

    def train_mode(self, **kwargs):
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return EngineEvalModeCtx(self, **kwargs)

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

    @staticmethod
    def _unpad_nested_embeds(embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = embeds.size(0)
        max_seq_len = max(embeds.offsets().diff())
        embed_dim = embeds.size(-1)
        embeds = torch.nested.to_padded_tensor(embeds, padding=0, output_size=(batch_size, max_seq_len, embed_dim))
        mask = torch.nested.to_padded_tensor(mask, padding=0, output_size=(batch_size, max_seq_len))
        return embeds, mask

    @staticmethod
    def _pad_embeds_for_sp(embeds: torch.Tensor, mask: torch.Tensor, sp_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = embeds.size(1)
        aligned_seq_len = (seq_len + sp_size - 1) // sp_size * sp_size
        if aligned_seq_len > seq_len:
            pad_len = aligned_seq_len - seq_len
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_len))
            mask = torch.nn.functional.pad(mask, (0, pad_len))
        return embeds, mask

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        latents = micro_batch["all_latents"]
        timesteps = micro_batch["all_timesteps"]
        prompt_embeds = micro_batch["prompt_embeds"]
        prompt_embeds_mask = micro_batch["prompt_embeds_mask"]
        negative_prompt_embeds = micro_batch["negative_prompt_embeds"]
        negative_prompt_embeds_mask = micro_batch["negative_prompt_embeds_mask"]
        sp_size = self.ulysses_sequence_parallel_size if self.use_ulysses_sp else 1

        if prompt_embeds.is_nested:
            prompt_embeds, prompt_embeds_mask = self._unpad_nested_embeds(prompt_embeds, prompt_embeds_mask)

        if sp_size > 1:
            prompt_embeds, prompt_embeds_mask = self._pad_embeds_for_sp(prompt_embeds, prompt_embeds_mask, sp_size)

        if isinstance(negative_prompt_embeds, torch.Tensor) and negative_prompt_embeds.is_nested:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._unpad_nested_embeds(
                negative_prompt_embeds, negative_prompt_embeds_mask
            )

        if isinstance(negative_prompt_embeds, torch.Tensor) and sp_size > 1:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._pad_embeds_for_sp(
                negative_prompt_embeds, negative_prompt_embeds_mask, sp_size
            )

        return prepare_model_inputs(
            module=self.module,
            model_config=self.model_config,
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            micro_batch=micro_batch,
            step=step,
        )

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        log_prob, prev_sample_mean, std_dev_t, sqrt_dt = output
        return {
            "log_probs": log_prob,
            "prev_sample_mean": prev_sample_mean,
            "std_dev_t": std_dev_t,
            "sqrt_dt": sqrt_dt,
        }

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, step):
        model_inputs, negative_model_inputs = self.prepare_model_inputs(micro_batch=micro_batch, step=step)
        raw_output = forward_and_sample_previous_step(
            module=self.module,
            scheduler=self.scheduler,
            model_config=self.model_config,
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
            scheduler_inputs=micro_batch,
            step=step,
        )
        model_output = self.prepare_model_outputs(output=raw_output, micro_batch=micro_batch)

        if loss_function is not None:
            data = tu.get_tensordict(
                {
                    "old_log_probs": micro_batch["old_log_probs"][:, step],
                    "advantages": micro_batch["advantages"][:, step],
                },
            )
            tu.assign_non_tensor(
                data,
                gradient_accumulation_steps=tu.get_non_tensor_data(
                    micro_batch, "gradient_accumulation_steps", default=None
                ),
                sp_size=tu.get_non_tensor_data(micro_batch, "sp_size", default=None),
            )

            if micro_batch.get("ref_log_prob", None) is not None:
                data["ref_log_prob"] = micro_batch["ref_log_prob"][:, step]

            if micro_batch.get("ref_prev_sample_mean", None) is not None:
                data["ref_prev_sample_mean"] = micro_batch["ref_prev_sample_mean"][:, step]

            if micro_batch.get("old_prev_sample_mean", None) is not None:
                data["old_prev_sample_mean"] = micro_batch["old_prev_sample_mean"][:, step]

            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device_name)
            metrics = {}

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

        return loss, output

    def forward_backward_batch(
        self, data: TensorDict, loss_function: Callable, forward_only: bool = False
    ) -> list[TensorDict]:
        num_timesteps = data["all_timesteps"].shape[1]
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)
        tu.assign_non_tensor(data, use_dynamic_bsz=False)

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        gradient_accumulation_steps = len(micro_batches) * num_timesteps
        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
            meta_info_lst = {"model_output": [], "loss": [], "metrics": []}
            with ctx:
                for step in range(num_timesteps):
                    loss, meta_info = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only, step=step
                    )

                    if not forward_only:
                        loss.backward()

                    for key, val in meta_info.items():
                        meta_info_lst[key].append(val)

            output_lst.append(meta_info_lst)

        return self.postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def postprocess_batch_func(self, output_lst, indices, data: TensorDict):
        model_output = {}
        losses = []
        aggregated_metrics = {}

        for output in output_lst:
            model_output_lst = {}
            if "model_output" in output:
                for model_output_dict in output["model_output"]:
                    for key, val in model_output_dict.items():
                        model_output_lst.setdefault(key, []).append(val)
                for key, val in model_output_lst.items():
                    model_output.setdefault(key, []).append(torch.stack(val, dim=1))

            if "loss" in output:
                losses.append(output["loss"])

            if "metrics" in output:
                for metrics in output["metrics"]:
                    append_to_dict(aggregated_metrics, metrics)

        for key, val in model_output.items():
            model_output[key] = torch.concat(val, dim=0)

        return {
            "model_output": model_output,
            "loss": losses,
            "metrics": aggregated_metrics,
        }

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

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

    def lr_scheduler_step(self):
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

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


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniDiffusionEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniDiffusionEngine)
        super().__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniDiffusionEngine)
        if self.engine.engine_config.fsdp_size > 1 and hasattr(self.engine.module, "reshard"):
            self.engine.module.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniDiffusionEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniDiffusionEngine)
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniDiffusionEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)
