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
from typing import Any, Callable, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import enable_full_determinism, prepare_micro_batches

from verl_omni.workers.config import (
    OmniModelConfig,
    VeOmniOmniEngineConfig,
    VeOmniOmniOptimizerConfig,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
device_name = get_device_name()

_NON_MODEL_KEYS = {
    "compute_loss",
    "disable_auto_offload",
    "global_token_num",
    "gradient_accumulation_steps",
    "max_token_len_per_gpu",
    "micro_batch_size_per_gpu",
    "mini_batch_size",
    "num_mini_batch",
    "sample_level_rewards",
    "sample_level_scores",
    "sp_size",
    "update_lr_scheduler",
    "use_dynamic_bsz",
    "use_fused_kernels",
    "use_remove_padding",
}


@EngineRegistry.register(model_type="omni_model", backend=["veomni"], device=["cuda"])
class VeOmniOmniEngine(BaseEngine):
    """VeOmni-backed Qwen3-Omni offline DPO engine."""

    def __init__(
        self,
        model_config: OmniModelConfig,
        engine_config: VeOmniOmniEngineConfig,
        optimizer_config: VeOmniOmniOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.rank = torch.distributed.get_rank()
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0 or self.model_config.lora_adapter_path is not None
        if self._is_lora:
            raise NotImplementedError(
                "VeOmni omni backend does not support LoRA injection yet. "
                "Set actor_rollout_ref.model.lora_rank=0 for the initial offline DPO path."
            )

        self._init_device_mesh()
        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def _init_device_mesh(self):
        from veomni.distributed import parallel_state

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

        self.dp_size = dp_size
        self.dp_replicate_size = dp_replicate_size
        self.dp_shard_size = dp_shard_size
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
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_parallel_size
        self.use_ulysses_sp = ps.sp_enabled

    def _build_ops_config(self):
        from veomni.arguments import OpsImplementationConfig

        ops_fields = {field.name for field in fields(OpsImplementationConfig)}
        ops_kwargs = {
            name: getattr(self.engine_config, name) for name in ops_fields if hasattr(self.engine_config, name)
        }
        return OpsImplementationConfig(**ops_kwargs)

    def _build_mixed_precision_config(self, enable: Optional[bool] = None):
        from veomni.arguments import MixedPrecisionConfig

        return MixedPrecisionConfig(
            enable=self.engine_config.mixed_precision if enable is None else enable,
            param_dtype=self.engine_config.mixed_precision_param_dtype,
            reduce_dtype=self.engine_config.mixed_precision_reduce_dtype,
            output_dtype=self.engine_config.mixed_precision_output_dtype,
            cast_forward_inputs=self.engine_config.mixed_precision_cast_forward_inputs,
        )

    def _build_model(self):
        from veomni.models import build_foundation_model

        ops_implementation = self.model_config.ops_implementation or self._build_ops_config()
        return build_foundation_model(
            config_path=self.model_config.config_path,
            weights_path=self.model_config.model_path,
            torch_dtype=self.engine_config.model_dtype,
            init_device=self.engine_config.init_device,
            encoder_data_balance=self.model_config.encoder_data_balance,
            encoder_data_balance_sorting_algo=self.model_config.encoder_data_balance_sorting_algo,
            ops_implementation=ops_implementation,
            config_kwargs=self.model_config.model_config,
        )

    def _parallelize_model(self, model, *, mixed_precision, enable_gradient_checkpointing: bool):
        from veomni.distributed.torch_parallelize import build_parallelize_model

        cpu_load_param_name = None
        if hasattr(model, "get_parallel_plan"):
            cpu_load_param_name = getattr(model.get_parallel_plan(), "cpu_load_param_name", None)

        return build_parallelize_model(
            model,
            init_device=self.engine_config.init_device,
            weights_path=self.model_config.model_path,
            enable_reshard_after_forward=self.engine_config.reshard_after_forward,
            mixed_precision=mixed_precision,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            basic_modules=list(
                set(getattr(model, "_no_split_modules", None) or []) | set(self.model_config.basic_modules)
            ),
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.forward_prefetch,
            enable_fsdp_offload=getattr(self.engine_config, "offload", False),
            broadcast_model_weights_from_rank0=True,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=getattr(self.engine_config, "max_load_broadcast_size", None),
        )

    def _build_model_optimizer(self):
        self.module = self._parallelize_model(
            self._build_model(),
            mixed_precision=self._build_mixed_precision_config(),
            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
        )
        self.model_fwd_context = nullcontext()
        self.model_bwd_context = nullcontext()

        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
        else:
            self.optimizer = torch.optim.AdamW(
                self.module.parameters(),
                lr=self.optimizer_config.lr,
                betas=tuple(self.optimizer_config.betas),
                eps=self.optimizer_config.eps,
                weight_decay=self.optimizer_config.weight_decay,
                fused=self.optimizer_config.fused,
            )
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)

        self.reference_model = self._parallelize_model(
            self._build_model(),
            mixed_precision=self._build_mixed_precision_config(enable=False),
            enable_gradient_checkpointing=False,
        )
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

        from veomni.data.data_collator import PostCollator

        self.post_forward = PostCollator()

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

    def _mixed_precision_forward_context(self):
        if not self.engine_config.mixed_precision:
            return nullcontext()
        return torch.autocast(
            device_type=device_name, dtype=getattr(torch, self.engine_config.mixed_precision_param_dtype)
        )

    def _concatenated_forward(self, model, micro_batch: TensorDict | dict[str, Any]):
        from veomni.distributed.parallel_state import get_parallel_state
        from veomni.distributed.sequence_parallel import gather_outputs
        from veomni.utils.constants import IGNORE_INDEX

        model_inputs = {key: value for key, value in micro_batch.items() if key not in _NON_MODEL_KEYS}
        outputs = model(**model_inputs, return_log_probs=True, use_cache=False)
        log_probs_packed = outputs.fused_linear_aux.log_probs.squeeze(0)
        seq_lens = self.post_forward.compute_seqlens_func(micro_batch)
        if self.use_ulysses_sp:
            log_probs_packed = gather_outputs(log_probs_packed, gather_dim=0, group=get_parallel_state().sp_group)
        log_probs_packed = log_probs_packed[: sum(seq_lens)]
        log_probs_list = list(log_probs_packed.split(seq_lens, dim=0))

        if self.use_ulysses_sp:
            all_labels = gather_outputs(micro_batch["labels"], gather_dim=-1, group=get_parallel_state().sp_group)
            all_labels = all_labels.view(-1)[: sum(seq_lens)]
            labels_list = list(all_labels.split(seq_lens))
        else:
            all_labels = micro_batch["labels"].view(-1)
            labels_list = []
            offset = 0
            for seq_len in seq_lens:
                seq_labels = all_labels[offset : offset + seq_len]
                labels_list.append(F.pad(seq_labels[1:], (0, 1), value=IGNORE_INDEX))
                offset += seq_len

        all_logps = []
        for seq_log_probs, seq_labels in zip(log_probs_list, labels_list, strict=True):
            loss_mask = seq_labels != IGNORE_INDEX
            logp = (seq_log_probs.float() * loss_mask).sum()
            if tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False):
                logp = logp / loss_mask.sum().clamp(min=1)
            all_logps.append(logp)
        all_logps_t = torch.stack(all_logps)
        return all_logps_t[0::2], all_logps_t[1::2]

    def _forward_backward_micro_batch(self, micro_batch: TensorDict, loss_function: Callable):
        from veomni.ops.batch_invariant_ops import set_batch_invariant_mode

        micro_batch = micro_batch.to(get_device_id())
        tu.assign_non_tensor(
            micro_batch,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self._concatenated_forward(self.reference_model, micro_batch)

        with self.model_fwd_context, self._mixed_precision_forward_context(), set_batch_invariant_mode(False):
            policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.module, micro_batch)

        model_output = {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": ref_chosen_logps,
            "reference_rejected_logps": ref_rejected_logps,
        }
        loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        with self.model_bwd_context, set_batch_invariant_mode(False):
            loss.backward()
        return loss.detach(), metrics

    def train_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        if loss_function is None:
            raise ValueError("VeOmniOmniEngine.train_batch requires a loss_function.")
        config = getattr(loss_function, "keywords", {}).get("config")
        loss_config = getattr(config, "omni_loss", None)
        tu.assign_non_tensor(data, average_log_prob=getattr(loss_config, "average_log_prob", False))
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        gradient_accumulation_steps = len(micro_batches)
        losses = []
        metrics: dict[str, list[torch.Tensor]] = {}
        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
            loss, micro_metrics = self._forward_backward_micro_batch(micro_batch, loss_function)
            losses.append(loss.item())
            for key, value in micro_metrics.items():
                metrics.setdefault(key, []).append(value)

        grad_norm = self.optimizer_step()
        self.optimizer_zero_grad()
        metrics = {key: torch.stack(value).mean().item() for key, value in metrics.items()}
        metrics["grad_norm"] = grad_norm
        return {"model_output": {}, "loss": losses, "metrics": metrics}

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        del loss_function
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        chosen_logps = []
        rejected_logps = []
        with torch.no_grad():
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                chosen, rejected = self._concatenated_forward(self.module, micro_batch)
                chosen_logps.append(chosen)
                rejected_logps.append(rejected)
        return {
            "model_output": {
                "chosen_logps": torch.cat(chosen_logps, dim=0),
                "rejected_logps": torch.cat(rejected_logps, dim=0),
            },
            "loss": [0.0],
            "metrics": {},
        }

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
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
            logger.warning("grad_norm is not finite: %s", grad_norm)
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        if self.lr_scheduler is None:
            return None
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        from veomni.distributed.offloading import (
            load_model_to_gpu,
            load_optimizer,
            offload_model_to_cpu,
            offload_optimizer,
        )

        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_model_to_gpu(self.module, get_device_id())
                load_model_to_gpu(self.reference_model, get_device_id())
            if optimizer and self.optimizer is not None:
                load_optimizer(self.optimizer, get_device_id())
            gc.collect()
        elif device == "cpu":
            if model:
                offload_model_to_cpu(self.module)
                offload_model_to_cpu(self.reference_model)
            if optimizer and self.optimizer is not None:
                offload_optimizer(self.optimizer)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        from veomni.distributed.offloading import load_model_to_gpu, offload_model_to_cpu

        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_model_to_gpu(self.module, get_device_id())
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_model_to_cpu(self.module)
        gc.collect()
        aggressive_empty_cache(force_sync=True)

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: int = True,
        **kwargs,
    ) -> None:
        from veomni.distributed.offloading import load_model_to_gpu, offload_model_to_cpu, offload_optimizer

        if self._is_offload_param:
            load_model_to_gpu(self.module, get_device_id())
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        from veomni.distributed.offloading import load_model_to_gpu, offload_model_to_cpu

        if self._is_lora:
            raise NotImplementedError("VeOmni omni backend does not support LoRA weight export yet.")
        load_model_to_gpu(self.module, get_device_id())
        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
        if self._is_offload_param:
            offload_model_to_cpu(self.module)
        device = get_device_id()
        export_dtype = PrecisionType.to_dtype(self.engine_config.model_dtype)

        def param_generator():
            for name, param in params.items():
                tensor = param.full_tensor() if isinstance(param, DTensor) else param
                tensor = tensor.to(device, non_blocking=True)
                if tensor.is_floating_point() and tensor.dtype != export_dtype:
                    tensor = tensor.to(export_dtype, non_blocking=True)
                yield name, tensor

        return param_generator(), None

    def disable_adapter(self):
        return nullcontext()


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniOmniEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniOmniEngine)
        super().__enter__()
        self.engine.module.eval()
        self.engine.reference_model.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniOmniEngine)
        if self.engine.engine_config.fsdp_size > 1 and hasattr(self.engine.module, "reshard"):
            self.engine.module.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniOmniEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniOmniEngine)
        super().__enter__()
        self.engine.module.train()
        self.engine.reference_model.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniOmniEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)
