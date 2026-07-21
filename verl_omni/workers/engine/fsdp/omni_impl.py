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
"""FSDP/FSDP2 engine for omni models (GSPO/PPO and offline paired DPO).

Model loading follows PR #258: ``AutoModelForMultimodalLM`` plus
``OmniModelBase.configure_model``.  Policy-gradient training reuses verl's
``FSDPEngineWithLMHead`` loop with omni-specific ``prepare_model_inputs``.
Offline paired DPO branches on ``model_config.trainer_type`` (``direct_preference``).
"""

import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    get_init_weight_context_manager,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.model import convert_weight_keys
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.utils import prepare_micro_batches

from verl_omni.pipelines.utils import (
    compute_omni_preference_logps,
    prepare_omni_model_inputs,
    prepare_omni_preference_inputs,
)
from verl_omni.utils.fsdp_utils import collect_lora_params
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(LoRAAdapterMixin, FSDPEngineWithLMHead):
    """FSDP/FSDP2 omni model engine for policy-gradient and offline paired DPO."""

    def __init__(
        self,
        model_config: OmniModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        self._is_lora = self.model_config.lora_rank > 0 or self.model_config.lora_adapter_path is not None
        self._trainer_type = getattr(model_config, "trainer_type", "policy_gradient")

    def _is_direct_preference(self) -> bool:
        return self._trainer_type == "direct_preference"

    def _build_module(self):
        from transformers import AutoModelForMultimodalLM

        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_config: OmniModelConfig
        architecture = self.model_config.architecture

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Umbrella config delegates tie_word_embeddings to sub-configs.
        if not hasattr(self.model_config.hf_config, "tie_word_embeddings"):
            self.model_config.hf_config.tie_word_embeddings = False

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            module = AutoModelForMultimodalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            adapter_cls = OmniModelBase.get_class_by_name(
                architecture,
                self.model_config.model_stage,
                self.model_config.get("external_lib"),
            )
            module = adapter_cls.configure_model(module, self.model_config)

            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_fsdp_module(self, module):
        saved_lora_rank = self.model_config.lora_rank
        lora_rank_changed = False
        if self._is_lora and saved_lora_rank <= 0:
            object.__setattr__(self.model_config, "lora_rank", 1)
            lora_rank_changed = True
        try:
            return super()._build_fsdp_module(module)
        finally:
            if lora_rank_changed:
                object.__setattr__(self.model_config, "lora_rank", saved_lora_rank)

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Apply omni adapter normalization, then run verl's LM packing path."""
        param_dtype = getattr(self, "_autocast_dtype", None)
        omni_inputs = prepare_omni_model_inputs(self.model_config, micro_batch, dtype=param_dtype)
        merged = {key: micro_batch.get(key) for key in micro_batch.keys()}
        merged.update(omni_inputs)
        batch_size = micro_batch.batch_size[0] if micro_batch.batch_size else omni_inputs["input_ids"].shape[0]
        return super().prepare_model_inputs(TensorDict.from_dict(merged, batch_size=[batch_size]))

    def _concatenated_forward(self, model, micro_batch: TensorDict | dict[str, Any]):
        model_inputs, labels, segment_ranges = prepare_omni_preference_inputs(
            self.model_config,
            micro_batch,
            dtype=next(self.module.parameters()).dtype,
        )
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return compute_omni_preference_logps(
            self.model_config,
            logits,
            labels,
            segment_ranges,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )

    def _forward_backward_preference_micro_batch(self, micro_batch: TensorDict, loss_function: Callable):
        micro_batch = micro_batch.to(get_device_id())
        tu.assign_non_tensor(
            micro_batch,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )
        policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.module, micro_batch)
        model_output = {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": micro_batch["reference_chosen_logps"],
            "reference_rejected_logps": micro_batch["reference_rejected_logps"],
        }
        loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        loss.backward()
        return loss.detach(), metrics

    def train_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        if not self._is_direct_preference():
            return super().train_batch(data, loss_function=loss_function)
        if loss_function is None:
            raise ValueError("OmniFSDPEngine.train_batch requires a loss_function for direct_preference training.")
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
        metrics: dict[str, list[Any]] = {}
        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
            loss, micro_metrics = self._forward_backward_preference_micro_batch(micro_batch, loss_function)
            losses.append(loss.item())
            for key, value in micro_metrics.items():
                metrics.setdefault(key, []).append(value)
        grad_norm = self.optimizer_step()
        self.optimizer_zero_grad()
        metrics["grad_norm"] = grad_norm
        return {"model_output": {}, "loss": losses, "metrics": metrics}

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        if not self._is_direct_preference():
            return super().infer_batch(data, loss_function=loss_function)
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

    def _model_module(self):
        return getattr(self.module, "_fsdp_wrapped_module", self.module)

    @contextmanager
    def disable_adapter(self):
        module = self._model_module()
        if not hasattr(module, "disable_adapters"):
            yield
            return
        module.disable_adapters()
        try:
            yield
        finally:
            module.enable_adapters()

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def lr_scheduler_step(self):
        if self.lr_scheduler is None:
            return None
        return super().lr_scheduler_step()

    def get_per_tensor_param(
        self,
        layered_summon=False,
        base_sync_done=False,
        adapter_name: str | None = None,
        **kwargs,
    ):
        from verl.utils.device import get_device_id

        load_fsdp_model_to_gpu(self.module)
        peft_config = None
        peft_model = self._model_module()
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            adapter_ctx = self.use_adapter(adapter_name) if adapter_name is not None else nullcontext()
            with adapter_ctx:
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                    is_diffusers=False,
                    adapter_name=adapter_name or "default",
                    layer_prefixes=self.model_config.fsdp_layer_prefixes,
                )
        else:
            params = self.module.state_dict()
        params = convert_weight_keys(params, peft_model)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        device = get_device_id()
        export_dtype = PrecisionType.to_dtype(self.engine_config.model_dtype)

        def param_generator():
            for name, param in params.items():
                tensor = param.full_tensor() if isinstance(param, DTensor) else param
                tensor = tensor.to(device, non_blocking=True)
                if tensor.is_floating_point() and export_dtype is not None and tensor.dtype != export_dtype:
                    tensor = tensor.to(export_dtype, non_blocking=True)
                yield name, tensor

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return param_generator(), peft_config_dict
