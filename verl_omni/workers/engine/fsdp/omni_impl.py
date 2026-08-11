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
"""FSDP engine for omni models, registered as ``model_type="omni_model"``."""

import logging
import warnings
from contextlib import nullcontext
from typing import Callable

import torch
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from transformers import AutoModelForMultimodalLM
from verl.utils import tensordict_utils as tu
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    get_init_weight_context_manager,
    load_fsdp_model_to_gpu,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import EngineTrainModeCtx, FSDPEngineWithLMHead
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

from verl_omni.utils.fsdp_utils import collect_lora_params
from verl_omni.workers.config import OmniModelConfig

logger = logging.getLogger(__name__)


class OmniTrainModeCtx(EngineTrainModeCtx):
    """Train-mode context with model-adapter hooks."""

    def __enter__(self):
        super().__enter__()
        from verl_omni.pipelines.model_base import OmniModelBase

        OmniModelBase.get_class(self.engine.model_config).configure_train_mode(self.engine.module)


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(FSDPEngineWithLMHead):
    """FSDP engine for omni models"""

    def train_mode(self, **kwargs):
        return OmniTrainModeCtx(self, **kwargs)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        # FSDP2 CPUOffloadPolicy owns CPU<->GPU placement; calling model.to(device) here
        # leaves the module half-moved and crashes state_dict() below (#5995). The
        # per-DTensor .to(device).full_tensor() below still produces GPU tensors.
        if not self._uses_fsdp2_cpu_offload_policy:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                # DIFF vs upstream: use verl_omni's fixed collect_lora_params
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                return self._merged_lora_per_tensor_param(), None
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=torch.device(get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None

        return per_tensor_param, peft_config_dict

    def _merged_lora_per_tensor_param(self):
        """Stream materialized merged weights before restoring the actor."""
        device = get_device_id()
        try:
            with merged_lora_context(self.module, backup_adapters=True):
                params = normalize_peft_param_name(self.module.state_dict())
                params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
                for name, param in params.items():
                    yield (
                        name,
                        param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                        if isinstance(param, DTensor)
                        else param.detach().clone(),
                    )
        finally:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

    def _build_module(self):
        from verl.utils.torch_dtypes import PrecisionType

        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_config: OmniModelConfig
        architecture = self.model_config.architecture

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        adapter_cls = OmniModelBase.get_class_by_name(
            architecture,
            self.model_config.model_stage,
            self.model_config.get("external_lib"),
        )

        module = adapter_cls.build_module(self.model_config, torch_dtype)
        if module is not None:
            if getattr(self.model_config, "enable_gradient_checkpointing", False) and hasattr(
                module, "gradient_checkpointing_enable"
            ):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return module.to(torch_dtype)

        # Use the stage sub-config for the meta-tensor decision; fall back to the umbrella config.
        stage_config = getattr(
            self.model_config.hf_config, f"{self.model_config.model_stage}_config", self.model_config.hf_config
        )
        tie_word_embeddings = getattr(stage_config, "tie_word_embeddings", False)
        if not hasattr(self.model_config.hf_config, "tie_word_embeddings"):
            self.model_config.hf_config.tie_word_embeddings = tie_word_embeddings

        init_context = get_init_weight_context_manager(use_meta_tensor=not tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if getattr(self.model_config, "use_liger", False):
                logger.warning("use_liger is set but not applied for omni models; this is a no-op.")
            if getattr(self.model_config, "use_fused_kernels", False):
                logger.warning("use_fused_kernels is set but not applied for omni models; this is a no-op.")

            module = AutoModelForMultimodalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            module = adapter_cls.configure_model(module, self.model_config)

            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module = super()._build_lora_module(module)

        lora_dtype = getattr(self.model_config, "lora_dtype", None)
        if lora_dtype is not None:
            from peft.tuners.tuners_utils import BaseTunerLayer
            from verl.utils.torch_dtypes import PrecisionType

            target_dtype = PrecisionType.to_dtype(lora_dtype)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    orig_dtype = param.dtype
                    param.data = param.data.to(target_dtype)
                    logger.debug("LoRA param %s: %s -> %s", name, orig_dtype, param.dtype)

            for submodule in module.modules():
                if isinstance(submodule, BaseTunerLayer):
                    submodule.cast_input_dtype_enabled = False

        return module


@EngineRegistry.register(model_type="omni_sft_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniSFTFSDPEngine(OmniFSDPEngine):
    """Omni-based FSDP engine for supervised fine-tuning."""

    def forward_backward_batch(
        self, data: TensorDict, loss_function: Callable, forward_only: bool = False
    ) -> list[TensorDict]:
        tu.assign_non_tensor(data, use_dynamic_bsz=False)
        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        output_lst = []
        gradient_accumulation_steps = len(micro_batches)
        ctx = torch.no_grad() if forward_only else nullcontext()
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(
                {
                    "loss": meta_info["loss"],
                    "metrics": meta_info["metrics"],
                }
            )

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def prepare_model_inputs(self, micro_batch: TensorDict):
        return {
            "input_ids": micro_batch["input_ids"],
            "attention_mask": micro_batch.get("attention_mask", None),
            "image_hidden_states": micro_batch.get("image_hidden_states", None),
            "timesteps": micro_batch.get("timesteps", None),
            "latent_pos_ids": micro_batch.get("latent_pos_ids", None),
        }

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        del micro_batch
        model_output = {"logits": output.logits}
        if output.image_velocity is not None:
            model_output["image_velocity"] = output.image_velocity
        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raw_output = self.module(**self.prepare_model_inputs(micro_batch=micro_batch))
        model_output = self.prepare_model_outputs(output=raw_output, micro_batch=micro_batch)

        if loss_function is not None:
            data_dict = {"labels": micro_batch["labels"]}
            if micro_batch.get("image_velocity_target", None) is not None:
                data_dict["image_velocity_target"] = micro_batch["image_velocity_target"]
            if micro_batch.get("image_loss_mask", None) is not None:
                data_dict["image_loss_mask"] = micro_batch["image_loss_mask"]
            data = tu.get_tensordict(data_dict)
            tu.assign_non_tensor(
                data,
                gradient_accumulation_steps=tu.get_non_tensor_data(
                    micro_batch, "gradient_accumulation_steps", default=None
                ),
                sp_size=tu.get_non_tensor_data(micro_batch, "sp_size", default=None),
            )
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=model_output["logits"].device)
            metrics = {}

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }
        return loss, output
