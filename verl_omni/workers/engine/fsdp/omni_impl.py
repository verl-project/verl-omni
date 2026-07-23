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

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoModelForMultimodalLM
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
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.utils.fsdp_utils import collect_lora_params
from verl_omni.workers.config import OmniModelConfig

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(FSDPEngineWithLMHead):
    """FSDP engine for omni models"""

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
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
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

        # DIFF vs upstream: delegate model-specific LoRA key normalization
        # to the registered OmniModelBase adapter.
        if peft_config is not None and base_sync_done:
            adapter = kwargs.get("adapter_name", "default")
            from verl_omni.pipelines.model_base import OmniModelBase

            adapter_cls = OmniModelBase.get_class(self.model_config)
            per_tensor_param = [
                (
                    adapter_cls.normalize_lora_key(
                        name.replace("_fsdp_wrapped_module.", "")
                        .replace(f"lora_A.{adapter}.weight", "lora_A.weight")
                        .replace(f"lora_B.{adapter}.weight", "lora_B.weight"),
                    ),
                    tensor,
                )
                for name, tensor in per_tensor_param
            ]

        return per_tensor_param, peft_config_dict

    def _build_module(self):
        from verl.utils.torch_dtypes import PrecisionType

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
