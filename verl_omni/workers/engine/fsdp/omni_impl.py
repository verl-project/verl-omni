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
"""FSDP engine for omni models, registered as ``model_type="omni"``."""

import warnings

import torch
from transformers import AutoModelForMultimodalLM
from verl.utils.fsdp_utils import get_init_weight_context_manager
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.workers.config import OmniModelConfig


@EngineRegistry.register(model_type="omni", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(FSDPEngineWithLMHead):
    """FSDP engine for omni models"""

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        """Like parent, but normalises LoRA weight keys for vLLM-Omni consumption."""
        per_tensor_param, peft_config = super().get_per_tensor_param(
            layered_summon=layered_summon, base_sync_done=base_sync_done, **kwargs
        )
        if peft_config is not None and base_sync_done:
            adapter = kwargs.get("adapter_name", "default")
            per_tensor_param = (
                (
                    name.replace("_fsdp_wrapped_module.", "")
                    .replace(f"lora_A.{adapter}.weight", "lora_A.weight")
                    .replace(f"lora_B.{adapter}.weight", "lora_B.weight"),
                    tensor,
                )
                for name, tensor in per_tensor_param
            )
        return per_tensor_param, peft_config

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
