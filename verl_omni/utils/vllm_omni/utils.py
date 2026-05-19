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


from typing import cast

import torch
from msgspec import field

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel

from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, logger
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
    DiffusersAdapterPipeline,
)
from vllm_omni.diffusion.registry import initialize_model
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest


class OmniTensorLoRARequest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMOmniHijack:
    """Monkey-patches vllm-omni internals to support in-memory LoRA tensors."""

    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: OmniTensorLoRARequest) -> tuple[LoRAModel, PEFTHelper]:
            """
            based on vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter,
            support load adapter with lora tensors

            Reason:
            VLLM-Omni does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            if not self._expected_lora_modules:
                raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

            logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

            lora_tensors = None

            if isinstance(lora_request, OmniTensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                logger.debug("Resolved LoRA path: %s", lora_path)

                peft_helper = PEFTHelper.from_local_dir(
                    lora_path,
                    max_position_embeddings=None,  # no need in diffusion
                    tensorizer_config_dict=lora_request.tensorizer_config_dict,
                )

            logger.info(
                "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
                peft_helper.r,
                peft_helper.lora_alpha,
                peft_helper.target_modules,
            )

            if isinstance(lora_request, OmniTensorLoRARequest):
                lora_model = LoRAModel.from_lora_tensors(
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",  # consistent w/ vllm's behavior
                    dtype=self.dtype,
                    model_vocab_size=None,
                    weights_mapper=None,
                )
            else:
                lora_model = LoRAModel.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules=self._expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",  # consistent w/ vllm's behavior
                    dtype=self.dtype,
                    model_vocab_size=None,
                    tensorizer_config_dict=lora_request.tensorizer_config_dict,
                    weights_mapper=None,
                )

            logger.info(
                "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
                lora_model.id,
                len(lora_model.loras),
                list(lora_model.loras.keys()),
            )

            for lora in lora_model.loras.values():
                lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

            return lora_model, peft_helper

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        def hijack_load_model(
            self,
            od_config: OmniDiffusionConfig,
            load_device: str,
            load_format: str | None = "default",
            custom_pipeline_name: str | None = None,
            device: torch.device | None = None,
        ):
            """
            based on vllm_omni.diffusion.model_loader.diffusers_loader.DiffusersPipelineLoader.load_model,
            fix sleep on custom_pipeline.

            Reason:
            Custom pipelines call HuggingFace `from_pretrained(...).to(device)` internally. If we construct
            them under `with target_device:` (CUDA), safetensors takes a direct-to-GPU fast path that calls
            `cudaMalloc` via the driver API and BYPASSES PyTorch's caching allocator. That makes those bytes
            invisible to CuMemAllocator, so `sleep()` cannot offload/unmap them and GPU memory stays pinned.

            Fix: build the custom pipeline on CPU first (no default device context), then explicitly move it
            to the target device. The subsequent `.to(target_device)` issues `torch.empty(..., device=cuda)`
            + `copy_`, which goes through the caching allocator and is fully tracked by CuMemAllocator.

            # TODO (long): drop this patch in vllm-omni next version upgrade.
            Ref: https://github.com/knlnguyen1802/vllm-omni/pull/25
            """
            if load_format is None:
                load_format = "default"
            self.od_config = od_config
            # CPU offload + FP8: load weights on device for FP8 quantization
            if load_device == "cpu" and od_config.quantization_config is not None:
                load_device = device.type
                logger.info(f"Quantization enabled with CPU offload, using {load_device} for weight loading")

            target_device = torch.device(load_device)
            with set_default_torch_dtype(od_config.dtype):
                if od_config.parallel_config.use_hsdp:
                    model = self._load_model_with_hsdp(
                        od_config,
                        target_device=device,
                        load_format=load_format,
                        custom_pipeline_name=custom_pipeline_name,
                    )
                else:
                    if load_format == "custom_pipeline":
                        from vllm_omni.diffusion.config import set_current_diffusion_config

                        model_cls = resolve_obj_by_qualname(custom_pipeline_name)
                        with set_current_diffusion_config(od_config):
                            model = model_cls(od_config=od_config)
                        if target_device.type != "cpu":
                            model.to(target_device)
                    else:
                        with target_device:
                            if load_format == "default":
                                model = initialize_model(od_config)
                            elif load_format == "diffusers":
                                model = DiffusersAdapterPipeline(od_config=od_config, device=target_device)
                            else:
                                raise ValueError(f"Unknown load_format: {load_format}")
                logger.debug("Loading weights on %s ...", load_device)
                if load_format == "diffusers":
                    cast(DiffusersAdapterPipeline, model).load_weights()
                elif self._is_gguf_quantization(od_config):
                    self._load_weights_with_gguf(model, od_config)
                else:
                    self.load_weights(model)

                self._process_weights_after_loading(model, target_device)

            return model.eval()

        do_hijack(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)
        do_hijack(DiffusersPipelineLoader, "load_model", hijack_load_model)
