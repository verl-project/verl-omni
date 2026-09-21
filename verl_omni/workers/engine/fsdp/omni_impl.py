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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor
from transformers import AutoModelForMultimodalLM
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.utils import get_sharding_strategy

from verl_omni.utils.fsdp_utils import apply_fsdp2, collect_lora_params
from verl_omni.workers.config import OmniModelConfig

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(FSDPEngineWithLMHead):
    """FSDP engine for omni models"""

    @staticmethod
    def _cast_dtensor_weight_for_sync(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            return tensor.to(dtype=torch.bfloat16, non_blocking=True)
        return tensor

    def prepare_model_inputs(self, micro_batch):
        """Prepare standard LM inputs, then add model-native replay fields."""
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        if not hasattr(self, "model_adapter_cls"):
            raise RuntimeError("Omni model inputs cannot be prepared before the model adapter is initialized.")
        model_inputs = self.model_adapter_cls.prepare_model_inputs(model_inputs, micro_batch, self.model_config)
        if not isinstance(model_inputs, dict):
            raise TypeError(
                f"OmniModelBase.prepare_model_inputs must return a dict, got {type(model_inputs).__name__}."
            )
        return model_inputs, output_args

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        # FSDP2 CPUOffloadPolicy owns CPU<->GPU placement; calling model.to(device) here
        # leaves the module half-moved and crashes state_dict() below (verl#5995). The
        # per-DTensor .to(device).full_tensor() below still produces GPU tensors.
        if not self._uses_fsdp2_cpu_offload_policy:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                adapter_name = kwargs.get("adapter_name", "default")
                peft_config = peft_model.peft_config.get(adapter_name, None)
                # DIFF vs upstream: use verl_omni's fixed collect_lora_params
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                    adapter_name=adapter_name,
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
            per_tensor_param = (
                (
                    name,
                    self._cast_dtensor_weight_for_sync(param.to(device, non_blocking=True).full_tensor())
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
                        self._cast_dtensor_weight_for_sync(param.to(device, non_blocking=True).full_tensor())
                        if isinstance(param, DTensor)
                        else param.detach().clone(),
                    )
        finally:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

    def _build_module(self):
        unsupported_options = [
            option for option in ("use_liger", "use_fused_kernels") if getattr(self.model_config, option, False)
        ]
        if unsupported_options:
            enabled_options = ", ".join(f"{option}=True" for option in unsupported_options)
            raise ValueError(
                f"Omni models do not support these enabled optimizations: {enabled_options}. "
                "Set them to false before starting the worker."
            )

        from verl.utils.torch_dtypes import PrecisionType

        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_config: OmniModelConfig
        architecture = self.model_config.architecture
        adapter_cls = OmniModelBase.get_class_by_name(
            architecture,
            self.model_config.model_stage,
            self.model_config.get("external_lib"),
        )
        self.model_adapter_cls = adapter_cls

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

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

            auto_model_cls = getattr(adapter_cls, "auto_model_class", None) or AutoModelForMultimodalLM
            module = auto_model_cls.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            module = adapter_cls.configure_model(module, self.model_config)

            if self.engine_config.strategy == "fsdp" and not self.engine_config.use_orig_params:
                trainability = {parameter.requires_grad for parameter in module.parameters()}
                if len(trainability) > 1:
                    raise ValueError(
                        "FSDP1 requires use_orig_params=true when a model adapter freezes only part of the model."
                    )

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

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        # Faithful copy of verl's FSDPEngine._build_fsdp_module; deltas are DIFF-marked.
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from verl.utils.torch_dtypes import PrecisionType

        # DIFF vs upstream: adapters declare frozen subtrees to leave unsharded (fsdp2 only)
        ignored_names = list(self.model_adapter_cls.get_fsdp_ignored_module_names(self.model_config))
        if ignored_names and self.engine_config.strategy != "fsdp2":
            raise NotImplementedError(
                f"{type(self).__name__}: FSDP2-ignored module names require strategy=fsdp2, "
                f"got {self.engine_config.strategy!r}."
            )

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        self._autocast_dtype = param_dtype
        # fp16 training requires loss scaling to avoid gradient underflow. Mirror the pattern
        # landed in #4036 for the legacy dp_actor path. bf16 / fp32 do not need a scaler.
        if param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh, zero3_enable=self.engine_config.reshard_after_forward)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            # - actor: offload_policy
            # - critic: offload_policy
            # - ref: CPUOffloadPolicy(pin_memory=True)
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)
                self._uses_fsdp2_cpu_offload_policy = True

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            # DIFF vs upstream: our apply_fsdp2 takes the ignored subtrees
            apply_fsdp2(module, fsdp_kwargs, self.engine_config, ignored_names=ignored_names)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module
