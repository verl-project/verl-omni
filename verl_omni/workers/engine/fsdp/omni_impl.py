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
        """FSDP2 build that keeps adapter-declared frozen towers unsharded.

        Copied from verl@fefb0802 ``FSDPEngine._build_fsdp_module`` (fsdp2
        branch) plus the wrap-target loop of ``verl.utils.fsdp_utils.apply_fsdp2``,
        so the root ``fully_shard`` call can pass torch's formal
        ``ignored_params`` for submodules that must not be FSDP-managed at all
        (MiniCPM-o's frozen Whisper ``apm``). Adapters declare them via a
        ``get_fsdp_ignored_module_names`` classmethod; every adapter without
        that method keeps verl's untouched behavior via ``super()``.

        Maintenance note: re-diff this override against the verl pin at every
        bump (upstream actively edits the copied code), and delete it once verl
        ships an FSDP2 ignore config key — being coordinated with wtomin in
        verl-omni#550. fsdp1-with-ignored-names fails closed; the FSDP1
        state-dict tail of the parent method is fsdp1-only and intentionally
        not copied.
        """
        ignored_names: list[str] = []
        adapter_cls = getattr(self, "model_adapter_cls", None)
        ignored_fn = getattr(adapter_cls, "get_fsdp_ignored_module_names", None)
        if callable(ignored_fn):
            ignored_names = list(ignored_fn(self.model_config))
        if not ignored_names:
            return super()._build_fsdp_module(module)
        if self.engine_config.strategy != "fsdp2":
            raise NotImplementedError(
                f"{type(self).__name__}: FSDP2-ignored module names require strategy=fsdp2, "
                f"got {self.engine_config.strategy!r}."
            )

        from torch.distributed.fsdp import FSDPModule, fully_shard
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        from verl.utils.activation_offload import enable_activation_offloading
        from verl.utils.fsdp_utils import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            _select_fsdp2_wrap_targets,
            fsdp2_load_full_state_dict,
            maybe_patch_fsdp_module,
        )
        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
        self._autocast_dtype = param_dtype
        if param_dtype == torch.float16:
            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = CPUOffloadPolicy(pin_memory=True)
            self._uses_fsdp2_cpu_offload_policy = True
        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "reshard_after_forward": self.engine_config.reshard_after_forward,
        }

        # ``apm`` must also cover PEFT-prefixed paths like base_model.model.apm.
        ignored_params = {
            param for name, param in module.named_parameters() if any(part in ignored_names for part in name.split("."))
        }

        transformer_cls_names = self.engine_config.get("wrap_policy", {}).get(
            "transformer_layer_cls_to_wrap", getattr(module, "_no_split_modules", None)
        )
        if isinstance(transformer_cls_names, str):
            transformer_cls_names = [transformer_cls_names]
        if isinstance(transformer_cls_names, set):
            transformer_cls_names = list(transformer_cls_names)
        assert transformer_cls_names, (
            "FSDP2 wrapping found no transformer_layer_cls_to_wrap (checked wrap_policy and module._no_split_modules)."
        )

        full_state = module.state_dict()
        # Nested fully_shard calls must NOT carry the root's ignored set (torch
        # rejects params a nested call does not own); only the root call below
        # passes ignored_params.
        wrap_targets = _select_fsdp2_wrap_targets(module, transformer_cls_names)
        for target in wrap_targets:
            with maybe_patch_fsdp_module(target):
                fully_shard(target, **fsdp_kwargs)
        with maybe_patch_fsdp_module(module):
            fully_shard(module, ignored_params=ignored_params, **fsdp_kwargs)
        fsdp2_load_full_state_dict(module, full_state, self.device_mesh, offload_policy)

        if self.engine_config.get("forward_prefetch", False):
            fsdp_modules = [m for m in wrap_targets if isinstance(m, FSDPModule)]
            for i, fsdp_module in enumerate(fsdp_modules):
                next_targets = fsdp_modules[i + 1 : i + 2]
                if next_targets and hasattr(fsdp_module, "set_modules_to_forward_prefetch"):
                    fsdp_module.set_modules_to_forward_prefetch(next_targets)

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        return module
