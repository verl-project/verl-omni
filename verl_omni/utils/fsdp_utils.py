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
"""
FSDP utilities for verl-omni
"""

from collections import OrderedDict

from verl.utils.fsdp_utils import (  # noqa: F401
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.fsdp_utils import collect_lora_params as _upstream_collect_lora_params


def _layered_summon_lora_params_diffusers(fsdp_module) -> OrderedDict:
    """Layered LoRA param collection for diffusers transformer-block models."""
    from peft.utils.save_and_load import get_peft_model_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.device import get_torch_device

    def _prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        # FSDP1
        "_fsdp_wrapped_module.transformer_blocks.",
        # FSDP2
        "transformer_blocks.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in _prefix_submodules(fsdp_module, prefix):
            block_prefix = name.replace("_fsdp_wrapped_module.", "")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{block_prefix}.{param_name}": (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                        for param_name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()
    return lora_params


def collect_lora_params(
    module,
    layered_summon: bool,
    base_sync_done: bool,
    is_diffusers: bool = False,
) -> OrderedDict:
    """Extended version of ``verl.utils.fsdp_utils.collect_lora_params``."""
    if is_diffusers and layered_summon and fsdp_version(module) > 0:
        if not base_sync_done:
            raise ValueError(
                "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                "rollout.load_format=safetensors"
            )
        return _layered_summon_lora_params_diffusers(module)
    return _upstream_collect_lora_params(module, layered_summon=layered_summon, base_sync_done=base_sync_done)
