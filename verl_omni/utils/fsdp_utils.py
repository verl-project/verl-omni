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
from collections.abc import Callable

from peft.utils.save_and_load import get_peft_model_state_dict
from verl.utils.fsdp_utils import collect_lora_params as _upstream_collect_lora_params
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.fsdp_utils import layered_summon_lora_params as _upstream_layered_summon_lora_params

__all__ = ["collect_lora_params"]


def _param_to_cpu(param):
    if hasattr(param, "full_tensor"):
        return param.full_tensor().detach().cpu()
    return param.detach().cpu()


def _peft_lora_params_to_cpu(peft_model, adapter_name: str) -> OrderedDict:
    lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
    return OrderedDict((name, _param_to_cpu(param)) for name, param in lora_params.items())


def _collect_base_weights_to_cpu(peft_model) -> OrderedDict:
    from verl.utils.device import get_device_name

    model = peft_model.base_model.model
    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
    model = model.to("cpu")
    lora_params = OrderedDict()
    for name, param in model.state_dict().items():
        if any(x in name for x in ["_flat_param", "lora_"]):
            continue
        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
        lora_params[name] = _param_to_cpu(param)
    model = model.to(orig_dev)
    return lora_params


def _collect_lora_params_with_adapter(
    module,
    layered_summon: bool,
    base_sync_done: bool,
    adapter_name: str,
    layered_summon_fn: Callable,
) -> OrderedDict:
    """Verl-style LoRA collection with explicit ``adapter_name`` for PEFT state."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.device import get_torch_device

    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            if layered_summon_fn is _upstream_layered_summon_lora_params:
                return layered_summon_fn(module)
            return layered_summon_fn(module, adapter_name=adapter_name)
        with FSDP.summon_full_params(module, writeback=False):
            if base_sync_done:
                lora_params = _peft_lora_params_to_cpu(peft_model, adapter_name)
            else:
                lora_params = _collect_base_weights_to_cpu(peft_model)
        get_torch_device().empty_cache()
        return lora_params

    if base_sync_done:
        return _peft_lora_params_to_cpu(peft_model, adapter_name)
    return _collect_base_weights_to_cpu(peft_model)


def _layered_summon_lora_params_diffusers(fsdp_module, adapter_name: str = "default") -> OrderedDict:
    """Layered LoRA param collection for diffusers transformer-block models."""
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
                    sub_lora_params = get_peft_model_state_dict(
                        peft_model, state_dict=submodule.state_dict(), adapter_name=adapter_name
                    )
                    sub_lora_params = {
                        f"{block_prefix}.{param_name}": _param_to_cpu(param)
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
    adapter_name: str = "default",
) -> OrderedDict:
    """Extended version of ``verl.utils.fsdp_utils.collect_lora_params``."""
    use_diffusers_layered = is_diffusers and layered_summon and fsdp_version(module) > 0
    if adapter_name == "default" and not use_diffusers_layered:
        return _upstream_collect_lora_params(module, layered_summon=layered_summon, base_sync_done=base_sync_done)

    layered_summon_fn = _layered_summon_lora_params_diffusers if is_diffusers else _upstream_layered_summon_lora_params
    return _collect_lora_params_with_adapter(
        module,
        layered_summon=layered_summon,
        base_sync_done=base_sync_done,
        adapter_name=adapter_name,
        layered_summon_fn=layered_summon_fn,
    )
