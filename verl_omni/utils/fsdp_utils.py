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

import logging
from collections import OrderedDict

from verl.utils.fsdp_utils import collect_lora_params as _upstream_collect_lora_params
from verl.utils.fsdp_utils import fsdp_version

logger = logging.getLogger(__name__)

__all__ = ["collect_lora_params", "get_rollout_weight_prefix"]


def get_rollout_weight_prefix(architecture: str | None) -> str:
    """Checkpoint key prefix expected by the colocated vLLM-Omni rollout worker.

    Qwen-Image and other diffusers pipelines expose ``pipeline.transformer.*``.
    BAGEL nests the trainable MoT stack under ``pipeline.transformer``
    (``self.transformer = self.language_model.model`` in BagelPipeline).
    The rollout-side LoRA manager iterates ``self.pipeline.transformer``
    and constructs module names as ``transformer.<path>``, so the prefix
    must be ``transformer.`` for both architectures.
    """
    # All diffusers-based pipelines in vllm-omni expose the trainable
    # transformer under ``pipeline.transformer``. The LoRA manager's
    # ``_replace_layers_with_lora`` iterates ``pipeline.transformer``
    # and builds full module names with the ``transformer.`` prefix.
    return "transformer."


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
        # Qwen-Image and other diffusers DiT models (FSDP1 / FSDP2)
        "_fsdp_wrapped_module.transformer_blocks.",
        "transformer_blocks.",
        # BAGEL MoT (BagelForTraining) — uses ``layers.N`` not ``transformer_blocks.N``
        "_fsdp_wrapped_module.base_model.model.layers.",
        "base_model.model.layers.",
        "_fsdp_wrapped_module.layers.",
        "layers.",
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
        lora_params = _layered_summon_lora_params_diffusers(module)
        if not lora_params:
            # Typical when FSDP only wrapped leaf LoRA modules (no per-layer FSDP).
            # Fall back to a full summon so rollout still receives adapter weights.
            logger.warning(
                "layered_summon collected 0 LoRA tensors; falling back to full summon. "
                "For BAGEL, set BagelForTraining._no_split_modules = ['BagelMoTLayer'] "
                "so FSDP wraps layers.N (required for efficient layered_summon)."
            )
            return _upstream_collect_lora_params(module, layered_summon=False, base_sync_done=base_sync_done)
        return lora_params
    return _upstream_collect_lora_params(module, layered_summon=layered_summon, base_sync_done=base_sync_done)
