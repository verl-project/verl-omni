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

import torch

from verl_omni.workers.config import DiffusionModelConfig

# veomni.lora implements Kaiming-uniform A / zero B only; other PEFT spellings are ignored.
_VEOMNI_SUPPORTED_LORA_INIT = ("true", "kaiming")


def _validate_veomni_lora_support(model_config: DiffusionModelConfig) -> None:
    """Reject LoRA settings that ``veomni.lora`` would silently ignore."""
    if model_config.lora.get("merge", False):
        raise NotImplementedError(
            "VeOmni diffusion backend does not support model.lora.merge=True yet; "
            "use adapter-only sync (model.lora.merge=False)."
        )

    policy_state_adapters = tuple(model_config.policy_state_adapters or ())
    # "reference" is a logical state served by disable_adapter, not a second adapter.
    if any(adapter not in ("default", "reference") for adapter in policy_state_adapters):
        raise NotImplementedError(
            "VeOmni diffusion backend supports the 'default' adapter and the logical "
            "'reference' state only "
            f"(veomni.lora.VeOmniLoraModel has no add_adapter/set_adapter); got "
            f"policy_state_adapters={policy_state_adapters}. Old/EMA-policy algorithms "
            "(e.g. diffusion_nft with rollout.rollout_adapter=old) require "
            "actor_rollout_ref.actor.strategy=fsdp2."
        )

    if model_config.lora_dtype is not None:
        raise NotImplementedError(
            f"VeOmni diffusion backend does not support model.lora_dtype "
            f"(got {model_config.lora_dtype!r}); veomni.lora creates adapters in the "
            "base-weight dtype. Drop lora_dtype or use actor_rollout_ref.actor.strategy=fsdp2."
        )

    if model_config.target_parameters:
        raise NotImplementedError(
            f"VeOmni diffusion backend does not support model.target_parameters "
            f"(got {model_config.target_parameters!r}); those adapters would be silently "
            "omitted. Use actor_rollout_ref.actor.strategy=fsdp2 for nn.Parameter LoRA."
        )

    init_weights = model_config.lora_init_weights
    # A loaded adapter overwrites the initial weights, so the init scheme is irrelevant.
    if model_config.lora_adapter_path is None and str(init_weights).strip().lower() not in _VEOMNI_SUPPORTED_LORA_INIT:
        raise NotImplementedError(
            "VeOmni diffusion backend initializes LoRA with Kaiming-uniform A and zero B; "
            f"model.lora_init_weights={init_weights!r} cannot be honored and would be "
            "silently ignored. Set actor_rollout_ref.model.lora_init_weights=true to "
            "acknowledge Kaiming initialization, or use "
            f"actor_rollout_ref.actor.strategy=fsdp2 for PEFT's {init_weights!r}."
        )


def _reject_veomni_moe_expert_lora(model: torch.nn.Module) -> None:
    """Reject MoE expert LoRA, which ``disable_adapter`` cannot bypass for the reference policy."""
    from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts

    if any(isinstance(module, LoraIndependentExperts | LoraSharedExperts) for module in model.modules()):
        raise NotImplementedError(
            "VeOmni diffusion backend supports dense LoRA only, not MoE expert LoRA; "
            "the reference policy would reuse the actor's expert adapters."
        )
