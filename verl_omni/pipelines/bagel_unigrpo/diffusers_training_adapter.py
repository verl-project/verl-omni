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

"""BAGEL (MoT) training-side adapter for UniGRPO (joint AR-thinking + image).

Registered as ``OmniBagelForConditionalGeneration`` / ``unigrpo`` in the
``DiffusionModelBase`` registry. It reuses ``bagel_flow_grpo``'s ``BagelDiffusion``
for the image (generation) branch -- scheduler, CFG, ``prepare_model_inputs`` and
``forward_and_sample_previous_step`` are inherited unchanged -- and only changes
what UniGRPO needs:

* ``build_module`` loads ``BagelForSFT`` (not ``BagelForTraining``) so the
  understanding (und) pathway is present for the AR-thinking rollout/replay.
* ``configure_trainable_params`` trains BOTH the und and the ``moe_gen`` experts
  (the whole MoT transformer); only the understanding-vision stack is frozen (the
  VAE is already frozen by ``BagelFrozenVAEEncoder``). The image-only flow_grpo hook
  freezes everything but ``moe_gen``, which would disable the AR-GRPO backward.

The joint 2-backwards->1-step update, the AR-thinking decode and the velocity-MSE
reference live in ``BagelUniGRPOHooks`` / ``UniGRPOJointUpdater``.
"""

from __future__ import annotations

import logging

import torch
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from ..bagel_flow_grpo.diffusers_training_adapter import BagelDiffusion

logger = logging.getLogger(__name__)

# Understanding-vision submodules unused by the reasoning->image recipe; kept frozen.
_FROZEN_VISION_SUBMODULES = ("vit_model", "connector", "vit_pos_embed")


@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="unigrpo")
class BagelUniGRPO(BagelDiffusion):
    """Joint AR-thinking + image UniGRPO adapter over one ``BagelForSFT`` (MoT)."""

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        from ..bagel_flow_grpo.bagel_sft_model import BagelForSFT

        logger.info("Loading BagelForSFT (UniGRPO) from %s", model_config.local_path)
        module = BagelForSFT.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)
        if hasattr(module, "enable_gradient_checkpointing"):
            module.enable_gradient_checkpointing()
        return module

    @classmethod
    def configure_trainable_params(cls, module, model_config: DiffusionModelConfig):
        """Train the whole MoT transformer (und + moe_gen); freeze only the vision stack.

        The SigLIP tower / connector / vit position embedding are unused by the
        reasoning->image recipe and are frozen; the VAE is frozen inside
        ``BagelFrozenVAEEncoder``. Every remaining (transformer) parameter stays
        trainable and is cast to fp32 for the master weights, mirroring flow_grpo.
        """
        inner = module.module if hasattr(module, "module") else module
        for name in _FROZEN_VISION_SUBMODULES:
            sub = getattr(inner, name, None)
            if sub is not None:
                for param in sub.parameters():
                    param.requires_grad_(False)
        for param in module.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    @classmethod
    def fsdp2_sharding_units(cls, module):
        """Shard every transformer block and each leaf reached outside root forward."""
        inner = getattr(module, "module", module)
        leaves = ("embed_tokens", "lm_head", "norm", "norm_moe_gen", "time_embedder", "vae2llm", "llm2vae")
        return list(inner.layers) + [getattr(inner, name) for name in leaves if getattr(inner, name, None) is not None]

    @classmethod
    def build_engine_hooks(cls, module, model_config, optimizer_config):
        """Attach joint backward and native sampling to the shared PPO engine."""
        from .hooks import BagelUniGRPOHooks

        return BagelUniGRPOHooks(module, model_config, optimizer_config)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler


__all__ = ["BagelUniGRPO"]
