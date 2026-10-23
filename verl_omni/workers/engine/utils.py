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

"""Shared helpers for verl-omni training engines."""

from __future__ import annotations

import logging
import types

import torch
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.pipelines.model_base import configure_composite_ar_module
from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)


def patch_composite_ar_engine_module_build(
    ar_engine: FSDPEngineWithLMHead,
    diffusion_model_config: DiffusionModelConfig,
    strategy: str = "fsdp2",
) -> None:
    """Monkey patch composite AR engine module build methods."""
    ar_engine._verl_omni_diffusion_model_config = diffusion_model_config
    ar_engine._original_build_module = ar_engine._build_module

    def _build_module(self):
        """Pipeline-specific AR setup goes through ``configure_composite_ar_module``.
        The DiT adapter is not consulted: this module is verl's language-model engine.
        """
        module = self._original_build_module()
        # pipeline specific AR setup
        configure_composite_ar_module(module, self._verl_omni_diffusion_model_config)
        return module

    def _build_fsdp_module(self, module):
        """Use ``DiffusersFSDPEngine._build_fsdp_module`` for the composite AR backend.

        verl's ``FSDPEngine._build_fsdp_module`` (``transformer_impl.py``) calls upstream
        ``verl.utils.fsdp_utils.apply_fsdp2`` without ``ignored_names``. verl-omni's diffusers
        engine passes adapter-declared subtrees into ``verl_omni.utils.fsdp_utils.apply_fsdp2``
        so frozen towers skipped on some micro-batches stay unsharded under FSDP2.
        """
        from verl.utils.activation_offload import enable_activation_offloading
        from verl.utils.torch_dtypes import PrecisionType

        from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine

        hf_model_config = self.model_config
        diffusion_cfg = self._verl_omni_diffusion_model_config

        # verl diff part
        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
        else:
            param_dtype = torch.bfloat16

        self._autocast_dtype = param_dtype
        if param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # apply verl-omni _build_fsdp_module
        # temporarily sets self.model_config to DiffusionModelConfig
        # so DiffusionModelBase.get_class(...).get_fsdp_ignored_module_names(...) resolves.
        saved_model_config = self.model_config
        self.model_config = diffusion_cfg
        try:
            module = DiffusersFSDPEngine._build_fsdp_module(self, module)
        finally:
            self.model_config = saved_model_config  # restore model config

        # verl diff part:
        if hf_model_config.enable_activation_offload:
            enable_activation_offloading(
                module,
                self.engine_config.strategy,
                hf_model_config.enable_gradient_checkpointing,
            )
        return module

    ar_engine._build_module = types.MethodType(_build_module, ar_engine)
    print("Patched {}._build_module to dispatch configure_composite_ar_module".format(ar_engine.__class__.__name__))

    if strategy == "fsdp2":
        ar_engine._build_fsdp_module = types.MethodType(_build_fsdp_module, ar_engine)
        print("Patched {}._build_fsdp_module to pass ignored_names".format(ar_engine.__class__.__name__))
