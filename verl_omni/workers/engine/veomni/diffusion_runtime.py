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

"""VeOmni runtime for a DiT receiving precomputed conditions from rollout."""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from veomni.models.model_runtime import VeOmniModelRuntime

from .lora_utils import _reject_veomni_moe_expert_lora


class VeOmniDiffusionRuntime(VeOmniModelRuntime):
    """Build just the trainable DiT; verl supplies conditioning, scheduling and checkpoints."""

    def __init__(self, *args, forward_only, fp32_modules, optimizer_config, configure_lora=None, **kwargs):
        self.forward_only = forward_only
        self.fp32_modules = fp32_modules
        self.optimizer_config = optimizer_config
        self.configure_lora = configure_lora
        super().__init__(*args, **kwargs)

    def _freeze_model_module(self):
        # Replay only changes trainable parameters when an adapter is requested.
        if self.args.lora_config:
            super()._freeze_model_module()
            _reject_veomni_moe_expert_lora(self.model)
            if self.configure_lora is not None:
                self.configure_lora(self.model.get_base_model())

    def _build_parallelized_model(self):
        if self.fp32_modules:
            base = self.model.get_base_model() if self.args.lora_config else self.model
            for path in self.fp32_modules:
                fully_shard(
                    base.get_submodule(path),
                    mesh=self.parallel_state.fsdp_mesh,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, cast_forward_inputs=True),
                    reshard_after_forward=self.args.accelerator.fsdp_config.reshard_after_forward,
                )
        super()._build_parallelized_model()

    def _build_optimizer(self, param_groups=None):
        if self.forward_only:
            return
        from veomni.optim import build_optimizer

        opt = self.args.optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=opt.lr,
            betas=opt.betas,
            eps=self.optimizer_config.eps,
            weight_decay=opt.weight_decay,
            fused=self.optimizer_config.fused,
            optimizer_type=opt.type,
            param_groups=param_groups,
            no_decay_modules=opt.no_decay_modules,
            no_decay_params=opt.no_decay_params,
            optimizer_config=opt,
        )

    def _build_model_assets(self):
        self.model_assets = [self.model_config]
