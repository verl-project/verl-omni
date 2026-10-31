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
from contextlib import nullcontext
from functools import wraps
from typing import Optional

import torch
from omegaconf import DictConfig
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.config import DistillationConfig

from verl_omni.workers.engine_workers import ActorRolloutRefWorker
from verl_omni.workers.utils.parameter_revision import parameter_access

__all__ = ["OmniDetachActorWorker"]


class OmniDetachActorWorker(ActorRolloutRefWorker, DetachActorWorker):
    """``DetachActorWorker`` routed through verl-omni's ``ActorRolloutRefWorker``.

    The omni worker comes first in the MRO so its LoRA-aware weight sync
    (adapter-only send, ``get_lora_peft_config``) wins over the upstream
    methods; ``DetachActorWorker`` contributes the CPU save/restore used by
    decoupled PPO.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        ActorRolloutRefWorker.__init__(self, config, role, distillation_config=distillation_config, **kwargs)
        self._strategy_handlers = None

    def _get_strategy_handlers(self):
        if self._strategy_handlers is None:
            copy_handler, restore_handler = super()._get_strategy_handlers()
            # verl's fsdp2 sharded save returns storage-sharing views of CPU-resident
            # (param_offload) parameters; the decoupled-PPO dance keeps several
            # snapshots live at once, so they must own their storage. The fsdp1 and
            # megatron save helpers already copy.
            if self.config.actor.strategy in ("fsdp2", "veomni"):
                copy_handler = _save_owning_storage(copy_handler)
            self._strategy_handlers = (copy_handler, restore_handler)
        return self._strategy_handlers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_version(self):
        """Explicitly enable tracking and return this worker's current version."""
        from verl_omni.workers.engine.megatron.omni_impl import OmniMegatronEngine

        if type(self.actor.engine) is not OmniMegatronEngine:
            raise ValueError("Parameter tracking requires the Omni Megatron engine")
        return self.actor.engine.enable_parameter_tracking().version()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @parameter_access("actor.engine")
    def save_model_to_cpu(self, n):
        return super().save_model_to_cpu(n)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        tracker = getattr(self.actor.engine, "_parameter_revision", None)
        with tracker.mutation("restore") if tracker is not None else nullcontext():
            if tracker is not None and n not in getattr(self, "cpu_saved_models", {}):
                raise ValueError("The requested CPU snapshot does not exist")
            return super().restore_model_from_cpu(n)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @parameter_access("actor.engine")
    def clear_cpu_model(self, n):
        return super().clear_cpu_model(n)


def _owning_cpu_state(value):
    """Rebuild a save-handler payload so every tensor owns its CPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _owning_cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(_owning_cpu_state(item) for item in value)
    return value


def _save_owning_storage(save_handler):
    @wraps(save_handler)
    def wrapper(model):
        return _owning_cpu_state(save_handler(model))

    return wrapper
