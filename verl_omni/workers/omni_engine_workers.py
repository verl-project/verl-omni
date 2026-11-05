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
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import (
    Dispatch,
    collect_all_to_all,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.workers.config import DistillationConfig

from verl_omni.workers.engine_workers import ActorRolloutRefWorker
from verl_omni.workers.utils.cohort_transport import cohort_transport
from verl_omni.workers.utils.parameter_revision import parameter_access
from verl_omni.workers.utils.step_extension_transport import step_extension_transport

__all__ = ["OmniDetachActorWorker"]

_STAGE_DISPATCH = {
    **make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"),
    "collect_fn": collect_all_to_all,
}


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
        self._strategy_handlers: tuple[Any, Any] | None = None

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
    @parameter_access("actor.engine")
    def save_model_to_cpu(self, n):
        return super().save_model_to_cpu(n)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        tracker = getattr(self.actor.engine, "_parameter_revision", None)
        with tracker.access(internal=True) if tracker is not None else nullcontext():
            if tracker is not None and n not in getattr(self, "cpu_saved_models", {}):
                raise ValueError("The requested CPU snapshot does not exist")
            with tracker.mutation("restore") if tracker is not None else nullcontext():
                return super().restore_model_from_cpu(n)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @parameter_access("actor.engine")
    def clear_cpu_model(self, n):
        return super().clear_cpu_model(n)

    @step_extension_transport
    @cohort_transport
    @register(dispatch_mode=_STAGE_DISPATCH, blocking=False)
    def stage_step_extension(self, data):
        """Dispatch an extension stage over the actor mesh."""
        from verl.utils import tensordict_utils as tu

        from verl_omni.pipelines.step_extension import SELECTOR_KEY, invoke_worker_extension

        selector = tu.get_non_tensor_data(data, SELECTOR_KEY, default=None)
        if not isinstance(selector, tuple) or len(selector) != 2:
            raise ValueError("Invalid worker extension selector")
        return invoke_worker_extension(selector[0], "stage", selector[1], self, data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def control_step_extension(self, name, operation, payload=None):
        """Dispatch a synchronous all-worker extension control operation."""
        from verl_omni.pipelines.step_extension import invoke_worker_extension

        return invoke_worker_extension(name, "control", operation, self, payload)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def commit_step_extension(self, name, operation, payload=None):
        """Dispatch an all-worker extension commit without implicit retries."""
        from verl_omni.pipelines.step_extension import invoke_worker_extension

        return invoke_worker_extension(name, "commit", operation, self, payload)


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
