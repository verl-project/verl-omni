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
from typing import Optional

from omegaconf import DictConfig
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.workers.config import DistillationConfig

from verl_omni.workers.engine_workers import ActorRolloutRefWorker


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
