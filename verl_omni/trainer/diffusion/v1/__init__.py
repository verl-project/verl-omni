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
"""verl-omni diffusion v1 trainer package.

Reuses upstream verl v1 infrastructure (ReplayBuffer, TransferQueue,
LLMServerManager, CheckpointEngineManager) while keeping the diffusion
``DataProto`` compute contract. Only the ``sync`` mode is implemented in this
draft; ``colocate_async`` and ``separate_async`` are intentionally omitted.
"""

from verl_omni.trainer.diffusion.v1.trainer_base import (
    DIFFUSION_TRAINER_REGISTRY,
    PolicyGradientDiffusionTrainerV1,
    get_diffusion_trainer_cls,
    register_diffusion_trainer,
)
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.trainer.diffusion.v1.tq_utils import (
    diffusion_tq_batch_to_dataproto,
    put_dataproto_fields_to_tq,
    sort_diffusion_tq_keys,
)

__all__ = [
    "DIFFUSION_TRAINER_REGISTRY",
    "PolicyGradientDiffusionTrainerV1",
    "PolicyGradientDiffusionTrainerV1Sync",
    "get_diffusion_trainer_cls",
    "register_diffusion_trainer",
    "diffusion_tq_batch_to_dataproto",
    "put_dataproto_fields_to_tq",
    "sort_diffusion_tq_keys",
]
