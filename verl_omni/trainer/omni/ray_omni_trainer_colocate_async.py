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
"""Omni colocate-async trainer — a thin ``PPOTrainerColocateAsync`` subclass (RFC #320)."""

import logging
import os

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.workers.config import OmniModelConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("omni_colocate_async")
class OmniPPOTrainerColocateAsync(PPOTrainerColocateAsync):
    """``PPOTrainerColocateAsync`` subclass that wires tokenizer/processor
    from ``OmniModelConfig``. Warmup and all async lifecycle hooks are
    inherited from verl and read the generic ``trainer.v1.colocate_async``
    config key (#320 §3.6).
    """

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor
