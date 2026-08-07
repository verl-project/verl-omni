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
"""Omni colocate-async trainer — a PPOTrainerColocateAsync subclass.

Registered via ``@register_trainer("omni_colocate_async")``. Overrides
``_init_tokenizer`` to wire the omni tokenizer/processor from
``OmniModelConfig`` (identical to ``OmniPPOTrainerSync``), and overrides
``on_train_begin`` to read warmup batches from the ``omni_colocate_async``
config key (consistent with the ``trainer_mode`` name). All other async
lifecycle hooks (abort/sleep/resume, weight sync) are inherited from
``PPOTrainerColocateAsync``.
"""

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
    from ``OmniModelConfig`` and reads warmup batches from the
    ``omni_colocate_async`` config key.
    """

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.omni_colocate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")
