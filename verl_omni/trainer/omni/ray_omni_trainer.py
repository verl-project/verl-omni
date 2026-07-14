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
"""V1 trainer for omni models — extends verl's PPOTrainer with omni hooks.

The omni trainer subclass replaces the monkey-patched ``hf_processor`` /
``hf_tokenizer`` calls in ``PPOTrainer._init_tokenizer`` by reusing the
tokenizer and processor already loaded by ``OmniModelConfig.__post_init__``
(via ``OmniModelBase`` adapter methods). All V1 infrastructure
(TransferQueue, ReplayBuffer, lifecycle hooks) is inherited unchanged.
"""

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@register_trainer("omni_sync")
class OmniPPOTrainerSync(PPOTrainerSync):
    """V1 sync trainer with omni model adapter hooks.

    Differences from stock ``PPOTrainerSync``:
    - Reuses processor/tokenizer loaded by ``OmniModelConfig.__post_init__``
      via ``OmniModelBase`` adapters (replaces monkey-patched
      ``hf_processor`` / ``hf_tokenizer``).
    - All V1 infrastructure (TransferQueue, ReplayBuffer, lifecycle hooks)
      is inherited unchanged.

    Selected via config::

        trainer:
          v1:
            trainer_mode: omni_sync

    Other omni trainer variants (e.g. ``omni_colocate_async``) can be
    added later by following the same pattern and inheriting from the
    corresponding verl trainer variant.
    """

    def _init_tokenizer(self):
        """Wire the trainer's tokenizer and processor from the model config.

        Tokenizer and processor are already loaded by
        ``OmniModelConfig.__post_init__`` via ``OmniModelBase`` adapter
        methods during Hydra config initialization.  We simply assign
        them to the trainer instance, where the parent's ``_setup()``
        consumes them for dataloader creation.

        This replaces the stock ``PPOTrainer._init_tokenizer`` which
        would call ``hf_processor`` / ``hf_tokenizer`` directly — a
        second load we avoid here.
        """
        model_config = self.config.actor_rollout_ref.model
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        model_config.trust_remote_code = trust_remote_code

        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor
