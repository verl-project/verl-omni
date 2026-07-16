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
"""Omni trainer — a PPOTrainerSync subclass registered via ``@register_trainer("omni_sync")``."""

import json
import os

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.fs import copy_to_local

from verl_omni.pipelines.model_base import OmniModelBase


@register_trainer("omni_sync")
class OmniPPOTrainerSync(PPOTrainerSync):
    """``PPOTrainerSync`` subclass that wires tokenizer/processor from ``OmniModelConfig``."""

    def _init_tokenizer(self):
        model_config = self.config.actor_rollout_ref.model
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        model_config.trust_remote_code = trust_remote_code

        model_path = model_config.path
        use_shm = model_config.get("use_shm", False)

        local_model_path = copy_to_local(model_path, use_shm=use_shm)

        # Auto-detect architecture from config.json (follows diffusion pattern).
        architecture = model_config.get("architecture")
        if not architecture:
            config_json = os.path.join(local_model_path, "config.json")
            with open(config_json) as f:
                architecture = json.load(f)["architectures"][0]

        tokenizer_path = model_config.get("tokenizer_path") or local_model_path
        local_tokenizer_path = copy_to_local(tokenizer_path, use_shm=use_shm)

        adapter_cls = OmniModelBase.get_class_by_name(
            architecture,
            model_config.model_stage,
            model_config.get("external_lib"),
        )
        self.tokenizer = adapter_cls.configure_tokenizer(local_tokenizer_path, model_config)
        self.processor = adapter_cls.configure_processor(local_model_path, model_config)
