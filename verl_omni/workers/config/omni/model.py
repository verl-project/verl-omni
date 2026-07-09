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
"""Configuration for omni-model training."""

import json
import os
from dataclasses import dataclass
from typing import Optional

from verl.workers.config.model import HFModelConfig

__all__ = ["OmniModelConfig"]


@dataclass
class OmniModelConfig(HFModelConfig):
    """HF model config with omni-stage metadata.

    Thinker-only Qwen3-Omni still trains as a language model in verl's engine.
    The additional fields let verl-omni resolve model adapters without adding
    algorithm-specific logic to the model config.
    """

    _mutable_fields = HFModelConfig._mutable_fields | {
        "architecture",
        "model_stage",
        "hf_config_name",
    }

    architecture: Optional[str] = None
    model_stage: str = "thinker"
    hf_config_name: Optional[str] = None
    max_image_tokens: int = 1024
    max_audio_tokens: int = 1500
    max_video_tokens: int = 2304

    def __post_init__(self):
        super().__post_init__()

        if self.architecture is None:
            config_path = os.path.join(self.local_hf_config_path, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    architectures = json.load(f).get("architectures", [])
                if architectures:
                    self.architecture = architectures[0]
