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
"""Configuration dataclass for omni (thinker/talker) model training."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs

from verl_omni.utils.fs import resolve_model_local_dir

__all__ = ["OmniModelConfig"]


@dataclass
class OmniModelConfig(BaseConfig):
    """Configuration for omni (thinker/talker) model training.

    Provides the fields that verl's FSDP engine needs for model loading
    (``path``, ``hf_config_name``, ``no_split_modules``, etc.) plus
    omni-specific fields (``architecture``, ``model_stage``).

    RL algorithm selection (GSPO, GRPO, RLOO, etc.) is handled by
    verl's ``actor.policy_loss.loss_mode`` and ``algorithm.adv_estimator``
    config — the config is algorithm-agnostic.
    """

    _mutable_fields = {
        "model_type",
        "architecture",
        "model_stage",
        "tokenizer_path",
        "tokenizer",
        "processor",
        "local_path",
        "local_tokenizer_path",
        "no_split_modules",
    }

    path: str = MISSING
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    override_config: dict = field(default_factory=dict)
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[Any] = "all-linear"  # "all-linear" or ["q_proj", "k_proj", ...]
    exclude_modules: Optional[str] = None
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True
    external_lib: Optional[str] = None
    custom_chat_template: Optional[str] = None

    model_type: str = "language_model"
    local_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    tokenizer: Any = None
    processor: Any = None

    load_tokenizer: bool = True
    use_shm: bool = False

    # HF config architectures[0].
    architecture: str = MISSING

    # Which stage to train: "thinker", "talker", or "all".
    model_stage: str = "thinker"

    # Sub-config key for the trainable component
    # (e.g. "thinker_config", "talker_config").
    hf_config_name: Optional[str] = None

    # FSDP layer class names.
    no_split_modules: list[str] = field(default_factory=list)

    # Force ``tie_word_embeddings=False`` for FSDP compatibility.
    tie_word_embeddings_override: Optional[bool] = None

    # Multimodal token budget
    max_image_tokens: Optional[int] = None
    max_audio_tokens: Optional[int] = None
    max_video_tokens: Optional[int] = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        self.local_path = resolve_model_local_dir(self.path, use_shm=self.use_shm)

        if self.tokenizer_path is None:
            tokenizer_path = os.path.join(self.local_path, "tokenizer")
            self.tokenizer_path = tokenizer_path if os.path.exists(tokenizer_path) else self.local_path

        if self.architecture is MISSING:
            config_path = os.path.join(self.local_path, "config.json")
            with open(config_path) as f:
                self.architecture = json.load(f)["architectures"][0]

        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            # Tokenizer/processor are loaded by the omni trainer via
            # OmniModelBase.configure_tokenizer / configure_processor.

    def get_processor(self):
        """Return the processor, or fall back to the tokenizer."""
        return self.processor if self.processor is not None else self.tokenizer
