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
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig
from verl.base_config import BaseConfig
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.workers.config.model import MtpConfig

from verl_omni.utils.fs import resolve_model_local_dir

__all__ = ["OmniModelConfig"]

logger = logging.getLogger(__name__)


@dataclass
class OmniModelConfig(BaseConfig):
    """Configuration for omni (thinker/talker) model training."""

    _mutable_fields = {
        "model_type",
        "architecture",
        "model_stage",
        "tokenizer_path",
        "tokenizer",
        "processor",
        "local_path",
        "local_tokenizer_path",
        "local_hf_config_path",
        "hf_config",
        "generation_config",
        "architectures",
        "share_embeddings_and_output_weights",
    }

    # note that we separate path, hf_config_path and tokenizer_path in case they are different
    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # model type
    model_type: str = "omni"

    # HF config architectures[0] (auto-detected from config.json if unset)
    architecture: str = MISSING
    architectures: Optional[list[str]] = None

    # which stage to train: "thinker", "talker", or "all"
    model_stage: str = "thinker"
    # sub-config key for the trainable component (e.g. "thinker_config", "talker_config")
    hf_config_name: Optional[str] = None

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None
    share_embeddings_and_output_weights: bool = False

    # whether to load tokenizer (set False when only model config is needed)
    load_tokenizer: bool = True

    # whether to use shared memory for model loading
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    # training flags
    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False
    use_remove_padding: bool = True

    # fsdp / megatron lora related
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj", "k_proj"]
    target_parameters: Optional[list[str]] = None  # for lora adapter on nn.Parameter
    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None

    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # TiledMLP configuration for memory-efficient MLP computation
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})

    # MTP (multi-token prediction / speculative decoding)
    mtp: MtpConfig = field(default_factory=MtpConfig)

    # multimodal token budgets
    max_image_tokens: Optional[int] = None
    max_audio_tokens: Optional[int] = None
    max_video_tokens: Optional[int] = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.path == MISSING:
            raise ValueError("OmniModelConfig.path is required but was not set.")

        self.local_path = resolve_model_local_dir(self.path, use_shm=self.use_shm)

        if self.hf_config_path is None:
            self.hf_config_path = self.path

        if self.tokenizer_path is None:
            tokenizer_path = os.path.join(self.local_path, "tokenizer")
            self.tokenizer_path = tokenizer_path if os.path.exists(tokenizer_path) else self.local_path

        if self.architecture == MISSING:
            config_path = os.path.join(self.local_path, "config.json")
            with open(config_path) as f:
                self.architecture = json.load(f)["architectures"][0]

        # Build hf_config so the FSDP engine can load and wrap the model.
        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=attn_implementation,
        )

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)
        self.architectures = getattr(self.hf_config, "architectures", None)

        if self.load_tokenizer:
            # Tokenizer/processor are loaded by the omni trainer via
            # OmniModelBase.configure_tokenizer / configure_processor.
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)

    def get_processor(self):
        """Return the processor, or fall back to the tokenizer."""
        return self.processor if self.processor is not None else self.tokenizer
