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

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.workers.config.model import MtpConfig

from verl_omni.utils.fs import resolve_model_local_dir

from .rollout import DiffusionPipelineConfig, DiffusionRolloutAlgoConfig

__all__ = ["DiffusionModelConfig"]


@dataclass
class DiffusionModelConfig(BaseConfig):
    _mutable_fields = {
        "model_type",
        "tokenizer_path",
        "tokenizer",
        "processor",
        "local_path",
        "local_tokenizer_path",
        "architecture",
    }

    path: str = MISSING
    architecture: Optional[str] = None
    algorithm: str = MISSING
    local_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # model type, e.g., "diffusion_model"
    model_type: str = "diffusion_model"

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    enable_gradient_checkpointing: bool = True
    attn_backend: str = "native"

    lora_rank: int = 0
    lora_alpha: int = 64
    lora_init_weights: str = "gaussian"
    target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj","k_proj"]
    target_parameters: Optional[list[str]] = None  # for lora adapter on nn.Parameter

    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None

    # dtype to convert LoRA parameters to (e.g., "fp32", "bf16"). Default None means no conversion.
    lora_dtype: Optional[str] = None

    mtp: Optional[MtpConfig] = field(default_factory=MtpConfig)

    pipeline: DiffusionPipelineConfig = field(default_factory=DiffusionPipelineConfig)

    algo: Optional[DiffusionRolloutAlgoConfig] = field(default_factory=DiffusionRolloutAlgoConfig)

    def __post_init__(self):
        import_external_libs(self.external_lib)

        valid_backends = {"native", "_native_npu"}
        if self.attn_backend not in valid_backends:
            raise ValueError(f"Invalid attn_backend: {self.attn_backend}. Must be one of {sorted(valid_backends)}")

        self.local_path = resolve_model_local_dir(self.path, use_shm=self.use_shm)
        if self.tokenizer_path is None:
            tokenizer_path = os.path.join(self.local_path, "tokenizer")
            self.tokenizer_path = tokenizer_path if os.path.exists(tokenizer_path) else self.local_path

        if self.architecture is None:
            import json

            model_index_path = os.path.join(self.local_path, "model_index.json")
            with open(model_index_path) as f:
                self.architecture = json.load(f)["_class_name"]

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(
                self.local_tokenizer_path, trust_remote_code=self.trust_remote_code, use_fast=True
            )
            if os.path.exists(os.path.join(self.local_path, "processor")):
                self.processor = hf_processor(
                    os.path.join(self.local_path, "processor"), trust_remote_code=self.trust_remote_code
                )
            else:
                self.processor = None

        # Ensure target_modules is a str or list[str] (only if not None)
        if self.target_modules is not None:
            if not isinstance(self.target_modules, (str | list)):
                raise TypeError(
                    "target_modules must be a string or a list of strings, "
                    f"but got {type(self.target_modules).__name__}"
                )
            if isinstance(self.target_modules, list):
                for x in self.target_modules:
                    if not isinstance(x, str):
                        raise TypeError(
                            f"All elements in target_modules list must be strings, but found {type(x).__name__}"
                        )

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
