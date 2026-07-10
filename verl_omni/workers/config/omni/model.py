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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.workers.config.model import MtpConfig

from verl_omni.utils.fs import resolve_model_local_dir

__all__ = ["OmniModelConfig"]


@dataclass
class OmniModelConfig(BaseConfig):
    _mutable_fields = {
        "model_type",
        "tokenizer_path",
        "tokenizer",
        "processor",
        "local_path",
        "local_tokenizer_path",
    }

    path: str = MISSING
    model_type: str = "omni_model"
    config_path: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    load_tokenizer: bool = True
    tokenizer: Any = None
    processor: Any = None
    use_shm: bool = False
    trust_remote_code: bool = True
    custom_chat_template: Optional[str] = None
    external_lib: Optional[str] = None

    enable_gradient_checkpointing: bool = True
    encoder_data_balance: bool = False
    encoder_data_balance_sorting_algo: str = "local"
    basic_modules: list[str] = field(default_factory=list)
    model_config: dict[str, Any] = field(default_factory=dict)
    ops_implementation: Any = None

    lora_rank: int = 0
    lora_alpha: int = 64
    lora_init_weights: str = "gaussian"
    target_modules: Optional[Any] = "all-linear"
    target_parameters: Optional[list[str]] = None
    exclude_modules: Optional[str] = None
    lora: dict[str, Any] = field(default_factory=dict)
    lora_adapter_path: Optional[str] = None
    policy_state_adapters: tuple[str, ...] = ("default",)
    lora_dtype: Optional[str] = None
    fsdp_layer_prefixes: list[str] = field(default_factory=lambda: ["thinker.model.layers."])
    mtp: Optional[MtpConfig] = field(default_factory=MtpConfig)

    def __post_init__(self):
        import_external_libs(self.external_lib)

        self.local_path = resolve_model_local_dir(self.path, use_shm=self.use_shm)
        if self.config_path is None:
            self.config_path = self.local_path
        if self.model_path is None:
            self.model_path = self.local_path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.local_path

        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(
                self.local_tokenizer_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
            )
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        if self.target_modules is not None and not isinstance(self.target_modules, (str | list)):
            raise TypeError(
                f"target_modules must be a string or a list of strings, but got {type(self.target_modules).__name__}"
            )

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
