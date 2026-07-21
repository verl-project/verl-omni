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

import json
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class DiffusionModelConfig(BaseConfig):
    _mutable_fields = {
        "model_type",
        "algorithm",
        "tokenizer_path",
        "tokenizer",
        "processor",
        "local_path",
        "local_tokenizer_path",
        "architecture",
        "transformer_config",
        "extra_tokenizer_map",
    }

    path: str = MISSING
    architecture: Optional[str] = None
    # diffusers ``transformer/config.json``; used by DiffusionFlopsCounter for MFU.
    transformer_config: Optional[dict[str, Any]] = None
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

    # Optional per-text-encoder tokenizers for models with multiple text encoders
    # (e.g. SD3.5 uses CLIP + T5). Maps a name to ``{path, max_length}``, where
    # ``path`` is absolute or relative to the model directory and ``max_length``
    # is the optional truncation length applied at agent-loop tokenization time.
    # The agent loop tokenizes the prompt once per entry and ships the token ids
    # to the rollout pipeline, so the pipeline never needs to decode/re-encode.
    extra_tokenizers: Optional[dict[str, Any]] = None

    # Loaded tokenizers built from ``extra_tokenizers``:
    # ``{name: {"tokenizer": PreTrainedTokenizer, "max_length": int | None}}``.
    extra_tokenizer_map: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    enable_gradient_checkpointing: bool = True
    attn_backend: str = "_flash_3_varlen_hub"

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

    # Named LoRA policy states required by the algorithm. "reference" uses disabled adapters.
    policy_state_adapters: tuple[str, ...] = ("default",)

    # dtype to convert LoRA parameters to (e.g., "fp32", "bf16"). Default None means no conversion.
    lora_dtype: Optional[str] = None

    mtp: Optional[MtpConfig] = field(default_factory=MtpConfig)

    pipeline: DiffusionPipelineConfig = field(default_factory=DiffusionPipelineConfig)

    algo: Optional[DiffusionRolloutAlgoConfig] = field(default_factory=DiffusionRolloutAlgoConfig)

    fsdp_layer_prefixes: list[str] = field(default_factory=lambda: ["transformer_blocks."])

    # Optional model config path. If unset, the backend uses
    # ``<local_path>/<transformer_subfolder>``.
    config_path: Optional[str] = None

    # Subfolder containing the diffusion transformer weights/config.
    transformer_subfolder: str = "transformer"

    def __post_init__(self):
        import_external_libs(self.external_lib)

        valid_backends = {"native", "_native_npu", "_flash_3_varlen_hub"}
        if self.attn_backend not in valid_backends:
            raise ValueError(f"Invalid attn_backend: {self.attn_backend}. Must be one of {sorted(valid_backends)}")

        if self.attn_backend == "_flash_3_varlen_hub":
            try:
                import kernels  # noqa: F401
            except ImportError as e:
                raise ImportError("attn_backend '_flash_3_varlen_hub' requires `kernels` package. ") from e

        self.local_path = resolve_model_local_dir(self.path, use_shm=self.use_shm)
        if self.tokenizer_path is None:
            tokenizer_path = os.path.join(self.local_path, "tokenizer")
            self.tokenizer_path = tokenizer_path if os.path.exists(tokenizer_path) else self.local_path

        if self.architecture is None:
            model_index_path = os.path.join(self.local_path, "model_index.json")
            with open(model_index_path) as f:
                self.architecture = json.load(f)["_class_name"]

        if self.transformer_config is None:
            config_path = os.path.join(self.local_path, "transformer", "config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as f:
                        self.transformer_config = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Diffusion MFU disabled: failed to read %s: %s", config_path, exc)
            else:
                logger.warning(
                    "Diffusion MFU disabled: transformer config not found at %s. "
                    "Expected the diffusers pipeline layout `<local_path>/transformer/config.json`.",
                    config_path,
                )

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(
                self.local_tokenizer_path, trust_remote_code=self.trust_remote_code, use_fast=True
            )
            processor_path = os.path.join(self.local_path, "processor")
            if os.path.exists(processor_path):
                self.processor = hf_processor(processor_path, trust_remote_code=self.trust_remote_code)
            else:
                self.processor = None

            self.extra_tokenizer_map = {}
            for name, spec in (self.extra_tokenizers or {}).items():
                path = spec.get("path") if hasattr(spec, "get") else None
                if not path:
                    raise ValueError(f"extra_tokenizers[{name!r}] must define a 'path' entry.")
                if not os.path.isabs(path) and not os.path.exists(path):
                    path = os.path.join(self.local_path, path)
                local_path = copy_to_local(path, use_shm=self.use_shm)
                tokenizer = hf_tokenizer(local_path, trust_remote_code=self.trust_remote_code, use_fast=True)
                max_length = spec.get("max_length")
                self.extra_tokenizer_map[name] = {
                    "tokenizer": tokenizer,
                    "max_length": int(max_length) if max_length is not None else None,
                }

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
