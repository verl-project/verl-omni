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
from typing import Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.engine import EngineConfig
from verl.workers.config.optimizer import OptimizerConfig

from .model import OmniModelConfig

__all__ = [
    "OmniLossConfig",
    "VeOmniOmniEngineConfig",
    "VeOmniOmniOptimizerConfig",
    "VeOmniOmniActorConfig",
]


@dataclass
class OmniLossConfig(BaseConfig):
    loss_mode: str = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False
    average_log_prob: bool = False
    refer_model_precision: str = "bfloat16"

    def __post_init__(self):
        if self.loss_mode not in {"dpo"}:
            raise ValueError(f"Unsupported omni loss_mode={self.loss_mode!r}; currently supported: ['dpo'].")
        if self.loss_type not in {"sigmoid", "ipo"}:
            raise ValueError(f"Invalid omni DPO loss_type={self.loss_type!r}; expected 'sigmoid' or 'ipo'.")
        if self.beta <= 0:
            raise ValueError(f"Omni DPO beta must be positive, got {self.beta}.")


@dataclass
class VeOmniOmniEngineConfig(EngineConfig):
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_parallel_size"}

    strategy: str = "veomni"
    fsdp_size: int = -1
    ulysses_parallel_size: int = 1
    expert_parallel_size: int = 1
    init_device: str = "meta"
    reshard_after_forward: bool = True
    forward_prefetch: bool = True
    model_dtype: str = "bfloat16"
    mixed_precision: bool = True
    mixed_precision_param_dtype: str = "bfloat16"
    mixed_precision_reduce_dtype: str = "float32"
    mixed_precision_output_dtype: Optional[str] = None
    mixed_precision_cast_forward_inputs: bool = True
    enable_reentrant: bool = False
    enable_activation_offload: bool = False
    activation_gpu_limit: float = 0.0
    attn_implementation: str = "eager"
    moe_implementation: str = "eager"
    cross_entropy_loss_implementation: str = "eager"
    rms_norm_implementation: str = "eager"
    swiglu_mlp_implementation: str = "eager"
    rotary_pos_emb_implementation: str = "eager"
    load_balancing_loss_implementation: str = "eager"
    rms_norm_gated_implementation: str = "eager"
    causal_conv1d_implementation: str = "eager"
    chunk_gated_delta_rule_implementation: str = "eager"

    def __post_init__(self):
        super().__post_init__()
        if self.strategy != "veomni":
            raise ValueError(f"VeOmni omni engine requires strategy='veomni', got {self.strategy!r}")


@dataclass
class VeOmniOmniOptimizerConfig(OptimizerConfig):
    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    eps: float = 1e-8
    fused: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.lr_scheduler_type not in {"constant", "linear", "cosine"}:
            raise ValueError(
                f"Invalid VeOmni lr_scheduler_type={self.lr_scheduler_type!r}; "
                "expected one of ['constant', 'linear', 'cosine']."
            )


@dataclass
class VeOmniOmniActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = "veomni"
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: int = MISSING
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    loss_scale_factor: Optional[float] = None
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    rollout_n: int = MISSING
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    global_batch_info: dict = field(default_factory=dict)
    omni_loss: OmniLossConfig = field(default_factory=OmniLossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    veomni_config: VeOmniOmniEngineConfig = field(default_factory=VeOmniOmniEngineConfig)
    optim: VeOmniOmniOptimizerConfig = field(default_factory=VeOmniOmniOptimizerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    model_config: OmniModelConfig = field(default_factory=BaseConfig)
    profiler: Optional[ProfilerConfig] = None

    def __post_init__(self):
        assert self.strategy == "veomni"
        assert self.rollout_n != MISSING
        self.engine = self.veomni_config
