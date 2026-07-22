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
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.engine import EngineConfig, FSDPEngineConfig
from verl.workers.config.optimizer import OptimizerConfig

from .model import DiffusionModelConfig

__all__ = [
    "DiffusionLossConfig",
    "VeOmniDiffusionEngineConfig",
    "VeOmniDiffusionOptimizerConfig",
    "DiffusionActorConfig",
    "FSDPDiffusionActorConfig",
    "VeOmniDiffusionActorConfig",
]


@dataclass
class DiffusionLossConfig(BaseConfig):
    loss_mode: str = "flow_grpo"
    clip_ratio: float = 0.0001
    adv_clip_max: float = 5.0
    mix_beta: float = 0.5
    ref_kl_coef: float = 0.0
    adaptive_weight_min: float = 1e-5
    dpo_beta: float = 2000.0
    kl_mask_threshold: float = 1e-5
    add_kl_coefficient: bool = True

    def __post_init__(self):
        """Validate diffusion loss configuration."""
        valid_modes = ["flow_grpo", "flow_dppo", "grpo_guard", "diffusion_nft", "dpo", "dance_grpo"]
        if self.loss_mode not in valid_modes:
            raise ValueError(f"Invalid diffusion loss_mode: {self.loss_mode}. Must be one of {valid_modes}")
        if self.adv_clip_max <= 0:
            raise ValueError(f"Diffusion adv_clip_max must be positive, got {self.adv_clip_max}.")
        if self.mix_beta <= 0:
            raise ValueError(f"mix_beta must be positive, got {self.mix_beta}.")
        if self.adaptive_weight_min <= 0:
            raise ValueError(f"adaptive_weight_min must be positive, got {self.adaptive_weight_min}.")
        if self.kl_mask_threshold <= 0:
            raise ValueError(f"kl_mask_threshold must be positive, got {self.kl_mask_threshold}.")


@dataclass
class VeOmniDiffusionEngineConfig(EngineConfig):
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_parallel_size"}

    # VeOmni diffusion backend only supports FSDP2.
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
            raise ValueError(f"VeOmni diffusion engine requires strategy='veomni', got {self.strategy!r}")
        if self.ulysses_parallel_size != 1:
            raise NotImplementedError("VeOmni Qwen-Image diffusion backend does not support Ulysses SP yet.")


@dataclass
class VeOmniDiffusionOptimizerConfig(OptimizerConfig):
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
class DiffusionActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: int = MISSING
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    diffusion_loss: DiffusionLossConfig = field(default_factory=DiffusionLossConfig)
    loss_scale_factor: Optional[float] = None
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: DiffusionModelConfig = field(default_factory=BaseConfig)
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    profiler: Optional[ProfilerConfig] = None

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Rollout Correction config.
    # When bypass_mode=True, ``diffusion_loss`` computes per-step RS from here.
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    def __post_init__(self):
        """Validate diffusion actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING


@dataclass
class FSDPDiffusionActorConfig(DiffusionActorConfig):
    # Training strategy: fsdp or fsdp2
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)

    def __post_init__(self):
        """Validate diffusion FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config
        # Sync strategy to engine config so engine_workers can pick the right FSDP version.
        # EngineConfig.strategy defaults to None, so without this, engine_workers.py always
        # falls back to FSDP1 even when actor.strategy="fsdp2".
        object.__setattr__(self.engine, "strategy", self.strategy)


@dataclass
class VeOmniDiffusionActorConfig(DiffusionActorConfig):
    strategy: str = "veomni"
    veomni_config: VeOmniDiffusionEngineConfig = field(default_factory=VeOmniDiffusionEngineConfig)
    optim: VeOmniDiffusionOptimizerConfig = field(default_factory=VeOmniDiffusionOptimizerConfig)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.veomni_config
