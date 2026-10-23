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
    dgpo_beta: float = 100.0
    dgpo_clip_range: float = 0.01
    # OmniNFT-only modality weights and reference regularization coefficients.
    video_weight: float = 1.0
    audio_weight: float = 1.0
    video_ref_kl_coef: float = 0.0
    audio_ref_kl_coef: float = 0.0
    # DiffusionOPSD relative trust-region radius for reward-gradient targets.
    opsd_rho: float = 0.10
    # DiffusionOPSD number of normalized reward-gradient steps per target.
    opsd_n_ascent: int = 2
    # DiffusionOPSD step-size multiplier inside the relative trust region.
    opsd_eta: float = 1.0
    # DiffusionOPSD positive/negative branch mixing coefficient (paper beta).
    opsd_branch_beta: float = 1.0
    # DiffusionOPSD requested low-noise query sigma in scheduler flow time.
    opsd_query_sigma: float = 0.278
    # DiffusionOPSD tolerance between requested and selected query sigma.
    opsd_query_sigma_tolerance: float = 0.15
    # DiffusionOPSD scheduler train-timestep scale (SD3.5 flow scheduler: 1000).
    opsd_num_train_timesteps: int = 1000
    # DiffusionOPSD detached residual-normalizer floor.
    opsd_eps_gamma: float = 1e-5
    # DiffusionOPSD differentiable target reward kind.
    opsd_reward_kind: str = "pickscore"
    # DiffusionOPSD differentiable reward checkpoint path; empty uses the scorer default.
    opsd_reward_model_path: str = ""
    # DiffusionOPSD target-construction dtype for the reward model.
    opsd_reward_dtype: str = "float32"
    # DiffusionOPSD trajectories per reward-gradient micro batch.
    opsd_target_micro_batch: int = 6

    def __post_init__(self):
        """Validate diffusion loss configuration."""
        valid_modes = [
            "flow_grpo",
            "flow_dppo",
            "grpo_guard",
            "diffusion_nft",
            "omni_nft",
            "dgpo",
            "dpo",
            "dmd2",
            "dance_grpo",
            "distill_kl",
            "distill_fm_mse",
            "opsd",
        ]
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
        if self.dgpo_beta <= 0:
            raise ValueError(f"dgpo_beta must be positive, got {self.dgpo_beta}.")
        if self.dgpo_clip_range < 0:
            raise ValueError(f"dgpo_clip_range must be non-negative, got {self.dgpo_clip_range}.")
        if not 0 < self.opsd_rho <= 1:
            raise ValueError(f"opsd_rho must be in (0, 1], got {self.opsd_rho}.")
        if self.opsd_n_ascent < 1:
            raise ValueError(f"opsd_n_ascent must be positive, got {self.opsd_n_ascent}.")
        if self.opsd_eta <= 0:
            raise ValueError(f"opsd_eta must be positive, got {self.opsd_eta}.")
        if self.opsd_branch_beta < 0:
            raise ValueError(f"opsd_branch_beta must be non-negative, got {self.opsd_branch_beta}.")
        if not 0 < self.opsd_query_sigma < 1:
            raise ValueError(f"opsd_query_sigma must be in (0, 1), got {self.opsd_query_sigma}.")
        if self.opsd_query_sigma_tolerance <= 0:
            raise ValueError(f"opsd_query_sigma_tolerance must be positive, got {self.opsd_query_sigma_tolerance}.")
        if self.opsd_num_train_timesteps <= 0:
            raise ValueError(f"opsd_num_train_timesteps must be positive, got {self.opsd_num_train_timesteps}.")
        if self.opsd_eps_gamma <= 0:
            raise ValueError(f"opsd_eps_gamma must be positive, got {self.opsd_eps_gamma}.")
        if self.opsd_target_micro_batch <= 0:
            raise ValueError(f"opsd_target_micro_batch must be positive, got {self.opsd_target_micro_batch}.")
        if self.loss_mode == "opsd":
            valid_opsd_rewards = ["pickscore"]
            if self.opsd_reward_kind not in valid_opsd_rewards:
                raise ValueError(
                    f"Invalid opsd_reward_kind: {self.opsd_reward_kind}. The first DiffusionOPSD integration "
                    f"ships only {valid_opsd_rewards}; add a differentiable scorer before enabling another reward."
                )
            if self.opsd_reward_dtype not in ["float32", "bfloat16", "float16"]:
                raise ValueError(f"Invalid opsd_reward_dtype: {self.opsd_reward_dtype}.")


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
    diffusion_loss: DiffusionLossConfig = field(default_factory=DiffusionLossConfig)
    loss_scale_factor: Optional[float] = None
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    use_distill_loss: bool = False
    distill_loss_mode: str = "distill_kl"
    distill_loss_coef: float = 1.0
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
        valid_distill_modes = ["distill_kl", "distill_fm_mse"]
        if self.use_distill_loss and self.distill_loss_mode not in valid_distill_modes:
            raise ValueError(
                f"Invalid distill_loss_mode: {self.distill_loss_mode}. Must be one of {valid_distill_modes}"
            )


@dataclass
class FSDPDiffusionActorConfig(DiffusionActorConfig):
    # Training strategy: fsdp or fsdp2
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    # Stage training inputs from CPU one timestep at a time.
    enable_timestep_staging: bool = False
    # Defer FSDP reduce-scatter until the last (micro-batch, timestep) backward.
    # Off by default: unsharded grads sit on GPU until that last backward.
    use_no_sync_for_gradient_accumulation: bool = False

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
