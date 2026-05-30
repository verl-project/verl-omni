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
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.model import MtpConfig
from verl.workers.config.rollout import (
    AgentLoopConfig,
    CheckpointEngineConfig,
    MultiTurnConfig,
    PrometheusConfig,
)

try:
    from verl.workers.config.disaggregation import DisaggregationConfig
except ModuleNotFoundError:

    @dataclass
    class DisaggregationConfig(BaseConfig):
        """Compatibility shim for older verl versions without disaggregation config."""

        pass


__all__ = [
    "DiffusionRolloutAlgoConfig",
    "DiffusionPipelineConfig",
    "DiffusionSamplingConfig",
    "DiffusionRolloutConfig",
]


@dataclass
class DiffusionRolloutAlgoConfig(BaseConfig):
    """Algorithm configuration for the SDE-based diffusion rollout."""

    noise_level: float = 1.0
    sde_type: str = "sde"
    sde_window_size: Optional[int] = None
    sde_window_range: Optional[list[int]] = None

    # MixGRPO-only configs
    sample_strategy: str = "random"
    iters_per_group: int = 1
    sde_window_seed: int = 0

    # Rollout data contract: trajectory for coupled RL, final_latent for decoupled forward-process RL
    collect_mode: str = "trajectory"

    def __post_init__(self):
        if self.sample_strategy not in ("random", "progressive"):
            raise ValueError(f"Unknown sample_strategy: {self.sample_strategy!r}")
        if self.sample_strategy == "progressive" and self.iters_per_group <= 0:
            raise ValueError(f"iters_per_group must be positive, got {self.iters_per_group}.")
        if self.collect_mode not in ("trajectory", "final_latent"):
            raise ValueError(
                f"Invalid diffusion rollout collect_mode: {self.collect_mode}. "
                "Must be one of ['trajectory', 'final_latent']."
            )


@dataclass
class DiffusionPipelineConfig(BaseConfig):
    # for pipeline specific sampling parameters
    height: int = 512
    width: int = 512
    num_inference_steps: int = 10
    true_cfg_scale: float = 1.0
    max_sequence_length: int = 512
    guidance_scale: Optional[float] = None


@dataclass
class DiffusionSamplingConfig(BaseConfig):
    # for validation only
    n: int = 1
    seed: int = 42
    pipeline: DiffusionPipelineConfig = field(default_factory=DiffusionPipelineConfig)
    algo: DiffusionRolloutAlgoConfig = field(default_factory=DiffusionRolloutAlgoConfig)


@dataclass
class DiffusionRolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format", "engine_kwargs", "prompt_length", "expert_parallel_size"}

    name: Optional[str] = MISSING
    mode: str = "async"
    nnodes: int = 0
    n_gpus_per_node: int = 8
    n: int = 1

    # Base seed for deterministic training rollout RNG. Per-step base is
    # ``seed + global_step - 1``. null disables rollout seeding.
    seed: Optional[int] = None

    prompt_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    enforce_eager: bool = False
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"

    val_kwargs: DiffusionSamplingConfig = field(default_factory=DiffusionSamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size_per_gpu: Optional[int] = None

    disable_log_stats: bool = True

    engine_kwargs: dict = field(default_factory=dict)

    pipeline: DiffusionPipelineConfig = field(default_factory=DiffusionPipelineConfig)

    calculate_log_probs: bool = False
    rollout_adapter: str = "default"

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    # Checkpoint Engine config for update weights from trainer to rollout
    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    skip_tokenizer_init: bool = True

    quantization: Optional[str] = None

    enable_rollout_routing_replay: bool = False

    enable_sleep_mode: bool = True

    mtp: Optional[MtpConfig] = field(default_factory=MtpConfig)

    profiler: Optional[ProfilerConfig] = None

    algo: Optional[DiffusionRolloutAlgoConfig] = field(default_factory=DiffusionRolloutAlgoConfig)

    disaggregation: DisaggregationConfig = field(default_factory=DisaggregationConfig)

    external_lib: Optional[str] = None

    def __post_init__(self):
        """Validate the diffusion rollout config"""
        if self.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` or remove the mode setting entirely."
            )
        if self.rollout_adapter not in ("default", "old"):
            raise ValueError(
                f"Invalid diffusion rollout rollout_adapter: {self.rollout_adapter}. Must be one of ['default', 'old']."
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm_omni":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )
