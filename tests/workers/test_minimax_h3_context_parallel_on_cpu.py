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
"""CPU contract tests for MiniMax H3 Diffusers context parallelism."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.special_e2e.run_flowgrpo_minimax_h3_tiny import _hydra_overrides, _validate_actor_sp
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


@pytest.mark.parametrize("adapter", [MiniMaxH3FlowGRPO, MiniMaxH3DiffusionNFT])
def test_minimax_h3_requests_uneven_ulysses_partitioning(adapter) -> None:
    """Every H3 task packs an unpadded joint sequence whose length may be uneven."""
    assert adapter.context_parallel_config_kwargs(MagicMock()) == {"ulysses_anything": True}


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
def test_fsdp_engine_forwards_minimax_h3_context_parallel_options(algorithm) -> None:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.use_ulysses_sp = True
    engine.ulysses_sequence_parallel_size = 2
    engine.ulysses_device_mesh = None
    engine.model_config = SimpleNamespace(
        architecture="MiniMaxH3Pipeline",
        algorithm=algorithm,
        external_lib=None,
    )
    module = MagicMock()
    module.config = SimpleNamespace(num_attention_heads=56)

    engine._enable_context_parallel(module)

    config = module.enable_parallelism.call_args.kwargs["config"]
    assert config.ulysses_degree == 2
    assert config.ulysses_anything is True


def test_minimax_h3_rejects_sp_that_does_not_divide_attention_heads() -> None:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.use_ulysses_sp = True
    engine.ulysses_sequence_parallel_size = 3
    engine.ulysses_device_mesh = None
    engine.model_config = SimpleNamespace(
        architecture="MiniMaxH3Pipeline",
        algorithm="flow_grpo",
        external_lib=None,
    )
    module = MagicMock()
    module.config = SimpleNamespace(num_attention_heads=56)

    with pytest.raises(ValueError, match="56 attention heads must be divisible"):
        engine._enable_context_parallel(module)

    module.enable_parallelism.assert_not_called()


def test_fsdp_engine_leaves_context_parallel_disabled_at_size_one() -> None:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.use_ulysses_sp = False
    module = MagicMock()

    engine._enable_context_parallel(module)

    module.enable_parallelism.assert_not_called()


@pytest.mark.parametrize(("task", "train_batch_size"), [("t2va", 8), ("fl2va", 8), ("ref2va", 4)])
def test_minimax_h3_smoke_config_enables_sp_for_each_task(task, train_batch_size) -> None:
    overrides = _hydra_overrides(
        tiny_model_dir="/tmp/model",
        train_parquet="/tmp/train.parquet",
        val_parquet="/tmp/val.parquet",
        reward_stub_path="/tmp/reward.py",
        output_dir="/tmp/output",
        task=task,
        num_gpus=4,
        actor_sp=2,
        rollout_tp=2,
        text_encoder_tp=1,
        total_training_steps=1,
        ray_num_cpus=4,
        height=160,
        width=288,
        num_frames=97,
        num_inference_steps=4,
    )

    assert "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2" in overrides
    assert f"data.train_batch_size={train_batch_size}" in overrides
    assert f"actor_rollout_ref.rollout.pipeline.task={task}" in overrides


def test_minimax_h3_smoke_config_rejects_invalid_sp_partition() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_actor_sp(num_gpus=4, actor_sp=3)
