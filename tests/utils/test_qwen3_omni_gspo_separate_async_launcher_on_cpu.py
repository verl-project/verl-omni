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
"""Launcher contract test for the Qwen3-Omni thinker GSPO MMK12 recipe on the
separate-async trainer.

Pins the disaggregation wiring and the colocated offload flags, not the
tunable training hyperparameters.
"""

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_separate_async_v1.sh"
)


def _active_settings() -> set[str]:
    """Recipe settings, comments dropped."""
    lines = [line for line in _SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    return {line.strip().removesuffix("\\").rstrip() for line in lines}


def _setting_value(key: str) -> str:
    """Value of the single exact ``key=...`` setting line."""
    matches = [line for line in _active_settings() if line.startswith(f"{key}=")]
    assert len(matches) == 1, f"expected exactly one {key}= line, got {matches}"
    return matches[0].split("=", 1)[1]


def test_launcher_selects_the_separate_async_trainer():
    settings = _active_settings()

    assert "trainer.v1.trainer_mode=omni_separate_async" in settings
    assert "trainer.v1.separate_async.num_warmup_batches=1" in settings
    assert "trainer.v1.separate_async.parameter_sync_step=8" in settings
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl" in settings
    # Adapter-only weight sync to the standalone replicas.
    assert "actor_rollout_ref.model.lora.merge=False" in settings


def test_launcher_batch_identity():
    # The upstream assert: 128 == parameter_sync_step * ppo_mini_batch_size.
    train_batch = int(_setting_value("data.train_batch_size"))
    sync_step = int(_setting_value("trainer.v1.separate_async.parameter_sync_step"))
    mini_batch = int(_setting_value("actor_rollout_ref.actor.ppo_mini_batch_size"))

    assert train_batch == sync_step * mini_batch


def test_launcher_keeps_colocated_offload():
    settings = _active_settings()

    # One standalone TP=2 replica (2 GPUs) plus the 2-GPU FSDP trainer pool.
    assert "actor_rollout_ref.rollout.nnodes=1" in settings
    assert "actor_rollout_ref.rollout.n_gpus_per_node=2" in settings
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in settings
    assert "trainer.n_gpus_per_node=2" in settings
    assert "trainer.nnodes=1" in settings

    # The colocated offload flags stay on; CPU snapshots own their storage,
    # so offload is safe at any parameter_sync_step.
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=true" in settings
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" in settings
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.8" in settings
