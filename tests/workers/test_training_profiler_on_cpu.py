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
"""Exercise the training worker with real CPU profiler traces."""

import gzip
import json
from contextlib import nullcontext
from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.profiler import DistProfiler, ProfilerConfig
from verl.utils.profiler.config import TorchProfilerScheduleConfig, TorchProfilerToolConfig

from verl_omni.workers import engine_workers
from verl_omni.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("scheduled", [False, True])
def test_update_trace_selects_mini_batches(tmp_path, discrete, scheduled):
    tool = TorchProfilerToolConfig(
        contents=["cpu"],
        discrete=discrete,
        schedule=TorchProfilerScheduleConfig(skip_first=1, active=1, repeat=1) if scheduled else None,
    )
    config = ProfilerConfig(tool="torch", enable=True, ranks=[0], save_path=str(tmp_path), tool_config={"torch": tool})
    inner = object.__new__(TrainingWorker)
    inner.profiler = DistProfiler(rank=0, config=config, tool_config=tool)
    inner.engine = SimpleNamespace(
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        train_mode=lambda **kwargs: nullcontext(),
        is_mp_src_rank_with_outputs=lambda: False,
    )
    completed = []

    def train_batch(batch):
        completed.append(batch["values"].square().sum().item())

    inner.train_batch = train_batch
    outer = SimpleNamespace(
        actor=inner,
        config=OmegaConf.create({"actor": {}}),
        enable_routing_replay=False,
        profiler=DistProfiler(rank=0, config=config, tool_config=tool),
    )
    data = TensorDict({"values": torch.arange(6, dtype=torch.float32)}, batch_size=[6])
    tu.assign_non_tensor(data, mini_batch_size=2, epochs=2, dataloader_kwargs={"shuffle": False})
    outer.profiler.start(role="train", profile_step=1)
    try:
        ActorRolloutRefWorker.update_actor(outer, data)
    finally:
        outer.profiler.stop(run_command=False)

    assert len(completed) == 6
    traces = list(tmp_path.glob("*.json.gz"))
    assert traces, "Selected profiling window did not emit a trace"
    names = {
        event["name"]
        for trace in traces
        for event in json.loads(gzip.decompress(trace.read_bytes()))["traceEvents"]
        if "name" in event
    }
    expected = {"mini_batch1"} if scheduled else {f"mini_batch{i}" for i in range(6)}
    assert {name for name in names if name.startswith("mini_batch")} == expected
    assert "aten::pow" in names


@pytest.mark.parametrize("role", ["actor", "ref"])
def test_init_model_forwards_typed_profiler_to_training_worker(monkeypatch, role):
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
                f"actor_rollout_ref.{role}.profiler.enable=True",
                f"actor_rollout_ref.{role}.profiler.ranks=[0]",
                f"actor_rollout_ref.{role}.profiler.tool_config.torch.discrete=True",
                f"+actor_rollout_ref.{role}.profiler.tool_config.torch.schedule="
                "{_target_:verl.utils.profiler.config.TorchProfilerScheduleConfig,skip_first:1,active:1,repeat:1}",
            ],
        ).actor_rollout_ref
    worker = object.__new__(ActorRolloutRefWorker)
    worker.role = role
    worker.config = config
    worker.distillation_enabled = False
    model = OmegaConf.create({"model_type": "diffusion_model", "mtp": {"enable": False}})
    convert = engine_workers.omega_conf_to_dataclass
    monkeypatch.setattr(
        engine_workers,
        "omega_conf_to_dataclass",
        lambda cfg: model if cfg is config.model else convert(cfg),
    )

    class WorkerConfigCaptured(Exception):
        pass

    construct = Mock(side_effect=WorkerConfigCaptured)
    monkeypatch.setattr(engine_workers, "TrainingWorker", construct)
    with pytest.raises(WorkerConfigCaptured):
        unwrap(ActorRolloutRefWorker.init_model)(worker)
    profiler = construct.call_args.kwargs["config"].profiler_config
    assert isinstance(profiler, ProfilerConfig)
    assert profiler.enable is True
    assert profiler.ranks == [0]
    assert isinstance(profiler.tool_config["torch"], TorchProfilerToolConfig)
    assert profiler.tool_config["torch"].discrete is True
    assert isinstance(profiler.tool_config["torch"].schedule, TorchProfilerScheduleConfig)
    assert profiler.tool_config["torch"].schedule.skip_first == 1
