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

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from peft import LoraConfig
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric

from verl_omni.workers.config import DiffusionDMDConfig
from verl_omni.workers.dmd_worker import DMDTrainingWorker
from verl_omni.workers.engine.fsdp import dmd_impl
from verl_omni.workers.engine.fsdp.dmd_impl import DMDDiffusersFSDPEngine
from verl_omni.workers.engine_workers import TrainingWorker


class TinyAdapters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = torch.nn.ParameterDict(
            {key: torch.nn.Parameter(torch.ones(2)) for key in ("default", "fake_score", "student_ema")}
        )
        self.peft_config = {key: LoraConfig(r=2) for key in self.adapters}
        self.set_adapter("default")

    def set_adapter(self, name):
        self.active_adapter = name
        for key, value in self.adapters.items():
            value.requires_grad_(key == name)


def constant_schedule(step):
    return 1.0


def cpu_device():
    return "cpu"


def no_collective(tensor, **kwargs):
    return None


def force_peer_nonfinite(tensor, **kwargs):
    tensor.zero_()


def microbatch_metrics(batch, loss_function, forward_only):
    mean = batch["value"].mean()
    return mean.clone().requires_grad_(), {
        "dmd/loss": Metric("mean", mean),
        "dmd/active_elements": float(len(batch)),
        "perf/forward_s": float(len(batch)),
    }


def engine_shell():
    engine = object.__new__(DMDDiffusersFSDPEngine)
    engine.module = TinyAdapters()
    engine.active_stage = "student"
    engine.dmd_config = DiffusionDMDConfig(ema_decay=0.5)
    engine.role_parameters = {
        "student": (engine.module.adapters["default"],),
        "fake_score": (engine.module.adapters["fake_score"],),
    }
    engine.optimizers = {key: torch.optim.SGD(values, lr=0.1) for key, values in engine.role_parameters.items()}
    engine.schedulers = {
        key: torch.optim.lr_scheduler.LambdaLR(opt, constant_schedule) for key, opt in engine.optimizers.items()
    }
    engine.optimizer_configs = {key: SimpleNamespace(clip_grad=1.0) for key in engine.optimizers}
    engine.optimizer_steps = {"student": 0, "fake_score": 0}
    engine.skipped_steps = {"student": 0, "fake_score": 0}
    engine.forward_finite = True
    engine.last_step_succeeded = False
    engine._is_offload_param = False
    engine.select_stage("student")
    return engine


class TestDMDOptimizer:
    @pytest.mark.parametrize("stage", ["student", "fake_score"])
    def test_nonfinite_gradient_does_not_step_scheduler_or_ema(self, monkeypatch, stage):
        monkeypatch.setattr(dmd_impl, "get_device_id", cpu_device)
        monkeypatch.setattr(torch.distributed, "all_reduce", no_collective)
        engine = engine_shell()
        engine.select_stage(stage)
        before = deepcopy(engine.module.state_dict())
        schedule = engine.lr_scheduler.last_epoch
        engine.role_parameters[stage][0].grad = torch.full((2,), float("nan"))
        engine.optimizer_step()
        assert not engine.last_step_succeeded
        assert engine.optimizer_steps[stage] == 0 and engine.skipped_steps[stage] == 1
        assert engine.lr_scheduler.last_epoch == schedule
        for name, value in engine.module.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_peer_skip_is_agreed_before_local_step(self, monkeypatch):
        monkeypatch.setattr(dmd_impl, "get_device_id", cpu_device)
        monkeypatch.setattr(torch.distributed, "all_reduce", force_peer_nonfinite)
        engine = engine_shell()
        engine.role_parameters["student"][0].grad = torch.ones(2)
        engine.optimizer_step()
        assert not engine.last_step_succeeded
        assert engine.optimizer_steps["student"] == 0
        torch.testing.assert_close(engine.module.adapters["default"], torch.ones(2))

    def test_success_steps_only_owner_and_ema(self, monkeypatch):
        monkeypatch.setattr(dmd_impl, "get_device_id", cpu_device)
        monkeypatch.setattr(torch.distributed, "all_reduce", no_collective)
        engine = engine_shell()
        engine.role_parameters["student"][0].grad = torch.ones(2)
        engine.optimizer_step()
        assert engine.last_step_succeeded
        assert engine.optimizer_steps == {"student": 1, "fake_score": 0}
        assert engine.schedulers["student"].last_epoch == 1
        assert engine.schedulers["fake_score"].last_epoch == 0
        torch.testing.assert_close(engine.module.adapters["fake_score"], torch.ones(2))
        torch.testing.assert_close(
            engine.module.adapters["student_ema"], (torch.ones(2) + engine.module.adapters["default"]) / 2
        )
        engine.lr_scheduler_step()
        assert engine.schedulers["student"].last_epoch == 1

    def test_inactive_gradients_fail_instead_of_cross_role_clipping(self):
        engine = engine_shell()
        engine.role_parameters["fake_score"][0].grad = torch.ones(2)
        with pytest.raises(RuntimeError, match="Gradient leaked"):
            engine.optimizer_step()


class TestDMDAccumulation:
    def test_metric_objects_and_unequal_microbatch_means(self, monkeypatch):
        monkeypatch.setattr(dmd_impl, "get_device_id", cpu_device)
        monkeypatch.setattr(
            dmd_impl,
            "get_torch_device",
            MagicMock(
                return_value=MagicMock(
                    max_memory_allocated=MagicMock(return_value=1024**3),
                    max_memory_reserved=MagicMock(return_value=2 * 1024**3),
                )
            ),
        )
        engine = engine_shell()
        engine.ulysses_sequence_parallel_size = 1
        engine.get_data_parallel_group = MagicMock(return_value=None)
        engine.forward_step = microbatch_metrics
        data = TensorDict({"value": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3])
        tu.assign_non_tensor(data, micro_batch_size_per_gpu=2)
        output = engine.forward_backward_batch(data, None)
        assert output["metrics"]["dmd/loss"].aggregate() == pytest.approx(2.0)
        assert output["metrics"]["dmd/active_elements"].aggregate() == 3
        assert output["metrics"]["perf/forward_s"].aggregate() == 3
        assert output["metrics"]["perf/max_memory_allocated_gib"].aggregate() == 1


class TestDMDWorker:
    def test_reuses_one_minibatch_and_selects_before_context(self, monkeypatch):
        result = tu.get_tensordict({}, {"metrics": {"loss": [1.0]}})
        train = MagicMock(return_value=result)
        monkeypatch.setattr(TrainingWorker, "train_mini_batch", train)
        worker = object.__new__(DMDTrainingWorker)
        worker.engine = MagicMock(active_stage="student", last_step_succeeded=False)
        worker.dmd_config = DiffusionDMDConfig(fake_score_micro_batch_size_per_gpu=2)
        data = TensorDict({}, batch_size=[4])
        tu.assign_non_tensor(data, dmd_stage="fake_score")
        output = worker.update_actor(data)
        train.assert_called_once()
        assert tu.get_non_tensor_data(data, "num_mini_batch", None) == 1
        assert tu.get_non_tensor_data(data, "epochs", None) == 1
        assert tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", None) == 2
        assert tu.get(output, "metrics")["dmd/update_applied"] == 0
        assert [call.args[0] for call in worker.engine.select_stage.call_args_list] == ["fake_score", "student"]
