# Copyright 2026 verl-omni contributors
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

"""Execute owned method bodies with explicit CPU engine/storage fixtures.

Registration and heavy engine imports are excluded. This covers worker logic,
not Ray dispatch, TransferQueue, distributed Megatron or end-to-end acceptance.
"""

import ast
import copy
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from codetiming import Timer
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric

from tests.workers.test_omni_output_version_on_cpu import CPUBackend, assert_equal
from verl_omni.workers.utils.parameter_revision import ParameterTrackingMixin, parameter_access
from verl_omni.workers.utils.training_output import prepare_single_update_batch, restore_single_update_output

ROOT = Path(__file__).resolve().parents[2]


def extracted(path, name, methods, base=object, extra=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.bases = [ast.Name(id="FixtureBase", ctx=ast.Load())]
    cls.decorator_list = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name in methods]
    for fn in cls.body:
        fn.decorator_list = [
            d
            for d in fn.decorator_list
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "parameter_access"
        ]
    scope = {
        **globals(),
        "FixtureBase": base,
        "chain": chain,
        "Timer": Timer,
        "NonTensorData": NonTensorData,
        "Metric": Metric,
        "parameter_access": parameter_access,
        "prepare_single_update_batch": prepare_single_update_batch,
        "restore_single_update_output": restore_single_update_output,
        **(extra or {}),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
    return scope[name]


class CPUEngine(ParameterTrackingMixin, CPUBackend):
    def train_mode(self, **kwargs):
        return nullcontext()

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def is_mp_src_rank_with_outputs(self):
        return True


class SnapshotFixture:
    def save_model_to_cpu(self, n):
        self.cpu_saved_models[n] = copy.deepcopy(self.actor.engine.model.state_dict())

    def restore_model_from_cpu(self, n):
        self.actor.engine.model.load_state_dict(self.cpu_saved_models[n])

    def clear_cpu_model(self, n):
        self.cpu_saved_models.pop(n, None)


def test_owned_snapshot_restore_invalidates_previous_version_and_retains_copy():
    cls = extracted(
        "verl_omni/workers/omni_engine_workers.py",
        "OmniDetachActorWorker",
        {"save_model_to_cpu", "restore_model_from_cpu", "clear_cpu_model"},
        SnapshotFixture,
    )
    worker = cls()
    engine = CPUEngine()
    engine.initialize()
    worker.actor = SimpleNamespace(engine=engine)
    worker.cpu_saved_models = {}
    tracker = engine.enable_parameter_tracking()
    version = tracker.version()
    worker.save_model_to_cpu(0)
    engine.backward(torch.ones(5, 3, dtype=torch.float64))
    engine.optimizer_step()
    worker.restore_model_from_cpu(0)
    assert tracker.version().revision == version.revision + 2
    assert_equal(engine.model.state_dict(), worker.cpu_saved_models[0])
    with pytest.raises(ValueError, match="Stale"):
        with tracker.update(version, "stale"):
            pass
    worker.clear_cpu_model(0)
    with pytest.raises(ValueError, match="does not exist"):
        worker.restore_model_from_cpu(0)
    with pytest.raises(RuntimeError, match="Partial"):
        tracker.version()


def append_to_dict(target, source):
    for key, value in source.items():
        target.setdefault(key, []).append(value)


class LoopFixture:
    def _check_training_output_support(self):
        # Explicit CPU test backend, not a claim of Megatron qualification.
        pass

    def train_batch(self, data):
        preupdate = self.engine.model(data["features"])
        detached = preupdate.detach().clone()
        self.observed_preupdate = detached[data["row"].argsort()]
        self.engine.optimizer.zero_grad()
        preupdate.square().mean().backward()
        self.engine.optimizer_step()
        tensors = {"log_probs": detached} if tu.get(data, "return_model_output", False) else {}
        return tu.get_tensordict(tensors, non_tensor_dict={"metrics": {"loss": 1.0}})


def worker():
    cls = extracted(
        "verl_omni/workers/engine_workers.py",
        "TrainingWorker",
        {"train_mini_batch"},
        LoopFixture,
        {"maybe_fix_3d_position_ids": lambda data: None},
    )
    result = cls()
    result.engine = CPUEngine()
    result.engine.initialize()
    return result


@pytest.mark.parametrize("shuffle", [False, True])
def test_owned_worker_returns_preupdate_detached_outputs_in_input_order(shuffle):
    base, candidate = worker(), worker()
    data = TensorDict(
        {"features": torch.arange(15, dtype=torch.float64).reshape(5, 3), "row": torch.arange(5)}, batch_size=[5]
    )
    tu.assign_non_tensor(data, mini_batch_size=5, epochs=1, seed=7, dataloader_kwargs={"shuffle": shuffle})
    normal = base.train_mini_batch(data.clone())
    tu.assign_non_tensor(data, return_model_output=True)
    actual = candidate.train_mini_batch(data)
    assert "log_probs" not in normal
    assert not actual["log_probs"].requires_grad
    assert torch.equal(actual["log_probs"], base.observed_preupdate)
    assert_equal(base.engine.model.state_dict(), candidate.engine.model.state_dict())
    assert_equal(base.engine.optimizer.state_dict(), candidate.engine.optimizer.state_dict())
    assert base.engine.calls == candidate.engine.calls == 1


def test_owned_worker_rejects_multiple_updates_before_forward():
    instance = worker()
    data = TensorDict({"features": torch.ones(5, 3, dtype=torch.float64)}, batch_size=[5])
    tu.assign_non_tensor(data, mini_batch_size=5, epochs=2, return_model_output=True)
    with pytest.raises(ValueError, match="one optimizer update"):
        instance.train_mini_batch(data)
    assert instance.engine.calls == 0
