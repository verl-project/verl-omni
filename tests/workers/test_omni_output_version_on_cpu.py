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

"""CPU contracts only: these do not qualify the V1 worker integration or GPU math."""

import asyncio
import copy

import pytest
import torch
from tensordict import TensorDict

from verl_omni.workers.utils import parameter_revision as revision
from verl_omni.workers.utils import training_output as outputs


class CPUBackend:
    def __init__(self):
        self.model = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        self.calls = 0

    def initialize(self):
        with torch.no_grad():
            self.model.weight.fill_(0.2)
            self.model.bias.zero_()

    def load_checkpoint(self, state):
        self.model.load_state_dict(state)

    def optimizer_step(self):
        self.calls += 1
        self.optimizer.step()

    def backward(self, data):
        self.optimizer.zero_grad()
        self.model(data).square().mean().backward()


class TrackedCPU(revision.ParameterTrackingMixin, CPUBackend):
    pass


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, tuple | list):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=False):
            assert_equal(left, right)
    else:
        assert a == b


def test_tracking_preserves_real_adam_updates_and_is_opt_in():
    baseline, candidate = CPUBackend(), TrackedCPU()
    baseline.initialize()
    candidate.initialize()
    assert not hasattr(candidate, "_parameter_revision")
    tracker = candidate.enable_parameter_tracking()
    assert candidate.enable_parameter_tracking() is tracker
    for step in range(3):
        batch = torch.arange(15, dtype=torch.float64).reshape(5, 3) / (step + 4)
        baseline.backward(batch)
        baseline.optimizer_step()
        with tracker.update(tracker.version(), f"cohort-{step}"):
            candidate.backward(batch)
            candidate.optimizer_step()
        assert_equal(baseline.model.state_dict(), candidate.model.state_dict())
        assert_equal(baseline.optimizer.state_dict(), candidate.optimizer.state_dict())
        for left, right in zip(baseline.model.parameters(), candidate.model.parameters(), strict=False):
            assert_equal(left.grad, right.grad)
    assert candidate.calls == 3


def test_load_and_snapshot_restore_invalidate_prior_versions():
    engine = TrackedCPU()
    tracker = engine.enable_parameter_tracking()
    engine.initialize()
    before = tracker.version()
    engine.load_checkpoint(copy.deepcopy(engine.model.state_dict()))
    with pytest.raises(ValueError, match="Stale"):
        with tracker.update(before, "old-load"):
            pytest.fail("stale update must not execute")
    before = tracker.version()
    with engine.parameter_mutation("restore"):
        with torch.no_grad():
            engine.model.weight.add_(0.1)
    with pytest.raises(ValueError, match="Stale"):
        with tracker.update(before, "old-restore"):
            pytest.fail("stale update must not execute")


@pytest.mark.parametrize("updates", [0, 2])
def test_requires_exactly_one_optimizer_write(updates):
    engine = TrackedCPU()
    tracker = engine.enable_parameter_tracking()
    with pytest.raises(RuntimeError, match="exactly one"):
        with tracker.update(tracker.version(), "bad-count"):
            for _ in range(updates):
                engine.optimizer_step()
    with pytest.raises(RuntimeError, match="Partial"):
        tracker.version()


def test_partial_writer_poisoned_and_not_retried():
    class FailingBackend(CPUBackend):
        def optimizer_step(self):
            super().optimizer_step()
            raise RuntimeError("after-write failure")

    class FailingTracked(revision.ParameterTrackingMixin, FailingBackend):
        pass

    engine = FailingTracked()
    tracker = engine.enable_parameter_tracking()
    with pytest.raises(RuntimeError, match="after-write"):
        with tracker.update(tracker.version(), "failed"):
            engine.optimizer_step()
    with pytest.raises(RuntimeError, match="Partial"):
        engine.optimizer_step()
    assert engine.calls == 1


def test_duplicate_or_unannotated_writer_refused():
    class AlreadyTracked(CPUBackend):
        @revision.parameter_writer("optimizer")
        def optimizer_step(self):
            return super().optimizer_step()

    class Duplicate(revision.ParameterTrackingMixin, AlreadyTracked):
        pass

    with pytest.raises(ValueError, match="already tracks"):
        Duplicate().enable_parameter_tracking()

    class Untracked(TrackedCPU):
        def optimizer_step(self):
            return CPUBackend.optimizer_step(self)

    with pytest.raises(ValueError, match="Untracked"):
        Untracked().enable_parameter_tracking()


def test_awaiting_export_excludes_other_async_parameter_work():
    async def scenario():
        tracker = revision.ParameterRevision()
        active = asyncio.Event()
        release = asyncio.Event()

        async def export():
            with tracker.access():
                active.set()
                await release.wait()

        task = asyncio.create_task(export())
        await active.wait()
        with pytest.raises(RuntimeError, match="Concurrent async"):
            tracker.version()
        release.set()
        await task
        tracker.version()

    asyncio.run(scenario())


@pytest.mark.parametrize("shuffle", [False, True])
def test_outputs_detached_and_restored_to_original_rows(shuffle):
    values = torch.arange(21, dtype=torch.float32).reshape(7, 3).requires_grad_()
    data = TensorDict({"log_probs": values}, batch_size=[7])
    batch, order = outputs.prepare_single_update_batch(
        data,
        mini_batch_size=7,
        epochs=1,
        seed=13,
        dataloader_kwargs={"shuffle": shuffle},
    )
    restored = outputs.restore_single_update_output(batch, order)
    assert torch.equal(restored["log_probs"], values)
    assert not restored["log_probs"].requires_grad
    assert "__omni_training_output_row__" not in data


@pytest.mark.parametrize("mini_batch_size,epochs", [(3, 1), (7, 2)])
def test_output_contract_rejects_multiple_updates(mini_batch_size, epochs):
    data = TensorDict({"log_probs": torch.ones(7, 3)}, batch_size=[7])
    with pytest.raises(ValueError, match="one complete"):
        outputs.prepare_single_update_batch(
            data, mini_batch_size=mini_batch_size, epochs=epochs, seed=0, dataloader_kwargs={}
        )


def test_duplicate_output_order_rejected():
    data = TensorDict({"log_probs": torch.ones(3, 2)}, batch_size=[3])
    with pytest.raises(ValueError, match="rows"):
        outputs.restore_single_update_output(data, torch.tensor([0, 0, 2]))


@pytest.mark.parametrize("per_token", [False, True])
def test_microbatch_detach_preserves_loss_gradients_and_releases_unused_graph(per_token):
    import gc
    import weakref

    def run(detach):
        x = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
        saved = []

        def pack(tensor):
            # A hook must not retain the original autograd tensor: that would
            # itself introduce a graph cycle and invalidate this lifetime test.
            packed = tensor.detach()
            if tensor.grad_fn is not None:
                saved.append(weakref.ref(packed))
            return packed

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            log_probs = x.square().sum(-1)
            diagnostic = x.exp().sum(-1)
            loss = log_probs.mean()
        reduced = {"model_output": {"log_probs": log_probs, "entropy": diagnostic}, "metrics": {"count": 3}}
        count = torch.tensor(3)
        result = (loss, count, reduced) if per_token else (loss, reduced)
        if detach:
            result = outputs.detach_microbatch_outputs(result)
            assert result[0] is loss
            if per_token:
                assert result[1] is count
            assert result[-1]["metrics"] is reduced["metrics"]
            for value in result[-1]["model_output"].values():
                assert not value.requires_grad and value.grad_fn is None
        del reduced, diagnostic, log_probs
        loss.backward()
        gc.collect()
        retained = sum(ref() is not None for ref in saved)
        return x.grad.clone(), retained, result[-1]["model_output"]

    baseline_grad, baseline_retained, baseline_output = run(False)
    actual_grad, actual_retained, actual_output = run(True)
    assert torch.equal(actual_grad, baseline_grad)
    assert baseline_retained > 0 and actual_retained == 0
    for key in baseline_output:
        assert torch.equal(actual_output[key], baseline_output[key])


def test_microbatch_detach_keeps_jagged_rows_and_metadata():
    values = torch.arange(5, dtype=torch.float64, requires_grad=True)
    nested = torch.nested.nested_tensor_from_jagged(values.square(), offsets=torch.tensor([0, 2, 2, 5]))
    loss = nested.values().sum()
    result = outputs.detach_microbatch_outputs((loss, {"model_output": {"log_probs": nested, "meta": ["rows", 3]}}))
    assert result[0] is loss
    assert result[-1]["model_output"]["meta"] == ["rows", 3]
    detached = result[-1]["model_output"]["log_probs"]
    assert torch.equal(detached.offsets(), nested.offsets())
    assert torch.equal(detached.values(), nested.values()) and not detached.requires_grad
    loss.backward()
    assert torch.equal(values.grad, 2 * values.detach())
