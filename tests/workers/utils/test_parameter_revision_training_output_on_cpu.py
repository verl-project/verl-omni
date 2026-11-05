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
import threading

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


def test_checkpoint_argument_error_does_not_invalidate_parameters():
    engine = TrackedCPU()
    tracker = engine.enable_parameter_tracking()
    before = tracker.version()
    with pytest.raises(TypeError):
        engine.load_checkpoint()
    assert tracker.version() == before
    engine.load_checkpoint(copy.deepcopy(engine.model.state_dict()))
    assert tracker.version() != before


@pytest.mark.parametrize("method", ["initialize", "optimizer_step"])
def test_writer_argument_error_does_not_invalidate_parameters(method):
    engine = TrackedCPU()
    tracker = engine.enable_parameter_tracking()
    before = tracker.version()
    with pytest.raises(TypeError):
        getattr(engine, method)(unexpected=True)
    assert tracker.version() == before


def test_checkpoint_backend_failure_still_invalidates_parameters():
    engine = TrackedCPU()
    tracker = engine.enable_parameter_tracking()
    with pytest.raises(RuntimeError):
        engine.load_checkpoint({})
    with pytest.raises(RuntimeError, match="Partial actor operation"):
        tracker.version()


def test_uninspectable_checkpoint_keeps_conservative_guard(monkeypatch):
    engine = TrackedCPU()
    tracker = engine.enable_parameter_tracking()

    def unavailable(_):
        raise ValueError("signature unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(revision.inspect, "signature", unavailable)
        with pytest.raises(TypeError):
            engine.load_checkpoint()
    with pytest.raises(RuntimeError, match="Partial actor operation"):
        tracker.version()


def test_disabled_tracking_does_not_inspect_backend(monkeypatch):
    engine = TrackedCPU()

    def unexpected(_):
        raise AssertionError("signature inspection must be opt-in")

    monkeypatch.setattr(revision.inspect, "signature", unexpected)
    engine.initialize()
    engine.load_checkpoint(copy.deepcopy(engine.model.state_dict()))
    engine.optimizer_step()


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


@pytest.mark.parametrize("operation", ["save_checkpoint", "infer_batch"])
def test_failed_read_only_access_leaves_tracker_usable(operation):
    tracker = revision.ParameterRevision()
    before = tracker.version()
    with pytest.raises(ValueError, match=operation):
        with tracker.access():
            raise ValueError(operation)
    assert tracker.version() == before
    with tracker.mutation("optimizer"):
        pass
    assert tracker.version().revision == before.revision + 1


def test_failed_engine_offload_poisons_even_inside_read_access():
    class PartialMoveBackend(CPUBackend):
        def to(self, *args, **kwargs):
            raise RuntimeError("partial device move")

    class TrackedMove(revision.ParameterTrackingMixin, PartialMoveBackend):
        pass

    engine = TrackedMove()
    tracker = engine.enable_parameter_tracking()
    with pytest.raises(RuntimeError, match="partial device move"):
        with tracker.access():
            engine.to("cpu")
    with pytest.raises(RuntimeError, match="Partial actor"):
        tracker.version()


def test_offload_on_export_thread_inside_actor_access():
    class MoveBackend(CPUBackend):
        def to(self, device):
            return device

    class TrackedMove(revision.ParameterTrackingMixin, MoveBackend):
        pass

    class Exporter:
        def __init__(self):
            self.engine = TrackedMove()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            return await asyncio.shield(revision.parameter_to_thread(self.engine.to, "cpu"))

    async def run():
        worker = Exporter()
        assert await worker.export() == "cpu"
        worker.tracker.version()

    asyncio.run(run())


def test_guarded_parameter_thread_requires_async_lifetime_manager():
    tracker = revision.ParameterRevision()
    with tracker.access(), pytest.raises(RuntimeError, match="live async access guard"):
        revision.parameter_to_thread(lambda: None)
    tracker.version()


def test_inherited_expired_access_cannot_start_a_parameter_thread():
    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()
            self.release = asyncio.Event()

        @revision.parameter_access()
        async def export(self):
            async def late_request():
                await self.release.wait()
                revision.parameter_to_thread(lambda: None)

            return asyncio.create_task(late_request())

    async def run():
        worker = Exporter()
        late = await worker.export()
        worker.release.set()
        with pytest.raises(RuntimeError, match="live async access guard"):
            await late
        worker.tracker.version()

    asyncio.run(run())


def test_delegated_parameter_threads_are_serialized():
    first_started, second_started, release = threading.Event(), threading.Event(), threading.Event()

    def first():
        first_started.set()
        assert release.wait(timeout=5)

    def second():
        second_started.set()

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            first_task = revision.parameter_to_thread(first)
            assert await asyncio.to_thread(first_started.wait, 5)
            second_task = revision.parameter_to_thread(second)
            await asyncio.gather(first_task, second_task)

    async def run():
        worker = Exporter()
        task = asyncio.create_task(worker.export())
        assert await asyncio.to_thread(first_started.wait, 5)
        try:
            await asyncio.sleep(0.05)
            assert not second_started.is_set()
        finally:
            release.set()
        await task
        assert second_started.is_set()
        worker.tracker.version()

    asyncio.run(run())


def test_access_drains_parameter_threads_registered_during_cleanup():
    first_started, first_release = threading.Event(), threading.Event()
    second_started, second_release = threading.Event(), threading.Event()

    def first():
        first_started.set()
        assert first_release.wait(timeout=5)

    def second():
        second_started.set()
        assert second_release.wait(timeout=5)

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.engine.enable_parameter_tracking()
            self.scheduled = asyncio.Event()

        @revision.parameter_access()
        async def export(self):
            revision.parameter_to_thread(first)

            async def schedule_during_cleanup():
                assert await asyncio.to_thread(first_started.wait, 5)
                revision.parameter_to_thread(second)
                self.scheduled.set()

            asyncio.create_task(schedule_during_cleanup())

    async def run():
        worker = Exporter()
        task = asyncio.create_task(worker.export())
        try:
            await asyncio.wait_for(worker.scheduled.wait(), timeout=5)
            first_release.set()
            assert await asyncio.to_thread(second_started.wait, 5)
            await asyncio.sleep(0.05)
            assert not task.done(), "Access ended with an active registered worker"
        finally:
            first_release.set()
            second_release.set()
            await task

    asyncio.run(run())


def test_nested_async_access_drains_only_its_own_parameter_threads():
    started, release = threading.Event(), threading.Event()

    def background_read():
        started.set()
        assert release.wait(timeout=5)

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()
            self.inner_done = asyncio.Event()

        @revision.parameter_access()
        async def inner(self):
            return None

        @revision.parameter_access()
        async def outer(self):
            task = revision.parameter_to_thread(background_read)
            assert await asyncio.to_thread(started.wait, 5)
            await self.inner()
            self.inner_done.set()
            await task

    async def run():
        worker = Exporter()
        task = asyncio.create_task(worker.outer())
        assert await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(0.05)
        observed_inner_done = worker.inner_done.is_set()
        release.set()
        await task
        assert observed_inner_done, "Nested read waited for an unrelated outer worker"
        worker.tracker.version()

    asyncio.run(run())


def test_direct_device_move_rejects_active_delegated_parameter_thread():
    started, release = threading.Event(), threading.Event()

    class MoveBackend(CPUBackend):
        def to(self, device):
            return device

    class TrackedMove(revision.ParameterTrackingMixin, MoveBackend):
        pass

    def background_read():
        started.set()
        assert release.wait(timeout=5)

    class Exporter:
        def __init__(self):
            self.engine = TrackedMove()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            task = revision.parameter_to_thread(background_read)
            assert await asyncio.to_thread(started.wait, 5)
            try:
                with pytest.raises(RuntimeError, match="Concurrent parameter thread"):
                    self.engine.to("cpu")
            finally:
                release.set()
            await task

    async def run():
        worker = Exporter()
        await worker.export()
        worker.tracker.version()

    asyncio.run(run())


def test_failed_delegated_offload_poisons_tracker():
    class PartialMoveBackend(CPUBackend):
        def to(self, device):
            raise RuntimeError("partial worker offload")

    class TrackedMove(revision.ParameterTrackingMixin, PartialMoveBackend):
        pass

    class Exporter:
        def __init__(self):
            self.engine = TrackedMove()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            return await asyncio.shield(revision.parameter_to_thread(self.engine.to, "cpu"))

    async def run():
        worker = Exporter()
        with pytest.raises(RuntimeError, match="partial worker offload"):
            await worker.export()
        with pytest.raises(RuntimeError, match="Partial actor"):
            worker.tracker.version()

    asyncio.run(run())


def test_cancelled_async_export_holds_access_until_worker_thread_finishes():
    started = threading.Event()
    release = threading.Event()

    def background_read():
        started.set()
        assert release.wait(timeout=5)

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            await asyncio.shield(revision.parameter_to_thread(background_read))

    async def run():
        worker = Exporter()
        task = asyncio.create_task(worker.export())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        try:
            await asyncio.sleep(0.05)
            assert not task.done(), "The parameter guard was released before the worker thread stopped"
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "Repeated cancellation released the parameter guard"
            with pytest.raises(RuntimeError, match="Concurrent"):
                worker.tracker.version()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        worker.tracker.version()

    asyncio.run(run())


def test_failed_resume_waits_for_parameter_thread_and_preserves_error():
    started = threading.Event()
    release = threading.Event()

    def background_read():
        started.set()
        assert release.wait(timeout=5)

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            revision.parameter_to_thread(background_read)
            assert await asyncio.to_thread(started.wait, 5)
            raise ValueError("resume failed")

    async def run():
        worker = Exporter()
        task = asyncio.create_task(worker.export())
        assert await asyncio.to_thread(started.wait, 5)
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
            with pytest.raises(RuntimeError, match="Concurrent"):
                worker.tracker.version()
        finally:
            release.set()
        with pytest.raises(ValueError, match="resume failed"):
            await task
        worker.tracker.version()

    asyncio.run(run())


def test_resume_error_remains_primary_when_background_worker_also_fails():
    started = threading.Event()
    release = threading.Event()

    def background_read():
        started.set()
        assert release.wait(timeout=5)
        raise OSError("gather failed")

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            revision.parameter_to_thread(background_read)
            assert await asyncio.to_thread(started.wait, 5)
            raise ValueError("resume failed")

    async def run():
        loop = asyncio.get_running_loop()
        diagnostics = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, details: diagnostics.append(details))
        worker = Exporter()
        task = asyncio.create_task(worker.export())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await asyncio.sleep(0.05)
            assert not task.done()
            release.set()
            with pytest.raises(ValueError, match="resume failed"):
                await task
            await asyncio.sleep(0)
            assert not diagnostics
            worker.tracker.version()
        finally:
            release.set()
            loop.set_exception_handler(previous)

    asyncio.run(run())


def test_cancelling_parameter_thread_wait_does_not_cancel_the_tracked_thread():
    started = threading.Event()
    release = threading.Event()

    def background_read():
        started.set()
        assert release.wait(timeout=5)

    class Exporter:
        def __init__(self):
            self.engine = TrackedCPU()
            self.tracker = self.engine.enable_parameter_tracking()

        @revision.parameter_access()
        async def export(self):
            handle = revision.parameter_to_thread(background_read)
            assert await asyncio.to_thread(started.wait, 5)
            handle.cancel()
            await handle

    async def run():
        worker = Exporter()
        task = asyncio.create_task(worker.export())
        assert await asyncio.to_thread(started.wait, 5)
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
            with pytest.raises(RuntimeError, match="Concurrent"):
                worker.tracker.version()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        worker.tracker.version()

    asyncio.run(run())


def test_collective_preflight_excludes_concurrent_writer_and_preserves_ticket_on_rejection():
    tracker = revision.ParameterRevision()
    version = tracker.version()
    results = []

    def competing_writer():
        try:
            with tracker.mutation("optimizer"):
                results.append("wrote")
        except RuntimeError as exc:
            results.append(str(exc))

    with pytest.raises(ValueError, match="rejected plan"):
        with tracker.preflight_update(version, "batch"):
            thread = threading.Thread(target=competing_writer)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert results == ["Concurrent parameter access"]
            raise ValueError("rejected plan")
    assert tracker.version() == version
    with tracker.update(version, "batch"):
        with tracker.mutation("optimizer"):
            pass


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


def test_untracked_parameter_thread_retains_native_task_cancellation():
    import asyncio
    import threading

    from verl_omni.workers.utils.parameter_revision import parameter_to_thread

    async def run():
        started, release = threading.Event(), threading.Event()

        def work():
            started.set()
            release.wait(timeout=5)

        task = parameter_to_thread(work)
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert isinstance(task, asyncio.Task)
            assert task.cancelled()
        finally:
            release.set()

    asyncio.run(run())


def test_invalid_device_move_arguments_do_not_poison_before_call():
    class Backend(CPUBackend):
        def to(self, device):
            raise AssertionError("The method must not be entered")

    class Engine(revision.ParameterTrackingMixin, Backend):
        pass

    engine = Engine()
    tracker = engine.enable_parameter_tracking()
    before = tracker.version()
    with pytest.raises(TypeError):
        engine.to("cpu", unexpected=True)
    assert tracker.version() == before


def test_device_move_without_inspectable_signature_still_executes():
    class Move:
        __signature__ = object()

        def __call__(self, device):
            return device

    class Backend(CPUBackend):
        to = Move()

    class Engine(revision.ParameterTrackingMixin, Backend):
        pass

    engine = Engine()
    tracker = engine.enable_parameter_tracking()
    assert engine.to("cpu") == "cpu"
    assert tracker.version().revision == 0


def test_uninspectable_failed_device_move_remains_fail_closed():
    class Move:
        __signature__ = object()

        def __call__(self, device):
            raise RuntimeError("uncertain partial move")

    class Backend(CPUBackend):
        to = Move()

    class Engine(revision.ParameterTrackingMixin, Backend):
        pass

    engine = Engine()
    tracker = engine.enable_parameter_tracking()
    with pytest.raises(RuntimeError, match="uncertain partial move"):
        engine.to("cpu")
    with pytest.raises(RuntimeError, match="Partial actor"):
        tracker.version()
