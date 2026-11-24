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
"""Pinned contracts of the framework-owned scorer microbatcher."""

import asyncio

import pytest
import pytest_asyncio
import torch

from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor, ScorerMicrobatcher


def _recording_scorer(calls, *, fail=None):
    def score_batch(requests):
        calls.append([request["i"] for request in requests])
        if fail is not None:
            raise fail
        return [{"score": request["i"]} for request in requests]

    return score_batch


@pytest.mark.parametrize("method", ["start", "submit", "pause", "close"])
def test_batcher_rejects_another_event_loop_before_mutation(method):
    loop = asyncio.new_event_loop()
    batcher = ScorerMicrobatcher(_recording_scorer([]))
    try:
        loop.run_until_complete(batcher.start())
        loop.run_until_complete(batcher.pause())
        call = getattr(batcher, method)
        with pytest.raises(RuntimeError, match="across event loops"):
            asyncio.run(call({"i": 1}) if method == "submit" else call())
        assert batcher._state == "paused"
        assert batcher._consumer is None
        assert not batcher._queue
    finally:
        loop.run_until_complete(batcher.close())
        loop.close()


@pytest_asyncio.fixture
async def microbatcher_factory():
    owned = []

    def create(*args, **kwargs):
        microbatcher = ScorerMicrobatcher(*args, **kwargs)
        owned.append(microbatcher)
        return microbatcher

    yield create
    for microbatcher in reversed(owned):
        consumer = microbatcher._consumer
        await asyncio.wait_for(microbatcher.close(), 5)
        assert consumer is None or consumer.done(), "owned consumer must settle before the test loop closes"
        assert not microbatcher._queue


@pytest.mark.asyncio
async def test_lone_request_scores_immediately_at_batch_one(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), name="m")
    await microbatcher.start()
    assert await microbatcher.submit({"i": 7}) == {"score": 7}
    assert calls == [[7]]


@pytest.mark.asyncio
@pytest.mark.parametrize("callable_instance", [False, True])
async def test_awaitable_forward_result_is_awaited(microbatcher_factory, callable_instance):
    async def result(requests):
        return [{"score": request["i"]} for request in requests]

    class AsyncCallable:
        async def __call__(self, requests):
            return await result(requests)

    def sync_factory(requests):
        return result(requests)

    batcher = microbatcher_factory(AsyncCallable() if callable_instance else sync_factory)
    await batcher.start()
    assert await batcher.submit({"i": 3}) == {"score": 3}


@pytest.mark.asyncio
async def test_cancelled_dependency_fails_only_its_batch(microbatcher_factory):
    entered = asyncio.Event()
    dependency = asyncio.get_running_loop().create_future()

    async def score(requests):
        if requests[0]["i"] == 0:
            entered.set()
            await dependency
        return [{"score": request["i"]} for request in requests]

    batcher = microbatcher_factory(score)
    await batcher.start()
    first = asyncio.create_task(batcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    second = asyncio.create_task(batcher.submit({"i": 1}))
    try:
        async with asyncio.timeout(5):
            while not batcher._queue:
                await asyncio.sleep(0)
        dependency.cancel()
        with pytest.raises(RuntimeError, match="cancelled its batch") as caught:
            await asyncio.wait_for(first, 5)
        assert isinstance(caught.value.__cause__, asyncio.CancelledError)
        assert await asyncio.wait_for(second, 5) == {"score": 1}
        assert batcher._state == "running"
        assert await batcher.submit({"i": 2}) == {"score": 2}
    finally:
        dependency.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_packs_concurrent_requests_and_preserves_request_result_mapping(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), max_batch_size=8, name="m")
    await microbatcher.start()
    results = await asyncio.gather(*(microbatcher.submit({"i": i}) for i in range(4)))
    assert results == [{"score": i} for i in range(4)]
    assert calls == [[0, 1, 2, 3]]


@pytest.mark.asyncio
async def test_packing_respects_max_batch_size(microbatcher_factory):
    calls = []

    def score_batch(requests):
        calls.append(len(requests))
        return [{"score": request["i"]} for request in requests]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=2, name="m")
    await microbatcher.start()
    results = await asyncio.gather(*(microbatcher.submit({"i": i}) for i in range(5)))
    assert results == [{"score": i} for i in range(5)]
    assert calls == [2, 2, 1]


@pytest.mark.asyncio
async def test_unbatched_scorer_scores_one_request_per_call(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), max_batch_size=None, name="m")
    await microbatcher.start()
    await asyncio.gather(*(microbatcher.submit({"i": i}) for i in range(3)))
    assert calls == [[0], [1], [2]]


@pytest.mark.asyncio
async def test_per_request_preprocessing_failure_is_isolated(microbatcher_factory):
    def score_batch(requests):
        return [
            ValueError(f"bad image {request['i']}") if request["i"] == 1 else {"score": request["i"]}
            for request in requests
        ]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=4, name="m")
    await microbatcher.start()
    outcomes = await asyncio.gather(
        *(microbatcher.submit({"i": i}) for i in range(3)),
        return_exceptions=True,
    )
    assert outcomes[0] == {"score": 0}
    assert isinstance(outcomes[1], ValueError) and "bad image 1" in str(outcomes[1])
    assert outcomes[2] == {"score": 2}


@pytest.mark.asyncio
async def test_forward_failure_fails_every_request_in_the_microbatch(microbatcher_factory):
    calls = []
    boom = OSError("cuda gone")
    microbatcher = microbatcher_factory(_recording_scorer(calls, fail=boom), max_batch_size=4, name="m")
    await microbatcher.start()
    outcomes = await asyncio.gather(
        *(microbatcher.submit({"i": i}) for i in range(2)),
        return_exceptions=True,
    )
    assert calls == [[0, 1]]
    assert outcomes == [boom, boom]


@pytest.mark.asyncio
async def test_wrong_length_output_fails_every_request(microbatcher_factory):
    def score_batch(requests):
        return [{"score": request["i"]} for request in requests[:-1]]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=4, name="m")
    await microbatcher.start()
    outcomes = await asyncio.gather(
        *(microbatcher.submit({"i": i}) for i in range(3)),
        return_exceptions=True,
    )
    assert all(isinstance(outcome, ValueError) for outcome in outcomes)


@pytest.mark.asyncio
async def test_full_queue_pause_fails_queued_and_capacity_blocked_retryably_and_reopens(microbatcher_factory):
    release = asyncio.Event()
    entered = asyncio.Event()
    accepted = []

    async def score_batch(requests):
        accepted.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    # Unbatched forward so exactly one request is in flight while others queue.
    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, max_inflight=2, name="m")
    await microbatcher.start()
    in_flight = asyncio.create_task(microbatcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    assert accepted == [0]

    queued = [asyncio.create_task(microbatcher.submit({"i": i})) for i in (1, 2)]
    blocked = asyncio.create_task(microbatcher.submit({"i": 3}))
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(microbatcher._queue) == 2

    try:
        pausing = asyncio.create_task(microbatcher.pause())
        outcomes = await asyncio.gather(*queued, blocked, return_exceptions=True)
        assert all(isinstance(outcome, RuntimeError) and "retry after wake_up" in str(outcome) for outcome in outcomes)

        release.set()
        assert await in_flight == {"score": 0}
        await pausing
        assert accepted == [0]

        await microbatcher.start()
        assert await microbatcher.submit({"i": 9}) == {"score": 9}
    finally:
        release.set()
        await asyncio.gather(in_flight, *queued, blocked, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_fails_unaccepted_terminally_and_settles_in_flight_forward(microbatcher_factory):
    release = asyncio.Event()
    entered = asyncio.Event()
    accepted = []

    async def score_batch(requests):
        accepted.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, max_inflight=2, name="m")
    await microbatcher.start()
    in_flight = asyncio.create_task(microbatcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    pending = [asyncio.create_task(microbatcher.submit({"i": i})) for i in (1, 2)]
    try:
        closing = asyncio.create_task(microbatcher.close())
        outcomes = await asyncio.gather(*pending, return_exceptions=True)
        assert all(
            isinstance(outcome, RuntimeError) and "closed before this request was accepted" in str(outcome)
            for outcome in outcomes
        )

        release.set()
        assert await in_flight == {"score": 0}
        await closing

        with pytest.raises(RuntimeError, match="closed"):
            await microbatcher.submit({"i": 4})
        with pytest.raises(RuntimeError, match="closed"):
            await microbatcher.start()
    finally:
        release.set()
        await asyncio.gather(in_flight, *pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_disturb_settlement_of_accepted_work(microbatcher_factory):
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = []

    async def score_batch(requests):
        entered.set()
        await release.wait()
        settled.extend(request["i"] for request in requests)
        return [{"score": request["i"]} for request in requests]

    microbatcher = microbatcher_factory(score_batch, name="m")
    await microbatcher.start()
    cancelled = asyncio.ensure_future(microbatcher.submit({"i": 0}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        cancelled.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not cancelled.done(), "settlement must precede the cancellation"

        follower = asyncio.create_task(microbatcher.submit({"i": 1}))
        await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert await follower == {"score": 1}
    assert settled == [0, 1]


@pytest.mark.asyncio
async def test_caller_cancellation_withdraws_a_queued_request_before_dequeue(microbatcher_factory):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def score_batch(requests):
        calls.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    # Unbatched forward so exactly one request is accepted while the other queues.
    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, name="m")
    await microbatcher.start()
    accepted = asyncio.create_task(microbatcher.submit({"i": 0}))
    try:
        await asyncio.wait_for(entered.wait(), 5)

        queued = asyncio.create_task(microbatcher.submit({"i": 1}))
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(microbatcher._queue) == 1

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert not microbatcher._queue, "the withdrawn request must leave the queue"

    finally:
        release.set()
    assert await accepted == {"score": 0}
    assert await microbatcher.submit({"i": 2}) == {"score": 2}
    assert calls == [0, 2], "a withdrawn request is never scored"


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_tensor_request_withdrawal_preserves_siblings_and_frees_capacity(microbatcher_factory, repeat_cancel):
    entered = asyncio.Event()
    release = asyncio.Event()
    blocked_started = asyncio.Event()
    calls = []

    async def score_batch(requests):
        calls.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    def request(index):
        return {"data_source": "same", "solution_image": torch.full((3, 2, 2), index), "i": index}

    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, max_inflight=2, name="m")
    await microbatcher.start()
    accepted = asyncio.create_task(microbatcher.submit(request(0)))
    await asyncio.wait_for(entered.wait(), 5)
    first = asyncio.create_task(microbatcher.submit(request(1)))
    cancelled = asyncio.create_task(microbatcher.submit(request(2)))

    async def submit_blocked():
        blocked_started.set()
        return await microbatcher.submit(request(3))

    blocked = None
    try:
        async with microbatcher._cond:
            await asyncio.wait_for(microbatcher._cond.wait_for(lambda: len(microbatcher._queue) == 2), 5)
        blocked = asyncio.create_task(submit_blocked())
        await asyncio.wait_for(blocked_started.wait(), 5)
        assert not blocked.done()
        async with microbatcher._cond:
            cancelled.cancel()
            await asyncio.sleep(0)
            if repeat_cancel:
                cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled, 5)
        async with microbatcher._cond:
            await asyncio.wait_for(microbatcher._cond.wait_for(lambda: len(microbatcher._queue) == 2), 5)
            assert [item["i"] for item, _ in microbatcher._queue] == [1, 3]
        release.set()
        assert await asyncio.wait_for(asyncio.gather(accepted, first, blocked), 5) == [
            {"score": 0},
            {"score": 1},
            {"score": 3},
        ]
        assert calls == [0, 1, 3]
    finally:
        release.set()
        tasks = [accepted, first, cancelled] + ([blocked] if blocked is not None else [])
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        await asyncio.wait_for(microbatcher.close(), 5)


@pytest.mark.asyncio
async def test_caller_cancellation_while_capacity_blocked_never_queues_the_request(microbatcher_factory):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def score_batch(requests):
        calls.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, max_inflight=1, name="m")
    await microbatcher.start()
    accepted = asyncio.create_task(microbatcher.submit({"i": 0}))
    try:
        await asyncio.wait_for(entered.wait(), 5)

        queued = asyncio.create_task(microbatcher.submit({"i": 1}))
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(microbatcher._queue) == 1
        blocked = asyncio.create_task(microbatcher.submit({"i": 2}))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not blocked.done(), "the submitter must park on the full queue"

        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked

    finally:
        release.set()
    assert await accepted == {"score": 0}
    assert await queued == {"score": 1}
    assert calls == [0, 1], "the capacity-blocked request is never scored"


@pytest.mark.asyncio
async def test_withdrawal_frees_the_capacity_slot_for_blocked_submitters(microbatcher_factory):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def score_batch(requests):
        calls.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, max_inflight=1, name="m")
    await microbatcher.start()
    accepted = asyncio.create_task(microbatcher.submit({"i": 0}))
    try:
        await asyncio.wait_for(entered.wait(), 5)

        queued = asyncio.create_task(microbatcher.submit({"i": 1}))
        for _ in range(10):
            await asyncio.sleep(0)
        blocked = asyncio.create_task(microbatcher.submit({"i": 2}))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not blocked.done(), "the submitter must park on the full queue"

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
    finally:
        release.set()
    assert await accepted == {"score": 0}
    assert await asyncio.wait_for(blocked, 5) == {"score": 2}, "the withdrawal must free the slot"
    assert calls == [0, 2]


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_abort_the_withdrawal(microbatcher_factory, monkeypatch):
    """A second cancel landing while _withdraw waits on the condition lock
    must not leave the cancelled request queued (and later scored)."""
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def score_batch(requests):
        calls.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    # Unbatched forward so the consumer stays busy while request 1 queues.
    microbatcher = microbatcher_factory(score_batch, max_batch_size=None, name="m")
    await microbatcher.start()
    accepted = asyncio.create_task(microbatcher.submit({"i": 0}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        queued = asyncio.create_task(microbatcher.submit({"i": 1}))
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(microbatcher._queue) == 1

        real_withdraw = microbatcher._withdraw
        attempts = []

        async def cancel_then_withdraw(entry):
            attempts.append(entry)
            if len(attempts) == 1:
                raise asyncio.CancelledError  # a second cancel hits mid-acquire
            return await real_withdraw(entry)

        monkeypatch.setattr(microbatcher, "_withdraw", cancel_then_withdraw)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert len(attempts) == 2, "the withdrawal must retry, not abort"
        assert not microbatcher._queue, "the cancelled request must leave the queue"

    finally:
        release.set()
    assert await accepted == {"score": 0}
    assert await microbatcher.submit({"i": 2}) == {"score": 2}
    assert calls == [0, 2], "the withdrawn request is never scored"


@pytest.mark.asyncio
async def test_consumer_cancellation_fails_the_batch_and_terminates_the_consumer(microbatcher_factory):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def score_batch(requests):
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    microbatcher = microbatcher_factory(score_batch, name="m")
    await microbatcher.start()
    consumer = microbatcher._consumer
    scoring = asyncio.create_task(microbatcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    queued = [asyncio.create_task(microbatcher.submit({"i": i})) for i in (1, 2)]

    consumer.cancel()
    outcomes = await asyncio.gather(scoring, *queued, return_exceptions=True)
    assert all(isinstance(outcome, RuntimeError) and "consumer was cancelled" in str(outcome) for outcome in outcomes)
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert not microbatcher._queue
    assert microbatcher._state == "closed"


@pytest.mark.asyncio
async def test_consumer_cancellation_fails_the_in_hand_request_and_what_is_still_queued(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), name="m")
    await microbatcher.start()
    consumer = microbatcher._consumer
    submits = [asyncio.create_task(microbatcher.submit({"i": i})) for i in range(3)]
    # Two loop passes: the submits enqueue; the consumer pops request 0 and
    # parks on its packing yield with requests 1 and 2 still queued.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    consumer.cancel()
    outcomes = await asyncio.gather(*submits, return_exceptions=True)
    assert all(isinstance(outcome, RuntimeError) and "consumer was cancelled" in str(outcome) for outcome in outcomes)
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert calls == [], "the cancelled forward must not run"
    assert not microbatcher._queue
    assert microbatcher._state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        await microbatcher.start()


@pytest.mark.asyncio
async def test_submit_outside_running_state_fails_fast(microbatcher_factory):
    microbatcher = microbatcher_factory(_recording_scorer([]), name="m")
    with pytest.raises(RuntimeError, match="not started"):
        await microbatcher.submit({"i": 0})

    await microbatcher.start()
    await microbatcher.pause()
    with pytest.raises(RuntimeError, match="retry after wake_up"):
        await microbatcher.submit({"i": 0})


@pytest.mark.asyncio
async def test_submit_fails_fast_when_the_consumer_was_cancelled_before_its_first_step(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), name="m")
    await microbatcher.start()
    consumer = microbatcher._consumer
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    async with asyncio.timeout(5):
        with pytest.raises(RuntimeError, match="consumer was cancelled"):
            await microbatcher.submit({"i": 0})
    assert calls == []
    assert microbatcher._state == "closed"


@pytest.mark.asyncio
async def test_submit_fails_fast_when_cancellation_is_requested_but_not_delivered(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), name="m")
    async with asyncio.timeout(5):
        await microbatcher.start()
        # No yield between the cancel request and admission: done() is still
        # False, and the consumer dies before its handler can run.
        microbatcher._consumer.cancel()
        with pytest.raises(RuntimeError, match="consumer was cancelled"):
            await microbatcher.submit({"i": 0})
    assert calls == []
    assert microbatcher._state == "closed"


@pytest.mark.asyncio
async def test_start_revival_survives_the_already_queued_settlement(microbatcher_factory):
    calls = []
    microbatcher = microbatcher_factory(_recording_scorer(calls), name="m")
    await microbatcher.start()
    dead = microbatcher._consumer
    dead.cancel()
    with pytest.raises(asyncio.CancelledError):
        # Awaiting the dead task first runs its done callback, which queues
        # the settlement without letting it land.
        await dead

    await microbatcher.start()
    revived = microbatcher._consumer
    # First pass runs the queued settlement (it must step aside); second
    # parks the revived consumer on its empty-queue wait.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert microbatcher._state == "running"
    assert revived is not dead and not revived.done()
    assert await microbatcher.submit({"i": 3}) == {"score": 3}
    assert calls == [[3]]
    await microbatcher.close()


def test_constructor_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="max_batch_size"):
        ScorerMicrobatcher(_recording_scorer([]), max_batch_size=0)
    with pytest.raises(ValueError, match="max_inflight"):
        ScorerMicrobatcher(_recording_scorer([]), max_inflight=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("second_action", ["start", "pause", "close"])
async def test_lifecycle_callers_join_the_same_live_forward(microbatcher_factory, second_action):
    entered, release = asyncio.Event(), asyncio.Event()

    async def score_batch(requests):
        entered.set()
        await release.wait()
        return [{"score": 1} for _ in requests]

    microbatcher = microbatcher_factory(score_batch)
    await microbatcher.start()
    scoring = asyncio.create_task(microbatcher.submit({}))
    async with asyncio.timeout(5):
        await entered.wait()
        first = asyncio.create_task(microbatcher.pause())
        await asyncio.sleep(0)
        second = asyncio.create_task(getattr(microbatcher, second_action)())
        await asyncio.sleep(0)
        try:
            assert not first.done()
            assert not second.done(), "every lifecycle caller must wait for the accepted forward"
        finally:
            release.set()
            await asyncio.gather(scoring, first, second)
    if second_action == "start":
        assert await microbatcher.submit({}) == {"score": 1}


@pytest.mark.asyncio
async def test_concurrent_close_and_cancelled_joiner_still_drain(microbatcher_factory):
    entered, release = asyncio.Event(), asyncio.Event()

    async def score_batch(requests):
        entered.set()
        await release.wait()
        return [{"score": 1} for _ in requests]

    microbatcher = microbatcher_factory(score_batch)
    await microbatcher.start()
    scoring = asyncio.create_task(microbatcher.submit({}))
    async with asyncio.timeout(5):
        await entered.wait()
        first = asyncio.create_task(microbatcher.close())
        await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(microbatcher.close())
        await asyncio.sleep(0)
        try:
            assert not first.done()
            assert not second.done()
        finally:
            release.set()
            outcomes = await asyncio.gather(scoring, first, second, return_exceptions=True)
        assert outcomes[0] == {"score": 1}
        assert isinstance(outcomes[1], asyncio.CancelledError)
        assert outcomes[2] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raised", [False, True])
async def test_stop_iteration_is_delivered_without_stranding_requests(microbatcher_factory, raised):
    def score_batch(requests):
        error = StopIteration("scorer stopped")
        if raised:
            raise error
        return [error, {"score": 2}]

    microbatcher = microbatcher_factory(score_batch, max_batch_size=2)
    await microbatcher.start()
    async with asyncio.timeout(5):
        outcomes = await asyncio.gather(microbatcher.submit({}), microbatcher.submit({}), return_exceptions=True)
    assert isinstance(outcomes[0], RuntimeError)
    assert isinstance(outcomes[0].__cause__, StopIteration)
    if raised:
        assert isinstance(outcomes[1], RuntimeError)
    else:
        assert outcomes[1] == {"score": 2}


@pytest.mark.parametrize("target", ["microbatcher", "hook", "legacy"])
def test_loop_shutdown_owns_sync_forward_until_thread_settles(target):
    import contextvars
    import threading
    from types import SimpleNamespace

    value = contextvars.ContextVar("score-context", default="missing")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    settled = []

    def score_batch(requests):
        entered.set()
        assert release.wait(5), "shutdown probe did not release its scorer thread"
        assert value.get() == "request-context"
        finished.set()
        return [{"score": 1}]

    async def run():
        value.set("request-context")
        if target == "microbatcher":
            microbatcher = ScorerMicrobatcher(score_batch)
            await microbatcher.start()
            microbatcher._consumer.add_done_callback(lambda _: settled.append(finished.is_set()))
            scoring = asyncio.create_task(microbatcher.submit({}))
        else:
            if target == "hook":
                call = NativeRewardExecutor._call(score_batch, [{}])
            else:
                owner = SimpleNamespace(_lock=asyncio.Lock(), _inflight=1, _idle=asyncio.Event())
                call = NativeRewardExecutor._infer_owned(owner, SimpleNamespace(infer=score_batch), ([{}],), {})
            scoring = asyncio.create_task(call)
            scoring.add_done_callback(lambda _: settled.append(finished.is_set()))
        scoring.add_done_callback(lambda task: task.cancelled() or task.exception())
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0)
        # Return while forward is active: asyncio.run cancels ALL wrapper tasks.
        threading.Timer(0.1, release.set).start()

    try:
        asyncio.run(run())
        assert settled == [True], "consumer shutdown detached a live synchronous forward"
    finally:
        release.set()


@pytest.mark.asyncio
async def test_dequeue_wakes_capacity_blocked_submitter(microbatcher_factory, monkeypatch):
    """Normal dequeue frees capacity without cancellation or a lifecycle transition."""
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    waiting = asyncio.Event()
    calls = []

    async def score(requests):
        index = requests[0]["i"]
        calls.append(index)
        if index < 2:
            entered[index].set()
            await release[index].wait()
        return [{"score": index}]

    batcher = microbatcher_factory(score, max_batch_size=None, max_inflight=1)
    await batcher.start()
    first = asyncio.create_task(batcher.submit({"i": 0}))
    queued = blocked = None
    try:
        await asyncio.wait_for(entered[0].wait(), 5)
        queued = asyncio.create_task(batcher.submit({"i": 1}))
        async with asyncio.timeout(5), batcher._cond:
            await batcher._cond.wait_for(lambda: len(batcher._queue) == 1)
        original_wait = batcher._cond.wait

        async def observe_wait():
            if asyncio.current_task() is blocked:
                waiting.set()
            return await original_wait()

        monkeypatch.setattr(batcher._cond, "wait", observe_wait)
        blocked = asyncio.create_task(batcher.submit({"i": 2}))
        await asyncio.wait_for(waiting.wait(), 5)
        release[0].set()
        await asyncio.wait_for(entered[1].wait(), 5)
        print("CAPACITY_DEQUEUE_WAKE_ASSERT_ENTERED", flush=True)
        async with asyncio.timeout(5), batcher._cond:
            await batcher._cond.wait_for(lambda: any(request["i"] == 2 for request, _ in batcher._queue))
        assert calls == [0, 1] and not blocked.done()
        release[1].set()
        results = await asyncio.wait_for(asyncio.gather(first, queued, blocked), 5)
        assert results == [{"score": 0}, {"score": 1}, {"score": 2}]
        assert calls == [0, 1, 2]
    finally:
        for event in release:
            event.set()
        await asyncio.wait_for(batcher.close(), 5)
        await asyncio.wait_for(
            asyncio.gather(*(task for task in (first, queued, blocked) if task is not None), return_exceptions=True), 5
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_cancel", [False, True])
@pytest.mark.parametrize("translation", ["cancel", "error", "return"])
async def test_forward_cancellation_translation_preserves_consumer_ownership(
    microbatcher_factory, consumer_cancel, translation
):
    entered = asyncio.Event()
    gate = asyncio.get_running_loop().create_future()
    calls = []

    async def score(requests):
        calls.extend(request["i"] for request in requests)
        if requests[0]["i"] == 0:
            entered.set()
            try:
                await gate
            except asyncio.CancelledError:
                if translation == "cancel":
                    raise
                if translation == "error":
                    raise OSError("scorer cleanup failed") from None
        return [{"score": request["i"]} for request in requests]

    batcher = microbatcher_factory(score, max_batch_size=None)
    await batcher.start()
    consumer = batcher._consumer
    first = asyncio.create_task(batcher.submit({"i": 0}))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(batcher.submit({"i": 1}))
        async with batcher._cond:
            await asyncio.wait_for(batcher._cond.wait_for(lambda: len(batcher._queue) == 1), 5)
        print("FORWARD_CANCEL_TRANSLATION_ASSERT_ENTERED", consumer_cancel, translation, flush=True)
        if consumer_cancel:
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(consumer), 5)
            outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
            assert all(isinstance(x, RuntimeError) and "consumer was cancelled" in str(x) for x in outcomes)
            assert batcher._state == "closed" and not batcher._queue
            assert calls == [0]
            with pytest.raises(RuntimeError, match="closed"):
                await batcher.submit({"i": 2})
        else:
            gate.cancel()
            if translation == "return":
                assert await first == {"score": 0}
            else:
                error = OSError if translation == "error" else RuntimeError
                with pytest.raises(error):
                    await first
            assert await second == {"score": 1}
            assert await batcher.submit({"i": 2}) == {"score": 2}
            assert batcher._state == "running"
    finally:
        if not gate.done():
            gate.set_result(None)
        if not consumer.done():
            consumer.cancel()
        await asyncio.gather(consumer, first, *([second] if second is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_consumer_cancel_cannot_abort_pending_settlement(microbatcher_factory, monkeypatch):
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    gate = asyncio.get_running_loop().create_future()

    async def score(requests):
        entered.set()
        await gate
        return [{"score": request["i"]} for request in requests]

    batcher = microbatcher_factory(score, max_batch_size=None)
    await batcher.start()
    consumer = batcher._consumer
    first = asyncio.create_task(batcher.submit({"i": 0}))
    second = None
    original = batcher._fail_pending
    owned = []

    async def observed_cleanup(batch, **kwargs):
        owned.extend(batch)
        cleanup_entered.set()
        return await original(batch, **kwargs)

    monkeypatch.setattr(batcher, "_fail_pending", observed_cleanup)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(batcher.submit({"i": 1}))
        async with batcher._cond:
            await asyncio.wait_for(batcher._cond.wait_for(lambda: len(batcher._queue) == 1), 5)
            consumer.cancel()
            await asyncio.wait_for(cleanup_entered.wait(), 5)
            print("REPEATED_CONSUMER_SETTLEMENT_ASSERT_ENTERED", flush=True)
            consumer.cancel()
            await asyncio.sleep(0)
            assert not consumer.done(), "accepted settlement must survive repeated consumer cancellation"
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(consumer), 5)
        outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
        assert all(isinstance(x, RuntimeError) and "consumer was cancelled" in str(x) for x in outcomes)
        assert batcher._state == "closed"
    finally:
        if not gate.done():
            gate.set_result(None)
        if not consumer.done():
            consumer.cancel()
        await original(owned)
        await asyncio.gather(consumer, first, *([second] if second is not None else []), return_exceptions=True)
