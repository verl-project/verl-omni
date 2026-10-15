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

from verl_omni.reward_loop.reward_model_executor import ScorerMicrobatcher


def _recording_scorer(calls, *, fail=None):
    def score_batch(requests):
        calls.append([request["i"] for request in requests])
        if fail is not None:
            raise fail
        return [{"score": request["i"]} for request in requests]

    return score_batch


@pytest.mark.asyncio
async def test_lone_request_scores_immediately_at_batch_one():
    calls = []
    microbatcher = ScorerMicrobatcher(_recording_scorer(calls), name="m")
    await microbatcher.start()
    assert await microbatcher.submit({"i": 7}) == {"score": 7}
    assert calls == [[7]]


@pytest.mark.asyncio
async def test_packs_concurrent_requests_and_preserves_request_result_mapping():
    calls = []
    microbatcher = ScorerMicrobatcher(_recording_scorer(calls), max_batch_size=8, name="m")
    await microbatcher.start()
    results = await asyncio.gather(*(microbatcher.submit({"i": i}) for i in range(4)))
    assert results == [{"score": i} for i in range(4)]
    assert calls == [[0, 1, 2, 3]]


@pytest.mark.asyncio
async def test_packing_respects_max_batch_size():
    calls = []

    def score_batch(requests):
        calls.append(len(requests))
        return [{"score": request["i"]} for request in requests]

    microbatcher = ScorerMicrobatcher(score_batch, max_batch_size=2, name="m")
    await microbatcher.start()
    results = await asyncio.gather(*(microbatcher.submit({"i": i}) for i in range(5)))
    assert results == [{"score": i} for i in range(5)]
    assert calls == [2, 2, 1]


@pytest.mark.asyncio
async def test_unbatched_scorer_scores_one_request_per_call():
    calls = []
    microbatcher = ScorerMicrobatcher(_recording_scorer(calls), max_batch_size=None, name="m")
    await microbatcher.start()
    await asyncio.gather(*(microbatcher.submit({"i": i}) for i in range(3)))
    assert calls == [[0], [1], [2]]


@pytest.mark.asyncio
async def test_per_request_preprocessing_failure_is_isolated():
    def score_batch(requests):
        return [
            ValueError(f"bad image {request['i']}") if request["i"] == 1 else {"score": request["i"]}
            for request in requests
        ]

    microbatcher = ScorerMicrobatcher(score_batch, max_batch_size=4, name="m")
    await microbatcher.start()
    outcomes = await asyncio.gather(
        *(microbatcher.submit({"i": i}) for i in range(3)),
        return_exceptions=True,
    )
    assert outcomes[0] == {"score": 0}
    assert isinstance(outcomes[1], ValueError) and "bad image 1" in str(outcomes[1])
    assert outcomes[2] == {"score": 2}


@pytest.mark.asyncio
async def test_forward_failure_fails_every_request_in_the_microbatch():
    calls = []
    boom = OSError("cuda gone")
    microbatcher = ScorerMicrobatcher(_recording_scorer(calls, fail=boom), max_batch_size=4, name="m")
    await microbatcher.start()
    outcomes = await asyncio.gather(
        *(microbatcher.submit({"i": i}) for i in range(2)),
        return_exceptions=True,
    )
    assert calls == [[0, 1]]
    assert outcomes == [boom, boom]


@pytest.mark.asyncio
async def test_wrong_length_output_fails_every_request():
    def score_batch(requests):
        return [{"score": request["i"]} for request in requests[:-1]]

    microbatcher = ScorerMicrobatcher(score_batch, max_batch_size=4, name="m")
    await microbatcher.start()
    outcomes = await asyncio.gather(
        *(microbatcher.submit({"i": i}) for i in range(3)),
        return_exceptions=True,
    )
    assert all(isinstance(outcome, ValueError) for outcome in outcomes)


@pytest.mark.asyncio
async def test_full_queue_pause_fails_queued_and_capacity_blocked_retryably_and_reopens():
    release = asyncio.Event()
    entered = asyncio.Event()
    accepted = []

    async def score_batch(requests):
        accepted.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    # Unbatched forward so exactly one request is in flight while others queue.
    microbatcher = ScorerMicrobatcher(score_batch, max_batch_size=None, max_inflight=2, name="m")
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
async def test_close_fails_unaccepted_terminally_and_settles_in_flight_forward():
    release = asyncio.Event()
    entered = asyncio.Event()
    accepted = []

    async def score_batch(requests):
        accepted.extend(request["i"] for request in requests)
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    microbatcher = ScorerMicrobatcher(score_batch, max_batch_size=None, max_inflight=2, name="m")
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
async def test_caller_cancellation_does_not_disturb_settlement_of_accepted_work():
    release = asyncio.Event()
    settled = []

    async def score_batch(requests):
        await release.wait()
        settled.extend(request["i"] for request in requests)
        return [{"score": request["i"]} for request in requests]

    microbatcher = ScorerMicrobatcher(score_batch, name="m")
    await microbatcher.start()
    cancelled = asyncio.ensure_future(microbatcher.submit({"i": 0}))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    follower = asyncio.create_task(microbatcher.submit({"i": 1}))
    await asyncio.sleep(0)
    release.set()
    assert await follower == {"score": 1}
    assert settled == [0, 1]


@pytest.mark.asyncio
async def test_shared_forward_lock_serializes_two_microbatchers():
    lock = asyncio.Lock()
    active = 0
    peak = 0

    async def score_batch(requests):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        active -= 1
        return [{"score": request["i"]} for request in requests]

    first = ScorerMicrobatcher(score_batch, name="a", forward_lock=lock)
    second = ScorerMicrobatcher(score_batch, name="b", forward_lock=lock)
    await asyncio.gather(first.start(), second.start())
    results = await asyncio.gather(
        first.submit({"i": 0}),
        first.submit({"i": 1}),
        second.submit({"i": 2}),
        second.submit({"i": 3}),
    )
    assert results == [{"score": i} for i in range(4)]
    assert peak == 1


@pytest.mark.asyncio
async def test_submit_outside_running_state_fails_fast():
    microbatcher = ScorerMicrobatcher(_recording_scorer([]), name="m")
    with pytest.raises(RuntimeError, match="not started"):
        await microbatcher.submit({"i": 0})

    await microbatcher.start()
    await microbatcher.pause()
    with pytest.raises(RuntimeError, match="retry after wake_up"):
        await microbatcher.submit({"i": 0})


def test_constructor_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="max_batch_size"):
        ScorerMicrobatcher(_recording_scorer([]), max_batch_size=0)
    with pytest.raises(ValueError, match="max_inflight"):
        ScorerMicrobatcher(_recording_scorer([]), max_inflight=0)
