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
"""CPU contracts for opt-in multi-reward concurrency."""

import asyncio
import inspect
import os
import threading
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager

REWARDS_PATH = "tests/reward_loop/test_multi_reward_concurrency_on_cpu.py"
_DEFAULT = object()


async def _async_score(data_source, **kwargs):
    assert "independent" not in kwargs
    return 1.0


async def _capture_engine_kwargs(**kwargs):
    return {"score": 1.0, "kwargs": kwargs}


def _sync_score(data_source, **kwargs):
    return 1.0


class _EngineExecutor:
    def reward_kwargs(self):
        return {"reward_router_address": "engine-router", "model_name": "engine"}


def _term(name="_async_score", **options):
    return {"path": REWARDS_PATH, "name": name, **options}


def _manager(terms, *, concurrency=_DEFAULT, models=None, configure=None):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.reward_functions = OmegaConf.create(terms)
    config.reward.models = OmegaConf.create(models or {})
    if concurrency is not _DEFAULT:
        config.reward.multi_reward_concurrency = concurrency
    if configure is not None:
        configure(config)
    return MultiVisualRewardManager(config, MagicMock(), compute_score=None)


def _sample(name):
    return DataProto.from_dict(
        tensors={"responses": torch.zeros((1, 3, 2, 2), dtype=torch.uint8)},
        non_tensors={
            "data_source": [name],
            "reward_model": [{"ground_truth": "ground truth"}],
            "extra_info": [{}],
        },
    )


def _replace(manager, key, scorer):
    sub = next(sub for sub in manager._sub_rewards if sub["key"] == key)
    sub["fn"] = scorer
    sub["sig"] = inspect.signature(scorer)
    sub["is_async"] = inspect.iscoroutinefunction(scorer)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_concurrency_requires_positive_integer(value):
    with pytest.raises(ValueError, match="multi_reward_concurrency must be a positive integer"):
        _manager({"one": _term()}, concurrency=value)


@pytest.mark.parametrize("value", [1, "true", None])
def test_independent_requires_strict_boolean(value):
    with pytest.raises(TypeError, match="independent must be a boolean"):
        _manager({"one": _term(independent=value)})


@pytest.mark.parametrize("concurrency", [1, 2])
def test_sync_and_native_terms_cannot_opt_in(concurrency):
    with pytest.raises(ValueError, match="independent requires an async scorer"):
        _manager({"one": _term("_sync_score", independent=True)}, concurrency=concurrency)
    with pytest.raises(ValueError, match="independent requires an async scorer"):
        _manager(
            {"one": _term(independent=True, model="native_model")},
            concurrency=concurrency,
            models={"native_model": {"backend": "native"}},
        )


def test_engine_term_opts_in_and_reserved_field_is_not_forwarded():
    manager = _manager(
        {"one": _term(independent=True, model="engine")},
        concurrency=2,
        models={"engine": {"backend": "engine"}},
    )
    manager.set_reward_executors({"engine": _EngineExecutor()}, None)
    result = manager.loop.run_until_complete(manager.run_single(_sample("engine-sample")))
    assert result["reward_score"] == 1.0


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("named_length", [None, 321])
@pytest.mark.parametrize("explicit", [False, True])
def test_named_engine_sampling_and_reserved_metadata_through_builder(concurrency, independent, named_length, explicit):
    def configure(config):
        config.data.seed = 73
        config.reward.reward_model.rollout.response_length = "${data.max_response_length}"
        config.reward.reward_model.rollout.seed = "${data.seed}"

    named_rollout = {"full_determinism": True}
    if named_length is not None:
        named_rollout["response_length"] = named_length
    term = _term("_capture_engine_kwargs", model="engine", independent=independent, use_rollout_sampling_params=True)
    if explicit:
        term["sampling_params"] = {"max_tokens": 17, "temperature": 0.2}
    manager = _manager(
        {"one": term},
        concurrency=concurrency,
        models={"engine": {"backend": "engine", "rollout": named_rollout}},
        configure=configure,
    )
    manager.set_reward_executors({"engine": _EngineExecutor()}, None)

    result = manager.loop.run_until_complete(manager.run_single(_sample("engine-sample")))
    kwargs = result["reward_extra_info"]["reward/one/kwargs"]
    if explicit:
        expected_sampling = {"max_tokens": 17, "temperature": 0.2}
    else:
        expected_sampling = {
            "max_tokens": named_length or manager.config.data.max_response_length,
            "seed": manager.config.data.seed,
        }
    assert kwargs["sampling_params"] == expected_sampling
    assert kwargs["reward_router_address"] == "engine-router"
    assert kwargs["model_name"] == "engine"
    assert "independent" not in kwargs
    assert "use_rollout_sampling_params" not in kwargs
    assert result["reward_score"] == 1.0


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("value", [1, "true", None])
def test_rollout_sampling_flag_requires_strict_boolean(concurrency, value):
    with pytest.raises(TypeError, match="use_rollout_sampling_params must be a boolean"):
        _manager(
            {"one": _term(model="engine", use_rollout_sampling_params=value)},
            concurrency=concurrency,
            models={"engine": {"backend": "engine"}},
        )


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("model,models", [(None, {}), ("native", {"native": {"backend": "native"}})])
def test_rollout_sampling_requires_named_engine(concurrency, model, models):
    term = _term(use_rollout_sampling_params=True)
    if model is not None:
        term["model"] = model
    with pytest.raises(ValueError, match="use_rollout_sampling_params requires a named engine reward model"):
        _manager({"one": term}, concurrency=concurrency, models=models)


def test_missing_engine_executor_stays_fatal_even_when_optional():
    manager = _manager(
        {"one": _term(independent=True, model="engine", required=False)},
        concurrency=2,
        models={"engine": {"backend": "engine"}},
    )
    with pytest.raises(RuntimeError, match="Reward model 'engine' is not available"):
        manager.loop.run_until_complete(manager.run_single(_sample("engine-sample")))


def test_default_mode_preserves_cross_sample_concurrency():
    manager = _manager({"one": _term()})
    assert manager.config.reward.multi_reward_concurrency == 1
    assert manager._multi_reward_concurrency == 1

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        active = 0

        async def scorer(data_source):
            nonlocal active
            active += 1
            if active == 2:
                entered.set()
            await release.wait()
            active -= 1
            return 1.0

        _replace(manager, "one", scorer)
        tasks = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(2)]
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert active == 2
        finally:
            release.set()
            await asyncio.gather(*tasks)

    manager.loop.run_until_complete(scenario())


def test_independent_terms_at_cap_one_stay_sequential_per_sample_but_samples_overlap():
    manager = _manager({"first": _term(independent=True), "second": _term(independent=True)}, concurrency=1)

    async def scenario():
        both_first = asyncio.Event()
        second_one = asyncio.Event()
        release = {"one": asyncio.Event(), "two": asyncio.Event()}
        first_started = set()
        second_started = set()

        async def first(data_source):
            first_started.add(data_source)
            if len(first_started) == 2:
                both_first.set()
            await release[data_source].wait()
            return 1.0

        async def second(data_source):
            second_started.add(data_source)
            if data_source == "one":
                second_one.set()
            return 1.0

        _replace(manager, "first", first)
        _replace(manager, "second", second)
        callers = [asyncio.create_task(manager.run_single(_sample(name))) for name in ("one", "two")]
        try:
            await asyncio.wait_for(both_first.wait(), 1)
            assert first_started == {"one", "two"}
            assert second_started == set()
            release["one"].set()
            await asyncio.wait_for(second_one.wait(), 1)
            assert second_started == {"one"}
        finally:
            release["one"].set()
            release["two"].set()
        results = await asyncio.gather(*callers)
        assert [result["reward_score"] for result in results] == [2.0, 2.0]
        assert second_started == {"one", "two"}

    manager.loop.run_until_complete(scenario())


def test_worker_cap_applies_across_samples_and_restores_permits():
    manager = _manager({"one": _term(independent=True)}, concurrency=2)
    assert manager.config.reward.multi_reward_concurrency == 2
    assert manager._multi_reward_concurrency == 2

    async def scenario():
        full = asyncio.Event()
        release = asyncio.Event()
        active = maximum = started = 0

        async def scorer(data_source):
            nonlocal active, maximum, started
            active += 1
            started += 1
            maximum = max(maximum, active)
            if active == 2:
                full.set()
            await release.wait()
            active -= 1
            return 1.0

        _replace(manager, "one", scorer)
        tasks = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(5)]
        try:
            await asyncio.wait_for(full.wait(), 1)
            assert started == 2
            assert active == 2
        finally:
            release.set()
            results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert started == 5
        assert maximum == 2
        assert active == 0
        assert (await manager.run_single(_sample("again")))["reward_score"] == 1.0

    manager.loop.run_until_complete(scenario())


def test_barriers_share_worker_cap_with_independent_terms_across_samples():
    manager = _manager({"independent": _term(independent=True), "barrier": _term()}, concurrency=2)

    async def scenario():
        two_barriers = asyncio.Event()
        release = asyncio.Event()
        third_independent = asyncio.Event()
        active = maximum = 0
        barrier_count = 0

        async def independent(data_source):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if data_source == "third":
                third_independent.set()
            active -= 1
            return 1.0

        async def barrier(data_source):
            nonlocal active, maximum, barrier_count
            active += 1
            maximum = max(maximum, active)
            barrier_count += 1
            if barrier_count == 2:
                two_barriers.set()
            try:
                await release.wait()
                return 1.0
            finally:
                active -= 1

        _replace(manager, "independent", independent)
        _replace(manager, "barrier", barrier)
        first_two = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(2)]
        try:
            await asyncio.wait_for(two_barriers.wait(), 1)
            assert active == 2
            third = asyncio.create_task(manager.run_single(_sample("third")))

            async def wait_for_waiter():
                while not manager._multi_reward_semaphore._waiters:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_waiter(), 1)
            assert not third_independent.is_set()
        finally:
            release.set()
        results = await asyncio.gather(*first_two, third)
        assert [result["reward_score"] for result in results] == [2.0, 2.0, 2.0]
        assert third_independent.is_set()
        assert maximum == 2
        assert active == 0

    manager.loop.run_until_complete(scenario())


def test_contiguous_terms_are_admitted_in_bounded_windows():
    manager = _manager({key: _term(independent=True) for key in ("first", "second", "third")}, concurrency=2)

    async def scenario():
        first_window = asyncio.Event()
        release = asyncio.Event()
        third_started = asyncio.Event()
        started = 0

        async def early(data_source):
            nonlocal started
            started += 1
            if started == 2:
                first_window.set()
            await release.wait()
            return 1.0

        async def third(data_source):
            third_started.set()
            return 1.0

        _replace(manager, "first", early)
        _replace(manager, "second", early)
        _replace(manager, "third", third)
        task = asyncio.create_task(manager.run_single(_sample("one")))
        try:
            await asyncio.wait_for(first_window.wait(), 1)
            assert started == 2
            assert not third_started.is_set()
        finally:
            release.set()
        assert (await task)["reward_score"] == 3.0
        assert third_started.is_set()

    manager.loop.run_until_complete(scenario())


def test_window_reduces_in_config_order_and_barrier_fences_neighbors():
    manager = _manager(
        {
            "first": _term(independent=True, weight=2),
            "second": _term(independent=True, weight=3),
            "barrier": _term(),
            "last": _term(independent=True),
        },
        concurrency=2,
    )

    async def scenario():
        first_started = asyncio.Event()
        second_finished = asyncio.Event()
        release_first = asyncio.Event()
        calls = []
        payload_ids = []

        async def first(data_source, solution_image):
            payload_ids.append(id(solution_image))
            first_started.set()
            await release_first.wait()
            calls.append("first")
            return {"score": 0.1, "detail": "first"}

        async def second(data_source, solution_image):
            payload_ids.append(id(solution_image))
            await first_started.wait()
            calls.append("second")
            second_finished.set()
            return {"score": 0.2, "detail": "second"}

        async def barrier(data_source):
            calls.append("barrier")
            return 0.3

        async def last(data_source):
            calls.append("last")
            return 0.4

        for key, scorer in (("first", first), ("second", second), ("barrier", barrier), ("last", last)):
            _replace(manager, key, scorer)
        task = asyncio.create_task(manager.run_single(_sample("one")))
        try:
            await asyncio.wait_for(second_finished.wait(), 1)
            assert calls == ["second"]
        finally:
            release_first.set()
        result = await task
        assert calls == ["second", "first", "barrier", "last"]
        assert len(set(payload_ids)) == 1
        assert list(result["reward_extra_info"]) == [
            "reward/first/detail",
            "reward/first",
            "reward/second/detail",
            "reward/second",
            "reward/barrier",
            "reward/last",
            "reward/combined",
        ]
        assert result["reward_score"] == pytest.approx(1.5)

    manager.loop.run_until_complete(scenario())


def test_optional_failure_zero_and_required_failure_drains_window():
    def run_case(required):
        manager = _manager(
            {"bad": _term(independent=True, required=required), "peer": _term(independent=True)},
            concurrency=2,
        )

        async def scenario():
            peer_started = asyncio.Event()
            bad_raised = asyncio.Event()
            release_peer = asyncio.Event()
            peer_done = asyncio.Event()

            async def bad(data_source):
                await peer_started.wait()
                bad_raised.set()
                raise ValueError("boom")

            async def peer(data_source):
                peer_started.set()
                try:
                    await release_peer.wait()
                    return 2.0
                finally:
                    peer_done.set()

            _replace(manager, "bad", bad)
            _replace(manager, "peer", peer)
            task = asyncio.create_task(manager.run_single(_sample("one")))
            await asyncio.wait_for(bad_raised.wait(), 1)
            assert not task.done()
            assert not peer_done.is_set()
            release_peer.set()
            if required:
                with pytest.raises(RuntimeError, match="Required sub-reward 'bad' failed: boom"):
                    await task
            else:
                result = await task
                assert result["reward_score"] == 2.0
                assert result["reward_extra_info"]["reward/bad/errors"] == 1
                assert result["reward_extra_info"]["reward/bad"] == 0.0
            assert peer_done.is_set()

        manager.loop.run_until_complete(scenario())

    run_case(False)
    run_case(True)


def test_first_configured_required_error_wins_after_opposite_completion_order():
    manager = _manager(
        {"first": _term(independent=True, required=True), "second": _term(independent=True, required=True)},
        concurrency=2,
    )

    async def scenario():
        second_raised = asyncio.Event()
        release_first = asyncio.Event()
        first_settled = asyncio.Event()

        async def first(data_source):
            try:
                await release_first.wait()
                raise ValueError("first error")
            finally:
                first_settled.set()

        async def second(data_source):
            second_raised.set()
            raise ValueError("second error")

        _replace(manager, "first", first)
        _replace(manager, "second", second)
        caller = asyncio.create_task(manager.run_single(_sample("one")))
        try:
            await asyncio.wait_for(second_raised.wait(), 1)
            assert not caller.done()
            assert not first_settled.is_set()
        finally:
            release_first.set()
        with pytest.raises(RuntimeError, match="Required sub-reward 'first' failed: first error"):
            await caller
        assert first_settled.is_set()

    manager.loop.run_until_complete(scenario())


def test_repeated_cancellation_drains_active_and_waiting_tasks():
    manager = _manager({"one": _term(independent=True)}, concurrency=2)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        active = 0
        settled = []

        async def scorer(data_source):
            nonlocal active
            active += 1
            if active == 2:
                entered.set()
            try:
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
                return 1.0
            finally:
                active -= 1
                settled.append(data_source)

        _replace(manager, "one", scorer)
        callers = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(3)]
        await asyncio.wait_for(entered.wait(), 1)
        assert active == 2

        async def wait_for_waiter():
            while not manager._multi_reward_semaphore._waiters:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_waiter(), 1)
        try:
            callers[0].cancel()
            callers[2].cancel()  # Its admitted scorer is waiting for the semaphore.
            with pytest.raises(asyncio.CancelledError):
                await callers[2]
            assert active == 2
            assert settled == []
            callers[0].cancel()
            await asyncio.sleep(0)
            assert not callers[0].done()
        finally:
            release.set()
            results = await asyncio.gather(*callers, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert isinstance(results[2], asyncio.CancelledError)
        assert results[1]["reward_score"] == 1.0
        assert active == 0
        assert sorted(settled) == ["0", "1"]
        assert (await manager.run_single(_sample("again")))["reward_score"] == 1.0
        assert active == 0

    manager.loop.run_until_complete(scenario())


def test_cancelled_sync_barrier_retains_permit_until_thread_exits():
    manager = _manager({"barrier": _term("_sync_score")}, concurrency=2)
    two_entered = threading.Event()
    release = threading.Event()
    entered = []
    exited = []
    lock = threading.Lock()

    def scorer(data_source):
        with lock:
            entered.append(data_source)
            if len(entered) == 2:
                two_entered.set()
        try:
            release.wait()
            return 1.0
        finally:
            with lock:
                exited.append(data_source)

    _replace(manager, "barrier", scorer)

    async def scenario():
        callers = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(2)]
        try:
            assert await asyncio.to_thread(two_entered.wait, 1)
            queued = asyncio.create_task(manager.run_single(_sample("queued")))

            async def wait_for_waiter():
                while not manager._multi_reward_semaphore._waiters:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_waiter(), 1)
            callers[0].cancel()
            await asyncio.sleep(0)
            callers[0].cancel()
            await asyncio.sleep(0)
            assert not callers[0].done()
            assert sorted(entered) == ["0", "1"]
            assert exited == []
            assert not queued.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await callers[0]
        assert (await callers[1])["reward_score"] == 1.0
        assert (await queued)["reward_score"] == 1.0
        assert sorted(entered) == ["0", "1", "queued"]
        assert sorted(exited) == ["0", "1", "queued"]
        assert (await manager.run_single(_sample("again")))["reward_score"] == 1.0

    manager.loop.run_until_complete(scenario())
