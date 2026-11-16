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

from verl_omni.pipelines.rollout_media import (
    ArtifactContractError,
    MediaArtifact,
    MediaSpec,
    artifact_fields,
    select_artifact,
)
from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.utils.reward_score.reward_utils import get_audio, video_reward_tensor

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


class _NativeExecutor:
    def __init__(self, model, omit_model=False):
        self.model = model
        self.omit_model = omit_model

    def reward_kwargs(self):
        return {} if self.omit_model else {"reward_model": self.model}


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
            "media_kind": ["image"],
            "extra_info": [{}],
        },
    )


def _named_sample():
    latent = torch.arange(128, dtype=torch.float16).reshape(16, 2, 2, 2)
    preview = torch.arange(36, dtype=torch.uint8).reshape(3, 3, 2, 2)
    audio = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    fields = artifact_fields(
        {
            "video_latent": MediaArtifact(MediaSpec("video", "latent", "CTHW"), latent),
            "video_preview": MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=24), preview),
            "audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=32000), audio),
        },
        "video_latent",
        "video_preview",
    )
    sample = DataProto.from_dict(
        tensors={
            "responses": latent.unsqueeze(0),
            **{key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)},
        },
        non_tensors={
            "data_source": ["named"],
            "reward_model": [{"ground_truth": "prompt"}],
            **{key: [value] for key, value in fields.items() if not isinstance(value, torch.Tensor)},
        },
    )
    return sample, latent, preview, audio


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


@pytest.mark.parametrize("independent", [False, True])
def test_batch_and_reserved_kwargs_through_serial_and_independent_paths(independent):
    manager = _manager(
        {
            "explicit": _term(independent=independent, routing_weights={"actor": 1.0}),
            "kwargs_only": _term(independent=independent, routing_weights={"actor": 1.0}),
        },
        concurrency=2,
    )
    sample = _sample("sample")
    seen = []

    async def explicit(batch, data_source, **kwargs):
        assert batch is sample
        assert data_source == "sample"
        assert "independent" not in kwargs
        assert "routing_weights" not in kwargs
        seen.append("explicit")
        return 1.0

    async def kwargs_only(**kwargs):
        assert "batch" not in kwargs
        assert "independent" not in kwargs
        assert "routing_weights" not in kwargs
        seen.append("kwargs_only")
        return 2.0

    _replace(manager, "explicit", explicit)
    _replace(manager, "kwargs_only", kwargs_only)
    result = manager.loop.run_until_complete(manager.run_single(sample))
    assert seen == ["explicit", "kwargs_only"]
    assert result["reward_score"] == 3.0
    assert list(result["reward_extra_info"]) == ["reward/explicit", "reward/kwargs_only", "reward/combined"]


@pytest.mark.parametrize("independent", [False, True])
def test_named_media_reaches_each_term_through_real_run_single(independent):
    manager = _manager(
        {"visual": _term(independent=independent), "audio": _term(independent=independent)},
        concurrency=2,
    )
    sample, latent, preview, audio = _named_sample()
    seen = []

    async def visual(batch, solution_image, extra_info):
        assert batch is sample
        torch.testing.assert_close(solution_image, latent)
        selected = video_reward_tensor(solution_image, extra_info, fps=24)
        torch.testing.assert_close(selected, preview)
        assert extra_info["primary_artifact"] == "video_latent"
        assert extra_info["preview_artifact"] == "video_preview"
        seen.append("visual")
        return {"score": 1.0}

    async def audio_term(batch, solution_image, extra_info):
        assert batch is sample
        torch.testing.assert_close(solution_image, latent)
        waveform, rate = get_audio(extra_info)
        torch.testing.assert_close(waveform, audio[0])
        assert rate == 32000
        seen.append("audio")
        return {"score": 2.0}

    _replace(manager, "visual", visual)
    _replace(manager, "audio", audio_term)
    result = manager.loop.run_until_complete(manager.run_single(sample))
    assert sorted(seen) == ["audio", "visual"]
    assert result["reward_score"] == 3.0
    assert list(result["reward_extra_info"]) == ["reward/visual", "reward/audio", "reward/combined"]


@pytest.mark.parametrize("independent", [False, True])
def test_optional_artifact_contract_error_is_fatal_in_both_execution_paths(independent):
    terms = {"bad": _term(independent=independent, required=False)}
    if independent:
        terms["peer"] = _term(independent=True)
    manager = _manager(terms, concurrency=2)
    sample, _, _, _ = _named_sample()

    async def scenario():
        bad_started = asyncio.Event()
        bad_finished = asyncio.get_running_loop().create_future()
        peer_started = asyncio.Event()
        peer_release = asyncio.Event()
        peer_settled = asyncio.Event()

        async def bad(extra_info):
            bad_started.set()
            if independent:
                await peer_started.wait()
            try:
                select_artifact(
                    extra_info["media_artifacts"],
                    name="missing_preview",
                    modality="video",
                    representation="decoded",
                )
            except ArtifactContractError:
                if independent:
                    loop = asyncio.get_running_loop()
                    asyncio.current_task().add_done_callback(lambda _: loop.call_soon(bad_finished.set_result, None))
                raise

        async def peer(batch):
            assert batch is sample
            peer_started.set()
            try:
                await peer_release.wait()
                return 2.0
            finally:
                peer_settled.set()

        _replace(manager, "bad", bad)
        if independent:
            _replace(manager, "peer", peer)
        caller = asyncio.create_task(manager.run_single(sample))
        try:
            await asyncio.wait_for(bad_started.wait(), 1)
            if independent:
                await asyncio.wait_for(peer_started.wait(), 1)
                await asyncio.wait_for(bad_finished, 1)
                assert not caller.done()
                assert not peer_settled.is_set()
        finally:
            peer_release.set()
            await asyncio.gather(caller, return_exceptions=True)
        with pytest.raises(ArtifactContractError, match="missing_preview.*absent"):
            await caller
        if independent:
            assert peer_settled.is_set()

    manager.loop.run_until_complete(scenario())


@pytest.mark.parametrize("independent", [False, True])
def test_artifact_contract_error_during_score_conversion_is_not_optional(independent):
    manager = _manager({"bad": _term(independent=independent, required=False)}, concurrency=2)

    class InvalidScore:
        def __float__(self):
            raise ArtifactContractError("score references an invalid artifact")

    async def bad(data_source):
        return {"score": InvalidScore()}

    _replace(manager, "bad", bad)
    with pytest.raises(ArtifactContractError, match="invalid artifact"):
        manager.loop.run_until_complete(manager.run_single(_sample("one")))


def test_native_barrier_keeps_batch_binding_and_valid_fps_after_independent_window():
    manager = _manager(
        {"first": _term(independent=True), "native": _term(model="native", required=False, fps=24.0)},
        concurrency=2,
        models={"native": {"backend": "native"}},
    )
    sample = _sample("sample")
    model = object()
    manager.set_reward_executors(None, {"native": _NativeExecutor(model)})
    calls = []

    async def first(data_source):
        calls.append("first")
        return 1.0

    async def native(batch, reward_model, fps):
        assert batch is sample
        assert reward_model is model
        assert fps == 24.0
        calls.append("native")
        return 2.0

    _replace(manager, "first", first)
    _replace(manager, "native", native)
    result = manager.loop.run_until_complete(manager.run_single(sample))
    assert calls == ["first", "native"]
    assert result["reward_score"] == 3.0


@pytest.mark.parametrize("missing_model, fps", [(True, 24.0), (False, 0.0)])
def test_native_argument_errors_after_independent_window_are_fatal_for_optional_term(missing_model, fps):
    manager = _manager(
        {"first": _term(independent=True), "native": _term(model="native", required=False, fps=fps)},
        concurrency=2,
        models={"native": {"backend": "native"}},
    )
    manager.set_reward_executors(None, {"native": _NativeExecutor(None if missing_model else object(), missing_model)})
    calls = []

    async def first(data_source):
        calls.append("first")
        return 1.0

    async def native(reward_model, fps):
        calls.append("native")
        return 2.0

    _replace(manager, "first", first)
    _replace(manager, "native", native)
    error = "reward_model" if missing_model else "fps must be finite and positive"
    with pytest.raises(ValueError, match=error):
        manager.loop.run_until_complete(manager.run_single(_sample("sample")))
    assert calls == ["first"]


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


def test_barriers_do_not_consume_independent_worker_cap_across_samples():
    manager = _manager({"independent": _term(independent=True), "barrier": _term()}, concurrency=2)

    async def scenario():
        two_barriers = asyncio.Event()
        release_barriers = asyncio.Event()
        release_third = asyncio.Event()
        third_independent = asyncio.Event()
        active = maximum = 0
        barrier_count = 0

        async def independent(data_source):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if data_source == "third":
                third_independent.set()
                try:
                    await release_third.wait()
                finally:
                    active -= 1
                return 1.0
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
                await release_barriers.wait()
                return 1.0
            finally:
                active -= 1

        _replace(manager, "independent", independent)
        _replace(manager, "barrier", barrier)
        first_two = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(2)]
        third = None
        try:
            await asyncio.wait_for(two_barriers.wait(), 1)
            assert active == 2
            third = asyncio.create_task(manager.run_single(_sample("third")))
            await asyncio.wait_for(third_independent.wait(), 1)
            assert active == 3
            assert barrier_count == 2
        finally:
            release_third.set()
            release_barriers.set()
            tasks = first_two + ([third] if third is not None else [])
            await asyncio.gather(*tasks, return_exceptions=True)
        assert third is not None
        results = await asyncio.gather(*first_two, third)
        assert [result["reward_score"] for result in results] == [2.0, 2.0, 2.0]
        assert third_independent.is_set()
        assert maximum == 3
        assert active == 0

    manager.loop.run_until_complete(scenario())


def test_barrier_only_samples_keep_cross_sample_concurrency_above_one():
    manager = _manager({"barrier": _term()}, concurrency=2)

    async def scenario():
        all_entered = asyncio.Event()
        release = asyncio.Event()
        active = 0

        async def barrier(data_source):
            nonlocal active
            active += 1
            if active == 3:
                all_entered.set()
            try:
                await release.wait()
                return 1.0
            finally:
                active -= 1

        _replace(manager, "barrier", barrier)
        callers = [asyncio.create_task(manager.run_single(_sample(str(i)))) for i in range(3)]
        try:
            await asyncio.wait_for(all_entered.wait(), 1)
            assert active == 3
        finally:
            release.set()
            await asyncio.gather(*callers, return_exceptions=True)
        results = await asyncio.gather(*callers)
        assert [result["reward_score"] for result in results] == [1.0, 1.0, 1.0]
        assert active == 0

    manager.loop.run_until_complete(scenario())


def test_contiguous_terms_are_admitted_in_bounded_windows():
    manager = _manager({key: _term(independent=True) for key in ("first", "second", "third")}, concurrency=2)

    async def scenario():
        first_finished = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        third_started = asyncio.Event()

        async def first(data_source):
            await second_started.wait()
            first_finished.set()
            return 1.0

        async def second(data_source):
            second_started.set()
            await release_second.wait()
            return 2.0

        async def third(data_source):
            third_started.set()
            return 3.0

        _replace(manager, "first", first)
        _replace(manager, "second", second)
        _replace(manager, "third", third)
        task = asyncio.create_task(manager.run_single(_sample("one")))
        try:
            await asyncio.wait_for(first_finished.wait(), 1)

            async def wait_for_free_permit():
                while manager._multi_reward_semaphore._value == 0:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_free_permit(), 1)
            await asyncio.sleep(0)
            assert manager._multi_reward_semaphore._value == 1
            assert not task.done()
            assert not third_started.is_set()
        except BaseException:
            task.cancel()
            raise
        finally:
            release_second.set()
            settled = asyncio.gather(task, return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(settled), 1)
        result = task.result()
        assert result["reward_score"] == 6.0
        assert list(result["reward_extra_info"].items()) == [
            ("reward/first", 1.0),
            ("reward/second", 2.0),
            ("reward/third", 3.0),
            ("reward/combined", 6.0),
        ]
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


def test_repeated_cancellation_drains_active_and_queued_terms_in_one_window():
    manager = _manager({"first": _term(independent=True), "second": _term(independent=True)}, concurrency=2)

    async def scenario():
        blocker_started = asyncio.Event()
        blocker_second_done = asyncio.Event()
        blocker_release = asyncio.Event()
        target_started = asyncio.Event()
        cleanup_entered = asyncio.Event()
        repeat_cancel_seen = asyncio.Event()
        cleanup_release = asyncio.Event()
        target_exited = asyncio.Event()
        fresh_started = asyncio.Event()
        target_second_calls = 0

        async def first(data_source):
            if data_source == "blocker":
                blocker_started.set()
                await blocker_release.wait()
            elif data_source == "target":
                target_started.set()
                try:
                    await cleanup_release.wait()
                except asyncio.CancelledError:
                    cleanup_entered.set()
                    while not cleanup_release.is_set():
                        try:
                            await cleanup_release.wait()
                        except asyncio.CancelledError:
                            repeat_cancel_seen.set()
                finally:
                    target_exited.set()
            else:
                assert target_exited.is_set()
                fresh_started.set()
            return 1.0

        async def second(data_source):
            nonlocal target_second_calls
            if data_source == "blocker":
                blocker_second_done.set()
            elif data_source == "target":
                target_second_calls += 1
            else:
                assert target_exited.is_set()
                fresh_started.set()
            return 1.0

        _replace(manager, "first", first)
        _replace(manager, "second", second)
        blocker = asyncio.create_task(manager.run_single(_sample("blocker")))
        target = fresh = None

        async def wait_for_queued(count):
            while len(manager._multi_reward_semaphore._waiters or ()) < count:
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(blocker_started.wait(), 1)
            await asyncio.wait_for(blocker_second_done.wait(), 1)
            target = asyncio.create_task(manager.run_single(_sample("target")))
            await asyncio.wait_for(target_started.wait(), 1)
            await asyncio.wait_for(wait_for_queued(1), 1)
            assert target_second_calls == 0

            target.cancel()
            await asyncio.wait_for(cleanup_entered.wait(), 1)
            assert not target.done()
            fresh = asyncio.create_task(manager.run_single(_sample("fresh")))
            await asyncio.wait_for(wait_for_queued(2), 1)
            target.cancel()
            await asyncio.wait_for(repeat_cancel_seen.wait(), 1)
            assert not target.done()
            assert not fresh_started.is_set()
            assert target_second_calls == 0

            cleanup_release.set()
            await asyncio.wait_for(fresh_started.wait(), 1)
            with pytest.raises(asyncio.CancelledError):
                await target
            assert target_exited.is_set()
            assert target_second_calls == 0

            blocker_release.set()
            assert (await asyncio.wait_for(blocker, 1))["reward_score"] == 2.0
            assert (await asyncio.wait_for(fresh, 1))["reward_score"] == 2.0
            assert manager._multi_reward_semaphore._value == 2
            assert not manager._multi_reward_semaphore._waiters
        finally:
            cleanup_release.set()
            blocker_release.set()
            if target is not None and not target.done():
                target.cancel()
            tasks = (task for task in (blocker, target, fresh) if task is not None)
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                1,
            )

    manager.loop.run_until_complete(scenario())


@pytest.mark.parametrize("concurrency", [1, 2])
def test_cancelled_sync_barrier_returns_before_executor_thread_exits(concurrency):
    manager = _manager({"barrier": _term("_sync_score")}, concurrency=concurrency)
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def scorer(data_source):
        entered.set()
        try:
            release.wait()
            return 1.0
        finally:
            exited.set()

    _replace(manager, "barrier", scorer)

    async def scenario():
        caller = asyncio.create_task(manager.run_single(_sample("blocked")))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(caller), 1)
            assert not exited.is_set()
        finally:
            release.set()
            assert await asyncio.to_thread(exited.wait, 1)
            await asyncio.gather(caller, return_exceptions=True)

    manager.loop.run_until_complete(scenario())
