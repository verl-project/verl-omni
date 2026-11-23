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
"""CPU tests for the class-based CLAP scorer."""

import asyncio
import os
import threading
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto

from verl_omni.pipelines.rollout_media import ArtifactContractError, MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_manager import multi as manager_module
from verl_omni.reward_loop.reward_manager.audio import AudioRewardManager
from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.reward_loop.reward_model import NativeManagedRewardModel
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor, build_native_reward_executors
from verl_omni.utils.reward_score.scorers import CLAPScorer
from verl_omni.utils.reward_score.scorers import clap as port
from verl_omni.utils.reward_score.scorers.registry import get_scorer_cls, peek_registered_scorers
from verl_omni.workers.config.reward import RewardModelSpec


class _FakeWeights:
    def __init__(self, device):
        self.device = torch.device(device)
        self.moves = []

    def to(self, device):
        self.device = torch.device(device)
        self.moves.append(self.device)
        return self


class _FakeAdapter:
    instances = []
    entered = threading.Event()
    release = threading.Event()

    def __init__(self, model_path, device, model_kwargs=None, processor_kwargs=None):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model_kwargs = model_kwargs
        self.processor_kwargs = processor_kwargs
        self.model = _FakeWeights(device)
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def infer(self, waveforms, prompts):
        self.calls.append((waveforms, list(prompts)))
        if "forward-error" in prompts:
            raise RuntimeError("shared CLAP forward failed")
        if "block" in prompts:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("blocked CLAP forward was never released")
        audio = torch.tensor(np.stack([waveform[:2] for waveform in waveforms]))
        prompt_features = {"right": [1.0, 0.0], "up": [0.0, 1.0], "block": [1.0, 0.0]}
        text = torch.tensor([prompt_features.get(prompt, [1.0, 0.0]) for prompt in prompts])
        return {"audio_embeddings": audio, "text_embeddings": text}

    def close(self):
        self.closed = True
        self.model = None
        self.device = None


@pytest.fixture
def fake_adapter(monkeypatch):
    _FakeAdapter.instances.clear()
    _FakeAdapter.entered.clear()
    _FakeAdapter.release.set()
    monkeypatch.setattr(port._frozen, "CLAPModel", _FakeAdapter)
    return _FakeAdapter.instances


def _artifact(waveform=(1.0, 0.0), sample_rate=48_000):
    return MediaArtifact(
        MediaSpec("audio", "decoded", "CT", sample_rate=sample_rate),
        torch.tensor(waveform, dtype=torch.float64).unsqueeze(0),
    )


def _request(prompt="right", waveform=(1.0, 0.0), sample_rate=48_000, **kwargs):
    request = {
        "data_source": "test",
        "solution_image": None,
        "ground_truth": prompt,
        "extra_info": {"media_artifacts": {"audio": _artifact(waveform, sample_rate)}},
    }
    request.update(kwargs)
    return request


def _executor(monkeypatch, **kwargs):
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(executor_module, "get_device_id", lambda: 0)
    return NativeRewardExecutor(
        RewardModelSpec(
            name="clap",
            backend="native",
            executor_config={
                "model": "pkg://verl_omni.utils.reward_score.scorers.clap:CLAPScorer",
                "kwargs": kwargs,
            },
        )
    )


def _manager_data(*, named_without_audio=False):
    responses = torch.zeros(1, 3, 2, 2, dtype=torch.uint8)
    tensors = {
        "responses": responses,
        "audio": torch.tensor([[[0.0, 1.0]]]),
        "audio_sample_rate": torch.tensor([48_000]),
    }
    non_tensors = {
        "data_source": ["test"],
        "reward_model": [{"ground_truth": "right"}],
        "reward_inputs": [{"text": {"caption": "up"}}],
        "media_kind": ["image"],
        "extra_info": [{}],
    }
    if named_without_audio:
        fields = artifact_fields(
            {"image": MediaArtifact(MediaSpec("image", "decoded", "CHW"), responses[0])}, primary="image"
        )
        for key, value in fields.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value.unsqueeze(0)
            else:
                non_tensors[key] = [value]
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_constructor_is_config_only_and_registry_exports_clap(monkeypatch):
    monkeypatch.setattr(port._frozen, "CLAPModel", lambda *args, **kwargs: pytest.fail("loaded in constructor"))
    scorer = CLAPScorer(model_path="weights", model_kwargs={"revision": "r"}, processor_kwargs={"fast": True})

    assert scorer.model_path == "weights"
    assert CLAPScorer().model_path == port._frozen._DEFAULT_MODEL
    assert scorer.max_batch_size == 16
    assert scorer.supports_cpu_offload is True
    assert peek_registered_scorers()["clap"] is CLAPScorer
    assert get_scorer_cls("clap") is CLAPScorer
    assert get_scorer_cls("pkg://verl_omni.utils.reward_score.scorers.clap:CLAPScorer") is CLAPScorer


@pytest.mark.parametrize(
    "count, expected_batches", [(0, []), (1, [1]), (3, [3]), (16, [16]), (17, [16, 1]), (33, [16, 16, 1])]
)
@pytest.mark.asyncio
async def test_actual_executor_batches_zero_single_burst_and_tail(monkeypatch, fake_adapter, count, expected_batches):
    if count == 0:
        assert CLAPScorer().score_batch([]) == []
        return
    executor = _executor(monkeypatch)
    await executor.wake_up()
    try:
        results = await asyncio.gather(*(executor.score(_request(waveform=(1.0, float(i)))) for i in range(count)))
        assert len(results) == count
        assert [result["score"] for result in results] == pytest.approx(
            [1.0 / (1.0 + i**2) ** 0.5 for i in range(count)]
        )
        assert [result["source_sample_rate"] for result in results] == [48_000] * count
        assert [len(call[1]) for call in fake_adapter[0].calls] == expected_batches
    finally:
        await executor.close()


@pytest.mark.parametrize(
    "legacy_audio", [torch.tensor([[9.0, 9.0], [7.0, 7.0]]), np.array([["invalid"]], dtype=object)]
)
def test_named_artifact_prompt_key_layout_rate_and_float32_mono(fake_adapter, monkeypatch, legacy_audio):
    resamples = []

    def resample(waveform, source_rate, target_rate):
        resamples.append((waveform.clone(), source_rate, target_rate))
        return waveform

    monkeypatch.setattr(port._frozen, "resample_audio", resample)
    scorer = CLAPScorer()
    scorer.load("cpu")
    sample = SimpleNamespace(
        batch={
            "audio": legacy_audio,
            "audio_sample_rate": 24_000,
            "audio_layout": "CT",
        },
        non_tensor_batch={"reward_inputs": {"text": {"caption": "up"}}},
    )
    request = _request(
        prompt="wrong",
        waveform=(0.0, 1.0),
        batch=[sample],
        prompt_key="caption",
        audio_layout="CT",
        score_scale=2.0,
        score_offset=0.25,
        score_min=-1.0,
        score_max=2.0,
    )

    (result,) = scorer.score_batch([request])

    assert result == {"score": pytest.approx(2.0), "source_sample_rate": 48_000}
    waveforms, prompts = fake_adapter[0].calls[0]
    assert prompts == ["up"]
    assert waveforms[0].dtype == np.float32
    np.testing.assert_array_equal(waveforms[0], np.array([0.0, 1.0], dtype=np.float32))
    assert len(resamples) == 1
    torch.testing.assert_close(resamples[0][0], torch.tensor([0.0, 1.0]))
    assert resamples[0][1:] == (48_000, 48_000)


def test_legacy_audio_is_resampled_after_channel_mean(fake_adapter, monkeypatch):
    seen = []

    def resample(waveform, source_rate, target_rate):
        seen.append((waveform.clone(), source_rate, target_rate))
        return waveform

    monkeypatch.setattr(port._frozen, "resample_audio", resample)
    scorer = CLAPScorer()
    scorer.load("cpu")
    request = _request()
    request["extra_info"] = {
        "audio": torch.tensor([[1.0, 0.0], [3.0, 2.0]], dtype=torch.float64),
        "audio_sample_rate": 24_000,
        "audio_layout": "CT",
    }

    (result,) = scorer.score_batch([request])

    torch.testing.assert_close(seen[0][0], torch.tensor([2.0, 1.0]))
    assert seen[0][1:] == (24_000, 48_000)
    assert fake_adapter[0].calls[0][0][0].dtype == np.float32
    assert result["source_sample_rate"] == 24_000


@pytest.mark.parametrize(
    "sample_rate", [np.array(48_000), np.array([48_000]), np.array([[48_000]])], ids=["0d", "1d", "2d"]
)
def test_legacy_audio_accepts_singleton_numpy_sample_rates(fake_adapter, sample_rate):
    scorer = CLAPScorer()
    scorer.load("cpu")
    request = _request(prompt="up")
    request["extra_info"] = {"audio": torch.tensor([0.0, 1.0]), "audio_sample_rate": sample_rate}

    (result,) = scorer.score_batch([request])

    assert result == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert request["extra_info"]["audio_sample_rate"] is sample_rate


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [np.array(48_000), np.array([48_000]), np.array([[48_000]])])
async def test_legacy_and_native_share_numpy_audio_and_calibration(fake_adapter, sample_rate):
    scorer = CLAPScorer()
    scorer.load("cpu")
    audio = np.array([["0.0", "1.0"]], dtype=object)
    extra_info = {"audio": audio, "audio_sample_rate": sample_rate, "audio_layout": "CT"}
    request = _request(
        prompt="up", extra_info=extra_info, score_scale=2.0, score_offset=0.25, score_min=-1.0, score_max=2.0
    )

    class _ManagedModel:
        async def infer(self, waveforms, prompts):
            return fake_adapter[0].infer(waveforms, prompts)

    native = scorer.score_batch([request])[0]
    legacy = await port._frozen.compute_score(
        data_source="test",
        solution_image=None,
        ground_truth="up",
        extra_info=extra_info,
        reward_model=_ManagedModel(),
        score_scale=2.0,
        score_offset=0.25,
        score_min=-1.0,
        score_max=2.0,
    )

    assert native == legacy == {"score": pytest.approx(2.0), "source_sample_rate": 48_000}
    assert extra_info["audio"] is audio
    assert extra_info["audio_sample_rate"] is sample_rate
    for waveforms, prompts in fake_adapter[0].calls:
        assert prompts == ["up"]
        np.testing.assert_array_equal(waveforms[0], np.array([0.0, 1.0], dtype=np.float32))


@pytest.mark.asyncio
async def test_legacy_and_native_reject_bad_object_audio_and_calibration_per_request(fake_adapter):
    scorer = CLAPScorer()
    scorer.load("cpu")
    malformed = {
        "audio": np.array([["invalid", "1.0"]], dtype=object),
        "audio_sample_rate": np.array([48_000]),
    }
    bad_calibration = {"audio": np.array([["1.0", "0.0"]], dtype=object), "audio_sample_rate": 48_000}
    requests = [
        _request(extra_info=malformed),
        _request(extra_info=bad_calibration, score_scale="bad"),
        _request(),
    ]
    results = scorer.score_batch(requests)
    assert isinstance(results[0], ValueError)
    assert isinstance(results[1], TypeError)
    assert results[2] == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert fake_adapter[0].calls[0][1] == ["right", "right"]

    class _ManagedModel:
        async def infer(self, waveforms, prompts):
            return fake_adapter[0].infer(waveforms, prompts)

    with pytest.raises(ValueError):
        await port._frozen.compute_score("test", None, "right", malformed, reward_model=_ManagedModel())
    with pytest.raises(TypeError):
        await port._frozen.compute_score(
            "test", None, "right", bad_calibration, reward_model=_ManagedModel(), score_scale="bad"
        )


@pytest.mark.parametrize(
    "sample_rate",
    [
        np.array([], dtype=np.int64),
        np.array([48_000, 48_000]),
        np.array([True]),
        np.array([float("nan")]),
        np.array([48_000.5]),
        np.array([0]),
        np.array(["48_000"]),
    ],
    ids=["empty", "multiple", "bool", "nonfinite", "fractional", "nonpositive", "string"],
)
def test_legacy_audio_rejects_invalid_numpy_sample_rates(fake_adapter, sample_rate):
    scorer = CLAPScorer()
    scorer.load("cpu")
    request = _request()
    request["extra_info"] = {"audio": torch.tensor([0.0, 1.0]), "audio_sample_rate": sample_rate}

    (result,) = scorer.score_batch([request])

    assert isinstance(result, ValueError)
    assert "sample rate" in str(result)
    assert request["extra_info"]["audio_sample_rate"] is sample_rate


def test_named_artifact_ignores_invalid_legacy_numpy_sample_rate(fake_adapter):
    scorer = CLAPScorer()
    scorer.load("cpu")
    request = _request()
    invalid_legacy_rate = np.array([0])
    request["extra_info"]["audio_sample_rate"] = invalid_legacy_rate

    (result,) = scorer.score_batch([request])

    assert result == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert request["extra_info"]["audio_sample_rate"] is invalid_legacy_rate


@pytest.mark.parametrize("audio_layout", ["T", "TC"])
def test_named_artifact_rejects_configured_noncanonical_layout_per_request(fake_adapter, audio_layout):
    scorer = CLAPScorer()
    scorer.load("cpu")

    invalid, valid = scorer.score_batch([_request(audio_layout=audio_layout), _request(audio_layout="CT")])

    assert isinstance(invalid, ArtifactContractError)
    assert "canonical CT audio" in str(invalid)
    assert valid == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert fake_adapter[0].calls[0][1] == ["right"]


def test_forward_uses_active_accelerator_device_context(fake_adapter, monkeypatch):
    events = []

    class _Context:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")

    class _FakeTorchDevice:
        def device(self, target):
            events.append(target)
            return _Context()

    monkeypatch.setattr(port, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(port, "get_torch_device", lambda: _FakeTorchDevice())
    scorer = CLAPScorer()
    scorer.load(torch.device("cuda:2"))

    (result,) = scorer.score_batch([_request()])

    assert result["score"] == pytest.approx(1.0)
    assert events == [torch.device("cuda:2"), "enter", "exit"]


@pytest.mark.asyncio
async def test_matches_frozen_managed_compute_score_math_and_calibration(fake_adapter):
    scorer = CLAPScorer()
    scorer.load("cpu")
    request = _request(
        waveform=(1.0, 1.0),
        score_scale=1.5,
        score_offset=-0.2,
        score_min=-0.5,
        score_max=0.6,
    )
    (actual,) = scorer.score_batch([request])

    class _Managed:
        async def infer(self, waveforms, prompts):
            return fake_adapter[0].infer(waveforms, prompts)

    expected = await port._frozen.compute_score(reward_model=_Managed(), **request)
    assert actual == {"score": pytest.approx(expected["score"]), "source_sample_rate": 48_000}


@pytest.mark.asyncio
async def test_executor_isolates_request_errors_and_propagates_artifact_contract(monkeypatch, fake_adapter):
    executor = _executor(monkeypatch)
    await executor.wake_up()
    invalid_artifact = _request()
    invalid_artifact["extra_info"] = {}
    invalid_calibration = _request(prompt="up", waveform=(0.0, 1.0), score_scale="bad")
    try:
        results = await asyncio.gather(
            executor.score(_request()),
            executor.score(invalid_artifact),
            executor.score(invalid_calibration),
            return_exceptions=True,
        )
    finally:
        await executor.close()

    assert results[0]["score"] == pytest.approx(1.0)
    assert isinstance(results[1], ArtifactContractError)
    assert isinstance(results[2], TypeError)
    assert len(fake_adapter[0].calls[0][1]) == 2


@pytest.mark.asyncio
async def test_shared_forward_error_fails_microbatch_and_later_request_recovers(monkeypatch, fake_adapter):
    executor = _executor(monkeypatch)
    await executor.wake_up()
    try:
        failed = await asyncio.gather(
            executor.score(_request("forward-error")),
            executor.score(_request("right")),
            return_exceptions=True,
        )
        assert all(isinstance(result, RuntimeError) for result in failed)

        recovered = await executor.score(_request("right"))
        assert recovered == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_manager_public_entry_injects_batch_for_prompt_audio_and_artifact_contract(monkeypatch, fake_adapter):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "reward.reward_model.enable=false",
                "+reward.models.clap.backend=native",
                "+reward.models.clap.placement.devices=[0]",
                "+reward.models.clap.executor.model=pkg://verl_omni.utils.reward_score.scorers.clap:CLAPScorer",
                "+reward.models.clap.executor.kwargs.model_path=fixture/clap",
                "+reward.models.clap.executor.kwargs.model_kwargs.revision=fixture-revision",
                "+reward.models.clap.executor.kwargs.processor_kwargs.use_fast=true",
                "+reward.reward_functions.clap.path=pkg://verl_omni.utils.reward_score.scorers.clap",
                "+reward.reward_functions.clap.name=compute_score_clap_batched",
                "+reward.reward_functions.clap.model=clap",
                "+reward.reward_functions.clap.prompt_key=caption",
                "+reward.reward_functions.clap.required=true",
            ],
        )
    manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
    managed_model = NativeManagedRewardModel("clap", config.reward.models.clap)
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(executor_module, "get_device_id", lambda: 0)
    executors = build_native_reward_executors({"clap": managed_model.executor_spec})
    executor = executors["clap"]
    manager.set_reward_executors(None, executors)
    await executor.wake_up()
    try:
        assert fake_adapter[0].model_path == "fixture/clap"
        assert fake_adapter[0].model_kwargs == {"revision": "fixture-revision"}
        assert fake_adapter[0].processor_kwargs == {"use_fast": True}
        result = await manager.run_single(_manager_data())
        assert result["reward_score"] == pytest.approx(1.0)
        assert result["reward_extra_info"]["reward/clap/source_sample_rate"] == 48_000
        waveforms, prompts = fake_adapter[0].calls[0]
        assert prompts == ["up"]
        np.testing.assert_array_equal(waveforms[0], np.array([0.0, 1.0], dtype=np.float32))

        with pytest.raises(ArtifactContractError, match="audio"):
            await manager.run_single(_manager_data(named_without_audio=True))
    finally:
        await executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["sleep", "close"])
async def test_cancelled_caller_drains_before_offload_or_close(monkeypatch, fake_adapter, lifecycle):
    _FakeAdapter.release.clear()
    executor = _executor(monkeypatch)
    await executor.wake_up()
    adapter = fake_adapter[0]
    weights = adapter.model
    scoring = asyncio.create_task(executor.score(_request("block")))
    sibling = asyncio.create_task(executor.score(_request()))
    transition = None
    entered = asyncio.Event()
    microbatcher = executor._microbatcher
    join_name = "pause" if lifecycle == "sleep" else "close"
    original_join = getattr(microbatcher, join_name)

    async def observed_join():
        entered.set()
        await original_join()

    monkeypatch.setattr(microbatcher, join_name, observed_join)
    try:
        assert await asyncio.to_thread(_FakeAdapter.entered.wait, 5)
        assert fake_adapter[0].calls[0][1] == ["block", "right"]
        scoring.cancel()
        transition = asyncio.create_task(getattr(executor, lifecycle)())
        await asyncio.wait_for(entered.wait(), 5)
        assert microbatcher._state == ("paused" if lifecycle == "sleep" else "closed")
        scoring.cancel()
        # Observe actual completion while the accepted forward remains gated.
        completed, _ = await asyncio.wait({scoring, sibling, transition}, timeout=0.05)
        assert not completed, "accepted work and lifecycle must stay owned until forward settlement"
        assert weights.moves == []
        assert not adapter.closed
        with pytest.raises(RuntimeError, match="not awake"):
            await executor.score(_request())
        _FakeAdapter.release.set()
        _, pending = await asyncio.wait({scoring, sibling, transition}, timeout=5)
        assert not pending, "released forward must settle every accepted caller and lifecycle join"
        with pytest.raises(asyncio.CancelledError):
            await scoring
        assert await sibling == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
        await transition
        if lifecycle == "sleep":
            assert weights.moves == [torch.device("cpu")]
            await executor.wake_up()
            assert await executor.score(_request()) == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
        else:
            assert adapter.closed
    finally:
        _FakeAdapter.release.set()
        cleanup = asyncio.create_task(executor.close())
        tasks = {scoring, sibling, cleanup}
        if transition is not None:
            tasks.add(transition)
        done, pending = await asyncio.wait(tasks, timeout=5)
        for task in done:
            if not task.cancelled():
                task.exception()
        for task in pending:
            task.cancel()
        assert not pending, "CLAP cancellation test cleanup left pending work"


@pytest.mark.asyncio
async def test_executor_load_sleep_wake_close_tracks_adapter_device_and_weights(monkeypatch, fake_adapter):
    executor = _executor(
        monkeypatch,
        model_path="weights",
        model_kwargs={"revision": "r"},
        processor_kwargs={"use_fast": True},
    )
    await executor.wake_up()
    scorer = executor._model
    adapter = fake_adapter[0]
    weights = adapter.model
    assert adapter.model_path == "weights"
    assert adapter.model_kwargs == {"revision": "r"}
    assert adapter.processor_kwargs == {"use_fast": True}

    await executor.sleep()
    assert adapter.device == torch.device("cpu")
    await executor.wake_up()
    assert adapter.device == scorer.device
    assert weights.moves == [torch.device("cpu"), scorer.device]

    await executor.close()
    assert scorer.model is None
    assert scorer._adapter is None
    assert adapter.closed


def test_fresh_executor_recreates_adapter_on_a_new_event_loop(monkeypatch, fake_adapter):
    async def score_and_close():
        executor = _executor(monkeypatch)
        await executor.wake_up()
        try:
            result = await executor.score(_request("up", waveform=(0.0, 1.0)))
            assert result == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
        finally:
            await executor.close()

    asyncio.run(score_and_close())
    asyncio.run(score_and_close())

    assert len(fake_adapter) == 2
    assert fake_adapter[0] is not fake_adapter[1]
    assert all(adapter.closed for adapter in fake_adapter)
    assert [len(adapter.calls) for adapter in fake_adapter] == [1, 1]


@pytest.mark.parametrize(
    "sample_request, error, message",
    [
        ({"extra_info": {}, "ground_truth": "right"}, ArtifactContractError, "absent"),
        (
            {
                "extra_info": {
                    "media_artifacts": {},
                    "audio": torch.ones(2),
                    "audio_sample_rate": 48_000,
                },
                "ground_truth": "right",
            },
            ArtifactContractError,
            "audio",
        ),
        (
            {
                "extra_info": {"audio": torch.ones(1, 2), "audio_sample_rate": 48_000, "audio_layout": "CT"},
                "ground_truth": "right",
                "audio_layout": "T",
            },
            ValueError,
            "Conflicting audio_layout",
        ),
        (
            {"extra_info": _request()["extra_info"], "ground_truth": "right", "prompt_key": "caption"},
            ValueError,
            "single-sample batch",
        ),
    ],
)
def test_preprocessing_failures_stay_in_their_slot(fake_adapter, sample_request, error, message):
    scorer = CLAPScorer()
    scorer.load("cpu")
    result, good = scorer.score_batch([sample_request, _request()])
    assert isinstance(result, error)
    assert message in str(result)
    assert good["score"] == pytest.approx(1.0)


def test_score_batch_all_invalid_preserves_error_objects_without_inference(fake_adapter, monkeypatch):
    scorer = CLAPScorer()
    scorer.load("cpu")
    requests = [_request(waveform=(float(index), 0.0)) for index in range(3)]
    errors = [ValueError("first"), RuntimeError("second"), TypeError("third")]
    errors_by_info = {id(request["extra_info"]): error for request, error in zip(requests, errors, strict=True)}

    def fail_preprocessing(extra_info, sample_batch, *, scorer):
        raise errors_by_info[id(extra_info)]

    monkeypatch.setattr(port._frozen, "audio_info_from_batch", fail_preprocessing)
    monkeypatch.setattr(fake_adapter[0], "infer", lambda *_: pytest.fail("adapter called for all-invalid batch"))

    results = scorer.score_batch(requests)

    assert all(result is error for result, error in zip(results, errors, strict=True))
    assert fake_adapter[0].calls == []


def test_score_batch_scatter_keeps_valid_prompt_audio_pairs_and_slot_errors(fake_adapter):
    scorer = CLAPScorer()
    scorer.load("cpu")
    missing_audio = _request("right")
    missing_audio["extra_info"] = {}
    requests = [
        _request("right", waveform=(1.0, 0.0)),
        missing_audio,
        _request("up", waveform=(0.0, 1.0)),
        _request("right", waveform=(0.6, 0.8), score_scale="bad"),
        _request("right", waveform=(0.0, 1.0)),
    ]

    results = scorer.score_batch(requests)

    assert results[0] == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert isinstance(results[1], ArtifactContractError)
    assert results[2] == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert isinstance(results[3], TypeError)
    assert results[4] == {"score": pytest.approx(0.0), "source_sample_rate": 48_000}
    assert len(fake_adapter[0].calls) == 1
    waveforms, prompts = fake_adapter[0].calls[0]
    assert prompts == ["right", "up", "right", "right"]
    np.testing.assert_array_equal(
        np.stack(waveforms),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_wrapper_forwards_the_original_request_kwargs_exactly():
    class _Executor:
        async def score(self, request):
            self.request = request
            return {"score": 0.5}

    executor = _Executor()
    batch = object()
    extra_info = {"audio": object()}
    result = await port.compute_score_clap_batched(
        data_source="source",
        solution_image=None,
        ground_truth="prompt",
        extra_info=extra_info,
        reward_model=executor,
        batch=batch,
        prompt_key="caption",
        score_scale=2.0,
        custom_option="kept",
    )

    assert result == {"score": 0.5}
    assert executor.request == {
        "data_source": "source",
        "solution_image": None,
        "ground_truth": "prompt",
        "extra_info": extra_info,
        "batch": batch,
        "prompt_key": "caption",
        "score_scale": 2.0,
        "custom_option": "kept",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_name", ["AudioRewardManager", "MultiAudioRewardManager"])
@pytest.mark.parametrize("transport", ["streamed", "finalized"])
@pytest.mark.parametrize("sample_rate", [48_000, np.array([48_000])], ids=["scalar", "numpy"])
async def test_audio_manager_calls_batched_entry_without_solution_image(
    monkeypatch, fake_adapter, manager_name, transport, sample_rate
):
    if manager_name == "MultiAudioRewardManager":
        manager_cls = getattr(manager_module, manager_name, None)
        if manager_cls is None:
            pytest.skip("MultiAudioRewardManager is introduced by #920")
    else:
        manager_cls = AudioRewardManager

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "reward.reward_model.enable=false",
                "+reward.models.clap.backend=native",
                "+reward.models.clap.placement.devices=[0]",
                "+reward.models.clap.executor.model=pkg://verl_omni.utils.reward_score.scorers.clap:CLAPScorer",
                "+reward.reward_functions.clap.path=pkg://verl_omni.utils.reward_score.scorers.clap",
                "+reward.reward_functions.clap.name=compute_score_clap_batched",
                "+reward.reward_functions.clap.model=clap",
                "+reward.reward_functions.clap.required=true",
            ],
        )
    executor = _executor(monkeypatch)
    compute_score = partial(port.compute_score_clap_batched, reward_model=executor)
    manager = manager_cls(config, MagicMock(), compute_score=compute_score)
    if manager_name == "MultiAudioRewardManager":
        manager.set_reward_executors(None, {"clap": executor})
    audio = {"audio": torch.tensor([[0.0, 1.0]]), "audio_sample_rate": sample_rate}
    metadata = (
        {"tool_extra_fields": [audio]} if transport == "streamed" else {key: [value] for key, value in audio.items()}
    )
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 4, dtype=torch.long)},
        non_tensors={"data_source": ["audio"], "reward_model": [{"ground_truth": "up"}], **metadata},
    )
    await executor.wake_up()
    try:
        result = await manager.run_single(data)
        assert result["reward_score"] == pytest.approx(1.0)
        rate_key = "source_sample_rate" if manager_name == "AudioRewardManager" else "reward/clap/source_sample_rate"
        assert result["reward_extra_info"][rate_key] == 48_000
        assert len(fake_adapter[0].calls) == 1
        waveforms, prompts = fake_adapter[0].calls[0]
        assert prompts == ["up"]
        np.testing.assert_array_equal(waveforms[0], np.array([0.0, 1.0], dtype=np.float32))
    finally:
        await executor.close()
