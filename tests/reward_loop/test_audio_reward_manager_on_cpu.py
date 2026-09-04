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
"""CPU tests for generic waveform reward routing."""

import importlib.util
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.reward_score import default_compute_score


def _load_audio_reward_manager():
    path = Path(__file__).parents[2] / "verl_omni/reward_loop/reward_manager/audio.py"
    spec = importlib.util.spec_from_file_location("audio_reward_manager_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AudioRewardManager


AudioRewardManager = _load_audio_reward_manager()


def _config():
    return OmegaConf.create({"reward": {}})


def _manager(compute_score):
    return AudioRewardManager(_config(), MagicMock(), compute_score=compute_score)


def _data(audio=None, sample_rate=24_000, *, layout="streamed"):
    fields = {}
    if audio is not None:
        fields = {"audio": audio, "audio_sample_rate": sample_rate}
    non_tensors = {
        "data_source": ["tts_reward"],
        "reward_model": [{"ground_truth": "ni3 hao3"}],
        "extra_info": [{"id": "sample-0"}],
        "tool_extra_fields": [fields if layout == "streamed" else {}],
    }
    if layout == "finalized" and audio is not None:
        non_tensors.update({"audio": [audio], "audio_sample_rate": [sample_rate]})
    return DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 4, dtype=torch.long)},
        non_tensors=non_tensors,
    )


def test_assemble_scores_preserves_one_reward_per_sample():
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.zeros(3, 2, dtype=torch.long),
            "responses": torch.zeros(3, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                ]
            ),
        }
    )

    scores = AudioRewardManager.assemble_rm_scores(data, [0.1, -1.0, 2.5])

    assert scores.shape == (3, 4)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(
        scores,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.1],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0],
            ]
        ),
    )


def test_run_single_rejects_multi_sample_batch():
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(2, 4, dtype=torch.long)},
        non_tensors={
            "data_source": ["tts_reward", "tts_reward"],
            "reward_model": [{"ground_truth": "first"}, {"ground_truth": "second"}],
        },
    )
    manager = _manager(lambda **kwargs: 0.0)

    with pytest.raises(ValueError, match="batch size 2"):
        manager.loop.run_until_complete(manager.run_single(data))


@pytest.mark.parametrize("compute_score", [default_compute_score, partial(default_compute_score)])
def test_default_text_reward_function_is_rejected(compute_score):
    with pytest.raises(ValueError, match="custom_reward_function"):
        _manager(compute_score)


def test_run_single_passes_waveform_and_returns_diagnostics():
    def compute_score(data_source, solution_audio, ground_truth, extra_info):
        waveform, sample_rate = solution_audio
        assert data_source == "tts_reward"
        assert waveform.dtype == np.float32
        assert waveform.shape == (24_000,)
        assert sample_rate == 24_000
        assert ground_truth == "ni3 hao3"
        assert extra_info["id"] == "sample-0"
        assert "global_steps" not in extra_info
        return {"score": 0.75, "pinyin_error_rate": 0.1}

    manager = _manager(compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_data(np.ones(24_000, dtype=np.float32))))

    assert result == {
        "reward_score": 0.75,
        "reward_extra_info": {"pinyin_error_rate": 0.1},
    }


def test_run_single_reads_finalized_top_level_audio_layout():
    def compute_score(solution_audio, extra_info, **kwargs):
        waveform, sample_rate = solution_audio
        np.testing.assert_array_equal(waveform, np.ones(8, dtype=np.float32))
        assert sample_rate == 16_000
        assert extra_info["id"] == "sample-0"
        return 0.5

    manager = _manager(compute_score)
    result = manager.loop.run_until_complete(
        manager.run_single(_data(np.ones(8, dtype=np.float32), 16_000, layout="finalized"))
    )

    assert result["reward_score"] == 0.5


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_data(), "requires extra_info\\['audio'\\]"),
        (_data([], 24_000), "empty waveform"),
        (_data([0.0, float("nan")], 24_000), "NaN or infinity"),
        (_data([0.0], 0), "positive integer"),
        (_data([0.0], 24_000.5), "positive integer"),
    ],
)
def test_invalid_audio_fails_closed(data, message):
    manager = _manager(lambda **kwargs: 0.0)

    with pytest.raises((KeyError, ValueError), match=message):
        manager.loop.run_until_complete(manager.run_single(data))


def test_missing_sample_rate_fails_closed():
    data = _data([0.0])
    data.non_tensor_batch["tool_extra_fields"][0].pop("audio_sample_rate")
    manager = _manager(lambda **kwargs: 0.0)

    with pytest.raises(KeyError, match="audio_sample_rate"):
        manager.loop.run_until_complete(manager.run_single(data))


def test_chunked_waveform_is_rejected_instead_of_guessed():
    manager = _manager(lambda **kwargs: 0.5)

    with pytest.raises(ValueError, match="could not convert"):
        manager.loop.run_until_complete(
            manager.run_single(_data([torch.tensor([0.1, 0.2]), torch.tensor([0.3])], 16_000))
        )


def test_two_dimensional_waveform_is_rejected_instead_of_downmixed_on_the_wrong_axis():
    manager = _manager(lambda **kwargs: 0.5)

    with pytest.raises(ValueError, match="one mono waveform"):
        manager.loop.run_until_complete(manager.run_single(_data(np.zeros((128, 2)), 16_000)))


@pytest.mark.asyncio
async def test_async_score_function_is_supported():
    async def compute_score(solution_audio, **kwargs):
        assert solution_audio[1] == 24_000
        return {"score": -0.25, "judge_margin": 3.0}

    result = await _manager(compute_score).run_single(_data(torch.ones(32)))

    assert result == {"reward_score": -0.25, "reward_extra_info": {"judge_margin": 3.0}}


@pytest.mark.asyncio
async def test_waveform_extraction_runs_off_the_event_loop(monkeypatch):
    event_loop_thread = threading.get_ident()
    extraction_threads = []

    def extract_audio(extra_info):
        extraction_threads.append(threading.get_ident())
        return np.zeros(8, dtype=np.float32), 24_000

    async def compute_score(**kwargs):
        return 0.5

    monkeypatch.setattr(AudioRewardManager, "_extract_audio", staticmethod(extract_audio))
    result = await _manager(compute_score).run_single(_data(np.ones(8, dtype=np.float32)))

    assert result["reward_score"] == 0.5
    assert extraction_threads and extraction_threads[0] != event_loop_thread


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_reward_fails_closed(score):
    manager = _manager(lambda **kwargs: score)

    with pytest.raises(ValueError, match="must be finite"):
        manager.loop.run_until_complete(manager.run_single(_data([0.0])))


def test_reward_dictionary_requires_score():
    manager = _manager(lambda **kwargs: {"metric": 1.0})

    with pytest.raises(ValueError, match="missing 'score'"):
        manager.loop.run_until_complete(manager.run_single(_data([0.0])))
