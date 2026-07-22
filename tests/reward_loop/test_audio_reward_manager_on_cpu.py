# Copyright 2026 Gulp AI Inc and/or its affiliates
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
"""CPU tests for AudioRewardManager and MultiAudioRewardManager (no models, no GPU)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.audio import AudioRewardManager
from verl_omni.reward_loop.reward_manager.multi import MultiAudioRewardManager

# Path to this file; sub-reward entries load the dummy functions below from here.
DUMMY_REWARDS_PATH = "tests/reward_loop/test_audio_reward_manager_on_cpu.py"


def reward_wav_seconds(solution_audio):
    """Returns the clip length in seconds, 0.0 when synthesis failed."""
    if solution_audio is None:
        return 0.0
    wav, sr = solution_audio
    return len(wav) / sr


def reward_with_extras(data_source, ground_truth):
    return {"score": 1.0, "detail": 7.0}


def _make_config(audio=None, reward_functions=None):
    reward = {"audio": audio or {}}
    if reward_functions is not None:
        reward["reward_functions"] = reward_functions
    return OmegaConf.create({"reward": reward, "actor_rollout_ref": {"model": {"path": "some/ckpt"}}})


def _build_manager(compute_score, audio=None) -> AudioRewardManager:
    return AudioRewardManager(_make_config(audio), MagicMock(), compute_score=compute_score)


def _make_data(audio=None, sr=24000, codes=None) -> DataProto:
    extra_info = {"id": "u0", "text": "hello world"}
    non_tensors = {
        "data_source": ["tts_reward"],
        "reward_model": [{"ground_truth": "hello world"}],
        "extra_info": [extra_info],
    }
    if codes is not None:
        extra_info["tts_audio_codes"] = codes
    if audio is not None:
        non_tensors["audio"] = [audio]
        non_tensors["sr"] = [sr]
    return DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 4, dtype=torch.long)},
        non_tensors=non_tensors,
    )


class TestInit:
    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError, match="Unknown reward.audio.codec"):
            _build_manager(compute_score=None, audio={"codec": "fish_s2"})

    def test_decoder_path_defaults_to_actor_model(self):
        manager = _build_manager(compute_score=None)
        assert manager._decoder_model_path == "some/ckpt"

    def test_assemble_rm_scores_shape(self):
        out = AudioRewardManager.assemble_rm_scores(MagicMock(), [0.1, -1.0, 2.5])
        assert out.shape == (3, 1)


class TestRunSingle:
    def test_dict_result_maps_to_extra_info(self):
        def compute_score(data_source, solution_audio, ground_truth, extra_info):
            assert data_source == "tts_reward"
            assert solution_audio is not None and solution_audio[1] == 24000
            assert ground_truth == "hello world"
            return {"score": 2.5, "text": 1.0, "sim": 0.5}

        manager = _build_manager(compute_score)
        wav = np.zeros(24000, dtype=np.float32)
        result = manager.loop.run_until_complete(manager.run_single(_make_data(audio=wav)))
        assert result["reward_score"] == 2.5
        assert result["reward_extra_info"] == {"text": 1.0, "sim": 0.5}

    def test_float_result_maps_to_acc(self):
        manager = _build_manager(lambda data_source, solution_audio, ground_truth, extra_info: 0.75)
        wav = np.zeros(24000, dtype=np.float32)
        result = manager.loop.run_until_complete(manager.run_single(_make_data(audio=wav)))
        assert result["reward_score"] == 0.75
        assert result["reward_extra_info"] == {"acc": 0.75}

    def test_no_audio_passes_none(self):
        seen = {}

        def compute_score(data_source, solution_audio, ground_truth, extra_info):
            seen["solution_audio"] = solution_audio
            return 0.0

        manager = _build_manager(compute_score)
        manager.loop.run_until_complete(manager.run_single(_make_data()))
        assert seen["solution_audio"] is None

    def test_codes_are_decoded(self):
        def compute_score(data_source, solution_audio, ground_truth, extra_info):
            wav, sr = solution_audio
            return {"score": float(len(wav)), "sr": sr}

        manager = _build_manager(compute_score)
        manager._decode = lambda codes: (np.ones(len(codes) * 100, dtype=np.float32), 24000)
        manager._codebook_max = 2047
        # 5 real frames, then eos at codec-0 of frame 5
        codes = torch.zeros(8, 16, dtype=torch.long)
        codes[5, 0] = manager._codec_eos
        result = manager.loop.run_until_complete(manager.run_single(_make_data(codes=codes.numpy())))
        assert result["reward_score"] == 500.0  # eos-trimmed to 5 frames
        assert result["reward_extra_info"]["sr"] == 24000


class TestMultiAudioRewardManager:
    def _build(self, reward_functions):
        config = _make_config(reward_functions=reward_functions)
        return MultiAudioRewardManager(config, MagicMock(), compute_score=None)

    def test_weighted_sum_over_decoded_audio(self):
        manager = self._build(
            {
                "seconds": {"path": DUMMY_REWARDS_PATH, "name": "reward_wav_seconds", "weight": 2.0},
                "extras": {"path": DUMMY_REWARDS_PATH, "name": "reward_with_extras", "weight": 1.0},
            }
        )
        wav = np.zeros(24000, dtype=np.float32)  # 1 second
        result = manager.loop.run_until_complete(manager.run_single(_make_data(audio=wav)))
        info = result["reward_extra_info"]
        assert info["reward/seconds"] == 1.0
        assert info["reward/extras"] == 1.0
        assert info["reward/extras/detail"] == 7.0
        assert result["reward_score"] == 2.0 * 1.0 + 1.0 * 1.0
        assert info["reward/combined"] == result["reward_score"]

    def test_failed_synthesis_passes_none(self):
        manager = self._build({"seconds": {"path": DUMMY_REWARDS_PATH, "name": "reward_wav_seconds"}})
        result = manager.loop.run_until_complete(manager.run_single(_make_data()))
        assert result["reward_extra_info"]["reward/seconds"] == 0.0

    def test_empty_reward_functions_raises(self):
        with pytest.raises(ValueError, match="non-empty reward.reward_functions"):
            self._build({})
