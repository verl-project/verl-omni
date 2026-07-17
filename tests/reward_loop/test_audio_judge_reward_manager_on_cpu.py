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
"""CPU tests for AudioJudgeRewardManager (mocked judge, no GPU, no network)."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.audio_judge import AudioJudgeRewardManager


def _make_config(num_workers=1, n=2):
    return OmegaConf.create(
        {
            "reward": {
                "num_workers": num_workers,
                "audio": {"codec": "qwen_tts"},
                "judge_urls": ["http://localhost:8901"],
                "judge_debias": True,
                "judge_timeout_s": 5,
            },
            "actor_rollout_ref": {"model": {"path": "some/ckpt"}, "rollout": {"n": n}},
        }
    )


def _make_item(uid="u0", audio=None, sr=24000) -> DataProto:
    extra_info = {"id": uid, "text": "hello world"}
    non_tensors = {
        "data_source": ["tts"],
        "reward_model": [{"ground_truth": "hello world"}],
        "extra_info": [extra_info],
        "uid": [uid],
        "audio": [audio if audio is not None else np.zeros(24000, dtype=np.float32)],
        "sr": [sr],
    }
    return DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 4, dtype=torch.long)},
        non_tensors=non_tensors,
    )


def test_num_workers_must_be_one():
    with pytest.raises(ValueError, match="num_workers=1"):
        AudioJudgeRewardManager(_make_config(num_workers=2), MagicMock(), compute_score=None)


def test_rendezvous_pairs_and_scores():
    manager = AudioJudgeRewardManager(_make_config(n=2), MagicMock(), compute_score=None)
    # Judge prefers candidate 0 over candidate 1.
    manager._judge_group = lambda text, blobs: [1.0, 0.0]

    d0 = _make_item(audio=np.ones(24000, dtype=np.float32))
    d1 = _make_item(audio=np.zeros(24000, dtype=np.float32))
    r0, r1 = manager.loop.run_until_complete(asyncio.gather(manager.run_single(d0), manager.run_single(d1)))
    assert r0["reward_extra_info"]["sj_score"] == 1.0
    assert r1["reward_extra_info"]["sj_score"] == 0.0
    assert r0["reward_score"] == 1.0 and r1["reward_score"] == 0.0
    assert r0["reward_extra_info"]["synth_ok"] == 1.0


def test_judge_receives_both_blobs():
    manager = AudioJudgeRewardManager(_make_config(n=2), MagicMock(), compute_score=None)
    seen = {}

    def fake_judge(text, blobs):
        seen["n"] = len(blobs)
        seen["text"] = text
        return [1.0, 0.0]

    manager._judge_group = fake_judge
    d0, d1 = _make_item(), _make_item()
    manager.loop.run_until_complete(asyncio.gather(manager.run_single(d0), manager.run_single(d1)))
    assert seen["n"] == 2  # the group's two candidates judged in one call
    assert seen["text"] == "hello world"
