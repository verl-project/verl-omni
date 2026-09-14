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
"""Generated media must reach scorers through both reward execution paths."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager import multi
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager
from verl_omni.trainer.diffusion.v1 import tq_utils


def _manager(monkeypatch, manager_cls, scorer):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "pt"}}},
            "reward": {"reward_functions": {"audio": {"path": "test", "name": "score", "required": True}}},
        }
    )
    monkeypatch.setattr(multi, "load_extern_object", lambda *args: scorer)
    return manager_cls(config, tokenizer=None, compute_score=scorer)


def _data(monkeypatch, transport):
    audio = torch.arange(16, dtype=torch.float16).reshape(1, 1, 16)
    tensors = {"responses": torch.zeros(1, 4, 3, 8, 8, dtype=torch.uint8)}
    non_tensors = {
        "data_source": ["test"],
        "reward_model": [{"ground_truth": "prompt"}],
        "extra_info": [{"dataset_field": "kept", "audio": "reference audio"}],
    }
    metadata = {"media_kind": "video", "audio_sample_rate": 32000}
    if transport == "tool":
        non_tensors["tool_extra_fields"] = [{"audio": audio[0], **metadata}]
    elif transport in ("top", "both"):
        tensors["audio"] = audio
        non_tensors.update({key: [value] for key, value in metadata.items()})
        if transport == "both":
            non_tensors["tool_extra_fields"] = [{"audio": audio[0].clone(), **metadata}]
    else:
        payload = {**tensors, **non_tensors, "audio": audio, "extra_fields": [metadata]}
        monkeypatch.setattr(tq_utils.tq, "kv_batch_get", lambda **kwargs: payload)
        return tq_utils.diffusion_tq_batch_to_dataproto(SimpleNamespace(keys=["sample"], partition_id="train")), audio
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), audio


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_cls", [VisualRewardManager, multi.MultiVisualRewardManager])
@pytest.mark.parametrize("transport", ["tool", "top", "tq", "both"])
async def test_generated_media_reaches_scorer(monkeypatch, manager_cls, transport):
    data, audio = _data(monkeypatch, transport)

    async def scorer(data_source, solution_image, ground_truth, extra_info):
        assert data_source == "test" and ground_truth == "prompt"
        assert solution_image.dtype == torch.uint8
        torch.testing.assert_close(extra_info["audio"], audio[0])
        assert extra_info["audio_sample_rate"] == 32000
        assert extra_info["media_kind"] == "video"
        assert extra_info["dataset_field"] == "kept"
        assert "responses" not in extra_info
        return {"score": 1.0}

    result = await _manager(monkeypatch, manager_cls, scorer).run_single(data)
    assert result["reward_score"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_cls", [VisualRewardManager, multi.MultiVisualRewardManager])
@pytest.mark.parametrize(
    "field, value", [("media_kind", "image"), ("audio_sample_rate", 24000), ("audio", torch.zeros(1, 16))]
)
async def test_conflicting_generated_media_sources_fail_before_scoring(monkeypatch, manager_cls, field, value):
    data, _ = _data(monkeypatch, "top")
    data.non_tensor_batch["tool_extra_fields"] = np.array([{field: value}], dtype=object)

    async def scorer(**kwargs):
        pytest.fail("Conflicting generated media must not reach the scorer")

    with pytest.raises(ValueError, match=f"Conflicting rollout media field.*{field}"):
        await _manager(monkeypatch, manager_cls, scorer).run_single(data)
