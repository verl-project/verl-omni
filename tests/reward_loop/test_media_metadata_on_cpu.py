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
async def test_latent_primary_and_decoded_artifacts_reach_real_reward_managers(monkeypatch, manager_cls):
    from verl_omni.pipelines.rollout_artifacts import MediaArtifact, artifact_fields
    from verl_omni.pipelines.rollout_media import MediaSpec
    from verl_omni.utils.reward_score.clap import _get_audio

    latent = torch.zeros(16, 2, 2, 2, dtype=torch.float16)
    artifacts = {
        "video_latent": MediaArtifact(MediaSpec("video", "latent", "CTHW"), latent),
        "video_preview": MediaArtifact(
            MediaSpec("video", "decoded", "TCHW", fps=24), torch.zeros(3, 3, 2, 2, dtype=torch.uint8)
        ),
        "audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=32000), torch.ones(2, 16)),
    }
    fields = artifact_fields(artifacts, "video_latent", "video_preview")
    data = DataProto.from_dict(
        tensors={
            "responses": latent.unsqueeze(0),
            **{key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)},
        },
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "prompt"}],
            **{key: [value] for key, value in fields.items() if not isinstance(value, torch.Tensor)},
        },
    )

    async def scorer(solution_image, extra_info, **kwargs):
        assert solution_image.dtype == torch.float16
        assert extra_info["media_artifacts"]["video_preview"].data.dtype == torch.uint8
        audio, rate = _get_audio(extra_info)
        assert rate == 32000 and audio.shape == (16,)
        return {"score": 1.0}

    manager = _manager(monkeypatch, manager_cls, scorer)
    manager.config.actor_rollout_ref.rollout.pipeline.output_type = "latent"
    result = await manager.run_single(data)
    assert result["reward_score"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_cls", [VisualRewardManager, multi.MultiVisualRewardManager])
@pytest.mark.parametrize("mismatch", [False, True])
async def test_decoded_audio_primary_is_self_describing_not_inferred_from_pixel_config(
    monkeypatch, manager_cls, mismatch
):
    from verl_omni.pipelines.rollout_artifacts import ArtifactContractError, MediaArtifact, artifact_fields
    from verl_omni.pipelines.rollout_media import MediaSpec
    from verl_omni.utils.reward_score.clap import _get_audio

    audio = torch.ones(2, 16)
    fields = artifact_fields(
        {"audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=32000), audio)}, "audio"
    )
    data = DataProto.from_dict(
        tensors={
            "responses": (audio + 1 if mismatch else audio).unsqueeze(0),
            **{key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)},
        },
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "sound"}],
            **{key: [value] for key, value in fields.items() if not isinstance(value, torch.Tensor)},
        },
    )

    async def scorer(solution_image, extra_info, **kwargs):
        assert solution_image.dtype == torch.float32
        waveform, rate = _get_audio(extra_info)
        assert waveform.shape == (16,) and rate == 32000
        return {"score": 1.0}

    manager = _manager(monkeypatch, manager_cls, scorer)
    if mismatch:
        with pytest.raises(ArtifactContractError, match="responses projection"):
            await manager.run_single(data)
    else:
        result = await manager.run_single(data)
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
