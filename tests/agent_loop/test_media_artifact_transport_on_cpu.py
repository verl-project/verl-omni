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
"""Named tensors and declarations survive both diffusion batching transports."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ
from verl_omni.pipelines.rollout_artifacts import MediaArtifact, artifact_fields, previews_from_batch
from verl_omni.pipelines.rollout_media import MediaSpec
from verl_omni.reward_loop.reward_manager.visual import _reward_extra_info
from verl_omni.trainer.diffusion.v1 import tq_utils
from verl_omni.utils.reward_score.clap import _get_audio
from verl_omni.utils.reward_score.imagebind import _to_tchw


def _internal():
    artifacts = {
        "video_preview": MediaArtifact(
            MediaSpec("video", "decoded", "TCHW", fps=12), torch.zeros(3, 3, 2, 2, dtype=torch.uint8)
        ),
        "video_latent": MediaArtifact(
            MediaSpec("video", "latent", "CTHW"), torch.ones(16, 2, 2, 2, dtype=torch.float16)
        ),
        "audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=24000), torch.ones(2, 16)),
    }
    fields = artifact_fields(artifacts, "video_latent", "video_preview")
    padded = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in fields.items()}
    return SimpleNamespace(
        prompt_ids=torch.tensor([[1, 2]]),
        response_diffusion_output=artifacts["video_latent"].data.unsqueeze(0),
        response_logprobs=None,
        reward_score=1.0,
        num_turns=1,
        metrics=SimpleNamespace(model_dump=lambda: {}),
        extra_fields=padded,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["ordinary", "tq"])
async def test_named_media_survives_to_preview_and_reward_consumers(monkeypatch, transport):
    if transport == "ordinary":
        worker = object.__new__(DiffusionAgentLoopWorker)
        data = worker._postprocess([_internal()])
    else:
        worker = object.__new__(DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class)
        stored = {}

        async def put(*, keys, fields, **kwargs):
            stored.update(fields=fields)

        monkeypatch.setattr(tq_utils.tq, "async_kv_batch_put", put)
        await worker._write_trajectory_to_tq(
            _internal(), uid="sample", session_id=0, trajectory={"step": 1}, validate=False
        )
        monkeypatch.setattr(tq_utils.tq, "kv_batch_get", lambda **kwargs: stored["fields"])
        data = tq_utils.diffusion_tq_batch_to_dataproto(SimpleNamespace(keys=["sample_0_0"], partition_id="train"))

    assert data.batch["media_artifact__video_latent"].dtype == torch.float16
    assert data.batch["media_artifact__video_latent"].shape == (1, 16, 2, 2, 2)
    previews = previews_from_batch(data)
    assert len(previews) == 1 and previews[0].data.dtype == torch.uint8
    assert previews[0].spec.fps == 12
    extra = _reward_extra_info(data[0])
    assert set(extra["media_artifacts"]) == {"video_preview", "video_latent", "audio"}
    audio, sample_rate = _get_audio(extra)
    assert sample_rate == 24000 and audio.shape == (16,)
    assert _to_tchw(extra["media_artifacts"]["video_preview"]).shape == (3, 3, 2, 2)
    assert data.batch["responses"].dtype == torch.float16  # preview selection never rewrites training responses


def test_named_scorers_reject_missing_stream_instead_of_legacy_fallback():
    with pytest.raises(ValueError, match="absent"):
        _get_audio({"media_artifacts": {}, "audio": torch.ones(16), "audio_sample_rate": 32000})
    with pytest.raises(ValueError, match="decoded video"):
        _to_tchw(MediaArtifact(MediaSpec("video", "latent", "CTHW"), torch.zeros(16, 3, 2, 2)))
