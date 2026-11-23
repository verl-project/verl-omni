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
"""Audio input selection through the shared reward manager."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.pipelines.rollout_media import ArtifactContractError, MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop.reward_manager import MultiAudioRewardManager, MultiVisualRewardManager


def score_visual(solution_image, **kwargs):
    assert "solution_audio" not in kwargs
    return float(solution_image.flatten()[0])


async def score_audio(solution_audio):
    waveform, rate = solution_audio
    assert waveform.dtype == np.float32 and waveform.ndim == 1
    return {"score": float(waveform.mean()), "rate": rate}


async def score_both(solution_image, *, solution_audio):
    result = await score_audio(solution_audio)
    result["score"] += float(solution_image.flatten()[0])
    return result


async def score_optional_audio(solution_audio=None):
    return 0.25 if solution_audio is None else await score_audio(solution_audio)


def _manager(manager_cls, **terms):
    functions = {key: {"path": __file__, "required": True, **term} for key, term in terms.items()}
    config = OmegaConf.create({"reward": {"reward_functions": functions}})
    return manager_cls(config, tokenizer=None, compute_score=None)


def _data(responses, *, tensors=None, **metadata):
    return DataProto.from_dict(
        tensors={"responses": responses.unsqueeze(0), **(tensors or {})},
        non_tensors={"data_source": ["test"], "reward_model": [{"ground_truth": "prompt"}], **metadata},
    )


def _named_visual_data(pixels, artifacts, **metadata):
    fields = artifact_fields(artifacts, "visual", "visual")
    tensors = {key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)}
    metadata.update({key: [value] for key, value in fields.items() if key not in tensors})
    return _data(pixels, tensors=tensors, **metadata)


@pytest.mark.parametrize(("layout", "fps", "audio_layout"), [("CHW", None, "T"), ("TCHW", 24, "CT")])
async def test_named_media_routes_only_requested_inputs(layout, fps, audio_layout):
    kind = "image" if fps is None else "video"
    pixels = torch.full((3, 2, 2) if fps is None else (2, 3, 2, 2), 4, dtype=torch.uint8)
    waveform = torch.tensor([0.0, 1.0]) if audio_layout == "T" else torch.tensor([[0.0], [1.0]])
    data = _named_visual_data(
        pixels,
        {
            "visual": MediaArtifact(MediaSpec(kind, "decoded", layout, fps=fps), pixels),
            "audio": MediaArtifact(MediaSpec("audio", "decoded", audio_layout, sample_rate=32_000), waveform),
        },
        tool_extra_fields=[{"audio": torch.zeros(2), "audio_sample_rate": 16_000, "audio_layout": "T"}],
    )
    manager = _manager(
        MultiVisualRewardManager,
        visual={"name": "score_visual", "weight": 0.5},
        both={"name": "score_both", "weight": 0.5},
    )

    result = await manager.run_single(data)
    assert result["reward_score"] == pytest.approx(4.25)
    assert result["reward_extra_info"]["reward/both/rate"] == 32_000


@pytest.mark.parametrize("named_audio", [False, True], ids=["visual-only", "invalid-named-audio"])
async def test_legacy_audio_fallback_with_named_visual_media(named_audio):
    pixels = torch.zeros(3, 2, 2, dtype=torch.uint8)
    waveform = torch.tensor([[0.0], [1.0]])
    artifacts = {"visual": MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels)}
    if named_audio:
        artifacts["audio"] = MediaArtifact(MediaSpec("audio", "latent", "CT"), waveform)
    data = _named_visual_data(
        pixels,
        artifacts,
        tool_extra_fields=[{"audio": np.array([[0.0], [1.0]], dtype=object), "audio_sample_rate": np.array(32_000)}],
    )
    manager = _manager(
        MultiVisualRewardManager,
        optional_audio={"name": "score_optional_audio", "required": False},
    )
    if named_audio:
        with pytest.raises(ArtifactContractError, match="expected audio/decoded"):
            await manager.run_single(data)
        return

    result = await manager.run_single(data)
    assert result["reward_score"] == 0.5
    assert result["reward_extra_info"]["reward/optional_audio/rate"] == 32_000


@pytest.mark.parametrize(
    ("extra_info", "error"),
    [
        ({}, "Two-dimensional audio requires explicit"),
        ({"audio_layout": "CT"}, "Conflicting.*audio_layout"),
    ],
    ids=["layout-mismatch", "conflicting-layouts"],
)
async def test_legacy_visual_audio_layout_is_projected(extra_info, error):
    pixels = torch.zeros(3, 2, 2, dtype=torch.uint8)
    data = _named_visual_data(
        pixels,
        {"visual": MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels)},
        audio_layout=["T"],
        extra_info=[extra_info],
        tool_extra_fields=[{"audio": torch.ones(2, 3), "audio_sample_rate": 32_000}],
    )
    manager = _manager(MultiVisualRewardManager, audio={"name": "score_audio", "required": False})
    with pytest.raises(ArtifactContractError, match=error):
        await manager.run_single(data)


async def test_audio_token_reward_uses_last_valid_response_token():
    manager = _manager(MultiAudioRewardManager, audio={"name": "score_audio", "weight": 0.5})
    data = _data(
        torch.tensor([1, 2, 0]),
        tensors={
            "prompts": torch.tensor([[1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "audio": torch.full((1, 4), 0.75),
        },
        audio_sample_rate=[48_000],
    )

    result = await manager.run_single(data)
    torch.testing.assert_close(
        manager.assemble_rm_scores(data, [result["reward_score"]]), torch.tensor([[0.0, 0.375, 0.0]])
    )


async def test_missing_audio_respects_scorer_signature():
    data = _data(torch.tensor([1]))
    defaulted = _manager(MultiAudioRewardManager, audio={"name": "score_optional_audio", "required": False})
    assert (await defaulted.run_single(data))["reward_score"] == 0.25

    requires_audio = _manager(MultiAudioRewardManager, audio={"name": "score_audio", "required": False})
    with pytest.raises(ArtifactContractError, match="audio"):
        await requires_audio.run_single(data)
