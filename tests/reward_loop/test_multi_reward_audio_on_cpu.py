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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from verl_omni.pipelines.rollout_media import ArtifactContractError, MediaArtifact, MediaSpec, artifact_fields
from verl_omni.utils.reward_score.reward_utils import visual_reward_frames


def score_visual(solution_image, extra_info, **solution_audio):
    assert "solution_audio" not in solution_audio
    return float(visual_reward_frames(solution_image, extra_info).flatten()[0])


async def score_audio(solution_audio):
    waveform, rate = solution_audio
    assert waveform.dtype == np.float32 and waveform.ndim == 1
    return {"score": float(waveform.mean()), "rate": rate}


def score_both(solution_image, *, solution_audio):
    return float(solution_image.flatten()[0]) + float(solution_audio[0].mean())


async def score_optional_audio(solution_audio=None):
    return 0.25 if solution_audio is None else await score_audio(solution_audio)


def _manager(name, *, models=None, **terms):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "pt"}}},
            "reward": {
                "reward_manager": {
                    "source": "importlib",
                    "name": name,
                    "module": {"path": "pkg://verl_omni.reward_loop.reward_manager"},
                },
                "reward_functions": {key: {"path": __file__, "required": True, **term} for key, term in terms.items()},
                "models": models or {},
            },
        }
    )
    return resolve_reward_manager_cls(config)(config, tokenizer=None, compute_score=None)


def _data(responses, *, tensors=None, **metadata):
    return DataProto.from_dict(
        tensors={"responses": responses.unsqueeze(0), **(tensors or {})},
        non_tensors={"data_source": ["test"], "reward_model": [{"ground_truth": "prompt"}], **metadata},
    )


@pytest.mark.parametrize(("layout", "fps", "audio_layout"), [("CHW", None, "T"), ("TCHW", 24, "CT")])
async def test_named_media_routes_only_requested_inputs(layout, fps, audio_layout):
    kind = "image" if fps is None else "video"
    pixels = torch.full((3, 2, 2) if fps is None else (2, 3, 2, 2), 4, dtype=torch.uint8)
    waveform = torch.tensor([0.0, 1.0]) if audio_layout == "T" else torch.tensor([[0.0], [1.0]])
    fields = artifact_fields(
        {
            "visual": MediaArtifact(MediaSpec(kind, "decoded", layout, fps=fps), pixels),
            "audio": MediaArtifact(MediaSpec("audio", "decoded", audio_layout, sample_rate=32_000), waveform),
        },
        "visual",
        "visual",
    )
    data = _data(
        pixels,
        tensors={key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)},
        tool_extra_fields=[{"audio": torch.zeros(2), "audio_sample_rate": 16_000}],
        **{key: [value] for key, value in fields.items() if not isinstance(value, torch.Tensor)},
    )
    manager = _manager(
        "MultiVisualRewardManager",
        visual={"name": "score_visual", "weight": 0.25},
        audio={"name": "score_audio", "weight": 0.5},
        both={"name": "score_both", "weight": 0.25},
    )

    result = await manager.run_single(data)
    assert result["reward_score"] == 2.375
    assert result["reward_extra_info"]["reward/audio/rate"] == 32_000


async def test_legacy_video_routes_channels_first_audio():
    data = _data(
        torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
        tool_extra_fields=[{"media_kind": "video", "audio": torch.tensor([[0.0], [1.0]]), "audio_sample_rate": 32_000}],
    )
    manager = _manager("MultiVisualRewardManager", audio={"name": "score_audio"})

    assert (await manager.run_single(data))["reward_score"] == 0.5


@pytest.mark.parametrize("named_audio", [False, True], ids=["visual-only", "invalid-named-audio"])
async def test_legacy_audio_fallback_with_named_visual_media(named_audio):
    pixels = torch.zeros(3, 2, 2, dtype=torch.uint8)
    waveform = torch.tensor([[0.0], [1.0]])
    artifacts = {"visual": MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels)}
    if named_audio:
        artifacts["audio"] = MediaArtifact(MediaSpec("audio", "latent", "CT"), waveform)
    fields = artifact_fields(artifacts, "visual", "visual")
    data = _data(
        pixels,
        tensors={key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)},
        tool_extra_fields=[{"audio": waveform, "audio_sample_rate": np.array(32_000)}],
        **{key: [value] for key, value in fields.items() if not isinstance(value, torch.Tensor)},
    )
    manager = _manager(
        "MultiVisualRewardManager",
        optional_audio={"name": "score_optional_audio", "required": False},
    )
    if named_audio:
        with pytest.raises(ArtifactContractError, match="expected audio/decoded"):
            await manager.run_single(data)
        return

    result = await manager.run_single(data)
    assert result["reward_score"] == 0.5
    assert result["reward_extra_info"]["reward/optional_audio/rate"] == 32_000


async def test_audio_token_rollout_supports_waveform_and_named_clap():
    infer = AsyncMock(
        return_value={"audio_embeddings": torch.tensor([[1.0, 0.0]]), "text_embeddings": torch.tensor([[1.0, 0.0]])}
    )
    manager = _manager(
        "MultiAudioRewardManager",
        models={"clap": {"backend": "native"}},
        audio={"name": "score_audio"},
        clap={"path": "pkg://verl_omni.utils.reward_score.clap", "name": "compute_score", "weight": 0.5},
    )
    manager.set_reward_executors(
        {}, {"clap": SimpleNamespace(reward_kwargs=lambda: {"reward_model": SimpleNamespace(infer=infer)})}
    )
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
    assert result["reward_score"] == 1.25
    infer.assert_awaited_once()
    torch.testing.assert_close(
        manager.assemble_rm_scores(data, [result["reward_score"]]), torch.tensor([[0.0, 1.25, 0.0]])
    )


async def test_missing_required_waveform_is_fatal_for_optional_term():
    manager = _manager("MultiAudioRewardManager", audio={"name": "score_audio", "required": False})
    data = _data(torch.tensor([1]), extra_info=[{"audio": [0.5], "audio_sample_rate": 16_000}])

    with pytest.raises(ArtifactContractError, match="audio"):
        await manager.run_single(data)


async def test_visual_only_rollout_allows_optional_audio_and_variadic_kwargs():
    manager = _manager(
        "MultiVisualRewardManager",
        visual={"name": "score_visual"},
        audio={"name": "score_optional_audio"},
    )
    data = _data(torch.full((3, 2, 2), 4, dtype=torch.uint8), media_kind=["image"])

    assert (await manager.run_single(data))["reward_score"] == 4.25
