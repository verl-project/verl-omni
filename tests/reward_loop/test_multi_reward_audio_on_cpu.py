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
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from verl_omni.pipelines.rollout_media import ArtifactContractError, MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop.reward_manager import AudioRewardManager
from verl_omni.utils.reward_score.reward_utils import visual_reward_frames


def score_visual(solution_image, extra_info, **kwargs):
    assert "solution_audio" not in kwargs
    return float(visual_reward_frames(solution_image, extra_info)[0, 0, 0, 0])


async def score_audio(solution_audio, **kwargs):
    waveform, sample_rate = solution_audio
    assert waveform.dtype == np.float32
    assert waveform.shape == (4,)
    return {"score": float(waveform.mean()), "sample_rate": sample_rate}


def score_both(solution_image, solution_audio):
    return float(solution_image.flatten()[0]) + float(solution_audio[0].mean())


def score_optional_audio(solution_audio=None):
    return 0.25 if solution_audio is None else float(solution_audio[0].mean())


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
                "reward_functions": {key: {"path": __file__, **entry} for key, entry in terms.items()},
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


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_scorers_select_visual_audio_or_both(kind):
    pixels = torch.full((3, 2, 2) if kind == "image" else (2, 3, 2, 2), 4, dtype=torch.uint8)
    visual_spec = MediaSpec(kind, "decoded", "CHW" if kind == "image" else "TCHW", fps=24 if kind == "video" else None)
    audio = torch.tensor([[0.25] * 4, [0.75] * 4])
    fields = artifact_fields(
        {
            "visual": MediaArtifact(visual_spec, pixels),
            "audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=32_000), audio),
        },
        "visual",
        "visual",
    )
    data = _data(
        pixels,
        tensors={key: value.unsqueeze(0) for key, value in fields.items() if isinstance(value, torch.Tensor)},
        **{key: [value] for key, value in fields.items() if not isinstance(value, torch.Tensor)},
    )
    manager = _manager(
        "MultiVisualRewardManager",
        visual={"name": "score_visual", "weight": 0.25},
        audio={"name": "score_audio", "weight": 0.5},
        both={"name": "score_both", "weight": 0.25},
    )

    assert await manager.run_single(data) == {
        "reward_score": 2.375,
        "reward_extra_info": {
            "reward/visual": 4.0,
            "reward/audio": 0.5,
            "reward/audio/sample_rate": 32_000,
            "reward/both": 4.5,
            "reward/combined": 2.375,
        },
    }


@pytest.mark.parametrize(
    ("audio", "layout"),
    [(torch.tensor([[0.25] * 4, [0.75] * 4]), {}), (torch.full((4,), 0.5), {"audio_layout": "T"})],
)
async def test_legacy_video_routes_audio(audio, layout):
    pixels = torch.full((2, 3, 2, 2), 4, dtype=torch.uint8)
    data = _data(
        pixels,
        tool_extra_fields=[
            {
                "media_kind": "video",
                "fps": 24,
                "audio": audio,
                "audio_sample_rate": 32_000,
                **layout,
            }
        ],
    )
    manager = _manager(
        "MultiVisualRewardManager",
        visual={"name": "score_visual"},
        audio={"name": "score_audio"},
    )

    assert (await manager.run_single(data))["reward_score"] == 4.5


@pytest.mark.parametrize("transport", ["streamed", "finalized"])
async def test_audio_multi_reward_preserves_legacy_waveform_and_token_score_placement(transport):
    audio = {"audio": np.full(4, 0.75, dtype=np.float32), "audio_sample_rate": 24_000}
    metadata = (
        {"tool_extra_fields": [audio]} if transport == "streamed" else {key: [value] for key, value in audio.items()}
    )
    data = _data(
        torch.tensor([1, 2, 0]),
        tensors={"prompts": torch.tensor([[1, 1]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])},
        **metadata,
    )
    manager = _manager("MultiAudioRewardManager", audio={"name": "score_audio"})
    legacy = AudioRewardManager(manager.config, tokenizer=None, compute_score=score_audio)
    legacy_result = await legacy.run_single(data)
    result = await manager.run_single(data)

    assert result["reward_score"] == legacy_result["reward_score"] == 0.75
    assert result["reward_extra_info"]["reward/audio/sample_rate"] == legacy_result["reward_extra_info"]["sample_rate"]
    torch.testing.assert_close(
        manager.assemble_rm_scores(data, [result["reward_score"]]), torch.tensor([[0.0, 0.75, 0.0]])
    )


@pytest.mark.parametrize("audio", [{}, {"audio": torch.tensor([float("nan")]), "audio_sample_rate": 24_000}])
async def test_required_waveform_contract_errors_are_fatal_for_optional_terms(audio):
    manager = _manager("MultiAudioRewardManager", audio={"name": "score_audio", "required": False})
    data = _data(
        torch.zeros(3, dtype=torch.long),
        tool_extra_fields=[audio],
        extra_info=[{"audio": [0.75] * 4, "audio_sample_rate": 16_000}],
    )

    with pytest.raises(ArtifactContractError, match="audio"):
        await manager.run_single(data)


async def test_optional_waveform_can_be_omitted_by_scorer():
    manager = _manager("MultiAudioRewardManager", audio={"name": "score_optional_audio"})
    data = _data(torch.zeros(3, dtype=torch.long))

    assert (await manager.run_single(data))["reward_score"] == 0.25


async def test_existing_clap_scorer_accepts_audio_only_with_named_executor():
    class Executor:
        def reward_kwargs(self):
            return {"reward_model": self}

        async def infer(self, waveforms, prompts):
            assert prompts == ["prompt"]
            np.testing.assert_array_equal(waveforms[0], np.full(4, 0.75, dtype=np.float32))
            return {"audio_embeddings": torch.tensor([[1.0, 0.0]]), "text_embeddings": torch.tensor([[1.0, 1.0]])}

    manager = _manager(
        "MultiAudioRewardManager",
        models={"clap": {"backend": "native"}},
        alignment={
            "path": "pkg://verl_omni.utils.reward_score.clap",
            "name": "compute_score",
            "model": "clap",
            "required": True,
        },
    )
    manager.set_reward_executors({}, {"clap": Executor()})
    data = _data(
        torch.zeros(3, dtype=torch.long),
        tool_extra_fields=[{"audio": torch.full((4,), 0.75), "audio_sample_rate": 48_000}],
    )

    result = await manager.run_single(data)

    assert result["reward_score"] == pytest.approx(2**-0.5)
    assert result["reward_extra_info"]["reward/alignment/source_sample_rate"] == 48_000
