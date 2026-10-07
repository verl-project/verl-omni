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
"""CPU integration for H3's declared native/decoded outputs and downstream export."""

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline
from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import MiniMaxH3PipelineWithLogProb
from verl_omni.pipelines.rollout_artifacts import MediaArtifact
from verl_omni.pipelines.rollout_media import MediaSpec
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


def _make_h3_final_res(monkeypatch, algorithm, output_type="pt", extra_args=None, fps=24):
    video = torch.zeros(1, 5, 8, 12, 3, dtype=torch.uint8)
    audio = torch.ones(1, 2, 320)
    video_latent = torch.ones(1, 24, 2, 2, 2, dtype=torch.float16)
    audio_latent = torch.full((2, 32, 8), 2, dtype=torch.bfloat16)
    prompt_embeds = torch.ones(1, 5, 8)
    latent_meta = torch.tensor([[2, 16, 2, 2, 2, 8]])
    joint = torch.ones(1, 704)
    pipeline_cls = MiniMaxH3DiffusionNFTPipeline if algorithm == "diffusion_nft" else MiniMaxH3PipelineWithLogProb
    pipeline = object.__new__(pipeline_cls)
    request = SimpleNamespace(
        request_id="h3-test",
        prompts=[{"prompt": "H3 prompt"}],
        sampling_params=SimpleNamespace(
            frame_rate=fps,
            output_type=output_type,
            extra_args=extra_args or {},
            num_outputs_per_prompt=1,
            max_sequence_length=5,
        ),
    )

    def forward(self, request):
        if algorithm == "diffusion_nft":
            self._nft_capture = {
                "video_latent": video_latent,
                "audio_latent": audio_latent,
                "text_embeddings": prompt_embeds[0],
                "text_tags": torch.ones(5, dtype=torch.long),
                "condition_video_rows": torch.zeros(0, 96),
                "keyframe_frame_indices": [],
                "task": "t2va",
                "latent_t": 2,
                "latent_h": 2,
                "latent_w": 2,
                "audio_t": 8,
                "num_steps": 10,
                "video_shift": 12.0,
            }
        else:
            self._flow_grpo_final_latents = (video_latent, audio_latent)
            self._flow_grpo_trajectory = {
                "all_latents": joint.unsqueeze(1),
                "all_next_latents": (joint + 1).unsqueeze(1),
                "all_timesteps": torch.tensor([[0.25]]),
                "all_log_probs": torch.tensor([[0.4]]),
                "latent_meta": latent_meta,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": torch.ones(1, 5, dtype=torch.long),
            }
        return DiffusionOutput(output=(video, audio))

    monkeypatch.setattr(MiniMaxH3Pipeline, "forward", forward)
    if algorithm == "diffusion_nft":
        result = pipeline.forward(request)
    else:
        result = pipeline.forward(
            SimpleNamespace(requests=[request], prompts=request.prompts, sampling_params=request.sampling_params)
        )
    return OmniRequestOutput.from_diffusion(
        images=result.output["payload"]["video"],
        multimodal_output={"metadata": result.output["metadata"]},
        trajectory_latents=result.trajectory_latents,
        trajectory_timesteps=result.trajectory_timesteps,
        trajectory_log_probs=result.trajectory_log_probs,
        request_id="h3-test",
    )


def _generate(monkeypatch, algorithm="diffusion_nft", output_type="pt", extra_args=None, fps=24):
    server = SimpleNamespace(
        global_steps=3, model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm=algorithm)
    )
    final_res = _make_h3_final_res(monkeypatch, algorithm, output_type, extra_args, fps)
    return DiffusionStrategy(server).process_output(
        final_res,
        None,
        {"output_type": (extra_args or {}).get("output_type", output_type), "logprobs": algorithm == "flow_grpo"},
    )


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_named_video_has_canonical_shape_and_preserves_native_latents(monkeypatch, algorithm):
    result = _generate(monkeypatch, algorithm)
    assert result.diffusion_output.shape == (5, 3, 8, 12)
    assert result.diffusion_output.dtype == torch.uint8
    video_latent = result.artifacts["video_latent"].data
    assert video_latent.shape == (24, 2, 2, 2)
    torch.testing.assert_close(video_latent, torch.ones_like(video_latent, dtype=torch.float16))
    audio_latent = result.artifacts["audio_latent"]
    assert audio_latent.spec.layout == "CLT" and audio_latent.data.shape == (2, 32, 8)
    torch.testing.assert_close(audio_latent.data, torch.full_like(audio_latent.data, 2, dtype=torch.bfloat16))


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_named_audio_is_not_unbatched_a_second_time(monkeypatch, algorithm):
    result = _generate(monkeypatch, algorithm)
    torch.testing.assert_close(result.extra_fields["audio"], torch.ones(2, 320))
    assert result.extra_fields["audio_sample_rate"] == 32000
    assert result.extra_fields["media_kind"] == "video"


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_training_fields_remain_separate_from_media(monkeypatch, algorithm):
    result = _generate(monkeypatch, algorithm)
    if algorithm == "diffusion_nft":
        assert result.extra_fields["latents_clean"].shape == (704,)
        assert result.extra_fields["train_timesteps"].shape == (9,)
    else:
        assert result.extra_fields["all_latents"].shape == (1, 704)
        torch.testing.assert_close(result.extra_fields["all_next_latents"], torch.full((1, 704), 2.0))
        torch.testing.assert_close(result.log_probs, torch.tensor([0.4]))
    for key in ("latent_meta", "prompt_embeds", "prompt_embeds_mask"):
        assert key in result.extra_fields
    assert set(result.artifacts) == {"video_preview", "audio", "video_latent", "audio_latent"}


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
@pytest.mark.parametrize(
    "output_type, override, primary",
    [("pt", None, "video_preview"), ("latent", None, "video_latent"), ("pt", "latent", "video_latent")],
)
@pytest.mark.parametrize("fps", [None, 23.976])
def test_adapter_selects_primary_and_preserves_decoded_preview(
    monkeypatch, algorithm, output_type, override, primary, fps
):
    extra = {"requested_outputs": ["video_preview", "audio"]}
    if override is not None:
        extra["output_type"] = override
    result = _generate(monkeypatch, algorithm, output_type=output_type, extra_args=extra, fps=fps)
    assert result.primary_artifact == primary
    assert result.preview_artifact == "video_preview"
    assert result.artifacts["video_preview"].spec.fps == (24 if fps is None else fps)
    torch.testing.assert_close(result.diffusion_output, result.artifacts[primary].data)


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_adapter_rejects_missing_requested_artifact(monkeypatch, algorithm):
    with pytest.raises(ValueError, match="requested artifacts absent.*missing"):
        _generate(monkeypatch, algorithm, extra_args={"requested_outputs": ["missing"]})


def test_rewards_reject_undeclared_channels_first_input():
    from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames

    video = torch.zeros(3, 5, 8, 12, dtype=torch.uint8)
    with pytest.raises(ValueError, match="T, 3, H, W"):
        video_tensor_to_pil_frames(video)
    artifact = MediaArtifact(MediaSpec("video", "decoded", "CTHW", fps=24), video)
    assert len(video_tensor_to_pil_frames(artifact.normalized(context="adapter", name="video_preview").data)) == 5


def test_named_video_is_exported_to_real_mp4(monkeypatch):
    import shutil

    from verl_omni.utils.tracking import wrap_val_samples_for_wandb

    preview = _generate(monkeypatch).artifacts["video_preview"]
    wrapped, temp, media = wrap_val_samples_for_wandb([("prompt", preview, 0.5)])
    try:
        assert wrapped[0][1] == "val/videos/sample_1"
        assert media and temp is not None
    finally:
        if temp is not None:
            shutil.rmtree(temp)


@pytest.mark.parametrize("layout,shape", [("CTHW", (3, 9, 8, 12)), ("TCHW", (9, 3, 8, 12))])
def test_dump_consumes_adapter_normalized_preview(layout, shape, monkeypatch, tmp_path):
    from verl_omni.trainer.diffusion import ray_diffusion_trainer as trainer

    previews = [
        MediaArtifact(MediaSpec("video", "decoded", layout, fps=24), torch.zeros(shape, dtype=torch.uint8)).normalized(
            context="adapter", name="video_preview"
        )
        for _ in range(2)
    ]
    seen = []

    def export(output, path, **kwargs):
        seen.append(tuple(output.data.shape))
        open(path, "wb").close()

    monkeypatch.setattr(trainer, "_export_video", export)
    trainer.BaseRayDiffusionTrainer._dump_generations(
        SimpleNamespace(global_steps=1),
        inputs=["a", "b"],
        outputs=torch.zeros(2, 16, 2, 2),
        gts=[None, None],
        scores=[1, 2],
        reward_extra_infos_dict={},
        dump_path=str(tmp_path),
        previews=previews,
    )
    assert seen == [(9, 3, 8, 12)] * 2


def test_dump_rejects_undeclared_extra_batch_axis(tmp_path):
    from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer

    with pytest.raises(ValueError, match="requires a rank-5 batch, got rank 6"):
        BaseRayDiffusionTrainer._dump_generations(
            SimpleNamespace(global_steps=1),
            inputs=["a"],
            outputs=torch.zeros(1, 1, 5, 3, 8, 12, dtype=torch.uint8),
            gts=[None],
            scores=[1],
            reward_extra_infos_dict={},
            dump_path=str(tmp_path),
            media_kind="video",
        )
