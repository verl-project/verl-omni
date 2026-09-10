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

from verl_omni.pipelines.diffusion_rollout_output import rollout_output
from verl_omni.pipelines.minimax_h3_diffusion_nft.artifacts import with_h3_artifacts
from verl_omni.pipelines.rollout_artifacts import MediaArtifact
from verl_omni.pipelines.rollout_media import MediaSpec
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


def _make_h3_final_res():
    base = rollout_output(
        media=None,
        rl={
            "latents_clean": torch.randn(1, 10, 96),
            "train_timesteps": torch.randn(1, 9),
            "latent_meta": torch.zeros(1, 6, dtype=torch.long),
        },
        prompt_embeddings={
            "prompt_embeds": torch.randn(1, 5, 8),
            "prompt_embeds_mask": torch.ones(1, 5, dtype=torch.long),
        },
    )
    result = with_h3_artifacts(
        base,
        video=torch.randint(256, (1, 5, 8, 12, 3), dtype=torch.uint8),
        audio=torch.randn(1, 2, 320),
        video_latent=torch.randn(1, 24, 2, 2, 2),
        audio_latent=torch.randn(2, 32, 8),
        sampling=SimpleNamespace(frame_rate=24, output_type="pt", extra_args={}),
        context="pipeline=H3, request_id=h3-test",
    )
    return SimpleNamespace(
        images=result.output["payload"]["video"],
        multimodal_output={"metadata": result.output["metadata"]},
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        request_output=None,
        request_id="h3-test",
    )


def _generate():
    server = SimpleNamespace(
        global_steps=3, model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm="flow_grpo")
    )
    return DiffusionStrategy(server).process_output(_make_h3_final_res(), None, {"output_type": "pt"})


def test_named_video_has_canonical_shape_and_preserves_native_latents():
    result = _generate()
    assert result.diffusion_output.shape == (5, 3, 8, 12)
    assert result.diffusion_output.dtype == torch.uint8
    assert result.artifacts["video_latent"].data.shape == (24, 2, 2, 2)
    assert result.artifacts["audio_latent"].spec.layout == "CLT"


def test_named_audio_is_not_unbatched_a_second_time():
    result = _generate()
    assert result.extra_fields["audio"].shape == (2, 320)
    assert result.extra_fields["audio_sample_rate"] == 32000
    assert result.extra_fields["media_kind"] == "video"


def test_training_fields_remain_separate_from_media():
    result = _generate()
    assert result.extra_fields["latents_clean"].shape == (10, 96)
    for key in ("train_timesteps", "latent_meta", "prompt_embeds", "prompt_embeds_mask"):
        assert key in result.extra_fields
    assert "latents_clean" not in result.artifacts


def test_rewards_reject_undeclared_channels_first_input():
    from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames

    video = torch.zeros(3, 5, 8, 12, dtype=torch.uint8)
    with pytest.raises(ValueError, match="T, 3, H, W"):
        video_tensor_to_pil_frames(video)
    artifact = MediaArtifact(MediaSpec("video", "decoded", "CTHW", fps=24), video)
    assert len(video_tensor_to_pil_frames(artifact.normalized(context="adapter", name="video_preview").data)) == 5


def test_named_video_is_exported_to_real_mp4():
    import shutil

    from verl_omni.utils.tracking import wrap_val_samples_for_wandb

    preview = _generate().artifacts["video_preview"]
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

    with pytest.raises(ValueError, match="canonical batched"):
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
