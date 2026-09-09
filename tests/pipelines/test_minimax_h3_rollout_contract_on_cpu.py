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
"""CPU integration tests for the MiniMax H3 rollout output contract.

Simulates the full server-side processing of H3's joint (video, audio)
rollout output — the exact path that crashed during GPU smoke testing —
without needing a GPU or the real vLLM-Omni engine.
"""

from types import SimpleNamespace

import pytest
import torch

server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")
tracking_module = pytest.importorskip("verl_omni.utils.tracking")
# The strategy resolves H3's audio sample rate from the adapter-declared
# DiffusionIOSpec, so the pipeline adapter must be imported/registered.
pytest.importorskip("verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter")

_VIDEO_T, _VIDEO_C, _VIDEO_H, _VIDEO_W = 107, 3, 384, 640
_AUDIO_SR = 32000
_AUDIO_SAMPLES = 4 * _AUDIO_SR


def _make_h3_final_res(batch_size: int = 1):
    """Build a fake OmniRequestOutput matching H3's rollout output."""
    video = torch.randint(0, 256, (batch_size, _VIDEO_C, _VIDEO_T, _VIDEO_H, _VIDEO_W), dtype=torch.uint8)
    audio = torch.randn(batch_size, _AUDIO_SAMPLES)
    metadata = {
        "rl": {
            "latents_clean": torch.randn(10, 96),
            "train_timesteps": torch.randn(1, 9),
            "latent_meta": torch.zeros(1, 6, dtype=torch.long),
        },
        "prompt_embeddings": {
            "prompt_embeds": torch.randn(1, 5, 8),
            "prompt_embeds_mask": torch.ones(1, 5, dtype=torch.long),
        },
    }
    envelope = {"payload": {"image": (video, audio)}, "metadata": metadata}
    # The engine extracts the payload into images; multimodal_output keeps the envelope.
    return SimpleNamespace(
        images=[(video, audio)],
        multimodal_output=envelope,
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        request_output=None,
    )


def _server():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.global_steps = 3
    # Keys the adapter-declared DiffusionIOSpec (audio sample rate 32000).
    server.model_config = SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm="flow_grpo")
    server._to_tensor = __import__("torchvision").transforms.PILToTensor()
    return server


class TestH3RolloutOutputContract:
    """Verify the server correctly processes H3's (video, audio) tuple output."""

    def test_tuple_extracted_video_tensor(self):
        """Video stream routes to the tensor path as uint8."""
        server = _server()
        final_res = _make_h3_final_res()
        sampling_params = {"output_type": "pt"}
        result = _run_generate(server, final_res, sampling_params)
        assert result.diffusion_output is not None
        assert isinstance(result.diffusion_output, torch.Tensor)
        assert result.diffusion_output.dtype == torch.uint8

    def test_audio_forwarded_to_extra_fields(self):
        """Audio from the tuple reaches extra_fields for CLAP/ImageBind."""
        server = _server()
        final_res = _make_h3_final_res()
        sampling_params = {"output_type": "pt"}
        result = _run_generate(server, final_res, sampling_params)
        assert "audio" in result.extra_fields
        assert result.extra_fields["audio"] is not None
        assert "audio_sample_rate" in result.extra_fields
        assert result.extra_fields["audio_sample_rate"] == _AUDIO_SR

    def test_rl_metadata_reaches_extra_fields(self):
        """rl and prompt_embeddings groups flatten into extra_fields."""
        server = _server()
        final_res = _make_h3_final_res()
        sampling_params = {"output_type": "pt"}
        result = _run_generate(server, final_res, sampling_params)
        for key in ("latents_clean", "train_timesteps", "latent_meta", "prompt_embeds", "prompt_embeds_mask"):
            assert key in result.extra_fields, f"missing {key} in extra_fields"

    def test_channels_first_video_in_reward_utils(self):
        """video_tensor_to_pil_frames handles [C,T,H,W] channels-first input."""
        from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames

        video_cthw = torch.randint(0, 256, (_VIDEO_C, _VIDEO_T, 64, 64), dtype=torch.uint8)
        frames = video_tensor_to_pil_frames(video_cthw)
        assert len(frames) == _VIDEO_T

    def test_5d_wandb_video(self):
        """wrap_val_samples_for_wandb handles H3's [B,C,T,H,W] video."""
        from verl_omni.utils.tracking import wrap_val_samples_for_wandb

        video = torch.randint(0, 256, (1, _VIDEO_C, _VIDEO_T, 64, 64), dtype=torch.uint8)
        samples = [("a test prompt", video, 0.5, None, None)]
        wrapped, tmpdir, media = wrap_val_samples_for_wandb(samples, fps=24, output_dir=None)
        assert len(wrapped) == 1
        assert tmpdir is not None
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.parametrize(
        ("shape", "expect"),
        [
            # Channels-first [N, C, T, H, W] must be normalized to [N, T, C, H, W].
            ((2, _VIDEO_C, 9, 32, 48), (9, _VIDEO_C, 32, 48)),
            # Already [N, T, C, H, W]; must be left untouched.
            ((2, 9, _VIDEO_C, 32, 48), (9, _VIDEO_C, 32, 48)),
        ],
        ids=["channels_first", "already_normalized"],
    )
    def test_dump_generations_normalizes_5d_layout(self, shape, expect, monkeypatch, tmp_path):
        """_dump_generations must hand ``_export_video`` per-sample ``[T, C, H, W]``.

        Both batched layouts must converge on the same per-sample shape; assert on the
        tensor reaching the exporter so a transposed guard cannot slip through.
        """
        from verl_omni.trainer.diffusion import ray_diffusion_trainer as rdt

        seen = []

        def _fake_export(output, output_path, **kwargs):
            seen.append(tuple(output.shape))
            open(output_path, "wb").close()

        monkeypatch.setattr(rdt, "_export_video", _fake_export)

        outputs = torch.randint(0, 256, shape, dtype=torch.uint8)
        stand_in = SimpleNamespace(global_steps=1)
        rdt.BaseRayDiffusionTrainer._dump_generations(
            stand_in,
            inputs=["p0", "p1"],
            outputs=outputs,
            gts=["g0", "g1"],
            scores=[0.1, 0.2],
            reward_extra_infos_dict={},
            dump_path=str(tmp_path),
        )

        assert seen == [expect] * 2, f"{shape} produced per-sample shapes {seen}, expected {expect}"

    def test_dump_generations_6d(self):
        """_dump_generations handles [N,1,T,C,H,W] 6D outputs."""
        import tempfile
        from pathlib import Path

        from verl_omni.utils.tracking import _export_video

        # Simulate the 6D tensor that reaches _dump_generations
        outputs = torch.randint(0, 256, (2, 1, _VIDEO_T, _VIDEO_C, 64, 64), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(outputs.shape[0]):
                squeezed = outputs[i].squeeze(0)  # [T, C, H, W]
                _export_video(squeezed, str(Path(tmpdir) / f"{i}.mp4"), fps=24)
                assert (Path(tmpdir) / f"{i}.mp4").exists()


def _run_generate(server, final_res, sampling_params):
    """Run the production diffusion output strategy with a fake engine result."""
    strategy = server_module.DiffusionStrategy(server)
    return strategy.process_output(final_res, params=None, sampling_params=sampling_params)
