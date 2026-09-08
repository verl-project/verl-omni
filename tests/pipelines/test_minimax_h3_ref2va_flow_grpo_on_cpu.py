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
"""CPU contracts for full-reference MiniMax H3 Ref2VA FlowGRPO."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    build_ref2va_layout_from_meta,
    ref2va_reference_image_short_edge,
    serialize_ref_blocks,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    H3_AUDIO_WIDTH,
    H3_VIDEO_WIDTH,
    h3_sigma_schedules,
    split_joint_latents,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import MiniMaxH3PipelineWithLogProb

_TEXT_LEN = 7
_TEXT_DIM = 16
_REF_BLOCKS = [
    {"kind": "image", "latent_h": 4, "latent_w": 4},
    {"kind": "image", "latent_h": 4, "latent_w": 4},
    {"kind": "video_audio", "ref_audio_t": 2, "latent_t": 2, "latent_h": 4, "latent_w": 4},
    {"kind": "video", "ref_audio_t": 0, "latent_t": 1, "latent_h": 4, "latent_w": 4},
    {"kind": "audio", "ref_audio_t": 3},
]


def test_reference_image_short_edge_environment_override_is_restored(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    constant = "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE"
    monkeypatch.setitem(_reference_image_shape.__globals__, constant, 2048)
    monkeypatch.setenv("REF_IMAGE_SHORT_EDGE", "1024")
    image = Image.new("RGB", (640, 400))

    with ref2va_reference_image_short_edge() as short_edge:
        assert short_edge == 1024
        assert _reference_image_shape(image) == (1632, 1024)
    assert min(_reference_image_shape(image)) == 2048


def test_request_short_edge_overrides_environment_temporarily(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    constant = "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE"
    monkeypatch.setitem(_reference_image_shape.__globals__, constant, 2048)
    monkeypatch.setenv("REF_IMAGE_SHORT_EDGE", "512")
    image = Image.new("RGB", (640, 400))
    seen = []

    def fake_forward(_self, _request):
        seen.append(min(_reference_image_shape(image)))
        raise RuntimeError("stop after observing the request-local override")

    monkeypatch.setattr(MiniMaxH3Pipeline, "forward", fake_forward)
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    object.__setattr__(pipeline, "_reference_image_short_edge", 512)
    object.__setattr__(pipeline, "_ensure_prompt_text", MagicMock())
    request = SimpleNamespace(
        requests=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_args={"reference_image_short_edge": 1024},
                    num_outputs_per_prompt=1,
                    max_sequence_length=1024,
                )
            )
        ]
    )

    with pytest.raises(RuntimeError, match="request-local override"):
        pipeline.forward(request)

    assert seen == [1024]
    assert min(_reference_image_shape(image)) == 2048


@pytest.mark.parametrize("value", ["invalid", "255", "1000", "2049"])
def test_reference_image_short_edge_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("REF_IMAGE_SHORT_EDGE", value)

    with pytest.raises(ValueError, match="REF_IMAGE_SHORT_EDGE"):
        validate_ref2va_reference_image_short_edge()


class _RefBranch:
    def __init__(self, *, packed, text_embeddings, token_tags, device):
        del text_embeddings
        self.img_pos = packed["img_pos"]
        self.audio_pos = packed["audio_pos"]
        self.update_mask_dev = packed["update_mask"].to(device)
        self.audio_update_mask_dev = packed["audio_update_mask"].to(device)
        self.static_kwargs = {"token_tags": token_tags.to(device)}

    def forward_kwargs(self, *, video_rows, audio_rows, **kwargs):
        del kwargs
        return {"hidden_states": video_rows, "audio_hidden_states": audio_rows}


class _Progress:
    def update(self):
        return None


def _run_ref_rollout(monkeypatch) -> tuple[MiniMaxH3PipelineWithLogProb, dict[str, torch.Tensor]]:
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_sde_contiguous = True
    pipeline._flow_grpo_seed = 42
    pipeline._h3_max_text_len = 8
    pipeline._initial_noise = MagicMock(
        return_value=(
            torch.zeros(4, H3_VIDEO_WIDTH),
            torch.zeros(6, H3_AUDIO_WIDTH),
        )
    )
    pipeline.record_denoise_step = MagicMock()
    pipeline.progress_bar = lambda total: nullcontext(_Progress())
    pipeline._resident_dit_layers_on_device = lambda enabled: nullcontext()

    calls: list[tuple[str, int]] = []

    video_inputs = []
    audio_inputs = []

    def transformer(**model_inputs):
        video_inputs.append(model_inputs["hidden_states"].clone())
        audio_inputs.append(model_inputs["audio_hidden_states"].clone())
        return (
            torch.zeros_like(model_inputs["hidden_states"]),
            torch.zeros_like(model_inputs["audio_hidden_states"]),
        )

    pipeline.transformer = transformer
    pipeline._transformer_for_task = MagicMock(return_value=transformer)

    video_anchor = torch.arange(20 * H3_VIDEO_WIDTH, dtype=torch.float32).reshape(20, H3_VIDEO_WIDTH)
    audio_anchor = torch.arange(10 * H3_AUDIO_WIDTH, dtype=torch.float32).reshape(10, H3_AUDIO_WIDTH)
    module = "verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter"
    monkeypatch.setattr(f"{module}.MiniMaxH3DenoiseBranch", _RefBranch)
    monkeypatch.setattr(f"{module}.minimax_h3_imgvid_cond_noise_aug_rows", lambda *args, **kwargs: video_anchor)
    monkeypatch.setattr(f"{module}.minimax_h3_audio_cond_noise_aug_rows", lambda *args, **kwargs: audio_anchor)
    monkeypatch.setattr(
        f"{module}.h3_sigma_schedules",
        lambda *args: ([1.0, 0.7, 0.3, 0.0], [1.0, 0.6, 0.2, 0.0]),
    )
    monkeypatch.setattr(f"{module}.configure_flow_scheduler", lambda *args: None)

    def transition(_scheduler, sample, _velocity, step, **kwargs):
        modality = "video" if sample.shape[-1] == H3_VIDEO_WIDTH else "audio"
        calls.append((modality, int(sample.shape[1])))
        log_prob = torch.tensor([0.2 if modality == "video" else 0.6]) if kwargs["return_log_prob"] else None
        return sample + float(step + 1), log_prob, sample + 0.5, torch.tensor(0.1), torch.tensor(0.2)

    monkeypatch.setattr(f"{module}.sample_h3_transition", transition)
    monkeypatch.setattr(f"{module}.minimax_h3_unpatchify_video_tokens", lambda rows, **kwargs: rows)
    monkeypatch.setattr(f"{module}.minimax_h3_unpack_audio_tokens", lambda rows, **kwargs: rows)

    video, audio = pipeline.diffuse(
        task="ref2va",
        text_embeddings=torch.randn(_TEXT_LEN, _TEXT_DIM),
        text_tags=torch.ones(_TEXT_LEN, dtype=torch.long),
        seed=7,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        num_frames=107,
        num_steps=4,
        video_shift=12.0,
        audio_shift=3.0,
        visual_condition=torch.randn(20, H3_VIDEO_WIDTH),
        visual_condition_shape=None,
        audio_condition=torch.randn(10, H3_AUDIO_WIDTH),
        ref_audio_t=None,
        ref_blocks=_REF_BLOCKS,
        visual_condition_shapes=[(1, 4, 4), (1, 4, 4), (2, 4, 4), (1, 4, 4)],
        audio_condition_lengths=[2, 3],
        keyframe_frame_indices=None,
        base_schedule=None,
    )

    assert calls == [("video", 4), ("audio", 6)] * 3
    assert video.shape == (4, H3_VIDEO_WIDTH)
    assert audio.shape == (6, H3_AUDIO_WIDTH)
    return pipeline, {
        "video_anchor": video_anchor,
        "audio_anchor": audio_anchor,
        "video_inputs": video_inputs,
        "audio_inputs": audio_inputs,
    }


def test_ref2va_rollout_keeps_all_reference_rows_fixed(monkeypatch):
    pipeline, anchors = _run_ref_rollout(monkeypatch)
    trajectory = pipeline._flow_grpo_trajectory

    assert trajectory["condition_video_row_count"].item() == 20
    assert trajectory["condition_audio_row_count"].item() == 10
    assert trajectory["ref_block_count"].item() == 5
    assert trajectory["ref_block_meta"].shape == (1, 12, 5)
    assert trajectory["all_latents"].shape == (
        1,
        3,
        1,
        4 * H3_VIDEO_WIDTH + 6 * H3_AUDIO_WIDTH,
    )
    assert trajectory["all_next_latents"].shape == trajectory["all_latents"].shape
    torch.testing.assert_close(trajectory["condition_video_rows"][0], anchors["video_anchor"])
    torch.testing.assert_close(trajectory["condition_audio_rows"][0], anchors["audio_anchor"])
    for video_input, audio_input in zip(anchors["video_inputs"], anchors["audio_inputs"], strict=True):
        torch.testing.assert_close(video_input[:20], anchors["video_anchor"])
        torch.testing.assert_close(audio_input[:10], anchors["audio_anchor"])
    torch.testing.assert_close(trajectory["all_log_probs"], torch.full((1, 3), 0.4))


def test_ref2va_actor_replays_full_layout_and_scores_only_targets(monkeypatch):
    pipeline, _ = _run_ref_rollout(monkeypatch)
    trajectory = pipeline._flow_grpo_trajectory
    trajectory["condition_video_rows"] = torch.nn.functional.pad(trajectory["condition_video_rows"], (0, 0, 0, 3))
    trajectory["condition_audio_rows"] = torch.nn.functional.pad(trajectory["condition_audio_rows"], (0, 0, 0, 2))
    micro_batch = TensorDict(trajectory, batch_size=[1])

    model_inputs, negative_inputs = MiniMaxH3FlowGRPO.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=trajectory["all_latents"],
        timesteps=trajectory["all_timesteps"],
        prompt_embeds=trajectory["prompt_embeds"],
        prompt_embeds_mask=trajectory["prompt_embeds_mask"],
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )

    assert negative_inputs is None
    assert model_inputs["hidden_states"].shape == (1, 24, H3_VIDEO_WIDTH)
    assert model_inputs["audio_hidden_states"].shape == (1, 16, H3_AUDIO_WIDTH)
    assert model_inputs["timestep"].tolist() == pytest.approx(
        [0.0, MINIMAX_H3_IMGVID_COND_TIMESTEP, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP]
    )

    module = MagicMock(
        return_value=(
            torch.zeros_like(model_inputs["hidden_states"]),
            torch.zeros_like(model_inputs["audio_hidden_states"]),
        )
    )
    calls: list[tuple[int, int]] = []

    def transition(_scheduler, sample, _velocity, _step, **kwargs):
        calls.append((sample.shape[1], sample.shape[2]))
        log_prob = torch.tensor([0.2 if sample.shape[-1] == H3_VIDEO_WIDTH else 0.6])
        return sample, log_prob, sample + 0.5, torch.tensor(0.1), torch.tensor(0.2)

    monkeypatch.setattr(
        "verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter.sample_h3_transition",
        transition,
    )
    model_config = SimpleNamespace(algo=SimpleNamespace(noise_level=0.8, sde_type="cps"))
    log_prob, mean, _, _ = MiniMaxH3FlowGRPO.forward_and_sample_previous_step(
        module=module,
        scheduler=(MagicMock(), MagicMock()),
        model_config=model_config,
        model_inputs=model_inputs,
        negative_model_inputs=None,
        scheduler_inputs=micro_batch,
        step=0,
    )

    assert calls == [(4, H3_VIDEO_WIDTH), (6, H3_AUDIO_WIDTH)]
    torch.testing.assert_close(log_prob, torch.tensor([0.4]))
    assert mean.shape == trajectory["all_latents"][:, 0].shape
    mean_video, mean_audio = split_joint_latents(mean, 4, 6)
    assert mean_video.shape == (1, 4, H3_VIDEO_WIDTH)
    assert mean_audio.shape == (1, 6, H3_AUDIO_WIDTH)


def test_ref2va_actor_rejects_invalid_block_count_before_transformer(monkeypatch):
    pipeline, _ = _run_ref_rollout(monkeypatch)
    trajectory = pipeline._flow_grpo_trajectory
    trajectory["ref_block_count"] = torch.tensor([[13]])
    module = MagicMock()

    with pytest.raises(ValueError, match="block count 13"):
        MiniMaxH3FlowGRPO.prepare_model_inputs(
            module=module,
            model_config=MagicMock(),
            latents=trajectory["all_latents"],
            timesteps=trajectory["all_timesteps"],
            prompt_embeds=trajectory["prompt_embeds"],
            prompt_embeds_mask=trajectory["prompt_embeds_mask"],
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=TensorDict(trajectory, batch_size=[1]),
            step=0,
        )

    module.assert_not_called()


def test_ref2va_prompt_keeps_original_ids_with_all_reference_modalities(monkeypatch):
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module

    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(pipeline_module, "_broadcast_tensor", lambda value, **kwargs: value)
    monkeypatch.setattr(
        pipeline_module,
        "sample_reference_video_frames",
        lambda path: {
            "frames": [np.zeros((4, 4, 3), dtype=np.uint8)] * 4,
            "block_timestamps": [0.2, 1.0],
        },
    )

    class Tokenizer:
        special = {"<|vision_start|>": 11, "<|image_pad|>": 12, "<|vision_end|>": 13, "<|video_pad|>": 14}

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [20 + index for index, _ in enumerate(text)]}

        def convert_tokens_to_ids(self, token):
            return self.special[token]

    class ImageProcessor:
        merge_size = 1

        def __call__(self, images, return_tensors):
            del return_tensors
            return {
                "pixel_values": torch.ones(len(images), 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            }

    class VideoProcessor:
        def __call__(self, videos, do_sample_frames, return_tensors):
            del do_sample_frames, return_tensors
            return {
                "pixel_values_videos": torch.ones(len(videos), 3),
                "video_grid_thw": torch.tensor([[2, 2, 2]] * len(videos)),
            }

    stub = object.__new__(MiniMaxH3PipelineWithLogProb)
    torch.nn.Module.__init__(stub)
    stub._h3_prompt_ids = torch.tensor([501, 502, 503])
    stub.tokenizer = Tokenizer()
    stub.processor = SimpleNamespace(image_processor=ImageProcessor(), video_processor=VideoProcessor())
    stub.text_encoder_tp_size = 1
    stub.device = torch.device("cpu")
    received = {}
    stub._distribute_encode_inputs = lambda ids, vision_kwargs: received.update(vision_kwargs) or ids
    stub._encode_text_hidden = lambda ids, vision_kwargs: ids[:, None].float()

    hidden, tags = stub.encode_prompt(
        task="ref2va",
        prompt="[pretokenized]",
        images=[object(), object()],
        prepared_videos=[{"prepared_path": "/tmp/reference.mp4", "input_has_audio": True}],
        condition_labels=[("image", 1), ("image", 2), ("audio", 1), ("video", 1), ("audio", 2)],
    )

    assert hidden[-3:, 0].tolist() == [501.0, 502.0, 503.0]
    assert tags[-3:].tolist() == [1, 1, 1]
    assert (tags[:-3] == 0).any()
    assert set(received) == {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}


def test_ref2va_metadata_round_trips_through_the_upstream_layout():
    metadata, count = serialize_ref_blocks(_REF_BLOCKS)
    actual = build_ref2va_layout_from_meta(
        [4, 6, 1, 4, 4, 3],
        _TEXT_LEN,
        metadata,
        count,
        text_token_tags=torch.ones(_TEXT_LEN, dtype=torch.long),
    )
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=_TEXT_LEN,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=_REF_BLOCKS,
    )
    used = int(packed["cu_seqlens"][1])

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = actual
    torch.testing.assert_close(position_ids, packed["img_position_ids"][:used], rtol=0, atol=0)
    assert torch.equal(token_tags, packed["token_tags"][:used])
    assert torch.equal(video_indices, packed["img_pos"])
    assert torch.equal(audio_indices, packed["audio_pos"])
    assert torch.equal(text_indices, packed["text_pos"])
    assert (num_cond_video, num_cond_audio) == (20, 10)
    assert metadata.shape == (12, 5)
    assert torch.count_nonzero(metadata[count:]) == 0


def test_ref2va_eta_zero_matches_the_official_denoiser(monkeypatch):
    device = torch.device("cpu")
    seed = 7
    text_embeddings = torch.randn(_TEXT_LEN, _TEXT_DIM)
    text_tags = torch.ones(_TEXT_LEN, dtype=torch.long)
    target_video = torch.randn(4, H3_VIDEO_WIDTH)
    target_audio = torch.randn(6, H3_AUDIO_WIDTH)
    clean_video = torch.randn(20, H3_VIDEO_WIDTH)
    clean_audio = torch.randn(10, H3_AUDIO_WIDTH)
    visual_shapes = [(1, 4, 4), (1, 4, 4), (2, 4, 4), (1, 4, 4)]
    audio_lengths = [2, 3]
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=_TEXT_LEN,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=_REF_BLOCKS,
    )
    tags = packed["token_tags"].clone()
    tags[packed["text_pos"]] = text_tags
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text_embeddings,
        token_tags=tags,
        device=device,
    )
    visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
        clean_video,
        condition_shapes=visual_shapes,
        target_latent_t=1,
        imgvid_cond_num_frames=4,
        seed=seed,
        noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
        clean_audio,
        condition_audio_t=audio_lengths,
        seed=seed,
        noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    initial_video = torch.zeros(24, H3_VIDEO_WIDTH)
    initial_video[branch.update_mask] = target_video
    initial_audio = torch.zeros(16, H3_AUDIO_WIDTH)
    initial_audio[branch.audio_update_mask] = target_audio
    video_sigmas, audio_sigmas = h3_sigma_schedules(4)

    def transformer(**kwargs):
        video = kwargs["x"][0, kwargs["img_pos_info"]["position_ids"]]
        audio = kwargs["audio_x"][0, kwargs["audio_pos_info"]["position_ids"]]
        return torch.tanh(video) * 0.05, torch.tanh(audio) * 0.03

    expected_video, expected_audio = minimax_h3_denoise_loop(
        model=transformer,
        positive=branch,
        initial_video_rows=initial_video,
        initial_audio_rows=initial_audio,
        keyframe_cond_rows=visual_anchor,
        audio_ref_rows=audio_anchor,
        sigmas_video=video_sigmas,
        sigmas_audio=audio_sigmas,
        device=device,
    )

    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    pipeline.device = device
    pipeline.transformer = transformer
    pipeline._transformer_for_task = MagicMock(return_value=transformer)
    pipeline._resident_dit_layers_on_device = lambda enabled: nullcontext()
    pipeline._initial_noise = MagicMock(return_value=(target_video.clone(), target_audio.clone()))
    pipeline.record_denoise_step = MagicMock()
    pipeline.progress_bar = lambda total: nullcontext(_Progress())
    pipeline._flow_grpo_noise_level = 0.0
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_sde_contiguous = True
    pipeline._flow_grpo_seed = 42
    pipeline._h3_max_text_len = 8
    module = "verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter"
    monkeypatch.setattr(f"{module}.minimax_h3_unpatchify_video_tokens", lambda rows, **kwargs: rows)
    monkeypatch.setattr(f"{module}.minimax_h3_unpack_audio_tokens", lambda rows, **kwargs: rows)

    actual_video, actual_audio = pipeline.diffuse(
        task="ref2va",
        text_embeddings=text_embeddings,
        text_tags=text_tags,
        seed=seed,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        num_frames=107,
        num_steps=4,
        video_shift=12.0,
        audio_shift=3.0,
        visual_condition=clean_video,
        visual_condition_shape=None,
        audio_condition=clean_audio,
        ref_audio_t=None,
        ref_blocks=_REF_BLOCKS,
        visual_condition_shapes=visual_shapes,
        audio_condition_lengths=audio_lengths,
        keyframe_frame_indices=None,
        base_schedule=None,
    )

    torch.testing.assert_close(actual_video, expected_video[branch.update_mask], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_audio, expected_audio[branch.audio_update_mask], rtol=0, atol=1e-6)


def test_ref2va_layout_masks_cover_every_reference_block():
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=_TEXT_LEN,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=_REF_BLOCKS,
    )

    assert packed["img_pos"].numel() == 24
    assert packed["audio_pos"].numel() == 16
    assert (~packed["update_mask"]).sum().item() == 20
    assert (~packed["audio_update_mask"]).sum().item() == 10
    assert packed["update_mask"].sum().item() == 4
    assert packed["audio_update_mask"].sum().item() == 6
    video_soundtrack = packed["audio_pos"][:4]
    video_visual = packed["img_pos"][8:16]
    standalone_audio = packed["audio_pos"][4:10]
    assert int(video_soundtrack[-1]) < int(video_visual[0])
    assert int(video_visual[-1]) < int(standalone_audio[0])
