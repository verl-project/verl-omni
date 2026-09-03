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
"""CPU contracts for MiniMax H3 multi-reference Ref2VA DiffusionNFT."""

from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    MAX_REF_BLOCKS,
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    TEXT_TAG,
    VIDEO_ROW_WIDTH,
    MiniMaxH3RolloutWeightSyncMixin,
    build_ref2va_layout_from_meta,
    pack_video_audio_rows,
    ref2va_reference_image_short_edge,
    serialize_ref_blocks,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline
from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

vllm_packed = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.packed_sequence")

# A mixed reference set: two images, one video+soundtrack, one silent video and
# one standalone audio clip. latent_t/h/w=1/4/4 and audio_t=3 give 20 video and
# 10 audio condition rows around 4 target video and 6 target audio rows.
_REF_BLOCKS = [
    {"kind": "image", "latent_h": 4, "latent_w": 4},
    {"kind": "image", "latent_h": 4, "latent_w": 4},
    {"kind": "video_audio", "ref_audio_t": 2, "latent_t": 2, "latent_h": 4, "latent_w": 4},
    {"kind": "video", "ref_audio_t": 0, "latent_t": 1, "latent_h": 4, "latent_w": 4},
    {"kind": "audio", "ref_audio_t": 3},
]
_IMAGE_REF_BLOCKS = [{"kind": "image", "latent_h": 4, "latent_w": 4}]
_META = [4, 6, 1, 4, 4, 3]
_NUM_COND_VIDEO = 20
_NUM_COND_AUDIO = 10
_TEXT_LEN = 7
_TEXT_DIM = 16


def test_reference_rows_are_minimally_padded_per_minibatch():
    video_rows = torch.randn(2, 8, VIDEO_ROW_WIDTH)
    video_mask = torch.tensor([[True, True, False, False, False, False, False, False], [True] * 5 + [False] * 3])
    audio_rows = torch.randn(2, 8, AUDIO_ROW_WIDTH)
    audio_mask = torch.tensor([[False] * 8, [True] * 3 + [False] * 5])
    micro_batch = TensorDict(
        {
            "condition_video_rows": video_rows,
            "condition_video_rows_mask": video_mask,
            "condition_video_row_count": torch.tensor([[2], [5]]),
            "condition_audio_rows": audio_rows,
            "condition_audio_rows_mask": audio_mask,
            "condition_audio_row_count": torch.tensor([[0], [3]]),
        },
        batch_size=2,
    )
    embeds_padding_2_no_padding(micro_batch)
    engine = object.__new__(NFTDiffusersFSDPEngine)

    engine._unpad_condition_rows(micro_batch)

    assert micro_batch["condition_video_rows"].shape == (2, 5, VIDEO_ROW_WIDTH)
    assert micro_batch["condition_audio_rows"].shape == (2, 3, AUDIO_ROW_WIDTH)
    assert micro_batch["condition_video_rows_mask"].sum(dim=1).tolist() == [2, 5]
    assert micro_batch["condition_audio_rows_mask"].sum(dim=1).tolist() == [0, 3]
    torch.testing.assert_close(micro_batch["condition_video_rows"][0, :2], video_rows[0, :2])
    torch.testing.assert_close(micro_batch["condition_audio_rows"][1, :3], audio_rows[1, :3])


def test_reference_row_minibatch_padding_rejects_count_mismatch():
    micro_batch = TensorDict(
        {
            "condition_video_rows": torch.randn(1, 4, VIDEO_ROW_WIDTH),
            "condition_video_rows_mask": torch.tensor([[True, True, False, False]]),
            "condition_video_row_count": torch.tensor([[3]]),
        },
        batch_size=1,
    )
    embeds_padding_2_no_padding(micro_batch)
    engine = object.__new__(NFTDiffusersFSDPEngine)

    with pytest.raises(ValueError, match="valid rows \\[2\\] do not match condition_video_row_count \\[3\\]"):
        engine._unpad_condition_rows(micro_batch)


def test_reference_image_short_edge_environment_override_is_restored(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    constant = "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE"
    monkeypatch.setitem(_reference_image_shape.__globals__, constant, 2048)
    monkeypatch.setenv("REF_IMAGE_SHORT_EDGE", "1024")
    image = Image.new("RGB", (640, 400))

    with ref2va_reference_image_short_edge() as short_edge:
        assert short_edge == 1024
        assert min(_reference_image_shape(image)) == 1024
    assert min(_reference_image_shape(image)) == 2048


def test_request_short_edge_overrides_environment_temporarily(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline, _reference_image_shape

    constant = "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE"
    monkeypatch.setitem(_reference_image_shape.__globals__, constant, 2048)
    monkeypatch.setenv("REF_IMAGE_SHORT_EDGE", "512")
    image = Image.new("RGB", (640, 400))
    monkeypatch.setattr(MiniMaxH3Pipeline, "forward", lambda _self, _request: min(_reference_image_shape(image)))
    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    object.__setattr__(pipeline, "_ensure_prompt_text", MagicMock())
    object.__setattr__(pipeline, "_nft_capture", None)
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={"reference_image_short_edge": 1024},
            num_outputs_per_prompt=1,
        )
    )

    assert pipeline.forward(request) == 1024
    assert min(_reference_image_shape(image)) == 2048


def test_reference_image_short_edge_serializes_concurrent_overrides(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    constant = "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE"
    monkeypatch.setitem(_reference_image_shape.__globals__, constant, 2048)
    image = Image.new("RGB", (640, 400))
    first_entered = Event()
    release_first = Event()
    second_started = Event()
    second_entered = Event()
    results = []

    def first_request():
        with ref2va_reference_image_short_edge(512):
            first_entered.set()
            release_first.wait(timeout=2)
            results.append(min(_reference_image_shape(image)))

    def second_request():
        second_started.set()
        with ref2va_reference_image_short_edge(1024):
            second_entered.set()
            results.append(min(_reference_image_shape(image)))

    first = Thread(target=first_request)
    second = Thread(target=second_request)
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert second_started.wait(timeout=2)
    try:
        assert not second_entered.wait(timeout=0.05)
    finally:
        release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert results == [512, 1024]
    assert min(_reference_image_shape(image)) == 2048


@pytest.mark.parametrize("value", ["invalid", "255", "1000", "2049"])
def test_reference_image_short_edge_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("REF_IMAGE_SHORT_EDGE", value)

    with pytest.raises(ValueError, match="REF_IMAGE_SHORT_EDGE"):
        validate_ref2va_reference_image_short_edge()


def _identity(**kwargs):
    return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def test_serialize_ref_blocks_is_fixed_size_with_a_valid_count():
    metadata, count = serialize_ref_blocks(_REF_BLOCKS)

    assert metadata.shape == (MAX_REF_BLOCKS, 5)
    assert count == len(_REF_BLOCKS)
    # Padding rows past the valid count stay zero so mismatched worker layouts stack cleanly.
    assert torch.count_nonzero(metadata[count:]) == 0


@pytest.mark.parametrize("count", [0, MAX_REF_BLOCKS + 1])
def test_serialize_ref_blocks_rejects_out_of_range_counts(count):
    with pytest.raises(ValueError, match="1-12 reference blocks"):
        serialize_ref_blocks([{"kind": "image", "latent_h": 4, "latent_w": 4}] * count)


def test_ref_block_metadata_rebuilds_the_multi_reference_layout():
    text_tags = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    metadata, count = serialize_ref_blocks(_REF_BLOCKS)

    actual = build_ref2va_layout_from_meta(
        _META,
        _TEXT_LEN,
        metadata,
        count,
        text_token_tags=text_tags,
    )
    upstream = vllm_packed.minimax_h3_packed_sequence_ref2va_blocks(
        text_len=_TEXT_LEN,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=_REF_BLOCKS,
    )
    expected_tags = upstream["token_tags"].clone()
    expected_tags[upstream["text_pos"]] = text_tags
    used = int(upstream["cu_seqlens"][1])

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = actual
    torch.testing.assert_close(position_ids, upstream["img_position_ids"][:used], rtol=0, atol=0)
    assert torch.equal(token_tags, expected_tags[:used])
    assert torch.equal(video_indices, upstream["img_pos"])
    assert torch.equal(audio_indices, upstream["audio_pos"])
    assert torch.equal(text_indices, upstream["text_pos"])
    assert (num_cond_video, num_cond_audio) == (_NUM_COND_VIDEO, _NUM_COND_AUDIO)


def test_ref2va_actor_slices_padded_reference_rows_and_returns_targets_only():
    video_rows = torch.randn(1, 4, VIDEO_ROW_WIDTH)
    audio_rows = torch.randn(1, 6, AUDIO_ROW_WIDTH)
    condition_video_rows = torch.randn(1, _NUM_COND_VIDEO, VIDEO_ROW_WIDTH)
    condition_audio_rows = torch.randn(1, _NUM_COND_AUDIO, AUDIO_ROW_WIDTH)
    metadata, count = serialize_ref_blocks(_REF_BLOCKS)
    # Simulate cross-worker padding: rows are padded past their true count.
    padded_video = torch.nn.functional.pad(condition_video_rows, (0, 0, 0, 3))
    padded_audio = torch.nn.functional.pad(condition_audio_rows, (0, 0, 0, 2))
    micro_batch = TensorDict(
        {
            "latent_meta": torch.tensor([_META], dtype=torch.long),
            "condition_video_rows": padded_video,
            "condition_audio_rows": padded_audio,
            "condition_video_row_count": torch.tensor([[_NUM_COND_VIDEO]], dtype=torch.long),
            "condition_audio_row_count": torch.tensor([[_NUM_COND_AUDIO]], dtype=torch.long),
            "ref_block_meta": metadata.unsqueeze(0),
            "ref_block_count": torch.tensor([[count]], dtype=torch.long),
            "prompt_token_tags": torch.full((1, _TEXT_LEN), TEXT_TAG, dtype=torch.long),
        },
        batch_size=1,
    )
    model_inputs, _ = MiniMaxH3DiffusionNFT.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=pack_video_audio_rows(video_rows, audio_rows),
        timesteps=torch.tensor([500.0]),
        prompt_embeds=torch.randn(1, _TEXT_LEN, _TEXT_DIM),
        prompt_embeds_mask=torch.ones(1, _TEXT_LEN, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )
    module = MagicMock(side_effect=_identity)
    module.config = None

    output = MiniMaxH3DiffusionNFT.forward(module, MagicMock(), model_inputs)

    call = module.call_args.kwargs
    assert call["hidden_states"].shape == (1, _NUM_COND_VIDEO + 4, VIDEO_ROW_WIDTH)
    assert call["audio_hidden_states"].shape == (1, _NUM_COND_AUDIO + 6, AUDIO_ROW_WIDTH)
    # Only the true (unpadded) reference rows are placed ahead of the target rows.
    torch.testing.assert_close(call["hidden_states"][0, :_NUM_COND_VIDEO], condition_video_rows[0])
    torch.testing.assert_close(call["audio_hidden_states"][0, :_NUM_COND_AUDIO], condition_audio_rows[0])
    # Targets denoise at the sampled timestep; visual references freeze at 0.999 and
    # reference audio at 1.0 (torch.unique returns them sorted ascending).
    assert call["timestep"].tolist() == pytest.approx([0.5, 0.999, 1.0])
    torch.testing.assert_close(output, -pack_video_audio_rows(video_rows, audio_rows))


def test_ref2va_actor_rejects_condition_rows_that_disagree_with_the_layout():
    video_rows = torch.randn(1, 4, VIDEO_ROW_WIDTH)
    audio_rows = torch.randn(1, 6, AUDIO_ROW_WIDTH)
    metadata, count = serialize_ref_blocks(_REF_BLOCKS)
    micro_batch = TensorDict(
        {
            "latent_meta": torch.tensor([_META], dtype=torch.long),
            "condition_video_rows": torch.randn(1, _NUM_COND_VIDEO, VIDEO_ROW_WIDTH),
            "condition_audio_rows": torch.randn(1, _NUM_COND_AUDIO, AUDIO_ROW_WIDTH),
            "condition_video_row_count": torch.tensor([[_NUM_COND_VIDEO - 1]], dtype=torch.long),
            "condition_audio_row_count": torch.tensor([[_NUM_COND_AUDIO]], dtype=torch.long),
            "ref_block_meta": metadata.unsqueeze(0),
            "ref_block_count": torch.tensor([[count]], dtype=torch.long),
            "prompt_token_tags": torch.full((1, _TEXT_LEN), TEXT_TAG, dtype=torch.long),
        },
        batch_size=1,
    )
    model_inputs, _ = MiniMaxH3DiffusionNFT.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=pack_video_audio_rows(video_rows, audio_rows),
        timesteps=torch.tensor([500.0]),
        prompt_embeds=torch.randn(1, _TEXT_LEN, _TEXT_DIM),
        prompt_embeds_mask=torch.ones(1, _TEXT_LEN, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )
    module = MagicMock(side_effect=_identity)
    module.config = None

    with pytest.raises(ValueError, match="condition video rows"):
        MiniMaxH3DiffusionNFT.forward(module, MagicMock(), model_inputs)


def test_ref2va_prompt_keeps_original_ids_with_all_reference_modalities(monkeypatch):
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module

    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(pipeline_module, "_broadcast_tensor", lambda value, **kwargs: value)
    monkeypatch.setattr(
        pipeline_module,
        "sample_reference_video_frames",
        lambda path: {
            "frames": [__import__("numpy").zeros((4, 4, 3), dtype="uint8")] * 4,
            "block_timestamps": [0.2, 1.0],
        },
        raising=False,
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

    stub = object.__new__(MiniMaxH3DiffusionNFTPipeline)
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

    # The Agent Loop text IDs survive verbatim at the tail after the reference spans.
    assert hidden[-3:, 0].tolist() == [501.0, 502.0, 503.0]
    assert tags[-3:].tolist() == [TEXT_TAG, TEXT_TAG, TEXT_TAG]
    assert (tags[:-3] == 0).any()
    assert set(received) == {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}


def test_ref2va_encode_prompt_restores_the_tokenizer_after_encoding():
    class Parent:
        def encode_prompt(self, *, task, prompt, image=None, images=None, **kwargs):
            # The upstream path sees the temporary token override that returns the
            # Agent Loop IDs for the pretokenized prompt, not the real tokenizer.
            assert self.tokenizer("[pretokenized]") == {"input_ids": [9, 9]}
            return "hidden", "tags"

    class Combined(MiniMaxH3RolloutWeightSyncMixin, Parent):
        def __init__(self):
            self._h3_prompt_ids = torch.tensor([9, 9])
            self.tokenizer = "real-tokenizer"

    combined = Combined()
    original = combined.tokenizer

    result = combined.encode_prompt(task="ref2va", prompt="[pretokenized]")

    assert result == ("hidden", "tags")
    assert combined.tokenizer is original


def test_ref2va_rollout_publishes_fixed_size_reference_replay_fields(monkeypatch):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    metadata, count = serialize_ref_blocks(_REF_BLOCKS)
    pipeline._nft_capture = {
        "video_latent": torch.randn(1, 24, 1, 4, 4),
        "audio_latent": torch.randn(2, 32, 3),
        "condition_video_rows": torch.randn(_NUM_COND_VIDEO, VIDEO_ROW_WIDTH),
        "condition_audio_rows": torch.randn(_NUM_COND_AUDIO, AUDIO_ROW_WIDTH),
        "keyframe_frame_indices": [],
        "ref_block_meta": metadata,
        "ref_block_count": count,
        "task": "ref2va",
        "text_embeddings": torch.randn(_TEXT_LEN, _TEXT_DIM),
        "text_tags": torch.full((_TEXT_LEN,), TEXT_TAG, dtype=torch.long),
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 3,
        "num_steps": 3,
        "video_shift": 12.0,
        "base_schedule": None,
    }
    monkeypatch.setattr(
        MiniMaxH3Pipeline,
        "forward",
        lambda self, request: DiffusionOutput(output=(torch.zeros(1), torch.zeros(1))),
    )
    request = MagicMock(
        prompts=[{"prompt_token_ids": [1, 2]}],
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            extra_args={MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True},
        ),
    )

    output = pipeline.forward(request)

    rl = output.output["metadata"]["rl"]
    assert rl["condition_video_rows"].shape == (1, _NUM_COND_VIDEO, VIDEO_ROW_WIDTH)
    assert rl["condition_audio_rows"].shape == (1, _NUM_COND_AUDIO, AUDIO_ROW_WIDTH)
    assert rl["condition_video_row_count"].tolist() == [[_NUM_COND_VIDEO]]
    assert rl["condition_audio_row_count"].tolist() == [[_NUM_COND_AUDIO]]
    assert rl["ref_block_meta"].shape == (1, MAX_REF_BLOCKS, 5)
    assert rl["ref_block_count"].tolist() == [[count]]


def test_ref2va_rollout_captures_multi_image_and_standalone_audio_anchors(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    video_latent = torch.randn(1, 24, 1, 4, 4)
    audio_latent = torch.randn(1, 2, 3, 16)
    monkeypatch.setattr(MiniMaxH3Pipeline, "diffuse", lambda self, **kwargs: (video_latent, audio_latent))

    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.default_video_shift = 12.0
    visual_shapes = [(1, 4, 4), (1, 4, 4), (2, 4, 4), (1, 4, 4)]
    audio_lengths = [2, 3]
    clean_visual = torch.randn(_NUM_COND_VIDEO, VIDEO_ROW_WIDTH)
    clean_audio = torch.randn(_NUM_COND_AUDIO, AUDIO_ROW_WIDTH)
    kwargs = {
        "task": "ref2va",
        "text_embeddings": torch.randn(_TEXT_LEN, _TEXT_DIM),
        "text_tags": torch.full((_TEXT_LEN,), TEXT_TAG, dtype=torch.long),
        "seed": 7,
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 3,
        "num_steps": 3,
        "video_shift": 12.0,
        "base_schedule": None,
        "visual_condition": clean_visual,
        "visual_condition_shape": None,
        "visual_condition_shapes": visual_shapes,
        "audio_condition": clean_audio,
        "audio_condition_lengths": audio_lengths,
        "ref_blocks": _REF_BLOCKS,
    }

    pipeline.diffuse(**kwargs)

    expected_video = minimax_h3_imgvid_cond_noise_aug_rows(
        clean_visual,
        condition_shapes=visual_shapes,
        target_latent_t=1,
        imgvid_cond_num_frames=len(visual_shapes),
        seed=7,
        noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    expected_audio = minimax_h3_audio_cond_noise_aug_rows(
        clean_audio,
        condition_audio_t=audio_lengths,
        seed=7,
        noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    capture = pipeline._nft_capture
    torch.testing.assert_close(capture["condition_video_rows"], expected_video)
    torch.testing.assert_close(capture["condition_audio_rows"], expected_audio)
    expected_meta, expected_count = serialize_ref_blocks(_REF_BLOCKS)
    assert torch.equal(capture["ref_block_meta"], expected_meta)
    assert capture["ref_block_count"] == expected_count


def test_ref2va_rollout_requires_reference_blocks(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    parent = MagicMock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(MiniMaxH3Pipeline, "diffuse", parent)
    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)

    with pytest.raises(ValueError, match="requires reference block metadata"):
        pipeline.diffuse(task="ref2va", ref_blocks=[], audio_condition=None)
    parent.assert_not_called()
