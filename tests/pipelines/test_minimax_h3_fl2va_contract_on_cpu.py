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
"""Parity checks against the pinned vLLM-Omni FL2VA packed contract."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MiniMaxH3RolloutWeightSyncMixin,
    build_layout_from_meta,
    keyframe_indices_to_anchors,
    messages_to_text,
)

vllm_packed = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.packed_sequence")


def test_merged_presentation_helper_reexport():
    """Guard the exact import path ``encode_prompt`` relies on."""
    pipeline_module = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3")
    preprocessing = pytest.importorskip("vllm_omni.model_executor.models.minimax_h3.preprocessing")
    assert pipeline_module.minimax_h3_multi_image_presentation is preprocessing.minimax_h3_multi_image_presentation


def test_nft_rollout_registry_resolves_custom_pipeline():
    from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import (
        MiniMaxH3DiffusionNFTPipeline,
    )
    from verl_omni.pipelines.model_base import VllmOmniPipelineBase

    assert VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", "diffusion_nft") is MiniMaxH3DiffusionNFTPipeline


def test_h3_prompt_text_ignores_structured_condition_images():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": object()},
                {"type": "image", "image": object()},
                {"type": "text", "text": "A sunrise becomes a starry night."},
            ],
        }
    ]
    assert messages_to_text(messages) == "A sunrise becomes a starry night."


@pytest.mark.parametrize("frame_indices", [[0], [-1], [0, -1]])
def test_actor_layout_matches_vllm_omni_fl2va(frame_indices):
    text_tags = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    latent_t, latent_h, latent_w, audio_t = 5, 8, 12, 7
    target_video_rows = latent_t * (latent_h // 2) * (latent_w // 2)
    target_audio_rows = audio_t * 2
    meta = [target_video_rows, target_audio_rows, latent_t, latent_h, latent_w, audio_t]

    upstream = vllm_packed.minimax_h3_packed_sequence(
        text_len=text_tags.numel(),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=True,
        keyframe_frame_indices=frame_indices,
        frame_count=17,
    )
    upstream_tags = upstream["token_tags"].clone()
    upstream_tags[upstream["text_pos"]] = text_tags

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, _ = build_layout_from_meta(
        meta,
        text_tags.numel(),
        keyframe_anchors=keyframe_indices_to_anchors(frame_indices),
        text_token_tags=text_tags,
    )
    used = int(upstream["cu_seqlens"][1])
    assert position_ids.shape[0] == used
    torch.testing.assert_close(position_ids, upstream["img_position_ids"][:used], rtol=0, atol=0)
    assert torch.equal(token_tags, upstream_tags[:used])
    assert torch.equal(video_indices, upstream["img_pos"])
    assert torch.equal(audio_indices, upstream["audio_pos"])
    assert torch.equal(text_indices, upstream["text_pos"])
    assert num_cond_video == int((~upstream["update_mask"]).sum())


def test_token_id_prompt_encoder_adds_vision_prefix_without_retokenizing_user_text(monkeypatch):
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module

    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(pipeline_module, "_broadcast_tensor", lambda value, **kwargs: value)

    class Tokenizer:
        special = {"<|vision_start|>": 11, "<|image_pad|>": 12, "<|vision_end|>": 13}

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [] if not text else [20 + index for index, _ in enumerate(text)]}

        def convert_tokens_to_ids(self, token):
            return self.special[token]

    class ImageProcessor:
        merge_size = 1

        def __call__(self, images, return_tensors):
            del images, return_tensors
            return {"pixel_values": torch.ones(1, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    class Stub(MiniMaxH3RolloutWeightSyncMixin):
        pass

    stub = Stub()
    stub._h3_prompt_ids = torch.tensor([101, 102])
    stub.tokenizer = Tokenizer()
    stub.processor = SimpleNamespace(image_processor=ImageProcessor())
    stub.text_encoder_tp_size = 1
    stub.device = torch.device("cpu")
    stub._distribute_encode_inputs = lambda ids, vision_kwargs: ids
    stub._encode_text_hidden = lambda ids, vision_kwargs: ids[:, None].float()

    hidden, tags = stub.encode_prompt(task="fl2va", prompt="[pretokenized]", images=[object()])

    assert hidden[-2:, 0].tolist() == [101.0, 102.0]
    assert tags[-2:].tolist() == [1, 1]
    assert (tags[:-2] == 0).any()


def test_nft_rollout_captures_official_fl2va_condition_state(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MINIMAX_H3_IMGVID_COND_TIMESTEP
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline

    video_latent = torch.randn(1, 24, 1, 4, 4)
    audio_latent = torch.randn(1, 2, 3, 16)
    monkeypatch.setattr(MiniMaxH3Pipeline, "diffuse", lambda self, **kwargs: (video_latent, audio_latent))

    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.default_video_shift = 12.0
    clean_condition = torch.randn(8, 96)
    kwargs = {
        "task": "fl2va",
        "text_embeddings": torch.randn(5, 16),
        "text_tags": torch.tensor([0, 0, 1, 1, 1]),
        "seed": 7,
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 3,
        "num_steps": 3,
        "video_shift": 12.0,
        "base_schedule": None,
        "visual_condition": clean_condition,
        "visual_condition_shape": None,
        "visual_condition_shapes": [(1, 4, 4), (1, 4, 4)],
        "keyframe_frame_indices": [0, -1],
    }

    pipeline.diffuse(**kwargs)

    expected_condition = minimax_h3_imgvid_cond_noise_aug_rows(
        clean_condition,
        condition_shapes=kwargs["visual_condition_shapes"],
        target_latent_t=1,
        imgvid_cond_num_frames=2,
        seed=7,
        noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    capture = pipeline._nft_capture
    torch.testing.assert_close(capture["condition_video_rows"], expected_condition)
    assert capture["keyframe_frame_indices"] == [0, -1]
    assert torch.equal(capture["text_tags"], kwargs["text_tags"])
