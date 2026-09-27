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

"""VeOmni-native MiniMax H3 FlowGRPO training helpers."""

from __future__ import annotations

import torch

__all__ = ["is_veomni_module", "predict_veomni"]


def is_veomni_module(module: torch.nn.Module) -> bool:
    """Return whether ``module`` wraps VeOmni's native fused H3 DiT."""
    config = getattr(module, "config", None)
    return getattr(config, "model_type", None) == "MiniMaxH3DiTModel"


def _native_forward_inputs(model_inputs: dict[str, torch.Tensor], use_gradient_checkpointing: bool) -> dict:
    video = model_inputs["hidden_states"]
    audio = model_inputs["audio_hidden_states"]
    prompt = model_inputs["encoder_hidden_states"]
    if video.shape[0] != 1 or audio.shape[0] != 1 or prompt.shape[0] != 1:
        raise ValueError("VeOmni MiniMax H3 FlowGRPO currently requires Actor micro-batch size 1.")

    video_indices = model_inputs["video_indices"].reshape(-1).to(video.device, dtype=torch.long)
    audio_indices = model_inputs["audio_indices"].reshape(-1).to(video.device, dtype=torch.long)
    text_indices = model_inputs["text_indices"].reshape(-1).to(video.device, dtype=torch.long)
    seq_len = int(model_inputs["position_ids"].shape[0])
    text_len = int(prompt.shape[1])
    if video.shape[1] != video_indices.numel() or audio.shape[1] != audio_indices.numel():
        raise ValueError("MiniMax H3 row tensors do not match their packed position indices.")
    if text_len != text_indices.numel():
        raise ValueError("MiniMax H3 prompt embeddings do not match their packed text indices.")
    if audio.shape[1] % 2:
        raise ValueError(f"MiniMax H3 audio rows must be divisible by two, got {audio.shape[1]}.")

    packed_video = video.new_zeros((1, seq_len, video.shape[-1]))
    packed_audio = audio.new_zeros((1, seq_len, audio.shape[-1]))
    packed_video[0, video_indices] = video[0]
    packed_audio[0, audio_indices] = audio[0]
    return {
        "x": packed_video,
        "audio_x": packed_audio,
        "img_position_ids": model_inputs["position_ids"].unsqueeze(0),
        "unique_timesteps": model_inputs["timestep"],
        "inverse_indices": model_inputs["timestep_indices"],
        "update_mask": None,
        "token_tags": model_inputs["token_tags"],
        "prompt_embeds": prompt[0],
        "img_pos_info": {"position_ids": video_indices},
        "audio_pos_info": {"position_ids": audio_indices},
        "text_pos_info": {"position_ids": text_indices},
        "img_pos_for_infer_output_info": {"position_ids": video_indices},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=video.device),
            "max_seqlen_q": seq_len,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, text_len], dtype=torch.int32, device=video.device),
            "max_seqlen_q": text_len,
        },
        "skip_mask_out_condition": True,
        "cond_rows": 0,
        "video_latent_shape": (video.shape[1], 1, 1),
        "audio_latent_shape": (2, audio.shape[1] // 2),
        "use_gradient_checkpointing": use_gradient_checkpointing,
    }


def predict_veomni(
    module: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    *,
    use_gradient_checkpointing: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run VeOmni's fused H3 wrapper and return Diffusers-compatible row velocities."""
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import pack_audio, patchify_video

    output = module(**_native_forward_inputs(model_inputs, use_gradient_checkpointing))
    predictions = getattr(output, "predictions", None)
    if not isinstance(predictions, list | tuple) or len(predictions) != 2:
        raise TypeError("VeOmni MiniMax H3 forward must return video and audio predictions.")
    video_prediction, audio_prediction = predictions
    video_rows = -patchify_video(video_prediction)
    audio_rows = -pack_audio(audio_prediction)
    expected_video = model_inputs["hidden_states"]
    expected_audio = model_inputs["audio_hidden_states"]
    if video_rows.shape != expected_video[0].shape or audio_rows.shape != expected_audio[0].shape:
        raise ValueError(
            "VeOmni MiniMax H3 output rows do not match replay rows: "
            f"video={tuple(video_rows.shape)}/{tuple(expected_video[0].shape)}, "
            f"audio={tuple(audio_rows.shape)}/{tuple(expected_audio[0].shape)}."
        )
    return video_rows.unsqueeze(0), audio_rows.unsqueeze(0)
