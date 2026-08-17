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

"""Shared LTX-2.3 FlowGRPO constants and numerical helpers."""

import torch

LTX2_LORA_TARGET_MODULES = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
]


def normalize_ltx_output_type(output_type: str | None) -> str | None:
    """Translate the generic image default to an LTX video tensor output."""
    return "pt" if output_type == "image" else output_type


def calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int,
    max_image_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    """Calculate the flow scheduler's dynamic timestep shift."""
    slope = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    intercept = base_shift - slope * base_image_seq_len
    return image_seq_len * slope + intercept


def apply_x0_cfg(
    sample: torch.Tensor,
    positive_velocity: torch.Tensor,
    negative_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply LTX-2.3 classifier-free guidance in clean-sample space."""
    positive_x0 = sample - sigma * positive_velocity
    negative_x0 = sample - sigma * negative_velocity
    guided_x0 = positive_x0 + (guidance_scale - 1.0) * (positive_x0 - negative_x0)
    return (sample - guided_x0) / sigma


# VeOmni checkpoint key → diffusers key.
# Order matters — longer/more-specific prefixes must come first to avoid
# partial string matches (e.g. "prompt_adaln_single." before "adaln_single.").
VEOMNI_TO_DIFFUSERS_KEY_MAP: list[tuple[str, str]] = [
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
    ("audio_adaln_single.", "audio_time_embed."),
    ("av_ca_video_scale_shift_adaln_single.", "av_cross_attn_video_scale_shift."),
    ("av_ca_audio_scale_shift_adaln_single.", "av_cross_attn_audio_scale_shift."),
    ("av_ca_a2v_gate_adaln_single.", "av_cross_attn_video_a2v_gate."),
    ("av_ca_v2a_gate_adaln_single.", "av_cross_attn_audio_v2a_gate."),
    ("adaln_single.", "time_embed."),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("patchify_proj", "proj_in"),
]


def remap_veomni_to_diffusers_key(name: str) -> str:
    """Map a VeOmni-style checkpoint key back to diffusers naming.

    VeOmni's ``LTX2VideoTransformer3DModel`` uses different parameter names
    than diffusers' ``LTXVideoTransformerModel`` (e.g. ``adaln_single`` →
    ``time_embed``).  This function converts VeOmni keys so that the
    exported state-dict matches diffusers conventions and can be loaded
    directly by the vLLM-Omni rollout pipeline.
    """
    for veomni_prefix, diffusers_prefix in VEOMNI_TO_DIFFUSERS_KEY_MAP:
        if veomni_prefix in name:
            return name.replace(veomni_prefix, diffusers_prefix)
    return name
