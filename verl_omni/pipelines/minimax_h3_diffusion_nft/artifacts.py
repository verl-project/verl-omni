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
"""The pinned H3 decoder and packed-token layouts, shared by NFT and FlowGRPO."""

from verl_omni.pipelines.diffusion_rollout_output import with_batched_media_artifacts
from verl_omni.pipelines.rollout_media import MediaSpec


def with_h3_artifacts(base, *, video, audio, video_latent, audio_latent, sampling, context):
    """Transport NTHWC/NCT decoded media and native NCTHW/CLT latents separately."""
    extra = sampling.extra_args or {}
    output_type = extra.get("output_type", sampling.output_type)
    specs = {
        "video_preview": MediaSpec(
            "video",
            "decoded",
            "THWC",
            fps=24 if sampling.frame_rate is None else sampling.frame_rate,
        ),
        "audio": MediaSpec("audio", "decoded", "CT", sample_rate=32000),
        "video_latent": MediaSpec("video", "latent", "CTHW"),
        "audio_latent": MediaSpec("audio", "latent", "CLT"),
    }
    return with_batched_media_artifacts(
        base,
        data={
            "video_preview": video,
            "audio": audio,
            "video_latent": video_latent,
            "audio_latent": audio_latent.unsqueeze(0),
        },
        specs=specs,
        primary="video_latent" if output_type == "latent" else "video_preview",
        preview="video_preview",
        audio="audio",
        context=context,
        requested=extra.get("requested_outputs"),
    )
