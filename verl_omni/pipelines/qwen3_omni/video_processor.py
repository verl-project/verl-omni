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

"""Preserve the sampling clock of decoded Qwen3-Omni videos."""

import math

from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor


class Qwen3OmniVideoProcessor(Qwen3OmniMoeProcessor):
    """Use the actual sampled FPS, as the pinned vLLM-Omni processor does.

    Transformers computes video_second_per_grid from one scalar FPS even
    when decoded frames come with metadata. NExT-QA supplies one video per
    request. Reject mixed clocks rather than silently mis-time another video.
    """

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        video_kwargs = dict(kwargs.get("videos_kwargs") or {})
        metadata = video_kwargs.get("video_metadata", kwargs.get("video_metadata"))
        if videos is not None and metadata is not None:
            rates = []
            for item in metadata:
                get = item.get if isinstance(item, dict) else lambda key, item=item: getattr(item, key, None)
                indices = get("frames_indices")
                duration = get("duration")
                if duration is None:
                    source_fps, frames = get("fps"), get("total_num_frames")
                    if source_fps is not None and source_fps > 0 and frames is not None:
                        duration = frames / source_fps
                if (
                    indices is None
                    or len(indices) == 0
                    or duration is None
                    or not math.isfinite(duration)
                    or duration <= 0
                ):
                    raise ValueError("Presampled video needs frame indices and a finite, positive duration.")
                rates.append(len(indices) / float(duration))
            if not rates or any(rate != rates[0] for rate in rates):
                raise ValueError("Qwen3-Omni requires one sampled FPS per request; use one video per NExT-QA row.")
            kwargs.pop("fps", None)
            kwargs.pop("do_sample_frames", None)
            video_kwargs.update(fps=rates[0], do_sample_frames=False)
            kwargs["videos_kwargs"] = video_kwargs
        return super().__call__(text=text, images=images, videos=videos, audio=audio, **kwargs)
