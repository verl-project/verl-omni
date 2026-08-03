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

"""Experiment-tracking helpers layered on verl.utils.tracking."""

import os
import tempfile

from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames


def wrap_val_samples_for_wandb(samples, fps=24):
    """Wrap ``(input, output, score)`` tuples for ``wandb`` validation logging.

    Video outputs ``[T, C, H, W]`` are encoded to a temp mp4 and passed to
    ``wandb.Video`` by path (no moviepy); other outputs become ``wandb.Image``.
    Returns the wrapped samples and the temp dir (``None`` if no video), which the
    caller removes after logging.
    """
    import wandb

    video_tmp_dir = None
    wrapped = []
    for inp, out, score in samples:
        if hasattr(out, "ndim") and out.ndim == 4:
            from diffusers.utils import export_to_video

            if video_tmp_dir is None:
                video_tmp_dir = tempfile.mkdtemp(prefix="val_video_")
            video_path = os.path.join(video_tmp_dir, f"{len(wrapped)}.mp4")
            export_to_video(video_tensor_to_pil_frames(out), video_path, fps=fps)
            media = wandb.Video(video_path, format="mp4")
        else:
            media = wandb.Image(out.float(), file_type="jpg")
        wrapped.append((inp, media, score))
    return wrapped, video_tmp_dir
