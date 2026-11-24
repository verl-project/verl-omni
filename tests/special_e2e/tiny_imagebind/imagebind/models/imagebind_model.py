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

"""Small checkpoint-backed model for the managed ImageBind trainer smoke."""

import os

import torch


class ModalityType:
    AUDIO = "audio"
    VISION = "vision"
    TEXT = "text"


class TinyImageBind(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("direction", torch.zeros(2))

    def forward(self, inputs):
        audio = inputs[ModalityType.AUDIO]
        video = inputs[ModalityType.VISION]
        if audio.device != self.direction.device or video.device != self.direction.device:
            raise ValueError("Expected both preprocessed modalities on the model device")
        if audio.device.type != os.environ["IMAGEBIND_SMOKE_DEVICE"]:
            raise ValueError(f"Unexpected ImageBind device {audio.device}")
        # Vary the embedding directions so cosine scoring distinguishes videos
        # even when the tiny H3 audio decoder produces silence.
        audio_x = self.direction[0] + 0.25 * audio.mean().tanh()
        video_x = 0.08 * video.mean().tanh()
        return {
            ModalityType.AUDIO: torch.stack((audio_x, self.direction[1])).unsqueeze(0),
            ModalityType.VISION: torch.stack((video_x, video.new_ones(()))).unsqueeze(0),
        }


def imagebind_huge(*, pretrained):
    if pretrained:
        raise ValueError("The trainer smoke loads its checkpoint through the native adapter")
    return TinyImageBind()
