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

"""Replay BAGEL's frozen image inputs with block-causal attention."""

import torch


def context_layout(length, image_start, vae_length, vit_length, device, target_length=0):
    """Return RoPE [L], generation routing [L], and visible attention pairs [L, L]."""
    total = length + target_length
    positions = torch.arange(total, device=device)
    generation = torch.zeros(total, dtype=torch.bool, device=device)
    visible = torch.ones(total, total, dtype=torch.bool, device=device).tril()
    cursor = image_start
    for block_length in (vae_length, vit_length):
        if block_length:
            if block_length < 2 or cursor + block_length > length:
                raise ValueError("Image replay block is outside its context.")
            positions[cursor : cursor + block_length] = positions[cursor].clone()
            positions[cursor + block_length :] -= block_length - 1
            visible[cursor : cursor + block_length, cursor : cursor + block_length] = True
            cursor += block_length
    if vae_length:
        generation[image_start + 1 : image_start + vae_length - 1] = True
    if target_length:
        positions[length:] = positions[length].clone()
        generation[length + 1 : total - 1] = True
        visible[length:, length:] = True
    return positions, generation, visible
