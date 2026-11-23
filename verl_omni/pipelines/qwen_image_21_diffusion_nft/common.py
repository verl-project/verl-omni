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
"""Qwen-Image 2.1 的训练与采样条件合约。"""

import torch


def align_prompt_tags(tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对齐 engine 去除全局 padding 后的 embedding 长度，不丢弃有效图像 slot。"""
    if tags.ndim != 2 or mask.ndim != 2 or tags.shape[0] != mask.shape[0] or tags.shape[1] < mask.shape[1]:
        raise ValueError("Prompt tags cannot align with the unpadded embedding mask.")
    if bool(tags[:, mask.shape[1] :].bool().any()):
        raise ValueError("Image slots were removed while unpadding prompt embeddings.")
    return tags[:, : mask.shape[1]]


def image_shapes(condition_shapes: torch.Tensor, target_shapes: torch.Tensor) -> list[list[tuple[int, int, int]]]:
    """将 (B, 4, 3) 条件形状和 (B, 3) 目标形状转换成上游布局。"""
    if condition_shapes.ndim != 3 or condition_shapes.shape[1:] != (4, 3):
        raise ValueError("condition_image_shapes must have shape (B, 4, 3).")
    if target_shapes.shape != (condition_shapes.shape[0], 3):
        raise ValueError("target_image_shape must have shape (B, 3).")
    result = []
    for conditions, target in zip(condition_shapes.tolist(), target_shapes.tolist(), strict=True):
        shapes = []
        padding = False
        for shape in conditions:
            if shape == [0, 0, 0]:
                padding = True
            elif padding or shape[0] != 1 or any(size <= 0 or size % 2 for size in shape[1:]):
                raise ValueError("Condition shapes require contiguous image slots with positive even spatial sizes.")
            else:
                shapes.append(tuple(shape))
        if target[0] != 1 or any(size <= 0 or size % 2 for size in target[1:]):
            raise ValueError("Target shape requires one frame and positive even spatial sizes.")
        result.append([*shapes, tuple(target)])
    return result


def joint_image_mask(tags: torch.Tensor, text_mask: torch.Tensor, shapes: list, target_tokens: int) -> torch.Tensor:
    """合并 (B, L) 的条件图 slot 标记与目标 slot，拒绝丢失或错位的条件。"""
    if tags.shape != text_mask.shape or tags.ndim != 2 or tags.shape[0] != len(shapes):
        raise ValueError("prompt_token_tags must align with the prompt embedding mask.")
    if bool((tags.bool() & ~text_mask.bool()).any()):
        raise ValueError("Image slots must not occur in prompt padding.")
    expected = tags.new_tensor([sum(f * h * w for f, h, w in row[:-1]) // 4 for row in shapes], dtype=torch.long)
    if not torch.equal(tags.bool().sum(-1), expected):
        raise ValueError("Prompt image slots do not match the condition latent shapes.")
    if target_tokens <= 0 or target_tokens % 4:
        raise ValueError("Target latent length must be a positive multiple of four.")
    return torch.cat([tags.bool(), tags.new_ones((tags.shape[0], target_tokens // 4), dtype=torch.bool)], dim=1)


def condition_target_latents(target: torch.Tensor, conditions: torch.Tensor, shapes: list) -> torch.Tensor:
    """组装 (B, condition_tokens + target_tokens, C)，不改变条件值。"""
    if target.ndim != 3 or conditions.ndim != 3:
        raise ValueError("Target and condition latents must have shape (B, tokens, channels).")
    if target.shape[0] != conditions.shape[0] or target.shape[2] != conditions.shape[2]:
        raise ValueError("Target and condition latents must share batch size and channels.")
    if len(shapes) != target.shape[0]:
        raise ValueError("Spatial metadata must contain one entry per sample.")
    for row in shapes:
        if sum(f * h * w for f, h, w in row[:-1]) != conditions.shape[1]:
            raise ValueError("Condition latent length does not match condition_image_shapes.")
        if row[-1][0] * row[-1][1] * row[-1][2] != target.shape[1]:
            raise ValueError("Target latent length does not match target_image_shape.")
    return torch.cat([conditions.to(target), target], dim=1)
