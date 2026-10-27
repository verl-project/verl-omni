# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 AlphaGRPO Authors.
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

"""Group selection and failure penalties from AlphaGRPO's reflect task."""

import torch

REFLECTION_CFG = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": (0.0, 1.0),
    "cfg_renorm_type": "text_channel",
    "cfg_renorm_min": 0.0,
}

REFLECTION_PROMPT = (
    "The user requires to generate image of `{}`. You tried to generate an image based on the user's request, "
    "but you failed to create the correct image.\n"
    "Reflect on what went wrong and write down the editing instruction that will help you do better "
    "and refine the image to meet user's request based on your own reflection."
)


def select_initial_images(rewards, uids, group_size):
    """Return the lowest-reward initial sample's index for each row [B]."""
    if rewards.ndim != 1 or len(uids) != rewards.numel() or not torch.isfinite(rewards).all():
        raise ValueError("Initial rewards must be finite [B] values aligned with prompt UIDs.")
    groups = {}
    for row, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(row)
    selected = torch.empty(len(uids), dtype=torch.long, device=rewards.device)
    for rows in groups.values():
        if len(rows) != group_size:
            raise ValueError("Reflection requires a complete candidate group for every prompt UID.")
        selected[rows] = rows[int(rewards[rows].argmin())]
    return selected


def penalize_failed_refinements(rewards, initial_rewards, uids):
    """Replace non-improving scores by the group's minimum initial score; return [B] and skip flag."""
    if rewards.shape != initial_rewards.shape or rewards.ndim != 1 or len(uids) != rewards.numel():
        raise ValueError("Refinement and initial rewards must align with prompt UIDs.")
    if not torch.isfinite(rewards).all() or not torch.isfinite(initial_rewards).all():
        raise ValueError("Refinement rewards must be finite.")
    improved = rewards > initial_rewards
    adjusted = rewards.clone()
    groups = {}
    for row, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(row)
    for rows in groups.values():
        row_ids = torch.tensor(rows, device=rewards.device)
        failed = row_ids[~improved[row_ids]]
        adjusted[failed] = initial_rewards[row_ids].min()
    return adjusted, not bool(improved.any())
