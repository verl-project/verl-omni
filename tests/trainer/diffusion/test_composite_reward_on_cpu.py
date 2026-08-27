# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""CPU tests for composite rewards from MultiVisualRewardManager,
and decouple rewards for AR.

Usage:

reward.reward_manager.name=MultiVisualRewardManager
reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager
"+reward.reward_functions.ar.path=$AR_REWARD_PATH"
'+reward.reward_functions.ar.name=compute_score
'+reward.reward_functions.ar.weight=1.0'
"+reward.reward_functions.dit.path=$DIT_REWARD_PATH"
'+reward.reward_functions.dit.name=compute_score'
'+reward.reward_functions.dit.weight=0.0'
"""

import pytest
import numpy as np
import torch
from verl import DataProto
from verl_omni.trainer.diffusion.dual_grpo_utils import (
    AR_REWARD_KEY,
    # average_grouped_rewards,
    extract_ar_rewards_from_colocate_batch,
)
from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

AR_REWARD_KEY = "reward/ar"
# def average_grouped_rewards(scores: np.ndarray, *, group_size: int) -> np.ndarray:
#     """Average ``group_size`` consecutive rows into one scalar per group."""
#     scores = np.asarray(scores)
#     if scores.ndim == 1:
#         scores = scores.reshape(-1, 1)
#     if scores.ndim == 2 and scores.shape[1] == 1:
#         scores = scores
#     # raise ValueError(f"Expected reward scores with shape (N,) or (N, 1), got {scores.shape}")
#     num_groups, remainder = divmod(scores.shape[0], group_size)
#     if remainder != 0:
#         raise ValueError(
#             f"Cannot average {scores.shape[0]} reward rows with group_size={group_size}."
#         )
#     return scores.reshape(num_groups, group_size, scores.shape[1]).mean(axis=1)

class TestExtractARRewardsFromColocateBatch:
    # def test_average_grouped_rewards_uses_non_overlapping_blocks(self):
    #     scores = np.array([0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    #     averaged = average_grouped_rewards(scores, group_size=4)
    #     assert averaged.shape == (2, 1)
    #     assert averaged[0, 0] == pytest.approx(0.15)
    #     assert averaged[1, 0] == pytest.approx(1.15)

    def test_extract_ar_rewards_populates_rm_scores_and_extra_keys(self):
        num_ar = 3
        rollout_m = 4
        ar_per_image_scores = []
        for ar_idx in range(num_ar):
            for image_idx in range(rollout_m):
                ar_per_image_scores.append(ar_idx + image_idx * 0.1)
        # [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3]

        batch_reward = DataProto.from_dict(
            tensors={"rm_scores": torch.zeros(num_ar * rollout_m, 1)},
            non_tensors={
                AR_REWARD_KEY: np.array(ar_per_image_scores, dtype=np.float32).reshape(-1, 1),
                f"{AR_REWARD_KEY}/semantic": np.array(ar_per_image_scores, dtype=np.float32),
                f"{AR_REWARD_KEY}/semantic/detail": (
                    np.array(["ar detail 1"]*rollout_m+["ar detail 2"]*rollout_m+["ar detail 3"]*rollout_m, dtype=np.float32)
                ),
            },
        )
        batch_reward.meta_info["reward_extra_keys"] = [AR_REWARD_KEY, f"{AR_REWARD_KEY}/semantic", f"{AR_REWARD_KEY}/semantic/detail"]
        ar_batch = DataProto.from_dict(
            tensors={"ar_response_ids": torch.zeros(num_ar, 5, dtype=torch.long)},
        )
        PolicyGradientRayTrainer._extract_ar_reward_tensor(batch_reward, ar_batch, avg_size=rollout_m)

        assert ar_batch.batch["rm_scores"].shape == (num_ar, 1)
        assert ar_batch.batch["rm_scores"][0].item() == pytest.approx(0.15)
        assert ar_batch.batch["rm_scores"][1].item() == pytest.approx(1.15)
        assert ar_batch.batch["rm_scores"][2].item() == pytest.approx(2.15)
        assert AR_REWARD_KEY not in batch_reward.non_tensor_batch
        assert f"{AR_REWARD_KEY}/semantic" in ar_batch.non_tensor_batch
        assert f"{AR_REWARD_KEY}/semantic/detail" in ar_batch.non_tensor_batch
        assert ar_batch.non_tensor_batch[f"{AR_REWARD_KEY}/semantic"][1] == pytest.approx(1.15)
        assert ar_batch.non_tensor_batch[f"{AR_REWARD_KEY}/semantic/detail"][1] == "ar detail 1"

    def test_extract_ar_rewards_raises_when_row_count_mismatches(self):
        batch_reward = DataProto.from_dict(
            tensors={"rm_scores": torch.zeros(2, 1)},
            non_tensors={AR_REWARD_KEY: np.array([0.0, 0.1], dtype=np.float32).reshape(-1, 1)},
        )
        batch_reward.meta_info["reward_extra_keys"] = [AR_REWARD_KEY]
        ar_batch = DataProto.from_dict(
            tensors={"ar_response_ids": torch.zeros(3, 5, dtype=torch.long)},
        )

        with pytest.raises(AssertionError, "reward_scores shape"):
            PolicyGradientRayTrainer._extract_ar_reward_tensor(batch_reward, ar_batch, avg_size=4)
