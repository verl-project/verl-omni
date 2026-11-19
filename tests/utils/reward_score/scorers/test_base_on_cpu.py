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
"""Default lifecycle hooks inherited from the RewardScorer base."""

import pytest
import torch

from verl_omni.utils.reward_score.scorers.base import RewardScorer


class _Weights:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(torch.device(device))
        return self


class _MinimalScorer(RewardScorer):
    supports_cpu_offload = True

    def load(self, device):
        self.model = _Weights()
        self.device = torch.device(device)

    def score_batch(self, batch):
        return [{"score": 0.0} for _ in batch]


def test_default_hooks_move_declared_model_between_device_and_host():
    scorer = _MinimalScorer()
    scorer.load(torch.device("meta"))

    scorer.sleep()
    assert scorer.model.moves == [torch.device("cpu")]

    scorer.wake_up()
    assert scorer.model.moves == [torch.device("cpu"), torch.device("meta")]

    scorer.close()
    assert scorer.model is None


def test_offload_hook_without_declared_model_fails_fast():
    scorer = _MinimalScorer()
    with pytest.raises(RuntimeError, match="assign self.model"):
        scorer.sleep()


def test_constructor_rejects_invalid_declarations():
    from verl_omni.utils.reward_score.scorers.registry import register_scorer

    with pytest.raises(ValueError, match="max_batch_size"):

        @register_scorer("bad-bounds-scorer")
        class BadBatchSize(RewardScorer):
            max_batch_size = 0

            def score_batch(self, batch):
                return [{"score": 0.0} for _ in batch]

    with pytest.raises(ValueError, match="supports_cpu_offload"):

        @register_scorer("bad-offload-scorer")
        class BadOffload(RewardScorer):
            supports_cpu_offload = "yes"

            def score_batch(self, batch):
                return [{"score": 0.0} for _ in batch]
