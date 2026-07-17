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
"""CPU unit tests for the online-DPO loss + pairing (no models, no GPU)."""

import math
from types import SimpleNamespace

import pytest
import torch
from verl import DataProto

from verl_omni.workers.utils import losses
from verl_omni.workers.utils.losses import build_online_dpo_pair_indices, tts_dpo_loss


def _config(beta=0.1, nll_lambda=0.0, use_dynamic_bsz=False):
    return SimpleNamespace(
        use_dynamic_bsz=use_dynamic_bsz,
        policy_loss={"dpo_beta": beta, "dpo_nll_lambda": nll_lambda},
    )


def _data(logp, ref, uids, scores):
    """A single-response-token batch; seq_logp == logp, seq_ref == ref (mask all ones)."""
    n = len(logp)
    dp = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(n, 1),
            "ref_log_prob": torch.tensor(ref, dtype=torch.float32).reshape(n, 1),
        },
        non_tensors={"uid": list(uids), "sj_score": list(scores)},
    )
    data = dp.to_tensordict()
    model_output = {"log_probs": torch.tensor(logp, dtype=torch.float32).reshape(n, 1)}
    return model_output, data


@pytest.fixture(autouse=True)
def _identity_padding(monkeypatch):
    # no_padding_2_padding turns flat log-probs into (B, resp_len); our fixtures are already (B, 1).
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda log_probs, data: log_probs)


class TestPairIndices:
    def test_groups_top_and_bottom(self):
        # uid a: idx0 (0.9) > idx1 (0.1); uid b: idx3 (0.7) > idx2 (0.3)
        assert build_online_dpo_pair_indices(["a", "a", "b", "b"], [0.9, 0.1, 0.3, 0.7]) == [0, 1, 3, 2]

    def test_skips_singleton_groups(self):
        assert build_online_dpo_pair_indices(["a", "a", "b"], [0.9, 0.1, 0.5]) == [0, 1]

    def test_keeps_ties(self):
        assert build_online_dpo_pair_indices(["a", "a"], [0.5, 0.5]) == [0, 1]


class TestLoss:
    def test_matches_hand_computed(self):
        # r = logp - ref = [2, 1, 1, 3]; pairs (a: 2 vs 1), (b: 3 vs 1); inside = beta*[1, 2]
        model_output, data = _data(
            [2.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0], ["a", "a", "b", "b"], [0.9, 0.1, 0.3, 0.7]
        )
        loss, metrics = tts_dpo_loss(_config(beta=0.1), model_output, data)
        expected = -(math.log(1 / (1 + math.exp(-0.1))) + math.log(1 / (1 + math.exp(-0.2)))) / 2
        assert abs(loss.item() - expected) < 1e-6
        assert metrics["actor/dpo_acc"].values[0] == 1.0  # both margins positive
        assert abs(metrics["actor/dpo_margin"].values[0] - 1.5) < 1e-6  # mean of [1, 2]

    def test_order_independent(self):
        rows = ([2.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0], ["a", "a", "b", "b"], [0.9, 0.1, 0.3, 0.7])
        loss_a, _ = tts_dpo_loss(_config(), *_data(*rows))
        # shuffle rows: (b_hi, a_lo, a_hi, b_lo)
        perm = ([3.0, 1.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0], ["b", "a", "a", "b"], [0.7, 0.1, 0.9, 0.3])
        loss_b, _ = tts_dpo_loss(_config(), *_data(*perm))
        assert abs(loss_a.item() - loss_b.item()) < 1e-6

    def test_tie_masks_the_pair(self):
        # uid a is a tie (equal scores) -> zero weight; only uid b contributes.
        model_output, data = _data(
            [2.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0], ["a", "a", "b", "b"], [0.5, 0.5, 0.3, 0.7]
        )
        loss, metrics = tts_dpo_loss(_config(beta=0.1), model_output, data)
        expected = -math.log(1 / (1 + math.exp(-0.2)))  # only pair b, inside = 0.1*2
        assert abs(loss.item() - expected) < 1e-6
        assert abs(metrics["actor/dpo_tie_frac"].values[0] - 0.5) < 1e-6

    def test_nll_anchor_adds_chosen_nll(self):
        rows = ([2.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0], ["a", "a", "b", "b"], [0.9, 0.1, 0.3, 0.7])
        base, _ = tts_dpo_loss(_config(beta=0.1, nll_lambda=0.0), *_data(*rows))
        anchored, _ = tts_dpo_loss(_config(beta=0.1, nll_lambda=0.5), *_data(*rows))
        # chosen seq_logp are [2, 3] over 1 token -> nll = [-2, -3]; term = mean = -2.5; +0.5*(-2.5)
        assert abs((anchored.item() - base.item()) - 0.5 * (-2.5)) < 1e-6

    def test_dynamic_bsz_raises(self):
        model_output, data = _data([2.0, 1.0], [0.0, 0.0], ["a", "a"], [0.9, 0.1])
        with pytest.raises(ValueError, match="use_dynamic_bsz"):
            tts_dpo_loss(_config(use_dynamic_bsz=True), model_output, data)

    def test_no_pairs_raises(self):
        model_output, data = _data([2.0, 1.0], [0.0, 0.0], ["a", "b"], [0.9, 0.1])  # no group has 2
        with pytest.raises(RuntimeError, match="no preference pairs"):
            tts_dpo_loss(_config(), model_output, data)
