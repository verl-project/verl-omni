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
"""CPU tests for the trainer's advantage gates: per-token credit and GRPO baseline exclusion.

``apply_advantage_gates`` is exercised as a pure function over a scored
``DataProto``, and ``OmniPPOTrainer._compute_advantage`` end to end over a
faked TransferQueue.
"""

import inspect
import textwrap

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack
from verl.protocol import DataProto
from verl.trainer.ppo import core_algos

from verl_omni.trainer.omni.advantage_gates import apply_advantage_gates


def _scored(
    uids: list[str], rewards: list[list[float]], masks: list[list[int]], *, norm_adv_by_std_in_grpo: bool
) -> DataProto:
    """A batch as verl's ``_compute_advantage`` leaves it: GRPO advantages over the whole groups."""
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor(rewards, dtype=torch.float32),
            "response_mask": torch.tensor(masks, dtype=torch.int64),
        },
        non_tensors={"uid": np.array(uids, dtype=object)},
    )
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        data.batch["token_level_rewards"],
        data.batch["response_mask"],
        data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _gates(data: DataProto, extra_fields: list, *, adv_estimator: str = "grpo", norm_adv_by_std_in_grpo: bool = False):
    return apply_advantage_gates(
        data, extra_fields, adv_estimator=adv_estimator, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
    )


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [False, True])
def test_clean_rows_are_rescored_against_the_clean_only_baseline(norm_adv_by_std_in_grpo):
    rewards = [[1.0, 0, 0, 0], [0.0, 0, 0, 0], [0.5, 0, 0, 0], [0.2, 0, 0, 0], [0.4, 0, 0, 0]]
    masks = [[1, 0, 1, 0]] * 5
    data = _scored(["g0", "g0", "g0", "g1", "g1"], rewards, masks, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo)
    untouched_g1 = data.batch["advantages"][3:].clone()
    metrics = _gates(
        data, [{}, {"advantage_exclude": True}, {}, {}, {}], norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
    )
    # g0's two clean rows equal verl's GRPO over the clean-only sub-batch; the excluded row is all zeros.
    clean_only = _scored(
        ["g0", "g0"], [rewards[0], rewards[2]], masks[:2], norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
    )
    torch.testing.assert_close(data.batch["advantages"][[0, 2]], clean_only.batch["advantages"])
    torch.testing.assert_close(data.batch["returns"][[0, 2]], clean_only.batch["returns"])
    assert torch.count_nonzero(data.batch["advantages"][1]) == 0
    assert torch.count_nonzero(data.batch["returns"][1]) == 0
    # The dropped member's 0 reward no longer pulls the baseline: mean 0.75 over
    # [1.0, 0.5], verl's unbiased std 0.5/√2 on the std branch.
    expected_first = 0.25 if not norm_adv_by_std_in_grpo else 0.25 / (0.5 / np.sqrt(2))
    assert data.batch["advantages"][0, 0] == pytest.approx(expected_first, abs=1e-5)
    # A group without an excluded member is untouched.
    torch.testing.assert_close(data.batch["advantages"][3:], untouched_g1)
    assert metrics == {"advantage/gated_token_share": 0.0, "advantage/excluded_sample_share": pytest.approx(0.2)}


def test_a_group_reduced_to_one_clean_member_follows_verls_singleton_convention():
    rewards = [[1.0, 0], [0.0, 0], [0.0, 0]]
    data = _scored(["g0", "g0", "g0"], rewards, [[1, 0]] * 3, norm_adv_by_std_in_grpo=False)
    assert data.batch["advantages"][0, 0] == pytest.approx(2 / 3)  # verl's mean 1/3 over all three members
    _gates(data, [{}, {"advantage_exclude": True}, {"advantage_exclude": True}])
    # verl's GRPO gives a singleton group mean 0: the lone clean row keeps its raw reward.
    assert data.batch["advantages"][0].tolist() == [1.0, 0.0]
    assert data.batch["advantages"][1:].count_nonzero() == 0


def test_a_fully_excluded_group_is_all_zeros():
    data = _scored(["g0", "g0"], [[1.0, 0], [0.0, 0]], [[1, 0]] * 2, norm_adv_by_std_in_grpo=False)
    metrics = _gates(data, [{"advantage_exclude": True}, {"advantage_exclude": True}])
    assert data.batch["advantages"].count_nonzero() == 0
    assert data.batch["returns"].count_nonzero() == 0
    assert metrics == {"advantage/gated_token_share": 0.0, "advantage/excluded_sample_share": 1.0}


def test_per_token_gate_pads_a_short_mask_with_zeros_and_truncates_a_long_one():
    data = _scored(["g0", "g0"], [[1.0, 0, 0, 0], [0.0, 0, 0, 0]], [[1, 0, 1, 1]] * 2, norm_adv_by_std_in_grpo=False)
    before = data.batch["advantages"].clone()
    metrics = _gates(data, [{"advantage_mask": [1, 0, 1]}, {"advantage_mask": [0, 0, 1, 1, 1, 1]}])
    expected = before * torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.float32)
    torch.testing.assert_close(data.batch["advantages"], expected)
    torch.testing.assert_close(data.batch["returns"], before)  # the gate scales advantages only
    # Six response-masked positions; the gate zeroed row 0's tail and row 1's first.
    assert metrics == {"advantage/gated_token_share": pytest.approx(1 / 3), "advantage/excluded_sample_share": 0.0}


def test_identity_when_neither_column_is_present():
    data = _scored(["g0", "g0", "g1"], [[1.0, 0], [0.0, 0], [0.3, 0]], [[1, 0]] * 3, norm_adv_by_std_in_grpo=False)
    before = {key: data.batch[key].clone() for key in ("advantages", "returns")}
    metrics = _gates(data, [{}, None, {"reward_extra_info": {"duplex_pacing_corrupted": 0.0}}])
    for key, value in before.items():
        torch.testing.assert_close(data.batch[key], value)
    assert metrics == {"advantage/gated_token_share": 0.0, "advantage/excluded_sample_share": 0.0}


def test_exclusion_is_refused_under_another_estimator_but_the_gate_is_not():
    data = _scored(["g0", "g0"], [[1.0, 0], [0.0, 0]], [[1, 0]] * 2, norm_adv_by_std_in_grpo=False)
    with pytest.raises(NotImplementedError, match="adv_estimator='gae'"):
        _gates(data, [{"advantage_exclude": True}, {}], adv_estimator="gae")
    gated = _gates(data, [{"advantage_mask": [0, 0]}, {}], adv_estimator="gae")
    assert data.batch["advantages"][0].count_nonzero() == 0
    assert gated["advantage/gated_token_share"] == pytest.approx(0.5)


def test_row_count_mismatch_is_refused():
    data = _scored(["g0", "g0"], [[1.0, 0], [0.0, 0]], [[1, 0]] * 2, norm_adv_by_std_in_grpo=False)
    with pytest.raises(ValueError, match="1 extra_fields rows for a batch of 2 rows"):
        _gates(data, [{}])


def test_trainer_compute_advantage_is_verl_plus_the_gates(monkeypatch):
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta

    from verl_omni.trainer.omni import OmniPPOTrainerSync

    # Three u0 members (one pacing-corrupted) and one u1 member, plus a padded
    # row whose extras deep-copy the corrupted member's.
    masks = [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0], [0]]
    scores = [[1.0, 0, 0], [0.0, 0, 0], [0.5, 0, 0], [0.2, 0], [0.0]]
    logprobs = [[-1.0] * len(mask) for mask in masks]
    jagged = lambda rows: torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)  # noqa: E731
    store = TensorDict(
        {
            "uid": NonTensorStack("u0", "u0", "u0", "u1", "pad0"),
            "response_mask": jagged(masks),
            "rm_scores": jagged(scores),
            "rollout_log_probs": jagged(logprobs),
            "old_log_probs": jagged(logprobs),
            "ref_log_prob": jagged(logprobs),
        },
        batch_size=[5],
    )
    corrupted = {"advantage_mask": [], "advantage_exclude": True}
    extras = NonTensorStack(
        {"advantage_mask": [1, 0, 0], "advantage_exclude": False},
        corrupted,
        {"advantage_mask": [1, 0, 1], "advantage_exclude": False},
        {"advantage_mask": [1, 0], "advantage_exclude": False},
        dict(corrupted),  # padding: a deep copy of a real sample's extras
    )
    gets: list[list[str]] = []

    def fake_get(*, keys, partition_id, select_fields):
        gets.append(list(select_fields))
        return (
            store.select(*select_fields)
            if "uid" in select_fields
            else TensorDict({"extra_fields": extras}, batch_size=[5])
        )

    written: dict = {}

    def fake_put(*, keys, partition_id, fields):
        written.update({key: value for key, value in fields.items()})
        return KVBatchMeta(keys=keys, partition_id=partition_id, tags=[{"is_padding": False}] * len(keys))

    monkeypatch.setattr(tq, "kv_batch_get", fake_get)
    monkeypatch.setattr(tq, "kv_batch_put", fake_put)
    trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "use_kl_in_reward": False,
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": False,
                "rollout_correction": None,
            },
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )
    trainer.use_critic = False
    keys = ["u0_s0_0", "u0_s1_0", "u0_s2_0", "u1_s0_0", "pad0_0_0"]
    batch = KVBatchMeta(keys=keys, partition_id="train", tags=[{"is_padding": False}] * 4 + [{"is_padding": True}])
    metrics: dict = {}
    out = trainer._compute_advantage(batch, metrics)
    assert gets == [
        ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob"],
        ["extra_fields"],
    ]
    assert out.keys == keys
    # verl's [0.5, -0.5, 0] over u0 becomes the clean-only [0.25, 0, -0.25], the
    # row-0 gate drops its third position, u1 is untouched, the padded row stays 0.
    advantages_written = [row.tolist() for row in written["advantages"]]
    for got, expected in zip(
        advantages_written,
        [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.25, 0.0, -0.25], [0.2, 0.0], [0.0]],
        strict=True,
    ):
        assert got == pytest.approx(expected)
    # Returns keep the ungated rescored values (the critic's targets, unused without a critic).
    for got, expected in zip(
        [row.tolist() for row in written["returns"]],
        [[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, -0.25], [0.2, 0.0], [0.0]],
        strict=True,
    ):
        assert got == pytest.approx(expected)
    assert metrics == {"advantage/gated_token_share": pytest.approx(3 / 7), "advantage/excluded_sample_share": 0.25}


def test_compute_advantage_is_a_faithful_copy_of_verl():
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    from verl_omni.trainer.omni.trainer_base import OmniPPOTrainer

    ours = textwrap.dedent(inspect.getsource(OmniPPOTrainer._compute_advantage)).splitlines()
    theirs = textwrap.dedent(inspect.getsource(PPOTrainer._compute_advantage)).splitlines()
    # Drop our docstring, the copy's annotation, and the DIFF block (comment + call + blank).
    start = next(i for i, line in enumerate(ours) if line.strip().startswith('"""'))
    end = next(i for i in range(start + 1, len(ours)) if ours[i].strip().endswith('"""'))
    ours[start : end + 1] = ['    """Compute the advantage of the batch."""']
    diff_at = next(i for i, line in enumerate(ours) if line.strip().startswith("# DIFF vs upstream"))
    assert ours[diff_at + 1].strip() == "metrics.update(self._apply_advantage_gates(batch, data))"
    del ours[diff_at : diff_at + 3]
    assert [line for line in ours if "Faithful copy" not in line] == theirs
