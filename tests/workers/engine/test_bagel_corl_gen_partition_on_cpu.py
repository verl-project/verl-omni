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

"""GEN ownership, padding and globally normalized gradient regressions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu


@pytest.fixture
def batch_ops():
    path = Path(__file__).resolve().parents[3] / "verl_omni/trainer/omni/bagel_corl_batch.py"
    spec = importlib.util.spec_from_file_location("corl_batch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gen_batch():
    data = TensorDict(
        {
            "parent_episode_id": torch.tensor([10, 20, 20, 30]),
            "gen_row_id": torch.arange(4),
            "all_latents": torch.ones(4, 4, 2),
            "all_timesteps": torch.full((4, 3), 0.8),
            "gen_valid": torch.ones(4),
            "gen_transition_valid": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            "old_log_probs": torch.zeros(4, 3),
            "advantages": torch.ones(4, 3),
        },
        batch_size=[4],
    )
    prompts = np.empty(4, dtype=object)
    prompts[:] = [[101], [201], [202], [301]]
    tu.assign_non_tensor(data, prompt_token_ids=prompts)
    return data


def test_each_gen_row_owned_once_across_shuffled_rank_minibatches(batch_ops):
    data = gen_batch()
    ids = []
    for episodes in [[30], [10], [99], [20]]:
        shard = batch_ops.select_gen_rows_for_episodes(data, torch.tensor(episodes))
        ids += shard["gen_row_id"].tolist()
        for row, prompt in zip(
            shard["gen_row_id"].tolist(), tu.get_non_tensor_data(shard, "prompt_token_ids", None), strict=True
        ):
            assert prompt == [101, 201, 202, 301][row : row + 1]
    assert sorted(ids) == [0, 1, 2, 3]


def test_duplicate_rows_and_episodes_rejected(batch_ops):
    data = gen_batch()
    data["gen_row_id"] = torch.tensor([0, 0, 2, 3])
    with pytest.raises(ValueError, match="unique"):
        batch_ops.select_gen_rows_for_episodes(data, torch.tensor([10]))
    with pytest.raises(ValueError, match="unique"):
        batch_ops.select_gen_rows_for_episodes(gen_batch(), torch.tensor([10, 10]))


def test_empty_rank_padding_is_safe_and_has_no_valid_units(batch_ops):
    data = gen_batch()
    shard = batch_ops.select_gen_rows_for_episodes(data, torch.tensor([99]))
    padded = batch_ops.pad_gen_shard(shard, target_rows=2, template=data[:1])
    assert len(padded) == 2
    assert not padded["gen_valid"].any()
    assert not padded["gen_transition_valid"].any()
    assert (padded["all_timesteps"] > 0).all()
    assert torch.isfinite(padded["all_latents"]).all()


def test_episode_hash_is_stable_and_nonnegative(batch_ops):
    assert batch_ops.episode_numeric_id("episode-x") == batch_ops.episode_numeric_id("episode-x")
    assert 0 <= batch_ops.episode_numeric_id("episode-x") < 2**63
    assert batch_ops.episode_numeric_id("episode-x") != batch_ops.episode_numeric_id("episode-y")


def test_freeze_ablation_excludes_und_parameters_from_gen_backward(batch_ops):
    model = torch.nn.ModuleDict({"text_lora": torch.nn.Linear(2, 2), "moe_gen": torch.nn.Linear(2, 2)})
    batch_ops.configure_corl_trainability(model, freeze_und=True, freeze_gen=False)
    assert all(not p.requires_grad for p in model["text_lora"].parameters())
    assert all(p.requires_grad for p in model["moe_gen"].parameters())
    model["moe_gen"](model["text_lora"](torch.ones(1, 2))).sum().backward()
    assert all(p.grad is None for p in model["text_lora"].parameters())
    assert all(p.grad is not None for p in model["moe_gen"].parameters())


def test_lane_gradient_norm_measures_increment_not_joint_norm(batch_ops):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[1.0, 2.0]])
    snapshot = batch_ops.gradient_snapshot(model)
    model.weight.grad += torch.tensor([[-3.0, 4.0]])
    norm = batch_ops.gradient_delta_norm(model, snapshot, dp_group=None, collective_device=torch.device("cpu"))
    assert norm == pytest.approx(5.0)


def test_global_ownership_checks_unknown_parents_and_padding(batch_ops):
    ids = batch_ops.validate_episode_ownership(torch.tensor([10, 20, 10]), torch.tensor([20]), [False, False, True])
    assert ids.tolist() == [10, 20, -1]
    with pytest.raises(ValueError, match="parent"):
        batch_ops.validate_episode_ownership(torch.tensor([10, 20]), torch.tensor([99]))
    with pytest.raises(ValueError, match="unique"):
        batch_ops.validate_episode_ownership(torch.tensor([10, 10]), torch.tensor([10]))
