# Copyright 2026 verl-omni contributors
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

"""CPU math/data contracts; no Ray, CUDA or new Omni V1 qualification."""

import copy

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss as native_ppo_loss
from verl.workers.utils.padding import response_from_nested

from tests.workers.test_omni_output_version_on_cpu import assert_equal as assert_state_equal
from verl_omni.workers.utils.cohort_data import CohortData, value_digest
from verl_omni.workers.utils.parameter_revision import ParameterRevision
from verl_omni.workers.utils.proximal_anchor import prepare_proximal_loss, proximal_anchor_loss_scope
from verl_omni.workers.utils.proximal_anchor import proximal_ppo_loss as ppo_loss


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def materialized():
    data = TensorDict(
        {
            "input_ids": nested([torch.arange(i + 2) for i in range(4)]),
            "response_mask": nested([torch.ones(i, dtype=torch.long) for i in range(4)]),
            "position_ids": nested([torch.arange(i + 2).repeat(3, 1) for i in range(4)]),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor_stack(
        data,
        "multi_modal_inputs",
        [
            {"image": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)},
            {"audio": np.arange(7, dtype=np.float32), "sample_rate": 16000},
            {"video": torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4), "clock": [0.0, 0.1]},
            {},
        ],
    )
    tu.assign_non_tensor_data(data, "temperature", 0.9)
    return data


def config_identity():
    return {"temperature": 0.9, "micro_batch_size": 2, "train_mode": True, "tp": 2, "sp": True}


def with_derived(data):
    data = data.clone()
    data["advantages"] = nested([torch.full((i,), float(i) - 1) for i in range(4)])
    data["returns"] = data["advantages"].clone()
    return data


@pytest.mark.parametrize("indices", [[0], [3, 1], [2, 0, 3, 1]])
def test_actual_dispatched_shards_keep_per_row_media_and_execution_identity(indices):
    data, keys, execution = materialized(), tuple("abcd"), config_identity()
    raw = CohortData.prepare(data, keys, execution)
    data = with_derived(data)
    bound = raw.bind_derived(data, keys, execution, ("advantages", "returns"))
    shard = bound.shard(indices)
    selected = tu.index_select_tensor_dict(data, torch.tensor(indices))
    observed = shard.observe(selected, [keys[i] for i in indices], execution)
    assert observed.keys == shard.keys
    assert observed.execution_digest == value_digest(execution)
    assert len(observed.raw_digest) == len(observed.derived_digest) == 64
    # Another independent materialization of the same logical rows agrees.
    other = tu.index_select_tensor_dict(with_derived(materialized()), torch.tensor(indices))
    assert shard.observe(other, shard.keys, execution) == observed


@pytest.mark.parametrize(
    "change", ["ids", "mrope", "media", "mask", "advantages", "returns", "keys", "config", "extra", "missing"]
)
def test_raw_derived_transport_and_execution_changes_are_rejected(change):
    data, keys, execution = materialized(), tuple("abcd"), config_identity()
    ticket = CohortData.prepare(data, keys, execution)
    data = with_derived(data)
    ticket = ticket.bind_derived(data, keys, execution, ("advantages", "returns"))
    if change in {"ids", "mrope", "mask"}:
        name = {"ids": "input_ids", "mrope": "position_ids", "mask": "response_mask"}[change]
        data[name].values().reshape(-1)[0] += 1
    elif change == "media":
        media = data.get("multi_modal_inputs").tolist()
        media[0]["image"][0, 0] += 1
    elif change in {"advantages", "returns"}:
        data[change].values()[0] += 1
    elif change == "keys":
        keys = keys[::-1]
    elif change == "config":
        execution["micro_batch_size"] = 4
    elif change == "extra":
        data["rollout_is_weights"] = torch.ones(4, 1)
    else:
        data.pop("returns")
    with pytest.raises((ValueError, TypeError, KeyError)):
        ticket.observe(data, keys, execution)


def test_derived_binding_cannot_bless_raw_change_or_be_repeated():
    data, keys, execution = materialized(), tuple("abcd"), config_identity()
    raw = CohortData.prepare(data, keys, execution)
    data = with_derived(data)
    bound = raw.bind_derived(data, keys, execution, ("advantages", "returns"))
    with pytest.raises(ValueError, match="exactly once"):
        bound.bind_derived(data, keys, execution, ("advantages", "returns"))
    data["input_ids"].values()[0] += 1
    with pytest.raises(ValueError, match="raw"):
        raw.bind_derived(data, keys, execution, ("advantages", "returns"))


@pytest.mark.parametrize("bad", [object(), {1: "wrong key"}, float("nan"), torch.empty(2, device="meta")])
def test_opaque_values_and_hidden_device_transfers_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        value_digest(bad)


def test_typed_digest_handles_bfloat16_layout_and_empty_media_without_collisions():
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    assert value_digest(tensor) == value_digest(tensor.t().contiguous().t())
    values = [tensor, tensor.float(), tensor.reshape(3, 2), [], (), b"", "", 1, True, 1.0]
    assert len({value_digest(value) for value in values}) == len(values)


def loss_data():
    data = TensorDict(
        {
            "prompts": nested([torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1, 2, 3])]),
            "responses": nested([torch.tensor([0, 2]), torch.tensor([], dtype=torch.long), torch.tensor([3])]),
            "response_mask": nested([torch.ones(2), torch.empty(0), torch.ones(1)]),
            "advantages": nested([torch.tensor([1.0, 1.0]), torch.empty(0), torch.tensor([-1.0])]),
        },
        batch_size=[3],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=3)
    return data


def output(model, entropy=True):
    logits = model(torch.arange(27, dtype=torch.float64).reshape(9, 3) / 13)
    logs = logits.log_softmax(-1)
    selected = logs.gather(-1, (torch.arange(9) % 4).unsqueeze(-1)).squeeze(-1)
    offsets = torch.tensor([0, 4, 5, 9])
    result = {"log_probs": torch.nested.nested_tensor_from_jagged(selected, offsets=offsets)}
    if entropy:
        ent = -(logs.exp() * logs).sum(-1)
        result["entropy"] = torch.nested.nested_tensor_from_jagged(ent, offsets=offsets)
    return result


def actor_config(mode, entropy, reference, calculate_entropy):
    return ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        ppo_mini_batch_size=3,
        clip_ratio_low=0.0003,
        clip_ratio_high=0.0004,
        policy_loss=PolicyLossConfig(loss_mode=mode),
        entropy_coeff=entropy,
        use_kl_loss=reference,
        calculate_entropy=calculate_entropy,
    )


def metric_values(metrics):
    return {
        key: (value.values, value.aggregation) if hasattr(value, "aggregation") else value
        for key, value in metrics.items()
    }


@pytest.mark.parametrize("layout", ["nested", "flat"])
@pytest.mark.parametrize("mode", ["vanilla", "gspo"])
@pytest.mark.parametrize(
    "entropy,reference,calculate_entropy",
    [(0.0, False, False), (0.0, False, True), (0.01, False, False), (0.01, True, False)],
)
def test_native_loss_three_adamw_updates_preserve_full_state_and_objectives(
    mode, entropy, reference, calculate_entropy, layout
):
    torch.manual_seed(9)
    baseline = torch.nn.Linear(3, 4, dtype=torch.float64)
    candidate = copy.deepcopy(baseline)
    base_optim = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    candidate_optim = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    guard = ParameterRevision()
    for step in range(3):
        data = loss_data()
        if reference:
            data["ref_log_prob"] = nested([torch.full((2,), -2.0), torch.empty(0), torch.full((1,), -2.0)])
        baseline_data = data.clone()
        with torch.no_grad():
            baseline_data["old_log_probs"] = response_from_nested(output(baseline)["log_probs"], data["response_mask"])
        base_optim.zero_grad()
        base_output = output(baseline, entropy=bool(entropy or calculate_entropy))
        if layout == "flat":
            base_output = {key: value.values() for key, value in base_output.items()}
        expected_loss, expected_metrics = native_ppo_loss(
            actor_config(mode, entropy, reference, calculate_entropy), base_output, baseline_data
        )
        expected_loss.backward()
        candidate_optim.zero_grad()
        with guard.update(guard.version(), f"loss-{step}"):
            with proximal_anchor_loss_scope(guard):
                # Always compute diagnostic entropy, even when it is absent
                # from the original zero-coefficient loss graph.
                model_output = output(candidate)
                if layout == "flat":
                    model_output = {key: value.values() for key, value in model_output.items()}
                actual_loss, actual_metrics = ppo_loss(
                    actor_config(mode, entropy, reference, calculate_entropy), model_output, data
                )
                actual_loss.backward()
                assert "old_log_probs" not in data
                assert_state_equal(expected_loss, actual_loss)
                assert_state_equal(metric_values(expected_metrics), metric_values(actual_metrics))
                assert_state_equal([p.grad for p in baseline.parameters()], [p.grad for p in candidate.parameters()])
            base_optim.step()
            with guard.mutation("optimizer"):
                candidate_optim.step()
        assert_state_equal(baseline.state_dict(), candidate.state_dict())
        assert_state_equal(base_optim.state_dict(), candidate_optim.state_dict())


def test_scope_requires_pending_actual_writer_and_restores_on_failure():
    guard = ParameterRevision()
    with pytest.raises(RuntimeError, match="pending"):
        with proximal_anchor_loss_scope(guard):
            pass
    cfg = actor_config("gspo", 0.0, False, False)
    model = torch.nn.Linear(3, 4, dtype=torch.float64)
    data, model_output = loss_data(), output(model)
    # Disabled default returns the exact original objects and does not create old.
    actual_output, actual_data = prepare_proximal_loss(cfg, model_output, data)
    assert actual_output is model_output and actual_data is data
    with pytest.raises(ValueError, match="stale"):
        with guard.update(guard.version(), "bad"):
            with proximal_anchor_loss_scope(guard):
                data["old_log_probs"] = response_from_nested(model_output["log_probs"], data["response_mask"])
                ppo_loss(cfg, model_output, data)
    assert prepare_proximal_loss(cfg, model_output, data) == (model_output, data)
    with pytest.raises(RuntimeError, match="Partial"):
        guard.version()
