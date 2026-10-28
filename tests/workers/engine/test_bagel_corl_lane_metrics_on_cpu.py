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

"""Production PPO reduction and diagnostic separation, without serving imports."""

import ast
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from test_bagel_corl_gen_alignment_on_cpu import config, install_real_ppo, tag_gen
from test_bagel_corl_gen_alignment_on_cpu import losses as imported_loss_fixture
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric


@pytest.fixture
def losses(monkeypatch):
    return imported_loss_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def real_ppo(losses, monkeypatch):
    return install_real_ppo(monkeypatch, entropy_coeff=0.07, kl_loss_coef=0.13)


def metric_sum(metrics, key):
    values = metrics[key]
    if not isinstance(values, list):
        values = [values]
    return sum(float(x.aggregate()) if isinstance(x, Metric) else float(x) for x in values)


def und_fixture(logp, entropy, indices, *, dp_size, global_zero=False, reference=0.11):
    lengths = [2, 8, 3]
    prompts = [torch.ones(1, dtype=torch.long) for _ in indices]
    responses = [torch.ones(lengths[i], dtype=torch.long) for i in indices]
    mask = torch.zeros(len(indices), 8)
    for row, i in enumerate(indices):
        if i != 2 and not global_zero:
            mask[row, : lengths[i]] = 1
    data = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor(prompts, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(responses, layout=torch.jagged),
            "response_mask": mask,
            "old_log_probs": torch.full_like(mask, -0.03),
            "advantages": torch.tensor([1.3, -0.7, 4.0])[indices, None].expand(-1, 8),
            "ref_log_prob": torch.full_like(mask, reference),
        },
        batch_size=[len(indices)],
    )
    tu.assign_non_tensor(
        data,
        max_response_len=8,
        dp_size=1,
        batch_num_tokens=None,
        global_batch_size=None,
        und_episode_normalizer=(1 if global_zero else 2) / dp_size,
        corl_dp_size=dp_size,
    )
    output = {
        "log_probs": torch.nested.as_nested_tensor([logp[i, : lengths[i] + 1] for i in indices], layout=torch.jagged),
        "entropy": torch.nested.as_nested_tensor([entropy[i, : lengths[i] + 1] for i in indices], layout=torch.jagged),
    }
    return data, output


@pytest.mark.parametrize("micro_size", [1, 2])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("global_zero", [False, True])
def test_real_ppo_kl_entropy_episode_mean_matches_serial_and_dp(losses, real_ppo, micro_size, dp_size, global_zero):
    logp = torch.full((3, 9), 0.02, requires_grad=True)
    entropy = torch.full((3, 9), 0.8, requires_grad=True)
    rank_metrics = []
    key_sets = []
    # Rank 1 owns a fully rejected episode as well as its eight-token episode.
    ranks = [[0, 1, 2]] if dp_size == 1 else [[0], [2, 1]]
    for rank in ranks:
        local_metrics = {}
        for start in range(0, len(rank), micro_size):
            data, output = und_fixture(
                logp, entropy, rank[start : start + micro_size], dp_size=dp_size, global_zero=global_zero
            )
            loss, metrics = losses["bagel_und_episode_loss"](real_ppo, output, data)
            assert torch.isfinite(loss)
            key_sets.append(set(metrics))
            for key in metrics:
                assert torch.isfinite(torch.tensor(metric_sum(metrics, key)))
                if key.startswith("actor/und/"):
                    local_metrics.setdefault(key, Metric(aggregation=AggregationType.SUM)).append(
                        metric_sum(metrics, key)
                    )
            (loss / dp_size).backward()
        rank_metrics.append(local_metrics)
    totals = {}
    for key in rank_metrics[0]:
        values = [rank[key] for rank in rank_metrics]
        width = max(len(value.values) for value in values)
        for value in values:
            value.extend([0.0] * (width - len(value.values)))
        totals[key] = Metric.aggregate_dp(values)
    assert all(keys == key_sets[0] for keys in key_sets)
    expected_logp = torch.zeros_like(logp)
    expected_entropy = torch.zeros_like(entropy)
    if not global_zero:
        for i, (length, advantage) in enumerate([(2, 1.3), (8, -0.7)]):
            expected_logp[i, :length] = (-torch.exp(torch.tensor(0.05)) * advantage + 0.13) / length / 2
            expected_entropy[i, :length] = -0.07 / length / 2
    torch.testing.assert_close(logp.grad, expected_logp)
    # Rejected outputs may be absent from the graph for entropy; valid episodes still contribute.
    if not global_zero:
        torch.testing.assert_close(entropy.grad, expected_entropy)
    else:
        assert entropy.grad is None or not entropy.grad.any()
    expected = {
        "actor/und/pg_loss": 0.0 if global_zero else -float(torch.exp(torch.tensor(0.05))) * 0.3,
        "actor/und/update_kl": 0.0 if global_zero else -0.05,
        "actor/und/ref_kl": 0.0 if global_zero else -0.09,
        "actor/und/entropy": 0.0 if global_zero else 0.8,
    }
    assert totals == pytest.approx(expected, abs=1e-6)


def test_reference_change_does_not_change_update_kl(losses, real_ppo):
    observed = []
    for reference in [0.11, 0.31]:
        data, output = und_fixture(
            torch.full((3, 9), 0.02), torch.full((3, 9), 0.8), [0], dp_size=1, reference=reference
        )
        _, metrics = losses["bagel_und_episode_loss"](real_ppo, output, data)
        observed.append({key: metric_sum(metrics, key) for key in metrics if key.startswith("actor/und/")})
    assert observed[0]["actor/und/update_kl"] == observed[1]["actor/und/update_kl"]
    assert observed[0]["actor/und/ref_kl"] != observed[1]["actor/und/ref_kl"]


def test_gen_lane_metrics_do_not_publish_und_values(losses):
    data = TensorDict({"old_log_probs": torch.zeros(2), "advantages": torch.tensor([1.0, -2.0])}, batch_size=[2])
    tag_gen(data, gradient_accumulation_steps=1)
    _, metrics = losses["bagel_composite_loss"](config(), {"log_probs": torch.tensor([0.01, -0.01])}, data)
    assert "actor/gen/update_kl" in metrics
    assert "actor/gen/pg_loss" in metrics
    assert not any(key.startswith("actor/und/") for key in metrics)
    assert not any("ref_kl" in key or "rollout_actor_logp_gap" in key for key in metrics)


def test_joint_dispatch_keeps_overlapping_legacy_metrics_in_separate_lane_namespaces(losses, real_ppo):
    real_ppo.diffusion_loss = config().diffusion_loss
    und_data, und_output = und_fixture(
        torch.full((3, 9), 0.02, requires_grad=True), torch.full((3, 9), 0.8, requires_grad=True), [0], dp_size=1
    )
    gen_data = TensorDict({"old_log_probs": torch.zeros(1), "advantages": torch.tensor([2.0])}, batch_size=[1])
    tu.assign_non_tensor(gen_data, bagel_corl_lane="gen")
    outer = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(
        outer,
        bagel_corl_und=und_data,
        bagel_corl_gen=gen_data,
        has_complete_gen_groups=True,
        skip_gen=False,
        num_gen_rows=1,
        gradient_accumulation_steps=1,
        sp_size=1,
    )
    loss, metrics = losses["bagel_composite_loss"](
        real_ppo, {"und": und_output, "gen": {"log_probs": torch.zeros(1, requires_grad=True)}}, outer
    )
    assert torch.isfinite(loss)
    assert metric_sum(metrics, "actor/und/pg_loss") == pytest.approx(-float(torch.exp(torch.tensor(0.05))) * 1.3 / 2)
    assert metric_sum(metrics, "actor/gen/pg_loss") == pytest.approx(-2.0)
    assert metric_sum(metrics, "actor/und/update_kl") == pytest.approx(-0.025)
    assert metric_sum(metrics, "actor/und/ref_kl") == pytest.approx(-0.045)
    assert metric_sum(metrics, "actor/gen/update_kl") == pytest.approx(0.0)


def test_trainer_rollout_gap_preserves_distinct_update_and_reference_metrics():
    # Execute the production metric publication suffix without initializing a trainer.
    tree = ast.parse(
        (Path(__file__).resolve().parents[3] / "verl_omni/trainer/omni/bagel_corl_trainer.py").read_text(
            encoding="utf-8"
        )
    )
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_update_actor")
    start = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "list(metrics.items())"
    )
    end = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_divergence_tripwire"
    )
    metrics = {
        "actor/und/update_kl": 0.03,
        "actor/und/ref_kl": 0.7,
        "actor/gen/update_kl": 0.002,
        "rollout_corr/kl": 0.19,
    }
    exec(
        compile(ast.Module(body=method.body[start:end], type_ignores=[]), "trainer_metric_suffix", "exec"),
        {"metrics": metrics, "math": math},
    )
    assert metrics["und/update_kl"] == 0.03
    assert metrics["und/ref_kl"] == 0.7
    assert metrics["gen/update_kl"] == 0.002
    assert metrics["und/rollout_actor_logp_gap"] == 0.19


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_gen_metrics_use_real_dp_average_for_ratio_reference_and_losses(losses, dp_size):
    # Eight transitions, two optimizer microsteps per rank, zero-weight padding
    # on the final rank. FSDP averages local losses; Metric.aggregate_dp also
    # averages ranks, even for SUM (which sums only microsteps).
    cfg = config()
    cfg.kl_loss_coef = 0.1
    cfg.diffusion_loss.use_gen_kl_loss = True
    per_rank = []
    expected_loss = 0.0
    total_valid = dp_size * 2 - 1
    for rank in range(dp_size):
        collected = {}
        for step in range(2):
            valid = float(rank != dp_size - 1 or step == 0)
            data = TensorDict(
                {
                    "old_log_probs": torch.zeros(1),
                    "advantages": torch.ones(1),
                    "loss_unit_weight": torch.tensor([valid]),
                    "ref_prev_sample_mean": torch.zeros(1, 1, 1),
                },
                batch_size=[1],
            )
            tag_gen(data, corl_loss_normalized=True, loss_normalizer=total_valid / dp_size, corl_dp_size=dp_size)
            loss, metrics = losses["bagel_composite_loss"](
                cfg,
                {
                    "log_probs": torch.tensor([0.01], requires_grad=True),
                    "prev_sample_mean": torch.full((1, 1, 1), 2.0, requires_grad=True),
                    "std_dev_t": torch.ones(1, 1, 1),
                },
                data,
            )
            expected_loss += float(loss.detach()) / dp_size
            for key, metric in metrics.items():
                if key.startswith("actor/gen/"):
                    collected.setdefault(key, Metric(aggregation=metric.aggregation)).append(metric)
        per_rank.append(collected)
    actual = {key: Metric.aggregate_dp([rank[key] for rank in per_rank]) for key in per_rank[0]}
    ratio = math.exp(0.01)
    assert actual["actor/gen/ratio_mean"] == pytest.approx(ratio)
    assert actual["actor/gen/ratio_square_mean"] == pytest.approx(ratio**2)
    assert actual["actor/gen/update_kl"] == pytest.approx(-0.01)
    assert actual["actor/gen/ref_kl"] == pytest.approx(2.0)
    assert actual["actor/gen/pg_loss"] == pytest.approx(-ratio)
    assert actual["actor/gen/total_loss"] == pytest.approx(expected_loss)
    assert math.sqrt(max(0.0, actual["actor/gen/ratio_square_mean"] - actual["actor/gen/ratio_mean"] ** 2)) < 0.001
