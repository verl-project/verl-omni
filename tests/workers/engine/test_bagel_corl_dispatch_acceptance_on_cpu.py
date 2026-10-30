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

"""Trace real composite dispatch, timestep extraction and loss under simulated DP.

CPU collectives return precomputed global counts; this proves ownership and math,
not real collective scheduling. The manual CUDA harness covers FSDP collectives.
"""

import sys
from contextlib import nullcontext
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from test_bagel_corl_gen_alignment_on_cpu import composite_dispatch, config, gen_trajectory, load_nodes, und_batch
from test_bagel_corl_gen_alignment_on_cpu import losses as imported_loss_fixture
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.workers.engine.utils import prepare_micro_batches


@pytest.fixture
def losses(monkeypatch):
    return imported_loss_fixture.__wrapped__(monkeypatch)


def production_engine(losses, monkeypatch, parameter, micro_size, captured):
    composite_dispatch(losses, monkeypatch)
    adv_module = ModuleType("verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.advantages")
    adv_module.select_gen_advantages_for_step = losses["select_gen_advantages_for_step"]
    monkeypatch.setitem(sys.modules, adv_module.__name__, adv_module)
    losses.update(get_device_id=lambda: "cpu", nullcontext=nullcontext, prepare_micro_batches=prepare_micro_batches)
    load_nodes(
        "verl_omni/workers/engine/fsdp/diffusers_impl.py",
        {"_run_forward_backward_batch", "_reslice_per_row_non_tensors"},
        losses,
        class_name="DiffusersFSDPEngine",
    )
    load_nodes(
        "verl_omni/workers/engine/fsdp/diffusers_impl.py",
        {"forward_backward_batch", "forward_step", "prepare_model_outputs"},
        losses,
        class_name="PPODiffusersFSDPEngine",
    )

    def adapter(**kwargs):
        batch, step = kwargs["scheduler_inputs"], kwargs["step"]
        captured.extend((int(row), step) for row in batch["gen_row_id"] if row >= 0)
        return (parameter[batch["gen_row_id"].clamp_min(0), step], None, None, None)

    losses["forward_and_sample_previous_step"] = adapter
    engine = SimpleNamespace(
        model_config={"composite_mode": "bagel_corl"},
        module=torch.nn.Linear(1, 1),
        scheduler=None,
        engine_config=SimpleNamespace(micro_batch_size_per_gpu=micro_size),
        ulysses_sequence_parallel_size=1,
        get_data_parallel_group=lambda: None,
        prepare_model_inputs=lambda **kwargs: ({}, {}),
        postprocess_batch_func=lambda **kwargs: {"loss": [], "metrics": {}},
        _reslice_per_row_non_tensors=losses["_reslice_per_row_non_tensors"],
    )
    for name in ("_run_forward_backward_batch", "forward_step", "prepare_model_outputs"):
        setattr(engine, name, MethodType(losses[name], engine))
    return engine


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("micro_size", [1, 2])
def test_four_tasks_eight_siblings_four_ranks_two_optimizer_minibatches(losses, monkeypatch, k, micro_size):
    batch_ops = sys.modules["verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.batch"]
    # Preserve the production ownership and normalization functions; replace collective I/O only.
    monkeypatch.setattr(batch_ops, "gradient_delta_norm", lambda *a, **kw: 0.0)
    monkeypatch.setattr(batch_ops, "gradient_snapshot", lambda *a: {})
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a: 4)
    episodes = torch.randperm(32, generator=torch.Generator().manual_seed(91)).reshape(4, 8)
    rows = 32 * k
    parent = torch.arange(32).repeat_interleave(k)
    valid = torch.zeros(rows, 3)
    valid[::2, :1] = 1
    valid[1::2, :] = 1
    adv = torch.linspace(-1.7, 2.1, rows * 3).reshape(rows, 3)
    gen = gen_trajectory(parent, adv, valid)
    parameter = torch.nn.Parameter(torch.zeros(rows, 3))
    captured = []
    engine = production_engine(losses, monkeypatch, parameter, micro_size, captured)
    expected_grad = torch.zeros_like(parameter)
    actual_loss = 0.0
    expected_loss = 0.0
    for minibatch in range(2):
        owners = [episodes[rank, minibatch * 4 : (minibatch + 1) * 4] for rank in range(4)]
        active = torch.isin(parent, torch.cat(owners))
        count = float(valid[active].sum())

        # all_reduce(SUM) gives the optimizer minibatch's valid transitions;
        # all_reduce(MAX) gives the equalized number of rows on any rank.
        def reduce(tensor, op=None, group=None, count=count):
            tensor.fill_(4 * k if op == torch.distributed.ReduceOp.MAX else count)

        monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
        for owner in owners:
            data = und_batch(owner, gen)

            def objective(**kwargs):
                nonlocal actual_loss
                loss, metrics = losses["bagel_composite_loss"](config(), **kwargs)
                actual_loss += float(loss.detach()) / 4
                return loss / 4, metrics  # Simulate averaged DP gradients.

            losses["forward_backward_batch"](engine, data, objective)
        # Independent serial objective: each minibatch is a mean of its own real transitions.
        reference = torch.zeros_like(parameter, requires_grad=True)
        serial = -(reference[active].exp() * adv[active] * valid[active]).sum() / count
        expected_loss += float(serial.detach())
        expected_grad += torch.autograd.grad(serial, reference)[0]
    assert sorted(captured) == [(row, step) for row in range(rows) for step in range(3)]
    torch.testing.assert_close(parameter.grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert parameter.grad.abs().sum() > 0
    assert actual_loss == pytest.approx(expected_loss, abs=1e-6)


@pytest.mark.parametrize("micro_size", [1, 2])
def test_four_real_transitions_empty_ranks_and_padding_preserve_gradient_and_metrics(losses, monkeypatch, micro_size):
    batch_ops = sys.modules["verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.batch"]
    monkeypatch.setattr(batch_ops, "gradient_delta_norm", lambda *a, **kw: 0.0)
    monkeypatch.setattr(batch_ops, "gradient_snapshot", lambda *a: {})
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a: 4)

    def reduce(tensor, op=None, group=None):
        tensor.fill_(1 if op == torch.distributed.ReduceOp.MAX else 4)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    valid = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    advantages = torch.tensor([[1.0, 7.0, 9.0], [-2.0, 3.0, -4.0]])
    gen = gen_trajectory(torch.tensor([10, 20]), advantages, valid)
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    captured, metrics_seen = [], []
    engine = production_engine(losses, monkeypatch, parameter, micro_size, captured)

    def objective(**kwargs):
        assert tu.get_non_tensor_data(kwargs["data"], "loss_normalizer", None) == 1.0  # 4 transitions / DP4
        loss, metrics = losses["bagel_composite_loss"](config(), **kwargs)
        metrics_seen.append(metrics)
        return loss / 4, metrics

    for episode in [10, 20, 30, 40]:
        data = und_batch(torch.tensor([episode]), gen)
        losses["forward_backward_batch"](engine, data, objective)
    torch.testing.assert_close(parameter.grad, -advantages * valid / 4)
    assert all(set(metric) == set(metrics_seen[0]) for metric in metrics_seen)
    for metric in metrics_seen:
        assert all(torch.isfinite(torch.as_tensor(value.aggregate())).all() for value in metric.values())
    for key, expected in [("actor/gen/pg_loss", 0.5), ("actor/gen/ratio_mean", 1.0)]:
        rank_metrics = []
        for start in range(0, len(metrics_seen), 3):
            metric = metrics_seen[start][key].init_list()
            for step_metrics in metrics_seen[start : start + 3]:
                metric.append(step_metrics[key])
            rank_metrics.append(metric)
        assert Metric.aggregate_dp(rank_metrics) == pytest.approx(expected)
