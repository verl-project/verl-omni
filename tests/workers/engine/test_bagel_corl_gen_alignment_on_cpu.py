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

"""Exercise real loss/materialization code without importing the serving registries."""

import ast
import importlib.util
import logging
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric

ROOT = Path(__file__).resolve().parents[3]


def load_nodes(path, names, namespace, class_name=None):
    """Load actual source nodes; leave math, selection and loss dispatch intact."""
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    nodes = [n for n in nodes if isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name in names]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])), path, "exec"), namespace
    )


class LossConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def losses(monkeypatch):
    spec = importlib.util.spec_from_file_location("corl_batch", ROOT / "verl_omni/trainer/omni/bagel_corl_batch.py")
    batch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_module)
    monkeypatch.setitem(sys.modules, "verl_omni.trainer.omni.bagel_corl_batch", batch_module)
    ns = dict(torch=torch, math=math, tu=tu, TensorDict=TensorDict, Metric=Metric, AggregationType=AggregationType)

    class Base:
        def validate_inputs(self, **kwargs):
            pass

    ns.update(
        DiffusionLossFn=Base,
        _loss_cfg_of=lambda c: c.diffusion_loss,
        DiffusionLossResult=lambda loss, metrics: SimpleNamespace(loss=loss, metrics=metrics, add_loss_metric=False),
    )
    load_nodes("verl_omni/trainer/diffusion/diffusion_algos.py", {"FlowGRPOLoss", "KLLoss"}, ns)
    ns["get_diffusion_loss_fn"] = lambda name: ns["KLLoss" if name == "kl" else "FlowGRPOLoss"]()
    ns["_LANE_SCALAR_TYPES"] = (bool, int, float, str, type(None))
    load_nodes(
        "verl_omni/workers/utils/losses.py",
        {
            "diffusion_loss",
            "bagel_composite_loss",
            "bagel_und_episode_loss",
            "_singleton_loss_tensor",
            "_bagel_corl_view",
            "_inherit_micro_batch_scalars",
            "_first_tensor_device",
            "_align_view_to_device",
            "_bagel_lane_weight",
        },
        ns,
    )
    load_nodes("verl_omni/workers/engine/fsdp/bagel_corl_composite.py", {"materialize_gen_train_batch"}, ns)
    load_nodes("verl_omni/trainer/omni/bagel_corl_gen_adv.py", {"select_gen_advantages_for_step"}, ns)
    ns.update(np=np, DataProto=DataProto, logger=logging.getLogger(__name__))
    load_nodes(
        "verl_omni/trainer/omni/bagel_corl_gen_adv.py",
        {"_logprob_1d", "_as_tensor", "_stack_padded", "_flowgrpo_group_index", "build_gen_flowgrpo_proto"},
        ns,
    )
    unused = ModuleType("verl.workers.utils.losses")
    unused.ppo_loss = lambda *a, **kw: pytest.fail("GEN must never enter UND loss")
    monkeypatch.setitem(sys.modules, "verl.workers.utils.losses", unused)
    return ns


def config():
    return SimpleNamespace(
        diffusion_loss=LossConfig(loss_mode="flow_grpo", clip_ratio=0.02, adv_clip_max=5.0),
        global_batch_info={},
        loss_scale_factor=None,
        use_kl_loss=False,
    )


def tag_gen(data, **metadata):
    tu.assign_non_tensor(
        data,
        bagel_corl_lane="gen",
        has_complete_gen_groups=True,
        skip_gen=False,
        num_gen_rows=len(data),
        sp_size=1,
        **metadata,
    )


def gen_trajectory(parents, advantages, valid):
    rows, steps = valid.shape
    return TensorDict(
        {
            "parent_episode_id": parents,
            "gen_row_id": torch.arange(rows),
            "all_latents": torch.ones(rows, steps + 1, 2),
            "all_timesteps": torch.full((rows, steps), 0.8),
            "gen_valid": torch.ones(rows),
            "gen_transition_valid": valid,
            "old_log_probs": torch.zeros(rows, steps),
            "advantages": advantages,
        },
        batch_size=[rows],
    )


def und_batch(episodes, gen):
    data = TensorDict(
        {
            "bagel_episode_id": episodes,
            "input_ids": torch.ones(len(episodes), 2),
            "old_log_probs": torch.zeros(len(episodes), 1),
            "advantages": torch.ones(len(episodes), 1),
        },
        batch_size=[len(episodes)],
    )
    tu.assign_non_tensor(data, bagel_corl_gen=gen, has_complete_gen_groups=True, skip_gen=False)
    return data


def composite_dispatch(losses, monkeypatch):
    module = ModuleType("verl_omni.workers.engine.fsdp.bagel_corl_composite")
    names = {"composite_forward_mode", "gen_view_has_traj", "is_bagel_corl_composite"}
    load_nodes("verl_omni/workers/engine/fsdp/bagel_corl_composite.py", names, losses)
    for name in names | {"materialize_gen_train_batch"}:
        setattr(module, name, losses[name])
    module.run_und_token_forward_backward = lambda *a: {"loss": [], "metrics": {}}
    module.merge_composite_outputs = lambda parts: parts[-1] if parts else {"loss": [], "metrics": {}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def install_real_ppo(monkeypatch, *, entropy_coeff, kl_loss_coef):
    from verl.workers.config import ActorConfig

    path = Path(importlib.util.find_spec("verl").submodule_search_locations[0]) / "workers/utils/losses.py"
    spec = importlib.util.spec_from_file_location("acceptance_verl_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys.modules["verl.workers.utils.losses"], "ppo_loss", module.ppo_loss)
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        entropy_coeff=entropy_coeff,
        use_kl_loss=True,
        kl_loss_coef=kl_loss_coef,
        kl_loss_type="kl",
    )


@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_each_gen_microbatch_has_its_own_signed_gradient(losses, order):
    view = TensorDict(
        {"old_log_probs": torch.zeros(2, 2), "advantages": torch.tensor([[1.0, 1.0], [-1.0, -1.0]])}, batch_size=[2]
    )[order]
    data = losses["materialize_gen_train_batch"](view, {"num_gen_rows": 2})
    assert tu.get_non_tensor_data(data, "bagel_corl_gen", default=None) is None
    for i, micro in enumerate(tu.chunk_tensordict(data, 2)):
        adv = losses["select_gen_advantages_for_step"](micro, 0, require_bagel_corl_gen=True)
        assert adv.shape == (1,)
        step = TensorDict({"old_log_probs": micro["old_log_probs"][:, 0], "advantages": adv}, batch_size=[1])
        tag_gen(step, gradient_accumulation_steps=4)
        new = torch.zeros(1, requires_grad=True)
        loss, _ = losses["bagel_composite_loss"](config(), {"log_probs": new}, step)
        loss.backward()
        assert new.grad.item() == pytest.approx(-view["advantages"][i, 0].item() / 4)


def test_gen_selector_rejects_untagged_und_advantages(losses):
    data = TensorDict({"advantages": torch.ones(1, 2)}, batch_size=[1])
    with pytest.raises(ValueError, match="GEN|gen"):
        losses["select_gen_advantages_for_step"](data, 0, require_bagel_corl_gen=True)


def test_flowgrpo_rejects_cross_row_or_timestep_broadcast(losses):
    with pytest.raises(ValueError, match="shape"):
        losses["FlowGRPOLoss"].compute_loss(
            old_log_prob=torch.zeros(2, 2), log_prob=torch.zeros(1), advantages=torch.ones(2, 2), config=config()
        )


def test_materialized_gen_uses_sentinel_for_current_timestep(losses):
    view = TensorDict(
        {
            "old_log_probs": torch.tensor([[11.0, 12.0], [21.0, 22.0]]),
            "advantages": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        batch_size=[2],
    )
    data = losses["materialize_gen_train_batch"](view, {})
    micro = tu.chunk_tensordict(data, 2)[1]
    assert losses["select_gen_advantages_for_step"](micro, 1, require_bagel_corl_gen=True).tolist() == [4.0]
    assert micro["old_log_probs"][:, 1].tolist() == [22.0]


@pytest.mark.parametrize("micro_size", [1, 2])
def test_valid_transition_gradient_is_microbatch_invariant(losses, micro_size):
    x = torch.zeros(2, 2, requires_grad=True)
    adv = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    valid = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    for start in range(0, 2, micro_size):
        for step in range(2):
            sl = slice(start, start + micro_size)
            loss, metrics = losses["FlowGRPOLoss"].compute_loss(
                old_log_prob=torch.zeros_like(x[sl, step]),
                log_prob=x[sl, step],
                advantages=adv[sl, step],
                config=config(),
                unit_weights=valid[sl, step],
                loss_normalizer=3.0,
            )
            loss.backward()
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
    torch.testing.assert_close(x.grad, -adv * valid / 3)


def test_masked_nonfinite_transition_cannot_poison_loss(losses):
    x = torch.tensor([0.0, float("nan")], requires_grad=True)
    loss, _ = losses["FlowGRPOLoss"].compute_loss(
        old_log_prob=torch.tensor([0.0, float("nan")]),
        log_prob=x,
        advantages=torch.tensor([2.0, float("nan")]),
        config=config(),
        unit_weights=torch.tensor([1.0, 0.0]),
        loss_normalizer=1.0,
    )
    loss.backward()
    assert loss.item() == -2.0
    torch.testing.assert_close(x.grad, torch.tensor([-2.0, 0.0]))


def test_trajectory_padding_has_explicit_valid_mask_and_safe_timesteps(losses):
    rows = [
        {
            "rollout_log_probs": torch.zeros(t),
            "rm_score": 0.7,
            "gen_group_uid": "call",
            "episode_uid": "episode",
            "gen_sample_uid": str(t),
            "prompt_token_ids": [42],
            "all_latents": torch.ones(t + 1, 2),
            "timesteps": torch.linspace(0.8, 0.4, t),
        }
        for t in [1, 3]
    ]
    proto = losses["build_gen_flowgrpo_proto"](rows)
    assert proto.batch["gen_transition_valid"].tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert (proto.batch["all_timesteps"] > 0).all()
    assert proto.batch["all_latents"].shape == (2, 4, 2)
    bad = dict(rows[0], all_latents=torch.zeros(1, 2))
    with pytest.raises(ValueError, match="transition|latent"):
        losses["build_gen_flowgrpo_proto"]([bad])


def test_optional_gen_reference_means_survive_trajectory_packing(losses):
    rows = [
        dict(
            rollout_log_probs=torch.zeros(t),
            rm_score=0.7,
            gen_group_uid="call",
            episode_uid="episode",
            prompt_token_ids=[42],
            all_latents=torch.ones(t + 1, 2),
            timesteps=torch.full((t,), 0.8),
            ref_prev_sample_mean=torch.full((t, 2), float(t)),
        )
        for t in [1, 3]
    ]
    proto = losses["build_gen_flowgrpo_proto"](rows)
    torch.testing.assert_close(proto.batch["ref_prev_sample_mean"], torch.tensor([[[1.0, 1.0]] * 3, [[3.0, 3.0]] * 3]))
    assert proto.batch["gen_transition_valid"].tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    rows[1].pop("ref_prev_sample_mean")
    with pytest.raises(ValueError, match="reference"):
        losses["build_gen_flowgrpo_proto"](rows)


def test_composite_engine_selects_gen_rows_for_current_und_minibatch(losses, monkeypatch):
    module = composite_dispatch(losses, monkeypatch)
    module.run_und_token_forward_backward = lambda *a: {"loss": [0.0], "metrics": {}}
    module.merge_composite_outputs = lambda parts: parts[-1]
    losses["get_device_id"] = lambda: "cpu"
    load_nodes(
        "verl_omni/workers/engine/fsdp/diffusers_impl.py",
        {"forward_backward_batch"},
        losses,
        class_name="PPODiffusersFSDPEngine",
    )
    gen = TensorDict(
        {
            "parent_episode_id": torch.tensor([10, 20, 20]),
            "gen_row_id": torch.arange(3),
            "all_latents": torch.ones(3, 3, 2),
            "all_timesteps": torch.ones(3, 2),
            "gen_valid": torch.ones(3),
            "gen_transition_valid": torch.ones(3, 2),
        },
        batch_size=[3],
    )
    data = TensorDict(
        {
            "input_ids": torch.ones(1, 2),
            "bagel_episode_id": torch.tensor([20]),
            "old_log_probs": torch.zeros(1, 1),
            "advantages": torch.ones(1, 1),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, bagel_corl_gen=gen, has_complete_gen_groups=True, skip_gen=False)
    captured = []

    def run(batch, *args, **kwargs):
        captured.append(batch)
        return {"loss": [], "metrics": {}}

    engine = SimpleNamespace(
        model_config={"composite_mode": "bagel_corl"},
        module=torch.nn.Linear(1, 1),
        engine_config=SimpleNamespace(micro_batch_size_per_gpu=1),
        get_data_parallel_group=lambda: None,
        _run_forward_backward_batch=run,
    )
    losses["forward_backward_batch"](engine, data, lambda **k: None, False)
    assert captured[0]["gen_row_id"].tolist() == [1, 2]
    assert tu.get_non_tensor_data(captured[0], "loss_normalizer", None) == 4


@pytest.mark.parametrize("micro_size", [1, 2])
def test_und_episode_mean_independent_of_response_lengths(losses, monkeypatch, micro_size):
    def ppo(config, output, data, **kwargs):
        mask = data["response_mask"].float()
        pg = -(output["log_probs"] * mask).sum() / mask.sum()
        return pg, {"actor/pg_loss": Metric(value=pg.detach().item(), aggregation=AggregationType.MEAN)}

    monkeypatch.setattr(sys.modules["verl.workers.utils.losses"], "ppo_loss", ppo)
    x = torch.zeros(2, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1.0]])
    for start in range(0, 2, micro_size):
        sl = slice(start, start + micro_size)
        data = TensorDict(
            {
                "response_mask": mask[sl],
                "old_log_probs": torch.zeros_like(mask[sl]),
                "advantages": torch.ones_like(mask[sl]),
            },
            batch_size=[len(mask[sl])],
        )
        tu.assign_non_tensor(
            data,
            bagel_corl_und=data,
            skip_gen=True,
            has_complete_gen_groups=False,
            gradient_accumulation_steps=2 // micro_size,
            und_episode_normalizer=2.0,
        )
        loss, _ = losses["bagel_composite_loss"](config(), {"und": {"log_probs": x[sl]}}, data)
        loss.backward()
    torch.testing.assert_close(x.grad, -mask / mask.sum(-1, keepdim=True) / 2)


def test_gen_metrics_exclude_padding_and_use_global_unit_count(losses):
    total = 0.0
    for ratio, valid in [(1.01, 1.0), (0.99, 1.0), (1.005, 1.0), (50.0, 0.0)]:
        data = TensorDict(
            {"old_log_probs": torch.zeros(1), "advantages": torch.ones(1), "loss_unit_weight": torch.tensor([valid])},
            batch_size=[1],
        )
        tag_gen(data, corl_loss_normalized=True, loss_normalizer=3.0, corl_dp_size=1, gradient_accumulation_steps=999)
        loss, metrics = losses["bagel_composite_loss"](config(), {"log_probs": torch.tensor([math.log(ratio)])}, data)
        total += float(metrics["actor/gen/ratio_mean"].aggregate())
    assert total == pytest.approx((1.01 + 0.99 + 1.005) / 3)


@pytest.mark.parametrize("micro_size", [1, 3])
def test_und_real_ppo_with_jagged_outputs_kl_entropy_and_rejected_row(losses, monkeypatch, micro_size):
    cfg = install_real_ppo(monkeypatch, entropy_coeff=0.05, kl_loss_coef=0.01)
    prompts = [torch.ones(n, dtype=torch.long) for n in [1, 2, 1]]
    responses = [torch.ones(n, dtype=torch.long) for n in [2, 4, 1]]
    leaves = [torch.zeros(p.numel() + r.numel(), requires_grad=True) for p, r in zip(prompts, responses, strict=True)]
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    for start in range(0, 3, micro_size):
        sl = slice(start, start + micro_size)
        data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor(prompts[sl], layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor(responses[sl], layout=torch.jagged),
                "response_mask": mask[sl],
                "old_log_probs": torch.zeros_like(mask[sl], dtype=torch.float),
                "advantages": torch.ones_like(mask[sl], dtype=torch.float),
                "ref_log_prob": torch.full_like(mask[sl], 0.1, dtype=torch.float),
            },
            batch_size=[len(leaves[sl])],
        )
        tu.assign_non_tensor(
            data,
            max_response_len=4,
            dp_size=1,
            batch_num_tokens=None,
            global_batch_size=None,
            und_episode_normalizer=2.0,
        )
        output = {
            "log_probs": torch.nested.as_nested_tensor(leaves[sl], layout=torch.jagged),
            "entropy": torch.nested.as_nested_tensor([x + 1 for x in leaves[sl]], layout=torch.jagged),
        }
        loss, metrics = losses["bagel_und_episode_loss"](cfg, output, data)
        assert torch.isfinite(loss)
        assert all(
            torch.isfinite(torch.as_tensor(v.aggregate() if isinstance(v, Metric) else v)).all()
            for v in metrics.values()
        )
        loss.backward()
    for i, (leaf, response) in enumerate(zip(leaves, responses, strict=True)):
        expected = torch.zeros_like(leaf)
        if i < 2:
            expected[-response.numel() - 1 : -1] = -1.04 / response.numel() / 2
        torch.testing.assert_close(leaf.grad, expected)


@pytest.mark.parametrize("reference_key", ["ref_log_prob", "ref_prev_sample_mean"])
def test_corl_explicit_gen_kl_requires_reverse_sde_inputs(losses, reference_key):
    data = TensorDict(
        {"old_log_probs": torch.zeros(1), "advantages": torch.ones(1), reference_key: torch.zeros(1)}, batch_size=[1]
    )
    tag_gen(data, gradient_accumulation_steps=1)
    cfg = config()
    cfg.use_kl_loss = True  # UND reference KL remains independently supported.
    cfg.diffusion_loss.use_gen_kl_loss = True
    with pytest.raises(ValueError, match="GEN reference"):
        losses["bagel_composite_loss"](cfg, {"log_probs": torch.zeros(1)}, data)


@pytest.mark.parametrize("micro_size", [1, 3])
def test_gen_reference_kl_masks_padding_and_is_microbatch_invariant(losses, micro_size):
    means = torch.tensor([[[2.0, 4.0]], [[4.0, 6.0]], [[float("nan"), float("nan")]]], requires_grad=True)
    valid = torch.tensor([1.0, 1.0, 0.0])
    sigmas = torch.tensor([1.0, 2.0, 0.0]).reshape(3, 1, 1)
    total = 0.0
    for start in range(0, 3, micro_size):
        sl = slice(start, start + micro_size)
        value, metrics = losses["KLLoss"].compute_loss(
            prev_sample_mean=means[sl],
            ref_prev_sample_mean=torch.zeros_like(means[sl]),
            std_dev_t=sigmas[sl],
            unit_weights=valid[sl],
            loss_normalizer=2.0,
        )
        assert torch.isfinite(value)
        assert math.isfinite(metrics["actor/kl_loss"])
        total += value.item()
        value.backward()
    assert total == pytest.approx((5 + 3.25) / 2)
    torch.testing.assert_close(means.grad, torch.tensor([[[0.5, 1.0]], [[0.25, 0.375]], [[0.0, 0.0]]]))


def test_gen_reference_switch_is_independent_from_und_kl(losses):
    cfg = config()
    cfg.use_kl_loss = False
    cfg.kl_loss_coef = 0.1
    cfg.diffusion_loss.use_gen_kl_loss = True
    data = TensorDict(
        {
            "old_log_probs": torch.zeros(1),
            "advantages": torch.ones(1),
            "ref_prev_sample_mean": torch.zeros(1, 1, 1),
            "loss_unit_weight": torch.ones(1),
        },
        batch_size=[1],
    )
    tag_gen(data, corl_loss_normalized=True, loss_normalizer=1.0, corl_dp_size=1)
    lp = torch.zeros(1, requires_grad=True)
    mean = torch.full((1, 1, 1), 2.0, requires_grad=True)
    value, metrics = losses["bagel_composite_loss"](
        cfg,
        {"log_probs": lp, "prev_sample_mean": mean, "std_dev_t": torch.ones(1, 1, 1)},
        data,
    )
    assert value.item() == pytest.approx(-0.8)
    assert metrics["actor/gen/pg_loss"].aggregate() == pytest.approx(-1.0)
    assert metrics["actor/gen/ref_kl"].aggregate() == pytest.approx(2.0)
    value.backward()
    torch.testing.assert_close(mean.grad, torch.full_like(mean, 0.2))


def test_und_kl_does_not_request_gen_reference_inputs(losses):
    cfg = config()
    cfg.use_kl_loss = True
    data = TensorDict({"old_log_probs": torch.zeros(1), "advantages": torch.ones(1)}, batch_size=[1])
    tag_gen(data, gradient_accumulation_steps=1)
    value, _ = losses["bagel_composite_loss"](cfg, {"log_probs": torch.zeros(1)}, data)
    assert value == -1


@pytest.mark.parametrize("bad", ["mean", "sigma", "weight", "normalizer"])
def test_gen_reference_rejects_invalid_valid_units(losses, bad):
    mean = torch.ones(1, 1, 2)
    sigma, weight, denominator = torch.ones(1, 1, 1), torch.ones(1), 1.0
    if bad == "mean":
        mean.fill_(float("nan"))
    elif bad == "sigma":
        sigma.zero_()
    elif bad == "weight":
        weight.fill_(float("nan"))
    else:
        denominator = 0.0
    with pytest.raises(ValueError, match="GEN reference"):
        losses["KLLoss"].compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=torch.zeros_like(mean),
            std_dev_t=sigma,
            unit_weights=weight,
            loss_normalizer=denominator,
        )


def test_unweighted_reference_kl_retains_legacy_mean(losses):
    mean = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]], requires_grad=True)
    sigma = torch.tensor([1.0, 2.0]).reshape(2, 1, 1)
    value, _ = losses["KLLoss"].compute_loss(
        prev_sample_mean=mean, ref_prev_sample_mean=torch.zeros_like(mean), std_dev_t=sigma
    )
    expected = (mean.square().mean((1, 2), keepdim=True) / (2 * sigma.square())).mean()
    torch.testing.assert_close(value, expected)


def test_corl_cfg_gates_each_rows_timestep_with_equal_forward_counts(losses):
    load_nodes(
        "verl_omni/pipelines/bagel_flow_grpo/diffusers_training_adapter.py",
        {"forward_and_sample_previous_step"},
        losses,
        class_name="BagelDiffusion",
    )
    cfg = LossConfig(composite_mode="bagel_corl", algo=SimpleNamespace(noise_level=0.8, sde_type="sde"))
    calls = []

    def model(**kwargs):
        calls.append(kwargs["cond"])
        return (torch.full((2, 1), 1.0 if kwargs["cond"] else 0.0),)

    class Adapter:
        _get_cfg_params = staticmethod(
            lambda c: {
                "cfg_interval_low": 0.5,
                "cfg_interval_high": 1.0,
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.0,
                "cfg_renorm_type": "global",
                "cfg_renorm_min": 0.0,
            }
        )
        _combine_cfg = staticmethod(lambda **kwargs: kwargs["v_t"] * 4)

    scheduler = SimpleNamespace(sample_previous_step=lambda **kw: (None, kw["model_output"], None, None, None))
    for times, expected in [([0.8, 0.2], [4.0, 1.0]), ([0.2, 0.2], [1.0, 1.0])]:
        inputs = {"all_latents": torch.ones(2, 2, 1), "all_timesteps": torch.tensor(times)[:, None]}
        fn = losses["forward_and_sample_previous_step"].__func__
        result = fn(Adapter, model, scheduler, cfg, {"cond": True}, {"cond": False}, inputs, 0)
        assert result[0].flatten().tolist() == expected
    assert calls == [True, False, True, False]
