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
"""CPU tests for the DGPO loss, its actor-batch preparation and the SD3 DGPO rollout loop.

Necessity: DGPO weights every sample by a sigmoid of its whole group's preference score,
so a silently split group, an unshared timestep or noise, or a stochastic rollout step
changes the objective without any error. These tests pin the loss to an independent
implementation of the paper formula and check each of those invariants directly.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import DGPOLoss, get_diffusion_loss_fn
from verl_omni.trainer.diffusion.diffusion_trainer_utils import old_policy_decay
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.utils.losses import diffusion_loss

C, H, W = 4, 3, 3


def make_actor(**loss_kwargs) -> DiffusionActorConfig:
    loss_cfg = DiffusionLossConfig(loss_mode="dgpo", **loss_kwargs)
    return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=loss_cfg)


def make_inputs(group_index, seed=0, t=0.6):
    generator = torch.Generator().manual_seed(seed)
    batch = len(group_index)
    x0 = torch.randn(batch, C, H, W, generator=generator)
    noise = torch.randn(batch, C, H, W, generator=generator)
    t_expanded = torch.full((batch, 1, 1, 1), t)
    xt = (1 - t_expanded) * x0 + t_expanded * noise
    target = noise - x0

    def prediction(scale):
        return target + scale * torch.randn(batch, C, H, W, generator=generator)

    return {
        "x0": x0,
        "xt": xt,
        "t_expanded": t_expanded,
        "target": target,
        "forward_prediction": prediction(0.5).requires_grad_(True),
        "old_prediction": prediction(0.5),
        "ref_forward_prediction": prediction(0.5),
    }


def reference_dgpo_loss(inputs, advantages, group_index, beta, clip_range, ref_kl_coef, adv_clip_max=5.0):
    """Direct transcription of the DGPO objective, one group at a time."""
    target = inputs["target"]
    advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
    dsm = ((target - inputs["forward_prediction"]) ** 2).flatten(1).mean(1)
    ref_dsm = ((target - inputs["ref_forward_prediction"]) ** 2).flatten(1).mean(1)
    old_dsm = ((target - inputs["old_prediction"]) ** 2).flatten(1).mean(1)
    weights = torch.empty_like(dsm)
    for group in set(group_index.tolist()):
        members = [i for i, g in enumerate(group_index.tolist()) if g == group]
        score = sum(advantages[i] * beta * (dsm[i].detach() - ref_dsm[i]) / len(members) for i in members)
        for i in members:
            weights[i] = torch.sigmoid(score)
    terms = []
    for i in range(len(dsm)):
        ratio = torch.exp(old_dsm[i] - dsm[i].detach())
        clipped = clip_range > 0 and (
            (advantages[i] > 0 and ratio > 1 + clip_range) or (advantages[i] < 0 and ratio < 1 - clip_range)
        )
        terms.append(weights[i] * advantages[i] * (dsm[i].detach() if clipped else dsm[i]))
    policy = torch.stack(terms).mean()
    ref_kl = ((inputs["forward_prediction"] - inputs["ref_forward_prediction"]) ** 2).mean()
    return policy + ref_kl_coef * ref_kl


def run_loss(inputs, advantages, group_index, group_size, **loss_kwargs):
    model_output = {key: inputs[key] for key in DGPOLoss.required_model_output_keys}
    data = TensorDict(
        {"advantages": advantages, "group_index": group_index, "group_size": group_size},
        batch_size=[len(group_index)],
    )
    return get_diffusion_loss_fn("dgpo")(config=make_actor(**loss_kwargs), model_output=model_output, data=data)


@pytest.mark.parametrize("clip_range", [0.0, 0.01])
@pytest.mark.parametrize("ref_kl_coef", [0.0, 0.02])
def test_dgpo_loss_and_gradient_match_the_paper_formula(clip_range, ref_kl_coef):
    group_index = torch.tensor([0, 0, 0, 1, 1, 1])
    group_size = torch.full((6,), 3)
    advantages = torch.tensor([1.2, -0.4, -0.8, 0.3, 0.9, -1.2])
    inputs = make_inputs(group_index)

    result = run_loss(
        inputs,
        advantages,
        group_index,
        group_size,
        dgpo_beta=100.0,
        dgpo_clip_range=clip_range,
        ref_kl_coef=ref_kl_coef,
    )
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])

    expected = reference_dgpo_loss(inputs, advantages, group_index, 100.0, clip_range, ref_kl_coef)
    (expected_grad,) = torch.autograd.grad(expected, inputs["forward_prediction"])
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(grad, expected_grad)
    assert 0.0 < result.metrics["actor/group_weight_dev"] < 0.5


def test_dgpo_target_is_recovered_from_xt():
    """The loss derives noise - x0 from (xt - x0) / t; a perfect prediction has zero error."""
    group_index = torch.tensor([0, 0])
    inputs = make_inputs(group_index, t=0.3)
    inputs["forward_prediction"] = inputs["target"].clone().requires_grad_(True)
    result = run_loss(inputs, torch.tensor([1.0, -1.0]), group_index, torch.full((2,), 2), ref_kl_coef=0.0)
    assert result.metrics["actor/dsm_loss"] == pytest.approx(0.0, abs=1e-10)


def test_dgpo_group_weight_value():
    """One group of two: the weight is sigmoid(sum A * beta * (dsm - ref_dsm) / 2)."""
    group_index = torch.tensor([0, 0])
    inputs = make_inputs(group_index)
    target = inputs["target"]
    inputs["forward_prediction"] = (target + 0.2).requires_grad_(True)  # dsm = 0.04 for both
    inputs["ref_forward_prediction"] = torch.stack([target[0] + 0.1, target[1] + 0.3])  # ref_dsm = 0.01, 0.09
    advantages = torch.tensor([1.0, -1.0])
    result = run_loss(inputs, advantages, group_index, torch.full((2,), 2), dgpo_beta=10.0, dgpo_clip_range=0.0)
    score = (1.0 * 10.0 * (0.04 - 0.01) + (-1.0) * 10.0 * (0.04 - 0.09)) / 2
    assert result.metrics["actor/group_weight_mean"] == pytest.approx(
        torch.sigmoid(torch.tensor(score)).item(), rel=1e-5
    )


def test_dgpo_clip_detaches_samples_outside_the_trust_region():
    group_index = torch.tensor([0, 0])
    inputs = make_inputs(group_index)
    target = inputs["target"]
    inputs["forward_prediction"] = (target + 0.1).requires_grad_(True)  # dsm = 0.01
    # Sample 0 (A > 0): old_dsm = 0.25 -> ratio exp(0.24) > 1.01 -> clipped.
    # Sample 1 (A < 0): old_dsm = 0.01 -> ratio 1 -> kept.
    inputs["old_prediction"] = torch.stack([target[0] + 0.5, target[1] + 0.1])
    result = run_loss(
        inputs, torch.tensor([1.0, -1.0]), group_index, torch.full((2,), 2), dgpo_clip_range=0.01, ref_kl_coef=0.0
    )
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
    assert result.metrics["actor/clip_frac"] == pytest.approx(0.5)
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[1]) > 0


def test_dgpo_rejects_a_group_split_across_micro_batches():
    group_index = torch.tensor([0, 0, 1, 1])
    inputs = make_inputs(group_index)
    with pytest.raises(ValueError, match="whole inside one micro batch"):
        run_loss(inputs, torch.tensor([1.0, -1.0, 0.5, -0.5]), group_index, torch.full((4,), 4))


def test_dgpo_through_the_worker_loss_entry_point():
    """The worker-side `diffusion_loss` dispatch reaches DGPO and divides by gradient accumulation."""
    group_index = torch.tensor([0, 0, 1, 1])
    group_size = torch.full((4,), 2)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    inputs = make_inputs(group_index)
    data = TensorDict({"advantages": advantages, "group_index": group_index, "group_size": group_size}, batch_size=[4])
    tu.assign_non_tensor(data, gradient_accumulation_steps=4, sp_size=1)
    model_output = {key: inputs[key] for key in DGPOLoss.required_model_output_keys}
    loss, metrics = diffusion_loss(make_actor(), model_output, data)
    direct = run_loss(inputs, advantages, group_index, group_size)
    torch.testing.assert_close(loss, direct.loss / 4)
    assert "actor/group_weight_mean" in metrics


def make_rollout_batch(uid, num_steps=10):
    batch_size = len(uid)
    schedule = torch.linspace(1000, 100, num_steps).round().long()
    return DataProto.from_dict(
        tensors={
            "latents_clean": torch.randn(batch_size, C, H, W),
            "all_timesteps": schedule.expand(batch_size, -1).clone(),
            "sample_level_scores": torch.arange(batch_size, dtype=torch.float32),
        },
        non_tensors={"uid": np.array(uid, dtype=object)},
    )


def make_config(**algorithm_kwargs):
    algorithm = SimpleNamespace(norm_adv_by_std_in_grpo=True, global_std=True, timestep_fraction=1.0)
    for key, value in algorithm_kwargs.items():
        setattr(algorithm, key, value)
    return SimpleNamespace(algorithm=algorithm)


def test_prepare_dgpo_actor_batch_groups_timesteps_and_noise():
    uid = ["b", "a", "b", "a", "c", "c"]
    batch = make_rollout_batch(uid)
    rewards = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.2, 0.8])
    scores_by_uid = {(u, float(r)) for u, r in zip(uid, rewards, strict=True)}

    torch.manual_seed(0)
    result = DGPOLoss.prepare_actor_batch(
        batch, rewards, make_config(train_timestep_range=[0, 7], train_timestep_count=4)
    )

    out_uid = list(result.non_tensor_batch["uid"])
    # Groups are numbered by first appearance and made contiguous.
    assert out_uid == ["b", "b", "a", "a", "c", "c"]
    # Rows moved together with their rewards.
    assert {
        (u, float(r)) for u, r in zip(out_uid, result.batch["sample_level_rewards"][:, 0], strict=True)
    } == scores_by_uid
    assert result.batch["group_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert result.batch["group_size"].tolist() == [2] * 6

    train_timesteps = result.batch["train_timesteps"]
    schedule = make_rollout_batch(uid).batch["all_timesteps"][0]
    assert train_timesteps.shape == (6, 4)
    assert (train_timesteps == train_timesteps[:1]).all()
    assert len(set(train_timesteps[0].tolist())) == 4
    assert set(train_timesteps[0].tolist()) <= set(schedule[:7].tolist())

    noise = result.batch["forward_noise"]
    assert noise.shape == (6, 4, C, H, W)
    for first, second in ((0, 1), (2, 3), (4, 5)):
        torch.testing.assert_close(noise[first], noise[second], rtol=0, atol=0)
    assert not torch.equal(noise[0], noise[2])

    advantages = result.batch["advantages"][:, 0]
    for group in range(3):
        assert advantages[result.batch["group_index"] == group].sum().abs() < 1e-5


def test_prepare_dgpo_actor_batch_defaults_to_the_full_schedule():
    batch = make_rollout_batch(["a", "a"], num_steps=6)
    result = DGPOLoss.prepare_actor_batch(batch, torch.tensor([0.0, 1.0]), make_config())
    assert sorted(result.batch["train_timesteps"][0].tolist()) == sorted(batch.batch["all_timesteps"][0].tolist())


def test_dgpo_config_validation():
    with pytest.raises(ValueError, match="dgpo_beta"):
        DiffusionLossConfig(loss_mode="dgpo", dgpo_beta=0.0)
    with pytest.raises(ValueError, match="dgpo_clip_range"):
        DiffusionLossConfig(loss_mode="dgpo", dgpo_clip_range=-0.1)
    with pytest.raises(ValueError, match="train_timestep_range"):
        DiffusionAlgoConfig(train_timestep_range=[3, 3])
    with pytest.raises(ValueError, match="train_timestep_count"):
        DiffusionAlgoConfig(train_timestep_count=0)


def test_linear_to_0_3_matches_the_dgpo_reference_ema():
    assert [old_policy_decay(step, "linear_to_0_3") for step in (0, 100, 300, 1000)] == pytest.approx(
        [0.0, 0.1, 0.3, 0.3]
    )


def test_sd3_dgpo_rollout_is_a_deterministic_euler_ode():
    """The DGPO rollout forces noise level 0 and drops the per-step trajectory."""
    pytest.importorskip("vllm_omni.diffusion.models.sd3.pipeline_sd3")
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
    from verl_omni.pipelines.sd3_dgpo import StableDiffusion3DGPOPipeline

    scheduler = FlowMatchSDEDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(5)
    # Bypass nn.Module.__init__: only the attributes the diffusion loop reads are needed.
    pipeline = object.__new__(StableDiffusion3DGPOPipeline)
    for name, value in {
        "scheduler": scheduler,
        "device": torch.device("cpu"),
        "_interrupt": False,
        "od_config": SimpleNamespace(dtype=torch.float32),
    }.items():
        object.__setattr__(pipeline, name, value)

    def velocity(hidden_states, timestep):
        return torch.tanh(hidden_states) * (timestep.view(-1, 1, 1, 1) / 1000.0)

    object.__setattr__(
        pipeline,
        "predict_noise_maybe_with_cfg",
        lambda do_cfg, scale, positive, negative, *_: velocity(positive["hidden_states"], positive["timestep"]),
    )

    start = torch.randn(2, C, H, W, generator=torch.Generator().manual_seed(1))
    latents, all_latents, all_log_probs, all_timesteps = pipeline.diffuse(
        None,
        None,
        None,
        None,
        start.clone(),
        scheduler.timesteps,
        False,
        1.0,
        noise_level=0.7,
        sde_window=(0, 2),
        sde_type="cps",
        generator=None,
        logprobs=True,
    )

    expected = start.clone()
    for i, timestep in enumerate(scheduler.timesteps):
        expected = expected + velocity(expected, timestep.expand(2)) * (scheduler.sigmas[i + 1] - scheduler.sigmas[i])
    torch.testing.assert_close(latents, expected)
    assert all_latents is None and all_log_probs is None
    torch.testing.assert_close(all_timesteps, scheduler.timesteps.expand(2, -1))


def test_dgpo_batch_through_the_nft_engine_step():
    """Prepared DGPO tensors reach the loss through the real engine step and SD3 DGPO adapter."""
    from contextlib import nullcontext
    from functools import partial

    from verl_omni.workers.engine.fsdp import diffusers_impl

    uid = ["a", "a", "b", "b"]
    batch = make_rollout_batch(uid, num_steps=6)
    torch.manual_seed(0)
    batch = DGPOLoss.prepare_actor_batch(
        batch, torch.tensor([0.2, 0.8, 0.5, 0.1]), make_config(train_timestep_range=[0, 4], train_timestep_count=2)
    )
    micro_batch = batch.batch.select(
        "latents_clean", "train_timesteps", "forward_noise", "advantages", "group_index", "group_size"
    )
    micro_batch["prompt_embeds"] = torch.randn(4, 5, 8)
    micro_batch["prompt_embeds_mask"] = torch.ones(4, 5, dtype=torch.int64)
    micro_batch["pooled_prompt_embeds"] = torch.randn(4, 8)
    tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=1, sp_size=1)

    engine = object.__new__(diffusers_impl.NFTDiffusersFSDPEngine)
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    engine.model_config = SimpleNamespace(
        architecture="StableDiffusion3Pipeline",
        algorithm="dgpo",
        external_lib=None,
        pipeline=SimpleNamespace(guidance_scale=4.5),
    )
    weight = torch.nn.Parameter(torch.tensor(0.5))
    calls = []

    def module(hidden_states, encoder_hidden_states, pooled_projections, timestep, joint_attention_kwargs):
        calls.append(timestep)
        return (weight * hidden_states + pooled_projections.mean() * 0.0,)

    engine.module = module
    engine.use_adapter = lambda name: nullcontext()
    engine.disable_adapter = nullcontext
    engine._set_adapter = lambda name: None

    seen = []

    def loss_function(model_output, data, dp_group):
        seen.append((model_output, data))
        return partial(diffusion_loss, config=make_actor())(model_output=model_output, data=data, dp_group=dp_group)

    step = 1
    loss, output = engine.forward_step(micro_batch, loss_function=loss_function, forward_only=False, step=step)
    loss.backward()

    model_output, data = seen[0]
    torch.testing.assert_close(data["advantages"], micro_batch["advantages"][:, step])
    assert data["group_index"].tolist() == [0, 0, 1, 1]
    assert data["group_size"].tolist() == [2, 2, 2, 2]
    t = micro_batch["train_timesteps"][:, step].float() / 1000
    expected_xt = (1 - t.view(-1, 1, 1, 1)) * micro_batch["latents_clean"] + t.view(-1, 1, 1, 1) * micro_batch[
        "forward_noise"
    ][:, step]
    torch.testing.assert_close(model_output["xt"], expected_xt)
    # Old, current and reference passes, each without a CFG branch.
    assert len(calls) == 3
    assert weight.grad is not None and torch.isfinite(weight.grad)
    assert "actor/group_weight_mean" in output["metrics"]


@pytest.mark.parametrize("loss_mode,expected_shuffle", [("dgpo", False), ("diffusion_nft", True)])
@pytest.mark.parametrize("trainer_version", ["v0", "v1"])
def test_actor_update_keeps_dgpo_groups_unshuffled(loss_mode, expected_shuffle, trainer_version):
    from unittest.mock import MagicMock

    from omegaconf import OmegaConf

    from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    config = OmegaConf.create(
        {
            "algorithm": {"paired_preference": False},
            "actor_rollout_ref": {
                "model": {"pipeline": {"height": 64, "width": 64}, "vae_scale_factor": 8},
                "actor": {
                    "ppo_mini_batch_size": 1,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": True,
                    "diffusion_loss": {"loss_mode": loss_mode},
                },
                "rollout": {"n": 2, "multi_turn": {"enable": False}},
            },
        }
    )
    actor = MagicMock()
    actor.update_actor.return_value = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})
    batch = DataProto.from_dict(tensors={"latents_clean": torch.zeros(2, C, H, W)})
    if trainer_version == "v0":
        trainer = DirectPreferenceRayTrainer.__new__(DirectPreferenceRayTrainer)
        trainer.config, trainer.actor_rollout_wg = config, actor
        trainer._update_actor(batch)
    else:
        trainer = SimpleNamespace(config=config, actor_rollout_wg=actor, _is_direct_preference=True)
        PolicyGradientDiffusionTrainerV1._update_actor(trainer, batch)

    sent = actor.update_actor.call_args.args[0]
    assert tu.get_non_tensor_data(sent, "dataloader_kwargs", default=None) == {"shuffle": expected_shuffle}


def test_prepare_dgpo_actor_batch_keeps_contiguous_groups_in_place():
    """Already-contiguous rollouts are not permuted, so driver-side per-row extras stay aligned."""
    uid = ["b", "b", "a", "a"]
    batch = make_rollout_batch(uid)
    latents = batch.batch["latents_clean"].clone()
    result = DGPOLoss.prepare_actor_batch(batch, torch.tensor([0.1, 0.9, 0.3, 0.5]), make_config())
    assert list(result.non_tensor_batch["uid"]) == uid
    assert result.batch["group_index"].tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(result.batch["latents_clean"], latents, rtol=0, atol=0)


def test_prepare_dgpo_actor_batch_rejects_mixed_schedules():
    batch = make_rollout_batch(["a", "a"])
    batch.batch["all_timesteps"][1, 0] += 1
    with pytest.raises(ValueError, match="same schedule"):
        DGPOLoss.prepare_actor_batch(batch, torch.tensor([0.0, 1.0]), make_config())


def test_dgpo_rejects_a_training_timestep_that_floors_to_zero():
    """A sub-unit schedule value would floor to t=0 and make the (x_t - x_0) / t target NaN."""
    schedule = torch.tensor([[900.0, 0.5], [900.0, 0.5]])
    with pytest.raises(ValueError, match="must be >= 1"):
        DGPOLoss._select_shared_timesteps(schedule, [1, 2], 1, 1.0)
    assert DGPOLoss._select_shared_timesteps(schedule, [0, 1], 1, 1.0).tolist() == [[900], [900]]


def make_trainer_config(**overrides):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "algorithm": {"train_timestep_range": None, "train_timestep_count": None},
            "data": {"train_batch_size": 8},
            "trainer": {
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "use_v1": True,
                "v1": {"trainer_mode": "sync", "sampler": {"drop_incomplete_groups": True}},
            },
            "actor_rollout_ref": {
                "model": {"lora_rank": 32, "policy_state_adapters": ["default", "old"]},
                "rollout": {"n": 8, "rollout_adapter": "old"},
                "actor": {
                    "ppo_mini_batch_size": 4,
                    "ppo_micro_batch_size_per_gpu": 8,
                    "use_dynamic_bsz": False,
                    "use_kl_loss": False,
                    "enable_timestep_staging": False,
                    "fsdp_config": {"ulysses_sequence_parallel_size": 1},
                },
            },
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    return config


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({}, None),
        ({"trainer.use_v1": False, "trainer.v1.sampler.drop_incomplete_groups": False}, None),
        ({"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 4}, "multiple of rollout.n"),
        ({"trainer.n_gpus_per_node": 8}, "ppo_mini_batch_size=4 must be divisible by dp size 8"),
        ({"trainer.n_gpus_per_node": 4, "data.train_batch_size": 6}, "6 prompts per actor update"),
        ({"actor_rollout_ref.actor.use_dynamic_bsz": True}, "use_dynamic_bsz"),
        ({"actor_rollout_ref.model.lora_rank": 0}, "lora_rank must be > 0"),
        ({"actor_rollout_ref.model.policy_state_adapters": ["default"]}, 'must include "old"'),
        ({"actor_rollout_ref.rollout.rollout_adapter": "default"}, "rollout_adapter must be old"),
        ({"actor_rollout_ref.actor.use_kl_loss": True}, "use_kl_loss must be False"),
        ({"actor_rollout_ref.actor.enable_timestep_staging": True}, "enable_timestep_staging must be False"),
        ({"trainer.v1.sampler.drop_incomplete_groups": False}, "drop_incomplete_groups must be True"),
    ],
)
def test_dgpo_trainer_config_requires_whole_groups(overrides, message):
    config = make_trainer_config(**overrides)
    if message is None:
        DGPOLoss.validate_trainer_config(config)
    else:
        with pytest.raises(ValueError, match=message):
            DGPOLoss.validate_trainer_config(config)


def test_other_losses_reject_the_dgpo_timestep_knobs():
    config = make_trainer_config(**{"algorithm.train_timestep_count": 4})
    with pytest.raises(ValueError, match="only used by the dgpo loss"):
        get_diffusion_loss_fn("diffusion_nft").validate_trainer_config(config)
    get_diffusion_loss_fn("diffusion_nft").validate_trainer_config(make_trainer_config())


DGPO_TRAINER_OVERRIDES = [
    "algorithm.trainer_type=direct_preference",
    "algorithm.sample_source=online",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=dgpo",
    "actor_rollout_ref.model.lora_rank=8",
    "actor_rollout_ref.model.policy_state_adapters=[default,old]",
    "actor_rollout_ref.rollout.rollout_adapter=old",
    "actor_rollout_ref.rollout.n=2",
    "actor_rollout_ref.actor.ppo_mini_batch_size=1",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
    "data.train_batch_size=1",
    "trainer.n_gpus_per_node=1",
    "trainer.v1.sampler.drop_incomplete_groups=true",
]


def make_dgpo_v1_trainer(extra_overrides=()):
    import os

    from hydra import compose, initialize_config_dir

    import verl_omni
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    config_dir = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=DGPO_TRAINER_OVERRIDES + list(extra_overrides))
    return PolicyGradientDiffusionTrainerV1Sync(config)


def test_v1_trainer_init_accepts_a_dgpo_config():
    """The real trainer accepts DGPO with the old adapter and loads the DGPO loss."""
    trainer = make_dgpo_v1_trainer()
    assert trainer._is_direct_preference
    assert trainer._has_old_adapter
    assert isinstance(trainer._loss_fn, DGPOLoss)


@pytest.mark.parametrize(
    "extra,message",
    [
        (["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"], "multiple of rollout.n"),
        (["actor_rollout_ref.model.policy_state_adapters=[default]"], 'must include "old"'),
        (["trainer.v1.sampler.drop_incomplete_groups=false"], "drop_incomplete_groups must be True"),
    ],
)
def test_v1_trainer_init_rejects_unsupported_dgpo_configs(extra, message):
    with pytest.raises(ValueError, match=message):
        make_dgpo_v1_trainer(extra)
