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
"""CPU tests for the AWM (Advantage Weighted Matching) loss and its actor-batch preparation.

Necessity: AWM turns the flow-matching error into a log-probability surrogate and feeds it
through a PPO-style clipped ratio, so a wrong sign, a clip that never binds or an EMA KL
term that is not zero when ``old == default`` would silently train the wrong objective.
These tests pin the loss to a transcription of the formula and check each property directly.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_algos import AWMLoss, get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.utils.losses import diffusion_loss

C, H, W = 4, 3, 3


def make_actor(**loss_kwargs) -> DiffusionActorConfig:
    """Build an actor config with the AWM loss and the given loss overrides."""
    loss_cfg = DiffusionLossConfig(loss_mode="awm", **loss_kwargs)
    return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=loss_cfg)


def make_inputs(batch, seed=0, t=0.6):
    """Build a forward-process batch: clean latents, noised latents, target velocity and three predictions."""
    generator = torch.Generator().manual_seed(seed)
    x0 = torch.randn(batch, C, H, W, generator=generator)
    noise = torch.randn(batch, C, H, W, generator=generator)
    t_expanded = torch.full((batch, 1, 1, 1), t)
    xt = (1 - t_expanded) * x0 + t_expanded * noise
    target = noise - x0

    def prediction(scale):
        """Return the target velocity plus Gaussian noise of the given scale."""
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


def reference_awm_loss(inputs, advantages, clip_range, awm_kl_coef, ref_kl_coef, adv_clip_max=5.0):
    """Direct transcription of the AWM objective, one sample at a time."""
    target = inputs["target"]
    advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
    dsm = ((target - inputs["forward_prediction"]) ** 2).flatten(1).mean(1)
    old_dsm = ((target - inputs["old_prediction"]) ** 2).flatten(1).mean(1)
    terms = []
    for i in range(len(dsm)):
        ratio = torch.exp(old_dsm[i] - dsm[i])  # exp(log p - log p_old) with log p = -dsm
        if clip_range > 0:
            clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            terms.append(torch.maximum(-advantages[i] * ratio, -advantages[i] * clipped))
        else:
            terms.append(-advantages[i] * ratio)
    policy = torch.stack(terms).mean()
    ema_kl = ((inputs["forward_prediction"] - inputs["old_prediction"]) ** 2).mean()
    ref_kl = ((inputs["forward_prediction"] - inputs["ref_forward_prediction"]) ** 2).mean()
    return policy + awm_kl_coef * ema_kl + ref_kl_coef * ref_kl


def run_loss(inputs, advantages, **loss_kwargs):
    """Run the registered AWM loss on the synthetic inputs and advantages."""
    model_output = {key: inputs[key] for key in AWMLoss.required_model_output_keys}
    data = TensorDict({"advantages": advantages}, batch_size=[len(advantages)])
    return get_diffusion_loss_fn("awm")(config=make_actor(**loss_kwargs), model_output=model_output, data=data)


@pytest.mark.parametrize("clip_range", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("awm_kl_coef,ref_kl_coef", [(0.0, 0.0), (0.1, 0.02)])
def test_awm_loss_and_gradient_match_the_formula(clip_range, awm_kl_coef, ref_kl_coef):
    """The loss value and its gradient match the reference transcription for every clip and KL setting."""
    advantages = torch.tensor([1.2, -0.4, -0.8, 0.3, 0.9, -1.2])
    inputs = make_inputs(len(advantages))

    result = run_loss(inputs, advantages, awm_clip_range=clip_range, awm_kl_coef=awm_kl_coef, ref_kl_coef=ref_kl_coef)
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])

    expected = reference_awm_loss(inputs, advantages, clip_range, awm_kl_coef, ref_kl_coef)
    (expected_grad,) = torch.autograd.grad(expected, inputs["forward_prediction"])
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(grad, expected_grad)
    assert result.metrics["actor/ema_kl_loss"] > 0


def test_awm_target_is_recovered_from_xt():
    """The loss derives noise - x0 from (xt - x0) / t; a perfect prediction has zero error."""
    inputs = make_inputs(2, t=0.3)
    inputs["forward_prediction"] = inputs["target"].clone().requires_grad_(True)
    result = run_loss(inputs, torch.tensor([1.0, -1.0]), awm_kl_coef=0.0, ref_kl_coef=0.0)
    assert result.metrics["actor/dsm_loss"] == pytest.approx(0.0, abs=1e-10)


def test_awm_advantage_sign_sets_the_gradient_direction():
    """Positive advantages pull the velocity toward the target; negative ones push it away."""
    inputs = make_inputs(2)
    inputs["old_prediction"] = inputs["forward_prediction"].detach().clone()  # ratio = 1 for both samples
    result = run_loss(inputs, torch.tensor([1.0, -1.0]), awm_clip_range=1.0, awm_kl_coef=0.0, ref_kl_coef=0.0)
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
    error = inputs["forward_prediction"].detach() - inputs["target"]
    # A gradient step (-grad) shrinks the error where A > 0 and grows it where A < 0.
    assert (grad[0] * error[0]).sum() > 0
    assert (grad[1] * error[1]).sum() < 0
    assert result.metrics["actor/ratio_mean"] == pytest.approx(1.0)
    assert result.metrics["actor/clip_frac"] == 0.0


def test_awm_clip_removes_the_policy_gradient_outside_the_trust_region():
    """A sample whose ratio left the trust region in the favourable direction contributes no policy gradient."""
    inputs = make_inputs(2)
    target = inputs["target"]
    inputs["forward_prediction"] = (target + 0.1).requires_grad_(True)  # dsm = 0.01
    # Sample 0 (A > 0): old_dsm = 0.25 -> ratio exp(0.24) > 1.01 -> clipped, constant surrogate.
    # Sample 1 (A < 0): old_dsm = 0.01 -> ratio 1 -> kept.
    inputs["old_prediction"] = torch.stack([target[0] + 0.5, target[1] + 0.1])
    result = run_loss(inputs, torch.tensor([1.0, -1.0]), awm_clip_range=0.01, awm_kl_coef=0.0, ref_kl_coef=0.0)
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
    assert result.metrics["actor/clip_frac"] == pytest.approx(0.5)
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[1]) > 0


def test_awm_zero_clip_range_disables_clipping():
    """`awm_clip_range=0` keeps the plain ratio surrogate: favourable moves still get a gradient."""
    inputs = make_inputs(2)
    target = inputs["target"]
    # Both samples moved favourably past any clip bound: dsm = [0.01, 0.25] vs old_dsm = [0.25, 0.01], so the
    # ratio is exp(0.24) where A > 0 and exp(-0.24) where A < 0.
    inputs["forward_prediction"] = torch.stack([target[0] + 0.1, target[1] + 0.5]).requires_grad_(True)
    inputs["old_prediction"] = torch.stack([target[0] + 0.5, target[1] + 0.1])
    result = run_loss(inputs, torch.tensor([1.0, -1.0]), awm_clip_range=0.0, awm_kl_coef=0.0, ref_kl_coef=0.0)
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
    assert result.metrics["actor/clip_frac"] == 0.0
    assert torch.count_nonzero(grad[0]) > 0
    assert torch.count_nonzero(grad[1]) > 0
    clipped = run_loss(inputs, torch.tensor([1.0, -1.0]), awm_clip_range=0.01, awm_kl_coef=0.0, ref_kl_coef=0.0)
    assert clipped.metrics["actor/clip_frac"] == pytest.approx(1.0)


def test_awm_ema_kl_is_zero_when_policy_matches_the_ema_adapter():
    """With old == default the EMA KL vanishes, the ratio is one and a zero advantage gives no gradient."""
    inputs = make_inputs(3)
    inputs["old_prediction"] = inputs["forward_prediction"].detach().clone()
    result = run_loss(inputs, torch.zeros(3), awm_kl_coef=0.5, ref_kl_coef=0.0)
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
    assert result.metrics["actor/ema_kl_loss"] == 0.0
    assert result.metrics["actor/ratio_mean"] == pytest.approx(1.0)
    torch.testing.assert_close(grad, torch.zeros_like(grad))
    # The term is live as soon as the policy moves away from the EMA adapter.
    inputs["old_prediction"] = inputs["old_prediction"] + 0.1
    moved = run_loss(inputs, torch.zeros(3), awm_kl_coef=0.5, ref_kl_coef=0.0)
    assert moved.metrics["actor/ema_kl_loss"] == pytest.approx(0.01, rel=1e-5)
    torch.testing.assert_close(moved.loss, torch.tensor(0.5 * 0.01), rtol=1e-5, atol=1e-7)


def test_awm_through_the_worker_loss_entry_point():
    """The worker-side `diffusion_loss` dispatch reaches AWM and divides by gradient accumulation."""
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    inputs = make_inputs(4)
    data = TensorDict({"advantages": advantages}, batch_size=[4])
    tu.assign_non_tensor(data, gradient_accumulation_steps=4, sp_size=1)
    model_output = {key: inputs[key] for key in AWMLoss.required_model_output_keys}
    loss, metrics = diffusion_loss(make_actor(), model_output, data)
    direct = run_loss(inputs, advantages)
    torch.testing.assert_close(loss, direct.loss / 4)
    assert "actor/ema_kl_loss" in metrics


def test_awm_config_validation():
    """The loss config exposes the AWM defaults and rejects negative clip ranges and KL coefficients."""
    assert DiffusionLossConfig(loss_mode="awm").awm_clip_range == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="awm_clip_range"):
        DiffusionLossConfig(loss_mode="awm", awm_clip_range=-0.1)
    with pytest.raises(ValueError, match="awm_kl_coef"):
        DiffusionLossConfig(loss_mode="awm", awm_kl_coef=-1.0)


def make_rollout_batch(uid, num_steps=10):
    """Build a final-latent rollout batch with a shared timestep schedule and per-row uids."""
    batch_size = len(uid)
    schedule = torch.linspace(1000, 100, num_steps).round().long()
    return DataProto.from_dict(
        tensors={
            "latents_clean": torch.randn(batch_size, C, H, W),
            "all_timesteps": schedule.expand(batch_size, -1).clone(),
        },
        non_tensors={"uid": np.array(uid, dtype=object)},
    )


def make_config(**algorithm_kwargs):
    """Build a minimal trainer config namespace with the algorithm fields the batch preparation reads."""
    algorithm = SimpleNamespace(norm_adv_by_std_in_grpo=True, global_std=False, timestep_fraction=1.0)
    for key, value in algorithm_kwargs.items():
        setattr(algorithm, key, value)
    return SimpleNamespace(algorithm=algorithm)


def test_prepare_awm_actor_batch_uses_grpo_advantages_and_shared_timesteps():
    """Batch preparation adds GRPO group advantages and batch-shared timesteps without group tensors."""
    uid = ["a", "b", "a", "b"]
    batch = make_rollout_batch(uid)
    rewards = torch.tensor([0.0, 0.5, 1.0, 0.5])

    torch.manual_seed(0)
    result = AWMLoss.prepare_actor_batch(
        batch, rewards, make_config(train_timestep_range=[0, 7], train_timestep_count=4)
    )

    # Rows stay in rollout order; groups need not be contiguous and no group tensors are added.
    assert list(result.non_tensor_batch["uid"]) == uid
    assert "group_index" not in result.batch.keys() and "forward_noise" not in result.batch.keys()
    train_timesteps = result.batch["train_timesteps"]
    schedule = make_rollout_batch(uid).batch["all_timesteps"][0]
    assert train_timesteps.shape == (4, 4)
    assert (train_timesteps == train_timesteps[:1]).all()
    assert set(train_timesteps[0].tolist()) <= set(schedule[:7].tolist())

    advantages = result.batch["advantages"]
    assert advantages.shape == (4, 4)
    # Group "a" is (0, 1): std-normalized advantages are -/+ 1/sqrt(2) up to epsilon; group "b" is constant.
    expected_a = 0.5 / (torch.tensor([0.0, 1.0]).std() + 1e-4)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([-expected_a, 0.0, expected_a, 0.0]))
    torch.testing.assert_close(result.batch["sample_level_rewards"][:, 0], rewards)


def make_trainer_config(**overrides):
    """Build a trainer config that passes AWM validation, with optional dotted-key overrides."""
    config = OmegaConf.create(
        {
            "algorithm": {"train_timestep_range": None, "train_timestep_count": None},
            "actor_rollout_ref": {
                "model": {"algorithm": "awm", "lora_rank": 32, "policy_state_adapters": ["default", "old"]},
                "rollout": {"n": 8, "rollout_adapter": "old", "pipeline": {"num_inference_steps": 10}},
                "actor": {"use_kl_loss": False, "enable_timestep_staging": False},
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
        ({"algorithm.train_timestep_range": [0, 7], "algorithm.train_timestep_count": 4}, None),
        ({"actor_rollout_ref.model.lora_rank": 0}, "lora_rank must be > 0"),
        (
            {"actor_rollout_ref.model.algorithm": "dgpo"},
            "model.algorithm must be awm when diffusion_loss.loss_mode is awm",
        ),
        ({"actor_rollout_ref.model.policy_state_adapters": ["default"]}, 'must include "old"'),
        ({"actor_rollout_ref.rollout.rollout_adapter": "default"}, "rollout_adapter must be old"),
        ({"actor_rollout_ref.actor.use_kl_loss": True}, "use_kl_loss must be False"),
        ({"actor_rollout_ref.actor.enable_timestep_staging": True}, "enable_timestep_staging must be False"),
        ({"algorithm.train_timestep_range": [0, 11]}, "0 <= start < end <= 10"),
        ({"algorithm.train_timestep_count": 11}, "train_timestep_count=11 must be between 1 and 10"),
    ],
)
def test_awm_trainer_config_validation(overrides, message):
    """Trainer validation accepts the timestep knobs and rejects a missing old adapter or a bad window."""
    config = make_trainer_config(**overrides)
    if message is None:
        AWMLoss.validate_trainer_config(config)
    else:
        with pytest.raises(ValueError, match=message):
            AWMLoss.validate_trainer_config(config)


AWM_TRAINER_OVERRIDES = [
    "algorithm.trainer_type=direct_preference",
    "algorithm.sample_source=online",
    "actor_rollout_ref.model.algorithm=awm",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=awm",
    "actor_rollout_ref.model.lora_rank=8",
    "actor_rollout_ref.model.policy_state_adapters=[default,old]",
    "actor_rollout_ref.rollout.rollout_adapter=old",
    "actor_rollout_ref.rollout.n=2",
    "actor_rollout_ref.actor.ppo_mini_batch_size=1",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
    "data.train_batch_size=1",
    "trainer.n_gpus_per_node=1",
]


def make_awm_v1_trainer(extra_overrides=()):
    """Build the real v1 sync trainer from the diffusion config with the AWM overrides."""
    import os

    from hydra import compose, initialize_config_dir

    import verl_omni
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    config_dir = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=AWM_TRAINER_OVERRIDES + list(extra_overrides))
    return PolicyGradientDiffusionTrainerV1Sync(config)


def test_v1_trainer_init_accepts_an_awm_config():
    """The real trainer accepts AWM with the old adapter and loads the AWM loss."""
    trainer = make_awm_v1_trainer()
    assert trainer._is_direct_preference
    assert trainer._has_old_adapter
    assert isinstance(trainer._loss_fn, AWMLoss)


@pytest.mark.parametrize(
    "extra,message",
    [
        (["actor_rollout_ref.model.algorithm=dgpo"], "model.algorithm must be awm"),
        (["actor_rollout_ref.actor.diffusion_loss.loss_mode=dgpo"], "model.algorithm must be dgpo"),
    ],
)
def test_v1_trainer_init_rejects_a_mismatched_algorithm_and_loss(extra, message):
    """A DGPO adapter with the AWM loss, or the reverse, is refused at startup rather than trained silently."""
    with pytest.raises(ValueError, match=message):
        make_awm_v1_trainer(extra)


def test_awm_sd3_adapters_are_registered_under_the_awm_package():
    """The (StableDiffusion3Pipeline, awm) pair resolves to the sd3_awm subclasses of the DGPO adapters."""
    from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
    from verl_omni.pipelines.sd3_awm import StableDiffusion3AWM, StableDiffusion3AWMPipeline
    from verl_omni.pipelines.sd3_dgpo import StableDiffusion3DGPO, StableDiffusion3DGPOPipeline

    assert DiffusionModelBase.peek_class("StableDiffusion3Pipeline", "awm") is StableDiffusion3AWM
    assert issubclass(StableDiffusion3AWM, StableDiffusion3DGPO)
    assert VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "awm") is StableDiffusion3AWMPipeline
    assert issubclass(StableDiffusion3AWMPipeline, StableDiffusion3DGPOPipeline)
    assert DiffusionModelBase.peek_class("StableDiffusion3Pipeline", "dgpo") is StableDiffusion3DGPO


def test_awm_batch_through_the_nft_engine_step():
    """Prepared AWM tensors reach the loss through the real engine step and the SD3 adapter registered for awm."""
    from contextlib import nullcontext
    from functools import partial

    from verl_omni.workers.engine.fsdp import diffusers_impl

    batch = make_rollout_batch(["a", "a", "b", "b"], num_steps=6)
    torch.manual_seed(0)
    batch = AWMLoss.prepare_actor_batch(
        batch, torch.tensor([0.2, 0.8, 0.5, 0.1]), make_config(train_timestep_range=[0, 4], train_timestep_count=2)
    )
    micro_batch = batch.batch.select("latents_clean", "train_timesteps", "advantages")
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
        algorithm="awm",
        external_lib=None,
        pipeline=SimpleNamespace(guidance_scale=4.5),
    )
    weight = torch.nn.Parameter(torch.tensor(0.5))
    calls = []

    def module(hidden_states, encoder_hidden_states, pooled_projections, timestep, joint_attention_kwargs):
        """Stand-in transformer: scale the latents by a trainable weight so the loss has a gradient path."""
        calls.append(timestep)
        return (weight * hidden_states + pooled_projections.mean() * 0.0,)

    engine.module = module
    engine.use_adapter = lambda name: nullcontext()
    engine.disable_adapter = nullcontext
    engine._set_adapter = lambda name: None

    seen = []

    def loss_function(model_output, data, dp_group):
        """Record the engine's loss inputs and dispatch to the worker-side diffusion loss."""
        seen.append((model_output, data))
        return partial(diffusion_loss, config=make_actor())(model_output=model_output, data=data, dp_group=dp_group)

    step = 1
    loss, output = engine.forward_step(micro_batch, loss_function=loss_function, forward_only=False, step=step)
    loss.backward()

    model_output, data = seen[0]
    torch.testing.assert_close(data["advantages"], micro_batch["advantages"][:, step])
    t = micro_batch["train_timesteps"][:, step].float() / 1000
    torch.testing.assert_close(model_output["t_expanded"].flatten(), t)
    # Old, current and reference passes, each without a CFG branch.
    assert len(calls) == 3
    assert weight.grad is not None and torch.isfinite(weight.grad)
    assert "actor/ema_kl_loss" in output["metrics"]
