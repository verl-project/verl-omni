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

"""L1 CPU tests for the DiffusionOPSD actor side: loss wiring, config safeguards,
batch preparation and the SD3.5 adapters.

Necessity: the OPSD loss, its ``opsd_*`` config and ``prepare_actor_batch`` form a
silent-failure chain. A mis-wired config default (wrong query sigma, non-detached
normalizer, ``adv_clip_max`` not applied), a query state selected from the wrong
captured step, an endpoint weight that is not group-normalized, or an SD3 adapter
that re-noises the endpoint instead of returning the stored query state all keep
training running with a finite loss while optimizing the wrong objective. These
tests pin each link to an independent transcription of the paper equations (Eq.
11-13), to the DiffusionNFT weight helpers the recipe reuses, and to the exact
rollout/actor-batch contract of the implementation.
"""

import json
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.sd3_opsd.diffusers_training_adapter import StableDiffusion3OPSD
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionNFTLoss, OPSDLoss, get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig

BATCH = 4
C, H, W = 4, 3, 3

OPSD_REQUIRED_MODEL_OUTPUT_KEYS = (
    "forward_prediction",
    "old_prediction",
    "xt",
    "t_expanded",
    "opsd_target_pos",
    "opsd_target_neg",
)
OPSD_METRIC_KEYS = {
    "actor/opsd_pos_loss",
    "actor/opsd_neg_loss",
    "actor/opsd_policy_loss",
    "actor/opsd_reward_prob_mean",
    "actor/opsd_total_loss",
    "actor/opsd_query_sigma",
    "actor/total_loss",
}

#: The SD3.5-Medium 10-step shift-3 schedule's trailing window (sigma = t / 1000).
OPSD_WINDOW_TIMESTEPS = (666, 562, 428, 250)


def make_actor(**loss_kwargs) -> DiffusionActorConfig:
    loss_cfg = DiffusionLossConfig(loss_mode="opsd", **loss_kwargs)
    return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=loss_cfg)


def _make_opsd_inputs(*, dtype=torch.float64, shape=(BATCH, C, H, W), seed=0):
    generator = torch.Generator().manual_seed(seed)
    forward_prediction = torch.randn(*shape, dtype=dtype, generator=generator, requires_grad=True)
    return {
        "forward_prediction": forward_prediction,
        "old_prediction": torch.randn(*shape, dtype=dtype, generator=generator),
        "xt": torch.randn(*shape, dtype=dtype, generator=generator),
        "t_expanded": torch.rand(*shape[:1], dtype=dtype, generator=generator).view(-1, *([1] * (len(shape) - 1))),
        "opsd_target_pos": torch.randn(*shape, dtype=dtype, generator=generator),
        "opsd_target_neg": torch.randn(*shape, dtype=dtype, generator=generator),
        "reward_prob": torch.rand(shape[0], dtype=dtype, generator=generator),
    }


def _reference_opsd_loss(
    *,
    forward_prediction,
    old_prediction,
    xt,
    t_expanded,
    target_pos,
    target_neg,
    reward_prob,
    beta,
    adv_clip_max,
    eps_gamma,
):
    """Literal Eq. 11-13 transcription, independent of ``opsd_two_branch_loss``.

    The two implicit branch velocities are formed against the detached old
    prediction, the per-sample normalizer is a detached float64 mean absolute
    residual clamped at ``eps_gamma``, and the batch mean of the reward-weighted
    per-sample regression is scaled by ``adv_clip_max``.
    """
    old = old_prediction.detach()
    v_pos = beta * forward_prediction + (1.0 - beta) * old
    v_neg = (1.0 + beta) * old - beta * forward_prediction
    y_pos = xt - t_expanded * v_pos
    y_neg = xt - t_expanded * v_neg

    reduce_dims = tuple(range(1, y_pos.ndim))
    gamma_pos = (y_pos.detach().double() - target_pos.detach().double()).abs().mean(dim=reduce_dims, keepdim=True)
    gamma_neg = (y_neg.detach().double() - target_neg.detach().double()).abs().mean(dim=reduce_dims, keepdim=True)
    gamma_pos = gamma_pos.clamp_min(eps_gamma).to(y_pos.dtype)
    gamma_neg = gamma_neg.clamp_min(eps_gamma).to(y_neg.dtype)

    pos = ((y_pos - target_pos.detach()) ** 2 / gamma_pos).mean(dim=reduce_dims)
    neg = ((y_neg - target_neg.detach()) ** 2 / gamma_neg).mean(dim=reduce_dims)
    # reduce_dims removes every non-batch axis, so each of pos/neg is (B,).
    weight = reward_prob.detach().reshape(-1)
    return (weight * pos + (1.0 - weight) * neg).mean() * adv_clip_max


def _run_opsd_loss(inputs, config, *, reward_prob=None):
    model_output = {key: inputs[key] for key in OPSDLoss.required_model_output_keys}
    prob = inputs["reward_prob"] if reward_prob is None else reward_prob
    data = TensorDict({"reward_prob": prob}, batch_size=[prob.shape[0]])
    return OPSDLoss()(config=config, model_output=model_output, data=data)


class TestOPSDLossContract:
    def test_registered_as_opsd_and_declares_the_worker_contract(self):
        loss_fn = get_diffusion_loss_fn("opsd")
        assert type(loss_fn) is OPSDLoss
        assert get_diffusion_loss_fn("opsd") is loss_fn
        assert loss_fn.uses_old_adapter is True
        # OPSD samples are independent; the group only normalizes the endpoint reward.
        assert loss_fn.requires_whole_groups is False
        # Exact input contract filled by the OPSD engine and prepare_actor_batch.
        assert OPSDLoss.required_model_output_keys == OPSD_REQUIRED_MODEL_OUTPUT_KEYS
        assert OPSDLoss.required_data_keys == ("reward_prob",)

    def test_validate_inputs_names_missing_model_output_keys(self):
        data = TensorDict({"reward_prob": torch.ones(2)}, batch_size=[2])
        incomplete = {key: torch.zeros(2) for key in OPSDLoss.required_model_output_keys if key != "opsd_target_neg"}
        with pytest.raises(KeyError, match="opsd_target_neg"):
            OPSDLoss().validate_inputs(loss_name="opsd", model_output=incomplete, data=data)
        with pytest.raises(KeyError, match="Available model_output keys"):
            OPSDLoss().validate_inputs(loss_name="opsd", model_output=incomplete, data=data)

    def test_validate_inputs_names_a_missing_data_key(self):
        model_output = {key: torch.zeros(2) for key in OPSDLoss.required_model_output_keys}
        with pytest.raises(KeyError, match="Missing data keys: \\['reward_prob'\\]"):
            OPSDLoss().validate_inputs(loss_name="opsd", model_output=model_output, data=TensorDict({}, batch_size=[2]))


class TestOPSDLossMath:
    @pytest.mark.parametrize("beta", [1.0, 0.5])
    def test_loss_and_gradient_match_the_reference_transcription(self, beta):
        inputs = _make_opsd_inputs(seed=0)
        config = make_actor(opsd_branch_beta=beta, opsd_eps_gamma=1e-5, adv_clip_max=5.0)

        reference = _reference_opsd_loss(
            forward_prediction=inputs["forward_prediction"],
            old_prediction=inputs["old_prediction"],
            xt=inputs["xt"],
            t_expanded=inputs["t_expanded"],
            target_pos=inputs["opsd_target_pos"],
            target_neg=inputs["opsd_target_neg"],
            reward_prob=inputs["reward_prob"],
            beta=beta,
            adv_clip_max=5.0,
            eps_gamma=1e-5,
        )
        result = _run_opsd_loss(inputs, config)
        (reference_grad,) = torch.autograd.grad(reference, inputs["forward_prediction"], retain_graph=True)
        (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])

        torch.testing.assert_close(result.loss, reference, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(grad, reference_grad, rtol=1e-9, atol=1e-12)
        assert result.add_loss_metric is True
        assert set(result.metrics) == OPSD_METRIC_KEYS
        assert result.metrics["actor/total_loss"] == pytest.approx(result.loss.item(), rel=1e-12)
        assert result.metrics["actor/opsd_policy_loss"] * 5.0 == pytest.approx(
            result.metrics["actor/total_loss"], rel=1e-6
        )
        assert result.metrics["actor/opsd_reward_prob_mean"] == pytest.approx(
            inputs["reward_prob"].mean().item(), rel=1e-6
        )
        assert result.metrics["actor/opsd_query_sigma"] == pytest.approx(inputs["t_expanded"].mean().item(), rel=1e-6)

    def test_surfaces_the_engine_target_construction_seconds(self):
        # The engine times target construction and stashes it on the model output; the loss must
        # report it so the OPSD-specific cost can be attributed inside the actor update.
        inputs = _make_opsd_inputs(seed=5)
        model_output = {key: inputs[key] for key in OPSDLoss.required_model_output_keys}
        model_output["opsd_target_seconds"] = torch.tensor(0.25)
        prob = inputs["reward_prob"]
        data = TensorDict({"reward_prob": prob}, batch_size=[prob.shape[0]])

        result = OPSDLoss()(config=make_actor(), model_output=model_output, data=data)

        assert result.metrics["actor/opsd_target_seconds"] == pytest.approx(0.25, rel=1e-6)
        assert set(result.metrics) == OPSD_METRIC_KEYS | {"actor/opsd_target_seconds"}

    def test_only_forward_prediction_carries_gradient(self):
        inputs = _make_opsd_inputs(seed=3)
        for key in ("old_prediction", "opsd_target_pos", "opsd_target_neg"):
            inputs[key].requires_grad_(True)
        inputs["reward_prob"].requires_grad_(True)
        result = _run_opsd_loss(inputs, make_actor())

        for detached_input in (
            inputs["old_prediction"],
            inputs["opsd_target_pos"],
            inputs["opsd_target_neg"],
            inputs["reward_prob"],
        ):
            assert torch.autograd.grad(result.loss, detached_input, allow_unused=True, retain_graph=True)[0] is None
        (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0

    def test_exact_targets_give_zero_loss_and_zero_gradient(self):
        inputs = _make_opsd_inputs(seed=1)
        beta = 1.0
        # Solve for the targets the current prediction reproduces exactly.
        with torch.no_grad():
            v_pos = beta * inputs["forward_prediction"] + (1.0 - beta) * inputs["old_prediction"]
            v_neg = (1.0 + beta) * inputs["old_prediction"] - beta * inputs["forward_prediction"]
            inputs["opsd_target_pos"] = inputs["xt"] - inputs["t_expanded"] * v_pos
            inputs["opsd_target_neg"] = inputs["xt"] - inputs["t_expanded"] * v_neg

        result = _run_opsd_loss(inputs, make_actor(opsd_branch_beta=beta))
        (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])

        assert result.loss.item() == 0.0
        assert result.metrics["actor/opsd_pos_loss"] == 0.0
        assert result.metrics["actor/opsd_neg_loss"] == 0.0
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)

    def test_eps_gamma_floors_the_detached_normalizer(self):
        # Branch residual mean |y - target| is 1.0, below the 1e3 floor, so the
        # normalizer must clamp to eps_gamma rather than use the residual itself.
        base = torch.full((2, C, H, W), 0.5, dtype=torch.float64)
        zero = torch.zeros_like(base)
        model_output = {
            "forward_prediction": zero.clone().requires_grad_(True),
            "old_prediction": zero,
            "xt": base,
            "t_expanded": torch.zeros(2, 1, 1, 1, dtype=torch.float64),
            "opsd_target_pos": base + 1.0,
            "opsd_target_neg": base + 1.0,
        }
        data = TensorDict({"reward_prob": torch.ones(2, dtype=torch.float64)}, batch_size=[2])
        result = OPSDLoss()(config=make_actor(opsd_eps_gamma=1e3), model_output=model_output, data=data)

        assert result.loss.item() == pytest.approx(5.0 / 1e3, rel=1e-12)

    def test_reward_prob_selects_branches_and_accepts_the_prepared_singleton_column(self):
        inputs = _make_opsd_inputs(seed=2)
        config = make_actor()
        ones = torch.ones(BATCH, dtype=torch.float64)
        zeros = torch.zeros(BATCH, dtype=torch.float64)

        result_pos = _run_opsd_loss(inputs, config, reward_prob=ones)
        result_neg = _run_opsd_loss(inputs, config, reward_prob=zeros)
        assert result_pos.loss.item() == pytest.approx(result_pos.metrics["actor/opsd_pos_loss"] * 5.0, rel=1e-9)
        assert result_neg.loss.item() == pytest.approx(result_neg.metrics["actor/opsd_neg_loss"] * 5.0, rel=1e-9)
        # Different branch losses: the reward weight really selects a branch.
        assert result_pos.metrics["actor/opsd_pos_loss"] != result_neg.metrics["actor/opsd_neg_loss"]

        # prepare_actor_batch stores reward_prob as (B, 1); both layouts must agree.
        result_column = _run_opsd_loss(inputs, config, reward_prob=inputs["reward_prob"][:, None])
        result_flat = _run_opsd_loss(inputs, config)
        torch.testing.assert_close(result_column.loss, result_flat.loss, rtol=1e-12, atol=0)

        # adv_clip_max is the loss scale (docs: not a plain learning-rate knob).
        doubled = _run_opsd_loss(inputs, make_actor(adv_clip_max=10.0))
        torch.testing.assert_close(doubled.loss, 2.0 * result_flat.loss, rtol=1e-12, atol=0)

    def test_call_forwards_target_displacement_diagnostics(self):
        inputs = _make_opsd_inputs(seed=4)
        model_output = {key: inputs[key] for key in OPSDLoss.required_model_output_keys}
        model_output["opsd_target_displacement_rel"] = torch.tensor(0.061)
        model_output["opsd_target_displacement_neg_rel"] = torch.tensor(0.058)
        data = TensorDict({"reward_prob": inputs["reward_prob"]}, batch_size=[BATCH])

        result = OPSDLoss()(config=make_actor(), model_output=model_output, data=data)

        assert result.metrics["actor/opsd_target_displacement_rel"] == pytest.approx(0.061, rel=1e-6)
        assert result.metrics["actor/opsd_target_displacement_neg_rel"] == pytest.approx(0.058, rel=1e-6)


class TestDiffusionLossConfigOPSD:
    def test_accepts_opsd_and_documents_the_recipe_defaults(self):
        config = DiffusionLossConfig(loss_mode="opsd")
        assert (config.opsd_rho, config.opsd_n_ascent, config.opsd_eta) == (0.1, 2, 1.0)
        assert config.opsd_branch_beta == pytest.approx(1.0)
        assert config.opsd_query_sigma == pytest.approx(0.278)
        assert config.opsd_query_sigma_tolerance == pytest.approx(0.15)
        assert config.opsd_num_train_timesteps == 1000
        assert config.opsd_eps_gamma == pytest.approx(1e-5)
        assert config.opsd_reward_kind == "pickscore"
        assert config.opsd_reward_model_path == ""
        assert config.opsd_reward_dtype == "float32"
        assert config.opsd_target_micro_batch == 6
        # The endpoint-weight scale is shared with the other forward-process losses.
        assert config.adv_clip_max == pytest.approx(5.0)

    def test_rejects_an_unknown_loss_mode(self):
        with pytest.raises(ValueError, match="Invalid diffusion loss_mode"):
            DiffusionLossConfig(loss_mode="opsd_typo")

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"opsd_rho": 0.0}, "opsd_rho must be in"),
            ({"opsd_rho": 1.5}, "opsd_rho must be in"),
            ({"opsd_n_ascent": 0}, "opsd_n_ascent must be positive"),
            ({"opsd_eta": 0.0}, "opsd_eta must be positive"),
            ({"opsd_branch_beta": -0.1}, "opsd_branch_beta must be non-negative"),
            ({"opsd_query_sigma": 0.0}, "opsd_query_sigma must be in"),
            ({"opsd_query_sigma": 1.0}, "opsd_query_sigma must be in"),
            ({"opsd_query_sigma_tolerance": 0.0}, "opsd_query_sigma_tolerance must be positive"),
            ({"opsd_num_train_timesteps": 0}, "opsd_num_train_timesteps must be positive"),
            ({"opsd_eps_gamma": 0.0}, "opsd_eps_gamma must be positive"),
            ({"opsd_target_micro_batch": 0}, "opsd_target_micro_batch must be positive"),
            ({"opsd_reward_kind": "lpips"}, "Invalid opsd_reward_kind"),
            ({"opsd_reward_dtype": "int8"}, "Invalid opsd_reward_dtype"),
        ],
    )
    def test_rejects_invalid_opsd_fields(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            DiffusionLossConfig(loss_mode="opsd", **overrides)

    @pytest.mark.parametrize("reward_kind", ["hpsv2", "clipscore"])
    def test_rejects_a_reward_kind_the_target_runtime_does_not_ship(self, reward_kind):
        # Only the PickScore v1 differentiable reward is implemented; another
        # allowlisted name would fail later when the engine builds the runtime.
        with pytest.raises(ValueError, match="Invalid opsd_reward_kind"):
            DiffusionLossConfig(loss_mode="opsd", opsd_reward_kind=reward_kind)

    def test_accepts_the_shipped_reward_kind(self):
        config = DiffusionLossConfig(loss_mode="opsd", opsd_reward_kind="pickscore")
        assert config.opsd_reward_kind == "pickscore"

    @pytest.mark.parametrize("reward_dtype", ["float32", "bfloat16", "float16"])
    def test_accepts_the_documented_reward_dtypes(self, reward_dtype):
        assert DiffusionLossConfig(loss_mode="opsd", opsd_reward_dtype=reward_dtype).opsd_reward_dtype == reward_dtype


def make_opsd_trainer_config(**overrides):
    config = OmegaConf.create(
        {
            "algorithm": {"train_timestep_range": None, "train_timestep_count": None},
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
            "actor_rollout_ref": {
                "model": {
                    "algorithm": "opsd",
                    "model_type": "diffusion_opsd_model",
                    "lora_rank": 32,
                    "policy_state_adapters": ["default", "old"],
                },
                "rollout": {"n": 8, "rollout_adapter": "old"},
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
        ({"actor_rollout_ref.model.algorithm": "diffusion_nft"}, "model.algorithm must be opsd"),
        ({"actor_rollout_ref.model.model_type": "diffusion_nft_model"}, "model_type must be diffusion_opsd_model"),
        ({"actor_rollout_ref.model.lora_rank": 0}, "lora_rank must be > 0"),
        ({"actor_rollout_ref.model.policy_state_adapters": ["default"]}, 'must include "old"'),
        ({"actor_rollout_ref.rollout.rollout_adapter": "default"}, "rollout_adapter must be old"),
        ({"actor_rollout_ref.rollout.n": 1}, "rollout.n must be > 1"),
        ({"actor_rollout_ref.actor.use_kl_loss": True}, "use_kl_loss must be False"),
        ({"actor_rollout_ref.actor.enable_timestep_staging": True}, "enable_timestep_staging must be False"),
        ({"trainer.use_v1": False}, "trainer.use_v1 must be true"),
        ({"trainer.v1.trainer_mode": "separate_async"}, "trainer.v1.trainer_mode must be sync"),
        # The shared guard names every loss that consumes these knobs, so match the stable prefix.
        ({"algorithm.train_timestep_range": [0, 7]}, "only used by the dgpo"),
        ({"algorithm.train_timestep_count": 4}, "only used by the dgpo"),
    ],
)
def test_opsd_trainer_config_guards(overrides, message):
    config = make_opsd_trainer_config(**overrides)
    if message is None:
        OPSDLoss.validate_trainer_config(config)
    else:
        with pytest.raises(ValueError, match=message):
            OPSDLoss.validate_trainer_config(config)


def make_model_dir_with_sd3_schedule(tmp_path, *, shift=3.0, dynamic_shifting=False):
    """Materialise a tiny checkpoint dir whose scheduler is a real SD3 flow schedule."""
    model_dir = tmp_path / "sd35m"
    (model_dir / "scheduler").mkdir(parents=True)
    payload = {"num_train_timesteps": 1000, "shift": shift, "_class_name": "FlowMatchEulerDiscreteScheduler"}
    if dynamic_shifting:
        payload.update(
            {
                "use_dynamic_shifting": True,
                "base_shift": 1.73,
                "max_shift": 5.0,
                "base_image_seq_len": 256,
                "max_image_seq_len": 4096,
            }
        )
    (model_dir / "scheduler" / "scheduler_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def make_window_config(model_dir, num_steps):
    config = make_opsd_trainer_config()
    OmegaConf.update(config, "actor_rollout_ref.model.path", str(model_dir))
    OmegaConf.update(config, "actor_rollout_ref.actor.diffusion_loss", {"loss_mode": "opsd"}, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.pipeline.num_inference_steps", num_steps)
    return config


def test_window_coverage_accepts_a_step_count_that_reaches_sigma(tmp_path):
    # 10 steps puts sigma*=0.278 exactly in the trailing window on the shift-3 schedule.
    model_dir = make_model_dir_with_sd3_schedule(tmp_path)
    OPSDLoss.validate_trainer_config(make_window_config(model_dir, num_steps=10))


def test_window_coverage_rejects_a_step_count_that_cannot_reach_sigma(tmp_path):
    # 100 steps moves the trailing window below sigma*=0.278 - tolerance; this used to fail at
    # the first training batch, wasting a model load, instead of at startup.
    model_dir = make_model_dir_with_sd3_schedule(tmp_path)
    with pytest.raises(ValueError, match="trailing query window"):
        OPSDLoss.validate_trainer_config(make_window_config(model_dir, num_steps=100))


def test_window_coverage_resolves_a_hub_repo_id(monkeypatch, tmp_path):
    # The published recipe points model.path at a hub repo id and validate_trainer_config runs
    # before the snapshot exists on disk, so requiring a local directory would silently skip the
    # check for every real run. Only the scheduler config is fetched.
    model_dir = make_model_dir_with_sd3_schedule(tmp_path)
    fetched = {}

    def fake_download(*, repo_id, filename):
        fetched["repo_id"] = repo_id
        fetched["filename"] = filename
        return str(model_dir / "scheduler" / "scheduler_config.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    config = make_window_config("stabilityai/stable-diffusion-3.5-medium", num_steps=100)
    with pytest.raises(ValueError, match="trailing query window"):
        OPSDLoss.validate_trainer_config(config)

    assert fetched == {
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "filename": "scheduler/scheduler_config.json",
    }


def test_window_coverage_is_skipped_when_the_hub_is_unreachable(monkeypatch):
    # Offline or gated repo: no schedule to derive, and the first-batch guard still applies.
    def unreachable(**_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unreachable)

    config = make_window_config("stabilityai/stable-diffusion-3.5-medium", num_steps=100)
    OPSDLoss.validate_trainer_config(config)


def test_window_coverage_is_skipped_for_dynamic_shifting(tmp_path):
    # A resolution-derived mu changes the grid and this probe does not model it, so it must not
    # reject a checkpoint it cannot reason about.
    model_dir = make_model_dir_with_sd3_schedule(tmp_path, dynamic_shifting=True)
    OPSDLoss.validate_trainer_config(make_window_config(model_dir, num_steps=100))


def test_window_coverage_is_skipped_without_a_checkpoint(tmp_path):
    # No model dir -> no schedule to derive; the runtime selection guard still covers this.
    config = make_opsd_trainer_config()
    OmegaConf.update(config, "actor_rollout_ref.rollout.pipeline.num_inference_steps", 100)
    OPSDLoss.validate_trainer_config(config)


def make_rollout_batch(uid, *, latents=None, timesteps=None):
    batch_size = len(uid)
    if timesteps is None:
        timesteps = torch.tensor(OPSD_WINDOW_TIMESTEPS, dtype=torch.long)
    if timesteps.ndim == 1:
        all_timesteps = timesteps.unsqueeze(0).expand(batch_size, -1).clone()
    else:
        all_timesteps = timesteps.clone()
    if latents is None:
        latents = torch.randn(batch_size, all_timesteps.shape[1] + 1, C, H, W)
    return DataProto.from_dict(
        tensors={
            "all_latents": latents,
            "all_timesteps": all_timesteps,
            # The v1 actor batch always carries tokenized prompts for validation decode, and
            # OPSD forwards them to the engine as ``opsd_prompt_ids``.
            "prompts": torch.zeros(batch_size, 8, dtype=torch.long),
        },
        non_tensors={"uid": np.array(uid, dtype=object)},
    )


def make_prepare_config(**loss_kwargs):
    actor = make_actor(**loss_kwargs)
    algorithm = SimpleNamespace(norm_adv_by_std_in_grpo=True, global_std=True, adv_mode="continuous")
    return SimpleNamespace(algorithm=algorithm, actor_rollout_ref=SimpleNamespace(actor=actor))


class TestSelectQueryIndex:
    def test_picks_the_closest_captured_sigma_per_row(self):
        sigmas = torch.tensor([[0.9, 0.6, 0.3, 0.1], [0.1, 0.3, 0.6, 0.9]])
        index = OPSDLoss._select_query_index(sigmas, requested_sigma=0.278, tolerance=0.2)
        assert index.tolist() == [2, 1]

    def test_rejects_a_sigma_outside_the_captured_window(self):
        sigmas = torch.tensor([[0.9, 0.7, 0.6, 0.5]])
        with pytest.raises(ValueError, match="could not find the requested query sigma"):
            OPSDLoss._select_query_index(sigmas, requested_sigma=0.278, tolerance=0.15)
        # The tolerance is a hard bound: the 0.222 gap fails even at tolerance 0.2.
        with pytest.raises(ValueError, match="closest captured 0.2220 away"):
            OPSDLoss._select_query_index(sigmas, requested_sigma=0.278, tolerance=0.2)
        assert OPSDLoss._select_query_index(sigmas, requested_sigma=0.278, tolerance=0.25).tolist() == [3]

    def test_rejects_non_matrix_sigmas(self):
        with pytest.raises(ValueError, match="shape \\[B, N\\]"):
            OPSDLoss._select_query_index(torch.tensor([0.9, 0.6, 0.3, 0.1]), requested_sigma=0.278, tolerance=0.15)


class TestPrepareActorBatch:
    def test_selects_the_query_state_and_reuses_the_nft_endpoint_weight(self):
        uid = ["a", "a", "b", "b"]
        rewards = torch.tensor([0.2, 0.8, 0.5, 0.1])
        latents = torch.arange(BATCH * 5 * C * H * W, dtype=torch.float32).reshape(BATCH, 5, C, H, W)
        batch = make_rollout_batch(uid, latents=latents)
        config = make_prepare_config()

        result = OPSDLoss.prepare_actor_batch(batch, rewards, config)

        # sigma = t / 1000, so the captured [0.666, 0.562, 0.428, 0.25] queries sigma*=0.278 at index 3.
        torch.testing.assert_close(result.batch["opsd_query_latents"], latents[:, 3])
        assert result.batch["opsd_query_latents"].shape == (BATCH, C, H, W)
        # The query timestep stays a float so fractional scheduler timesteps survive fitting.
        torch.testing.assert_close(result.batch["train_timesteps"], torch.full((BATCH, 1), 250.0))

        expected_advantages = DiffusionNFTLoss._compute_group_advantages(
            rewards=rewards,
            uid=np.array(uid, dtype=object),
            norm_by_std=config.algorithm.norm_adv_by_std_in_grpo,
            global_std=config.algorithm.global_std,
        )
        expected_reward_prob = DiffusionNFTLoss._advantage_to_reward_prob(
            expected_advantages,
            adv_clip_max=config.actor_rollout_ref.actor.diffusion_loss.adv_clip_max,
            adv_mode=config.algorithm.adv_mode,
        )
        torch.testing.assert_close(result.batch["reward_prob"], expected_reward_prob[:, None])
        torch.testing.assert_close(result.batch["advantages"], expected_advantages[:, None])
        torch.testing.assert_close(result.batch["returns"], result.batch["advantages"])
        torch.testing.assert_close(result.batch["sample_level_rewards"], rewards[:, None])

        # The endpoint weight is group-centered and monotone in the raw reward.
        group_advantage_sums = result.batch["advantages"][:, 0].reshape(2, 2).sum(dim=1)
        torch.testing.assert_close(group_advantage_sums, torch.zeros(2), rtol=0, atol=1e-6)
        assert result.batch["reward_prob"][1, 0] > 0.5 > result.batch["reward_prob"][0, 0]

    def test_selection_is_per_row(self):
        uid = ["a", "b"]
        latents = torch.arange(2 * 5 * C * H * W, dtype=torch.float32).reshape(2, 5, C, H, W)
        timesteps = torch.tensor([[900, 700, 500, 278], [900, 278, 500, 300]], dtype=torch.long)
        batch = make_rollout_batch(uid, latents=latents, timesteps=timesteps)

        result = OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.0, 1.0]), make_prepare_config())

        expected = torch.stack([latents[0, 3], latents[1, 1]])
        torch.testing.assert_close(result.batch["opsd_query_latents"], expected)
        assert result.batch["train_timesteps"][:, 0].tolist() == [278, 278]

    def test_singleton_groups_get_a_neutral_weight(self):
        # rollout.n == 1 is rejected by validate_trainer_config; prepare_actor_batch must
        # not invent a signal for it: centered advantages are zero, so omega == 0.5.
        batch = make_rollout_batch(["a", "b"])
        result = OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.1, 0.9]), make_prepare_config())
        torch.testing.assert_close(result.batch["reward_prob"], torch.full((2, 1), 0.5))
        torch.testing.assert_close(result.batch["advantages"], torch.zeros(2, 1))

    def test_forwards_prompt_ids_to_the_target_runtime(self):
        batch = make_rollout_batch(["a", "a"])
        prompts = torch.arange(16, dtype=torch.long).reshape(2, 8)
        batch.batch["prompts"] = prompts.clone()

        result = OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.2, 0.8]), make_prepare_config())

        torch.testing.assert_close(result.batch["opsd_prompt_ids"], prompts)

    def test_rejects_an_actor_batch_without_prompts(self):
        # The differentiable reward needs the prompt behind each query state; without it the
        # engine cannot decode prompts and would fail deeper inside target construction.
        batch = make_rollout_batch(["a", "a"])
        del batch.batch["prompts"]

        with pytest.raises(ValueError, match="requires `prompts` token ids"):
            OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.2, 0.8]), make_prepare_config())

    def test_publishes_the_selection_scale_for_the_adapter(self):
        # Selection runs in the trainer and fitting in the adapter; the scale used here travels
        # with the batch so the adapter can reject a config that disagrees with its scheduler.
        batch = make_rollout_batch(["a", "a"])
        result = OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.2, 0.8]), make_prepare_config())
        torch.testing.assert_close(result.batch["opsd_timestep_scale"], torch.full((2,), 1000, dtype=torch.long))

    def test_keeps_fractional_query_timesteps_for_fitting(self):
        # SD3.5's shifted schedule yields fractional timesteps; truncating them would fit a
        # different sigma than the one selected during query selection.
        uid = ["a", "a"]
        all_timesteps = torch.tensor(
            [[1000.0, 500.0, 250.4, 100.0], [1000.0, 500.0, 250.4, 100.0]], dtype=torch.float32
        )
        batch = DataProto.from_dict(
            tensors={
                "all_latents": torch.zeros(2, all_timesteps.shape[1] + 1, C, H, W),
                "all_timesteps": all_timesteps,
                "prompts": torch.zeros(2, 8, dtype=torch.long),
            },
            non_tensors={"uid": np.array(uid, dtype=object)},
        )

        result = OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.2, 0.8]), make_prepare_config())

        # sigma = 0.2504 is the captured value closest to the requested 0.278.
        torch.testing.assert_close(result.batch["train_timesteps"], torch.full((2, 1), 250.4))

    def test_rejects_missing_rollout_fields(self):
        uid = ["a", "a"]
        rewards = torch.tensor([0.0, 1.0])
        timesteps = make_rollout_batch(uid).batch["all_timesteps"]

        without_latents = DataProto.from_dict(
            tensors={"all_timesteps": timesteps},
            non_tensors={"uid": np.array(uid, dtype=object)},
        )
        with pytest.raises(ValueError, match="`all_latents` and `all_timesteps`"):
            OPSDLoss.prepare_actor_batch(without_latents, rewards, make_prepare_config())

        without_timesteps = DataProto.from_dict(
            tensors={"all_latents": torch.randn(2, 5, C, H, W)},
            non_tensors={"uid": np.array(uid, dtype=object)},
        )
        with pytest.raises(ValueError, match="`all_latents` and `all_timesteps`"):
            OPSDLoss.prepare_actor_batch(without_timesteps, rewards, make_prepare_config())

        without_uid = DataProto.from_dict(
            tensors={"all_latents": torch.randn(2, 5, C, H, W), "all_timesteps": timesteps}
        )
        with pytest.raises(ValueError, match="requires `uid`"):
            OPSDLoss.prepare_actor_batch(without_uid, rewards, make_prepare_config())

    def test_rejects_a_rollout_window_that_does_not_hold_the_requested_sigma(self):
        timesteps = torch.tensor([900, 700, 600, 500], dtype=torch.long)
        batch = make_rollout_batch(["a", "a"], timesteps=timesteps)
        rewards = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match="could not find the requested query sigma"):
            OPSDLoss.prepare_actor_batch(batch, rewards, make_prepare_config())

        # The tolerance is the configured one, not a hardcoded constant.
        result = OPSDLoss.prepare_actor_batch(batch, rewards, make_prepare_config(opsd_query_sigma_tolerance=0.3))
        assert result.batch["train_timesteps"][:, 0].tolist() == [500, 500]

    @pytest.mark.parametrize(
        "tensors,message",
        [
            (
                {
                    "all_latents": torch.randn(2, 4, C, H, W),
                    "all_timesteps": torch.zeros(2, 4, dtype=torch.long),
                },
                "all_latents",
            ),
            (
                {
                    "all_latents": torch.randn(2, 4, C, H, W),
                    "all_timesteps": torch.zeros(2, 4, 1, dtype=torch.long),
                },
                "all_timesteps",
            ),
            (
                {
                    "all_latents": torch.randn(2, 5, C, H, W),
                    "all_timesteps": torch.zeros(2, 5, dtype=torch.long),
                },
                "one more state",
            ),
        ],
    )
    def test_rejects_inconsistent_rollout_shapes(self, tensors, message):
        batch = DataProto.from_dict(tensors=tensors, non_tensors={"uid": np.array(["a", "b"], dtype=object)})
        with pytest.raises(ValueError, match=message):
            OPSDLoss.prepare_actor_batch(batch, torch.tensor([0.0, 1.0]), make_prepare_config())


class TestStableDiffusion3OPSDTrainingAdapter:
    def test_training_adapter_is_registered_for_sd3_opsd(self):
        model_config = SimpleNamespace(
            architecture="StableDiffusion3Pipeline",
            algorithm="opsd",
            external_lib=None,
        )
        assert DiffusionModelBase.get_class(model_config) is StableDiffusion3OPSD

    def test_returns_the_stored_query_state_instead_of_re_noising(self):
        query = torch.randn(2, C, H, W)
        timesteps = torch.tensor([[900, 500, 250], [800, 400, 200]], dtype=torch.long)
        micro_batch = TensorDict(
            {
                "opsd_query_latents": query,
                "train_timesteps": timesteps,
                # The NFT/DGPO path would re-noise latents_clean with forward_noise; neither
                # may be read when the batch already carries the query state.
                "latents_clean": torch.full_like(query, 99.0),
                "forward_noise": torch.full_like(query, -99.0),
            },
            batch_size=[2],
        )
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

        xt, timestep, loss_context = StableDiffusion3OPSD.prepare_training_inputs(micro_batch, 1, scheduler)

        assert timestep.tolist() == [500, 400]
        torch.testing.assert_close(xt, query)
        torch.testing.assert_close(loss_context["xt"], query)
        expected_time = timesteps[:, 1].float() / 1000.0
        torch.testing.assert_close(loss_context["t_expanded"], expected_time.view(-1, 1, 1, 1))
        assert tu.get_non_tensor_data(micro_batch, "num_train_timesteps", default=None) == 1000

    def test_rejects_a_selection_scale_that_disagrees_with_the_scheduler(self):
        # The query sigma and the fitted flow time must come from one scale; a mismatch would
        # silently fit a different sigma than the one selected.
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        micro_batch = TensorDict(
            {
                "opsd_query_latents": torch.randn(2, C, H, W),
                "train_timesteps": torch.tensor([[562.0], [562.0]]),
                "opsd_timestep_scale": torch.full((2,), 2000, dtype=torch.long),
            },
            batch_size=[2],
        )

        with pytest.raises(ValueError, match="disagree on the timestep scale"):
            StableDiffusion3OPSD.prepare_training_inputs(micro_batch, 0, scheduler)

    def test_accepts_a_matching_selection_scale(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        micro_batch = TensorDict(
            {
                "opsd_query_latents": torch.randn(2, C, H, W),
                "train_timesteps": torch.tensor([[250.0], [250.0]]),
                "opsd_timestep_scale": torch.full((2,), 1000, dtype=torch.long),
            },
            batch_size=[2],
        )

        _, _, loss_context = StableDiffusion3OPSD.prepare_training_inputs(micro_batch, 0, scheduler)
        torch.testing.assert_close(loss_context["t_expanded"], torch.full((2, 1, 1, 1), 0.25))

    def test_preserves_fractional_query_timesteps(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        timesteps = torch.tensor([[250.4], [199.6]])
        micro_batch = TensorDict(
            {"opsd_query_latents": torch.randn(2, C, H, W), "train_timesteps": timesteps},
            batch_size=[2],
        )

        _, timestep, loss_context = StableDiffusion3OPSD.prepare_training_inputs(micro_batch, 0, scheduler)

        torch.testing.assert_close(loss_context["t_expanded"], (timesteps / 1000.0).view(-1, 1, 1, 1))
        assert timestep.dtype.is_floating_point, "fractional query timesteps must not be truncated"

    def test_slices_the_query_state_from_a_stacked_window(self):
        query = torch.randn(2, 3, C, H, W)
        timesteps = torch.tensor([[900, 500, 250], [800, 400, 200]], dtype=torch.long)
        micro_batch = TensorDict(
            {"opsd_query_latents": query, "train_timesteps": timesteps},
            batch_size=[2],
        )
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

        xt, timestep, loss_context = StableDiffusion3OPSD.prepare_training_inputs(micro_batch, 2, scheduler)

        torch.testing.assert_close(xt, query[:, 2])
        assert timestep.tolist() == [250, 200]
        torch.testing.assert_close(loss_context["t_expanded"], (timesteps[:, 2].float() / 1000.0).view(-1, 1, 1, 1))

    def test_rejects_an_actor_batch_without_the_stored_query_state(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        without_latents = TensorDict(
            {"train_timesteps": torch.tensor([[250], [250]], dtype=torch.long)}, batch_size=[2]
        )
        with pytest.raises(KeyError, match="opsd_query_latents"):
            StableDiffusion3OPSD.prepare_training_inputs(without_latents, 0, scheduler)

        without_timesteps = TensorDict({"opsd_query_latents": torch.randn(2, C, H, W)}, batch_size=[2])
        with pytest.raises(KeyError, match="train_timesteps"):
            StableDiffusion3OPSD.prepare_training_inputs(without_timesteps, 0, scheduler)

    def test_rejects_mismatched_or_malformed_query_states(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        # train_timesteps must be [B, T].
        flat_timesteps = TensorDict(
            {"opsd_query_latents": torch.randn(2, C, H, W), "train_timesteps": torch.tensor([250, 250])},
            batch_size=[2],
        )
        with pytest.raises(ValueError, match="shape \\[B, T\\]"):
            StableDiffusion3OPSD.prepare_training_inputs(flat_timesteps, 0, scheduler)

        # Query latents must be [B, C, H, W] or [B, T, C, H, W].
        malformed = TensorDict(
            {
                "opsd_query_latents": torch.randn(2, C, H),
                "train_timesteps": torch.tensor([[250], [250]], dtype=torch.long),
            },
            batch_size=[2],
        )
        with pytest.raises(ValueError, match="must have shape \\[B, C, H, W\\]"):
            StableDiffusion3OPSD.prepare_training_inputs(malformed, 0, scheduler)

        # Latents and timesteps must agree on the batch size. TensorDict enforces one
        # consistent leading dim itself, so use a plain mapping to reach the adapter check.
        mismatched = {
            "opsd_query_latents": torch.randn(3, C, H, W),
            "train_timesteps": torch.tensor([[250], [250]], dtype=torch.long),
        }
        with pytest.raises(ValueError, match="disagree on batch size"):
            StableDiffusion3OPSD.prepare_training_inputs(mismatched, 0, scheduler)

    def test_requires_the_scheduler_timestep_scale(self):
        micro_batch = TensorDict(
            {
                "opsd_query_latents": torch.randn(2, C, H, W),
                "train_timesteps": torch.tensor([[250], [250]], dtype=torch.long),
            },
            batch_size=[2],
        )
        with pytest.raises(ValueError, match="num_train_timesteps"):
            StableDiffusion3OPSD.prepare_training_inputs(micro_batch, 0, SimpleNamespace(config=SimpleNamespace()))

    def test_prepare_loss_inputs_slices_per_step_and_forwards_targets(self):
        reward_prob = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        advantages = 10.0 * reward_prob
        target_pos = torch.randn(2, C, H, W)
        target_neg = torch.randn(2, C, H, W)
        micro_batch = TensorDict(
            {
                "reward_prob": reward_prob,
                "advantages": advantages,
                "opsd_target_pos": target_pos,
                "opsd_target_neg": target_neg,
            },
            batch_size=[2],
        )

        inputs = StableDiffusion3OPSD.prepare_loss_inputs(micro_batch, 1)

        torch.testing.assert_close(inputs["reward_prob"], reward_prob[:, 1])
        torch.testing.assert_close(inputs["advantages"], advantages[:, 1])
        torch.testing.assert_close(inputs["opsd_target_pos"], target_pos)
        torch.testing.assert_close(inputs["opsd_target_neg"], target_neg)
        assert "group_index" not in inputs

    def test_prepare_loss_inputs_omits_absent_targets(self):
        micro_batch = TensorDict({"reward_prob": torch.ones(2, 1)}, batch_size=[2])
        inputs = StableDiffusion3OPSD.prepare_loss_inputs(micro_batch, 0)
        assert set(inputs) == {"reward_prob"}


class TestStableDiffusion3OPSDPipeline:
    def test_rollout_adapter_is_registered_with_the_query_window(self):
        pytest.importorskip("vllm_omni")
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase
        from verl_omni.pipelines.sd3_opsd.vllm_omni_rollout_adapter import StableDiffusion3OPSDPipeline

        assert VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "opsd") is StableDiffusion3OPSDPipeline
        # The trailing window is what makes sigma*=0.278 reachable on the 10-step schedule.
        assert StableDiffusion3OPSDPipeline.query_window_steps == 4
