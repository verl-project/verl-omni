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
"""CPU tests for the DiffusionART loss wiring, actor-batch preparation and config validation.

Necessity: the loss reads cached ODE states, a per-prompt guidance tensor and the next flow time from
the batch; a wrong gather, a guidance assigned to the wrong prompt, or a config that silently falls
back to a stochastic rollout would train without any error. These tests pin the batch contract and
the start-up validation to what ``ARTDiffusersFSDPEngine`` and the recipe assume.
"""

import json
import logging
import pathlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

import verl_omni
from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss, get_diffusion_loss_fn
from verl_omni.trainer.diffusion.diffusion_art import art_loss
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.utils.losses import diffusion_loss

C, H, W = 4, 3, 3


def make_actor(**loss_kwargs) -> DiffusionActorConfig:
    loss_cfg = DiffusionLossConfig(loss_mode="diffusion_art", **loss_kwargs)
    return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=loss_cfg)


def make_inputs(batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    old = torch.randn(batch, C, H, W, generator=generator)
    raw = torch.randn(batch, C, H, W, generator=generator)
    guidance = raw / raw.flatten(1).square().mean(dim=1).sqrt()[:, None, None, None]
    t = torch.linspace(1.0, 0.1, batch)
    return {
        "forward_prediction": (old + 0.01 * torch.randn(batch, C, H, W, generator=generator)).requires_grad_(True),
        "old_prediction": old,
        "t_expanded": t.view(batch, 1, 1, 1),
        "art_guidance": guidance,
        "sigma_next": (t - 0.1).clamp_min(0.0),
    }


def run_loss(inputs, **loss_kwargs):
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    data = TensorDict(
        {key: inputs[key] for key in DiffusionARTLoss.required_data_keys}, batch_size=[len(inputs["sigma_next"])]
    )
    return get_diffusion_loss_fn("diffusion_art")(
        config=make_actor(**loss_kwargs), model_output=model_output, data=data
    )


def test_registry_and_contract():
    loss_fn = get_diffusion_loss_fn("diffusion_art")
    assert isinstance(loss_fn, DiffusionARTLoss)
    assert loss_fn.uses_old_adapter and not loss_fn.requires_whole_groups
    assert DiffusionLossConfig(loss_mode="diffusion_art").art_mask_metric == "sde_kl"


@pytest.mark.parametrize("backend", ["fsdp", "fsdp2"])
def test_art_engine_is_registered_for_both_fsdp_backends(backend):
    from verl.workers.engine.base import EngineRegistry

    from verl_omni.workers.engine import ARTDiffusersFSDPEngine, NFTDiffusersFSDPEngine

    engine_cls = EngineRegistry.get_engine_cls(model_type="diffusion_art_model", backend=backend)
    assert engine_cls is ARTDiffusersFSDPEngine
    assert issubclass(engine_cls, NFTDiffusersFSDPEngine)


def test_loss_matches_the_pure_function_and_reports_metrics():
    inputs = make_inputs()
    result = run_loss(inputs, art_guidance_gain=0.5, art_mask_metric="fixed_mse", art_mask_threshold=1e-3)
    terms = art_loss(
        inputs["forward_prediction"],
        inputs["old_prediction"],
        inputs["art_guidance"],
        inputs["t_expanded"].flatten(),
        inputs["sigma_next"],
        guidance_gain=0.5,
        mask_metric="fixed_mse",
        mask_threshold=1e-3,
    )
    torch.testing.assert_close(result.loss, terms["per_item_loss"].mean())
    for key in (
        "actor/policy_loss",
        "actor/unmasked_loss",
        "actor/active_frac",
        "actor/mask_frac",
        "actor/outward_frac",
        "actor/old_deviate",
        "actor/movement_score",
        "actor/guidance_rms",
        "actor/total_loss",
    ):
        assert key in result.metrics, key
    assert result.metrics["actor/active_frac"] == 1.0
    assert result.metrics["actor/anchor_frac"] == 0.0
    assert result.metrics["actor/guidance_rms"] == pytest.approx(1.0, rel=1e-5)


def test_loss_splits_audio_video_segments_with_modality_weights():
    """With ``video_seq_len`` the loss is the weighted mean of two per-modality masked MSEs."""
    inputs = make_inputs()
    video_seq_len = torch.full((4,), 2, dtype=torch.long)  # dim 1 is C=4 here: tokens [0:2] video, [2:4] audio
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    data = TensorDict(
        {"art_guidance": inputs["art_guidance"], "sigma_next": inputs["sigma_next"], "video_seq_len": video_seq_len},
        batch_size=[4],
    )
    result = get_diffusion_loss_fn("diffusion_art")(
        config=make_actor(
            art_video_weight=3.0, art_audio_weight=1.0, art_mask_metric="fixed_mse", art_mask_threshold=1e9
        ),
        model_output=model_output,
        data=data,
    )
    sigma = inputs["t_expanded"].flatten()
    expected = torch.zeros(4)
    for rows, weight in ((slice(0, 2), 0.75), (slice(2, 4), 0.25)):
        terms = art_loss(
            inputs["forward_prediction"][:, rows],
            inputs["old_prediction"][:, rows],
            inputs["art_guidance"][:, rows],
            sigma,
            inputs["sigma_next"],
            mask_metric="fixed_mse",
            mask_threshold=1e9,
        )
        expected = expected + weight * terms["per_item_loss"]
    torch.testing.assert_close(result.loss, expected.mean())
    for modality in ("video", "audio"):
        for name in ("unmasked_loss", "active_frac", "mask_frac", "guidance_rms", "old_deviate"):
            assert f"actor/{modality}/{name}" in result.metrics
    with pytest.raises(ValueError, match="video_seq_len"):
        data["video_seq_len"] = torch.tensor([2, 2, 3, 2])
        get_diffusion_loss_fn("diffusion_art")(config=make_actor(), model_output=model_output, data=data)


def test_loss_restores_anchor_rows_toward_old_and_logs_rollout_gap():
    inputs = make_inputs()
    inputs["art_guidance"][1] = 0.0  # pairwise tie: no guidance
    anchor = torch.tensor([False, True, False, False])
    rollout_velocity = inputs["old_prediction"] + 0.1
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    data = TensorDict(
        {
            "art_guidance": inputs["art_guidance"],
            "sigma_next": inputs["sigma_next"],
            "art_anchor": anchor,
            "rollout_velocity": rollout_velocity,
        },
        batch_size=[4],
    )
    result = get_diffusion_loss_fn("diffusion_art")(config=make_actor(), model_output=model_output, data=data)
    (grad,) = torch.autograd.grad(result.loss, inputs["forward_prediction"])
    assert result.metrics["actor/anchor_frac"] == pytest.approx(0.25)
    assert torch.count_nonzero(grad[1]) > 0  # anchor row trains (toward v_old)
    drift = inputs["forward_prediction"][1].detach() - inputs["old_prediction"][1]
    torch.testing.assert_close(grad[1], 2 * inputs["t_expanded"].flatten()[1] * drift / (4 * C * H * W))
    assert result.metrics["actor/old_rollout_gap"] == pytest.approx(0.01, rel=1e-4)


def test_loss_accepts_full_shape_t_expanded_and_requires_its_inputs():
    inputs = make_inputs()
    inputs["t_expanded"] = inputs["t_expanded"].expand(-1, C, H, W)
    assert torch.isfinite(run_loss(inputs).loss)
    with pytest.raises(KeyError, match="art_guidance"):
        get_diffusion_loss_fn("diffusion_art")(
            config=make_actor(),
            model_output={key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys},
            data=TensorDict({"sigma_next": inputs["sigma_next"]}, batch_size=[4]),
        )


def test_through_the_worker_loss_entry_point_divides_by_gradient_accumulation():
    inputs = make_inputs()
    data = TensorDict({key: inputs[key] for key in DiffusionARTLoss.required_data_keys}, batch_size=[4])
    tu.assign_non_tensor(data, gradient_accumulation_steps=4, sp_size=1)
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    loss, metrics = diffusion_loss(make_actor(), model_output, data)
    torch.testing.assert_close(loss, run_loss(inputs).loss / 4)
    assert "actor/mask_frac" in metrics


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"art_guidance_gain": 0.0}, "art_guidance_gain.*positive"),
        ({"art_loss_space": "x0"}, "art_loss_space"),
        ({"art_mask_metric": "kl"}, "art_mask_metric"),
        ({"art_mask_threshold": 0.0}, "art_mask_threshold.*positive"),
        ({"art_reference_noise_level": 0.0}, "art_reference_noise_level.*positive"),
    ],
)
def test_invalid_loss_config_values_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DiffusionLossConfig(loss_mode="diffusion_art", **kwargs)


def test_algorithm_config_exposes_art_knobs():
    cfg = DiffusionAlgoConfig()
    assert cfg.art_pairwise_reward is False and cfg.art_exclude_final_step is True
    assert cfg.art_log_rollout_velocity_gap is False and cfg.art_modality_reward_weights is None
    assert DiffusionAlgoConfig(art_modality_reward_weights={"video": {"a": 1.0}}).art_modality_reward_weights == {
        "video": {"a": 1.0}
    }
    with pytest.raises(ValueError, match="art_modality_reward_weights"):
        DiffusionAlgoConfig(art_modality_reward_weights={"text": {"a": 1.0}})
    with pytest.raises(ValueError, match="non-negative"):
        DiffusionAlgoConfig(art_modality_reward_weights={"audio": {"a": -1.0}})
    with pytest.raises(ValueError, match="art_video_weight"):
        DiffusionLossConfig(loss_mode="diffusion_art", art_video_weight=0.0, art_audio_weight=0.0)


def make_rollout_batch(uid, num_steps=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(uid)
    schedule = torch.linspace(1000.0, 200.0, num_steps)
    return DataProto.from_dict(
        tensors={
            "all_latents": torch.randn(batch_size, num_steps + 1, C, H, W, generator=generator),
            "all_timesteps": schedule.expand(batch_size, -1).clone(),
            "all_log_probs": torch.zeros(batch_size, num_steps),
            "prompt_embeds": torch.zeros(batch_size, 7, 8),
        },
        non_tensors={"uid": np.array(uid, dtype=object)},
    )


def make_config(**algorithm_kwargs):
    algorithm = SimpleNamespace(
        norm_adv_by_std_in_grpo=True,
        global_std=True,
        timestep_fraction=1.0,
        art_pairwise_reward=False,
        art_exclude_final_step=False,
        art_log_rollout_velocity_gap=False,
        art_modality_reward_weights=None,
    )
    reward = algorithm_kwargs.pop("reward", None)
    for key, value in algorithm_kwargs.items():
        setattr(algorithm, key, value)
    algorithm.get = lambda key, default=None: getattr(algorithm, key, default)
    actor = SimpleNamespace(data_loader_seed=42)
    return SimpleNamespace(algorithm=algorithm, actor_rollout_ref=SimpleNamespace(actor=actor), reward=reward)


def test_prepare_actor_batch_writes_cached_states_and_shared_guidance():
    uid = ["b", "a", "b", "a", "c", "c"]
    batch = make_rollout_batch(uid)
    all_latents = batch.batch["all_latents"].clone()
    all_timesteps = batch.batch["all_timesteps"].clone()
    rewards = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.2, 0.2])

    result = DiffusionARTLoss.prepare_actor_batch(batch, rewards, make_config())

    num_steps = all_timesteps.shape[1]
    assert result.batch["train_latents"].shape == (6, num_steps, C, H, W)
    assert result.batch["train_timesteps"].shape == (6, num_steps)
    assert result.batch["art_guidance"].shape == (6, C, H, W)
    assert "all_latents" not in result.batch.keys() and "all_log_probs" not in result.batch.keys()
    assert "prompt_embeds" in result.batch.keys()
    torch.testing.assert_close(result.batch["latents_clean"], all_latents[:, -1])

    # Every selected state is the latent the rollout visited at the selected timestep, and the next
    # timestep follows the schedule (0 after the last step).
    for row in range(6):
        for slot in range(num_steps):
            step = (all_timesteps[row] == result.batch["train_timesteps"][row, slot]).nonzero().item()
            torch.testing.assert_close(result.batch["train_latents"][row, slot], all_latents[row, step])
            expected_next = all_timesteps[row, step + 1] if step + 1 < num_steps else torch.tensor(0.0)
            assert result.batch["train_timesteps_next"][row, slot] == expected_next
    assert sorted(result.batch["train_timesteps"][0].tolist()) == sorted(all_timesteps[0].tolist())

    guidance = result.batch["art_guidance"]
    torch.testing.assert_close(guidance[0], guidance[2])  # same prompt "b"
    torch.testing.assert_close(guidance[1], guidance[3])  # same prompt "a"
    assert torch.count_nonzero(guidance[4]) == 0 and torch.count_nonzero(guidance[5]) == 0  # constant rewards
    assert guidance[0].square().mean().sqrt().item() == pytest.approx(1.0, rel=1e-5)
    assert result.batch["advantages"].shape == (6, num_steps)
    assert result.batch["sample_level_rewards"].shape == (6, num_steps)


def test_prepare_actor_batch_subsamples_states_and_supports_pairwise_labels():
    uid = ["a", "a", "a", "a"]
    batch = make_rollout_batch(uid, num_steps=6)
    rewards = torch.tensor([0.9, 0.1, 0.4, 0.6])
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, rewards, make_config(timestep_fraction=0.5, art_pairwise_reward=True)
    )
    assert result.batch["train_latents"].shape == (4, 3, C, H, W)
    assert result.batch["train_timesteps"].shape == (4, 3)
    assert result.batch["art_guidance"][0].square().mean().sqrt().item() == pytest.approx(1.0, rel=1e-5)


def test_prepare_actor_batch_excludes_the_final_ode_position_by_default():
    batch = make_rollout_batch(["a", "a", "a", "a"], num_steps=5)
    schedule = batch.batch["all_timesteps"][0].clone()
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, torch.tensor([0.1, 0.2, 0.3, 0.4]), make_config(art_exclude_final_step=True)
    )
    assert result.batch["train_timesteps"].shape == (4, 4)
    assert schedule[-1] not in result.batch["train_timesteps"][0]
    assert "train_latents_next" not in result.batch.keys()


def test_prepare_actor_batch_draws_a_fresh_state_subset_per_rollout():
    """timestep_fraction < 1: the subset changes with global_steps and is reproducible for the same step."""
    uid = ["a", "a", "a", "a"]
    rewards = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def selection(step):
        batch = make_rollout_batch(uid, num_steps=8)
        batch.meta_info["global_steps"] = step
        result = DiffusionARTLoss.prepare_actor_batch(batch, rewards, make_config(timestep_fraction=0.5))
        return result.batch["train_timesteps"]

    first, second, first_again = selection(1), selection(2), selection(1)
    assert first.shape == (4, 4)
    torch.testing.assert_close(first, first_again)
    assert not torch.equal(first, second)
    # Without global_steps the uids (fresh every rollout) seed the draw deterministically.
    batch = make_rollout_batch(uid, num_steps=8)
    once = DiffusionARTLoss.prepare_actor_batch(batch, rewards, make_config(timestep_fraction=0.5)).batch[
        "train_timesteps"
    ]
    again = DiffusionARTLoss.prepare_actor_batch(
        make_rollout_batch(uid, num_steps=8), rewards, make_config(timestep_fraction=0.5)
    )
    torch.testing.assert_close(once, again.batch["train_timesteps"])


def test_prepare_actor_batch_pairwise_marks_ties_as_restore_anchors():
    uid = ["a", "a", "a", "a", "b", "b", "b", "b"]
    batch = make_rollout_batch(uid)
    # Group a: whichever random pairing is drawn, exactly one pair ties (0.9 vs 0.9) and one is decided.
    rewards = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.3, 0.3, 0.3, 0.3])
    result = DiffusionARTLoss.prepare_actor_batch(batch, rewards, make_config(art_pairwise_reward=True))
    anchor = result.batch["art_anchor"]
    guidance = result.batch["art_guidance"]
    assert anchor.shape == (8,)
    decided = ~anchor
    # Tie rows carry no guidance but are anchors; decided rows share the prompt's unit-RMS direction.
    assert torch.count_nonzero(guidance[anchor]) == 0
    assert guidance[decided].flatten(1).square().mean(dim=1).sqrt().allclose(torch.ones(int(decided.sum())), atol=1e-5)
    assert anchor[:4].sum() == 2 and anchor[4:].all()  # group b is all ties -> all anchors (batch has decided pairs)


def make_av_rollout_batch(uid, num_steps=4, video_tokens=5, audio_tokens=3, channels=2):
    """LTX-style layout: unified [video | audio] tokens, all_latents (B, T, ...) plus all_next_latents."""
    generator = torch.Generator().manual_seed(1)
    batch_size = len(uid)
    tokens = video_tokens + audio_tokens
    schedule = torch.linspace(1000.0, 250.0, num_steps)
    return DataProto.from_dict(
        tensors={
            "all_latents": torch.randn(batch_size, num_steps, tokens, channels, generator=generator),
            "all_next_latents": torch.randn(batch_size, num_steps, tokens, channels, generator=generator),
            "all_timesteps": schedule.expand(batch_size, -1).clone(),
            "video_seq_len": torch.full((batch_size,), video_tokens, dtype=torch.long),
        },
        non_tensors={
            "uid": np.array(uid, dtype=object),
            "reward/video_align": np.array([0.1, 0.9, 0.3, 0.5, 0.2, 0.8], dtype=np.float32),
            "reward/clap": np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.9], dtype=np.float32),
            "reward/desync": np.array([0.4, 0.2, 0.9, 0.1, 0.6, 0.3], dtype=np.float32),
        },
    )


def test_prepare_actor_batch_routes_rewards_per_modality_for_audio_video_latents():
    uid = ["a", "a", "a", "b", "b", "b"]
    batch = make_av_rollout_batch(uid)
    all_latents = batch.batch["all_latents"].clone()
    all_next = batch.batch["all_next_latents"].clone()
    combined = torch.tensor([0.5, 0.6, 0.7, 0.3, 0.2, 0.9])
    routing = {"video": {"video_align": 1.0, "desync": 1.0}, "audio": {"clap": 1.0, "desync": 1.0}}
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, combined, make_config(art_modality_reward_weights=routing, art_log_rollout_velocity_gap=True)
    )
    guidance = result.batch["art_guidance"]
    assert guidance.shape == (6, 8, 2)
    video, audio = guidance[:, :5], guidance[:, 5:]
    torch.testing.assert_close(video.flatten(1).square().mean(dim=1).sqrt(), torch.ones(6))
    torch.testing.assert_close(audio.flatten(1).square().mean(dim=1).sqrt(), torch.ones(6))
    torch.testing.assert_close(guidance[0], guidance[1])
    assert not torch.allclose(video[0], video[3])
    # Endpoint = last next-state; states are the visited ones; next states shipped for the velocity gap.
    torch.testing.assert_close(result.batch["latents_clean"], all_next[:, -1])
    assert result.batch["train_latents"].shape == (6, 4, 8, 2)
    torch.testing.assert_close(result.batch["train_latents"].sort(dim=1).values, all_latents.sort(dim=1).values)
    assert result.batch["train_latents_next"].shape == (6, 4, 8, 2)
    assert "all_next_latents" not in result.batch.keys() and "all_latents" not in result.batch.keys()


OMNINFT_REWARD_METRIC_SUFFIXES = ("mean", "std", "min", "max")
AV_NAMES = ["clap", "desync", "video_align"]


def expected_reward_metric_keys(names):
    keys = {"reward/sum/mean", "reward/sum/std"}
    for name in names:
        keys.update(f"reward/{name}/{stat}" for stat in OMNINFT_REWARD_METRIC_SUFFIXES)
    return keys


def expected_guidance_metric_keys(modalities):
    keys = set()
    for modality in modalities:
        prefix = f"{modality}/" if modality else ""
        keys.update(
            f"{prefix}{name}"
            for name in ("inactive_group_frac", "covariance_rms", "routed_reward_mean", "routed_reward_std")
        )
    return keys


def attached_metric_names(result):
    prefix = DiffusionARTLoss.batch_metric_prefix
    return {k[len(prefix) :] for k in result.batch.keys() if isinstance(k, str) and k.startswith(prefix)}


def test_prepare_actor_batch_attaches_omninft_reward_summaries_and_art_guidance_stats():
    uid = ["a", "a", "a", "b", "b", "b"]
    batch = make_av_rollout_batch(uid)
    raw = {n: torch.as_tensor(batch.non_tensor_batch[f"reward/{n}"]) for n in AV_NAMES}
    prefix = DiffusionARTLoss.batch_metric_prefix
    scalar_reward = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.2, 0.8])
    result = DiffusionARTLoss.prepare_actor_batch(batch, scalar_reward, make_config())
    assert attached_metric_names(result) == expected_reward_metric_keys(AV_NAMES) | expected_guidance_metric_keys(
        ["video", "audio"]
    )
    # OmniNFT semantics: raw (un-normalised) scores, whole rollout batch, population std, replicated per sample.
    torch.testing.assert_close(result.batch[f"{prefix}reward/clap/mean"], raw["clap"].mean().expand(6))
    torch.testing.assert_close(result.batch[f"{prefix}reward/clap/std"], raw["clap"].std(correction=0).expand(6))
    torch.testing.assert_close(result.batch[f"{prefix}reward/video_align/max"], raw["video_align"].max().expand(6))
    torch.testing.assert_close(result.batch[f"{prefix}reward/sum/mean"], sum(raw.values()).mean().expand(6))
    # Both prompts have varying rewards -> no inactive group; covariance RMS is a positive scalar per modality.
    assert result.batch[f"{prefix}video/inactive_group_frac"][0] == 0.0
    assert result.batch[f"{prefix}audio/covariance_rms"][0] > 0
    # Without routing info the scalar reward feeds both modalities, so routed stats equal the scalar's.
    torch.testing.assert_close(result.batch[f"{prefix}video/routed_reward_mean"][0], scalar_reward.mean())
    torch.testing.assert_close(result.batch[f"{prefix}audio/routed_reward_std"][0], scalar_reward.std(correction=0))
    # The loss reports them under OmniNFT's metric names from any micro batch slice.
    data = TensorDict(
        {k: v[:2] for k, v in result.batch.items() if isinstance(k, str) and k.startswith(prefix)}, batch_size=[2]
    )
    metrics = DiffusionARTLoss.collect_batch_metrics(data)
    assert {k for k in metrics if k.startswith("actor/reward/")} == {
        f"actor/{k}" for k in expected_reward_metric_keys(AV_NAMES)
    }
    assert metrics["actor/reward/clap/mean"] == pytest.approx(raw["clap"].mean().item())
    assert "actor/video/covariance_rms" in metrics and "actor/audio/inactive_group_frac" in metrics


def test_prepare_actor_batch_reward_summaries_and_inactive_groups():
    """Component summaries come from the reward/<name> extras; a constant-reward prompt is an inactive group."""
    uid = ["a", "a", "a", "b", "b", "b"]
    batch = make_av_rollout_batch(uid)
    for n in AV_NAMES:
        scores = np.asarray(batch.non_tensor_batch[f"reward/{n}"], dtype=np.float32).copy()
        scores[3:] = 0.5  # prompt b: constant rewards -> inactive group
        batch.non_tensor_batch[f"reward/{n}"] = scores
    matrix = torch.stack([torch.as_tensor(batch.non_tensor_batch[f"reward/{n}"]) for n in AV_NAMES], dim=1)
    prefix = DiffusionARTLoss.batch_metric_prefix
    result = DiffusionARTLoss.prepare_actor_batch(
        batch,
        matrix.sum(dim=1),
        make_config(art_modality_reward_weights={"video": {"video_align": 1.0}, "audio": {"clap": 1.0}}),
    )
    assert attached_metric_names(result) == expected_reward_metric_keys(AV_NAMES) | expected_guidance_metric_keys(
        ["video", "audio"]
    )
    torch.testing.assert_close(result.batch[f"{prefix}reward/sum/std"], matrix.sum(dim=1).std(correction=0).expand(6))
    assert result.batch[f"{prefix}video/inactive_group_frac"][0] == pytest.approx(0.5)
    assert result.batch[f"{prefix}video/routed_reward_std"][0] > 0
    # A 2-D reward tensor is not a supported input (no preserve_components path in this tree).
    with pytest.raises(ValueError, match=r"\(B,\) reward tensor"):
        DiffusionARTLoss.prepare_actor_batch(make_av_rollout_batch(uid), matrix, make_config())
    # Single-modality batch without component infos: only the modality-free guidance stats are attached.
    plain = DiffusionARTLoss.prepare_actor_batch(
        make_rollout_batch(["a", "a"]), torch.tensor([0.0, 1.0]), make_config()
    )
    assert attached_metric_names(plain) == expected_guidance_metric_keys([""])


def test_prepare_actor_batch_writes_per_sample_records(tmp_path):
    uid = ["a", "a", "a", "b", "b", "b"]
    batch = make_av_rollout_batch(uid)
    batch.meta_info["global_steps"] = 7
    batch.non_tensor_batch["extra_info"] = np.array([{"index": 10 + i // 3} for i in range(6)], dtype=object)
    routing = {"video": {"video_align": 1.0, "desync": 1.0}, "audio": {"clap": 1.0, "desync": 1.0}}
    reward = torch.tensor([0.5, 0.6, 0.7, 0.3, 0.2, 0.9])
    DiffusionARTLoss.prepare_actor_batch(
        batch,
        reward,
        make_config(art_modality_reward_weights=routing, art_sample_log_dir=str(tmp_path / "train_samples")),
    )
    path = tmp_path / "train_samples" / "step_7.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 6
    assert set(rows[0]) == {
        "sample",
        "uid",
        "group_index",
        "reward",
        "prompt_index",
        "reward_raw",
        "reward_standardized",
        "video_routed",
        "video_group_active",
        "audio_routed",
        "audio_group_active",
        "anchor",
    }
    assert rows[0]["uid"] == "a" and rows[3]["group_index"] == 1 and rows[3]["prompt_index"] == 11
    assert set(rows[0]["reward_raw"]) == set(AV_NAMES) == set(rows[0]["reward_standardized"])
    assert rows[0]["reward_raw"]["clap"] == pytest.approx(0.5) and rows[0]["reward"] == pytest.approx(0.5)
    assert rows[0]["video_group_active"] is True and rows[0]["anchor"] == [False, False]
    index = [json.loads(line) for line in (tmp_path / "train_samples" / "index.jsonl").read_text().splitlines()]
    assert index == [{"step": 7, "step_key": "7", "seed": 42 + 7, "file": "step_7.jsonl", "n_rows": 6}]
    # Writer alone: deterministic file name, one line per row, index appended (uid-hash fallback keeps step null).
    out = DiffusionARTLoss.write_sample_records(str(tmp_path / "w"), 3, [{"a": 1}, {"b": 2.5}], seed=45)
    assert out.endswith("step_3.jsonl") and pathlib.Path(out).read_text() == '{"a": 1}\n{"b": 2.5}\n'
    DiffusionARTLoss.write_sample_records(str(tmp_path / "w"), "uid123", [{"a": 1}], seed=123)
    index = [json.loads(line) for line in (tmp_path / "w" / "index.jsonl").read_text().splitlines()]
    assert [(e["step"], e["step_key"], e["file"], e["n_rows"]) for e in index] == [
        (3, "3", "step_3.jsonl", 2),
        (None, "uid123", "step_uid123.jsonl", 1),
    ]


def make_post_rollout_av_batch(uid, step=3):
    """What the v0 trainer hands prepare_actor_batch on an LTX-2.3 run: trajectory tensors, rollout media fields,
    per-sample `global_steps` stamped by the rollout strategy, reward component infos and the component matrix."""
    batch = make_av_rollout_batch(uid)
    batch_size = len(uid)
    names = AV_NAMES
    matrix = torch.stack([torch.as_tensor(batch.non_tensor_batch[f"reward/{n}"]) for n in names], dim=1)
    extra = DataProto.from_dict(
        tensors={
            "prompt_embeds": torch.zeros(batch_size, 7, 8),
            "audio": torch.zeros(batch_size, 2400),
            "audio_sample_rate": torch.full((batch_size,), 24000),
            "fps": torch.full((batch_size,), 24.0),
            "input_ids": torch.zeros(batch_size, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
        },
        non_tensors={
            "reward_inputs": np.array([{"text": {"video": "p", "audio": "p"}}] * batch_size, dtype=object),
            "extra_info": np.array([{"index": i // 3} for i in range(batch_size)], dtype=object),
            "data_source": np.array(["vggsound"] * batch_size, dtype=object),
            "global_steps": np.array([step] * batch_size, dtype=object),
        },
    )
    batch = batch.union(extra)
    batch.meta_info["reward_names"] = list(names)
    return batch, matrix


def test_prepared_batch_converts_to_the_worker_tensordict(tmp_path):
    """The real driver->worker conversion must accept everything prepare_actor_batch adds (no key collisions)."""
    from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict

    batch, matrix = make_post_rollout_av_batch(["a", "a", "a", "b", "b", "b"], step=3)
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, matrix.sum(dim=1), make_config(art_log_rollout_velocity_gap=True, art_sample_log_dir=str(tmp_path))
    )
    td = _to_diffusion_worker_tensordict(result)
    assert "art_guidance" in td.keys() and "train_latents_next" in td.keys()
    # The step comes from the per-sample rollout field, so the per-sample file is step_<N>.jsonl.
    assert (tmp_path / "step_3.jsonl").exists()
    assert "global_steps" not in result.meta_info


def test_meta_info_key_colliding_with_a_batch_field_is_rejected_by_the_conversion():
    """Pins the hazard: DataProto.to_tensordict folds meta_info into the TensorDict and asserts no collision, so
    trainers must not write e.g. meta_info['global_steps'] (the rollout already stamps it per sample)."""
    from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict

    batch, _ = make_post_rollout_av_batch(["a", "a", "a", "b", "b", "b"], step=3)
    batch.meta_info["global_steps"] = 3
    with pytest.raises(AssertionError):
        _to_diffusion_worker_tensordict(batch)


def test_step_is_read_from_the_per_sample_rollout_field():
    batch, matrix = make_post_rollout_av_batch(["a", "a", "a", "b", "b", "b"], step=11)
    assert DiffusionARTLoss._global_step(batch) == 11
    assert DiffusionARTLoss._step_seed(batch, 42) == 53
    without = make_av_rollout_batch(["a", "a", "a", "b", "b", "b"])
    assert DiffusionARTLoss._global_step(without) is None


def test_loss_metric_key_set_is_identical_across_micro_batches():
    """dp aggregation needs the same keys (one value each) on every micro batch and rank: empty buckets and
    all-inactive batches must still emit every key (0.0 + n_items_<bucket> = 0)."""

    def run(inputs):
        model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
        data = TensorDict(
            {
                "art_guidance": inputs["art_guidance"],
                "sigma_next": inputs["sigma_next"],
                "video_seq_len": torch.full((4,), 2, dtype=torch.long),
            },
            batch_size=[4],
        )
        return get_diffusion_loss_fn("diffusion_art")(config=make_actor(), model_output=model_output, data=data).metrics

    normal = run(make_inputs())  # t = 1.0, 0.7, 0.4, 0.1 -> every bucket populated
    one_bucket = make_inputs()
    one_bucket["t_expanded"] = torch.full((4, 1, 1, 1), 0.9)  # only t_high populated
    one_bucket["sigma_next"] = torch.full((4,), 0.8)
    inactive = make_inputs()
    inactive["art_guidance"] = torch.zeros_like(inactive["art_guidance"])  # every group inactive
    for other in (run(one_bucket), run(inactive)):
        assert set(other) == set(normal)
        assert all(isinstance(v, float) for v in other.values())
    empty = run(one_bucket)
    assert empty["actor/video/n_items_t_low"] == 0.0 and empty["actor/video/mask_frac_t_low"] == 0.0
    assert empty["actor/video/n_items_t_high"] == 4.0
    assert run(inactive)["actor/audio/n_items_t_high"] == 0.0 and run(inactive)["actor/audio/active_frac"] == 0.0


def test_loss_reports_per_modality_policy_loss_and_flow_time_buckets():
    inputs = make_inputs()  # t = 1.0, 0.7, 0.4, 0.1 -> one item per bucket boundary case
    video_seq_len = torch.full((4,), 2, dtype=torch.long)
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    data = TensorDict(
        {"art_guidance": inputs["art_guidance"], "sigma_next": inputs["sigma_next"], "video_seq_len": video_seq_len},
        batch_size=[4],
    )
    result = get_diffusion_loss_fn("diffusion_art")(config=make_actor(), model_output=model_output, data=data)
    for modality in ("video", "audio"):
        assert f"actor/{modality}/policy_loss" in result.metrics
        for bucket in ("t_high", "t_mid", "t_low"):
            assert f"actor/{modality}/movement_score_{bucket}" in result.metrics
            assert f"actor/{modality}/mask_frac_{bucket}" in result.metrics
            assert result.metrics[f"actor/{modality}/n_items_{bucket}"] >= 1.0
    video_losses = result.metrics["actor/video/policy_loss"]
    audio_losses = result.metrics["actor/audio/policy_loss"]
    assert result.metrics["actor/policy_loss"] == pytest.approx(0.5 * (video_losses + audio_losses), rel=1e-5)


def test_prepare_actor_batch_pairwise_k2_labels_each_modality_from_its_routed_reward():
    """DiffusionART-Pairwise with K=2 (paper Sec. 4.4, App. C.3): one pair per prompt; per modality the label
    comes from that modality's routed reward, the guidance is the unit-RMS winner-minus-loser direction of that
    modality's endpoint tokens, and a tied pair (constant routed reward) makes both rows restore anchors."""
    uid = ["a", "a", "b", "b", "c", "c"]
    batch = make_av_rollout_batch(uid)
    endpoints = batch.batch["all_next_latents"][:, -1].clone()
    # Video route = video_align, audio route = clap: prompt b ties on video_align, prompt c ties on clap.
    batch.non_tensor_batch["reward/video_align"] = np.array([0.9, 0.1, 0.5, 0.5, 0.2, 0.8], dtype=np.float32)
    batch.non_tensor_batch["reward/clap"] = np.array([0.1, 0.9, 0.3, 0.7, 0.4, 0.4], dtype=np.float32)
    routing = {"video": {"video_align": 1.0}, "audio": {"clap": 1.0}}
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, torch.zeros(6), make_config(art_pairwise_reward=True, art_modality_reward_weights=routing)
    )
    guidance, anchor = result.batch["art_guidance"], result.batch["art_anchor"]
    assert anchor.shape == (6, 2)  # one anchor flag per modality
    video, audio = guidance[:, :5], guidance[:, 5:]

    def unit(x):
        return x / x.square().mean().sqrt()

    # Prompt a: row 0 wins on video, row 1 wins on audio -> opposite directions per modality, same for both rows.
    torch.testing.assert_close(video[0], unit(endpoints[0, :5] - endpoints[1, :5]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(video[1], video[0])
    torch.testing.assert_close(audio[0], unit(endpoints[1, 5:] - endpoints[0, 5:]), rtol=1e-5, atol=1e-6)
    assert anchor[0].tolist() == [False, False]
    # Prompt b: video tie -> zero video guidance and video anchors; audio decided.
    assert torch.count_nonzero(video[2:4]) == 0 and anchor[2].tolist() == [True, False]
    assert audio[2].square().mean().sqrt().item() == pytest.approx(1.0, rel=1e-5)
    # Prompt c: audio tie -> audio anchors, video decided.
    assert torch.count_nonzero(audio[4:6]) == 0 and anchor[4].tolist() == [False, True]
    # Odd group sizes are rejected.
    with pytest.raises(ValueError, match="even group size"):
        DiffusionARTLoss.prepare_actor_batch(
            make_av_rollout_batch(["a", "a", "a", "b", "b", "b"]),
            torch.zeros(6),
            make_config(art_pairwise_reward=True, art_modality_reward_weights=routing),
        )


def test_prepare_actor_batch_reads_routing_weights_from_reward_functions():
    from omegaconf import OmegaConf

    uid = ["a", "a", "a", "b", "b", "b"]
    reward = OmegaConf.create(
        {
            "reward_functions": {
                "video_align": {"routing_weights": {"video": 1.0, "audio": 0.0}},
                "clap": {"routing_weights": {"video": 0.0, "audio": 1.0}},
                "desync": {"routing_weights": {"video": 1.0, "audio": 1.0}},
            }
        }
    )
    explicit = DiffusionARTLoss.prepare_actor_batch(
        make_av_rollout_batch(uid),
        torch.zeros(6),
        make_config(
            art_modality_reward_weights={
                "video": {"video_align": 1.0, "desync": 1.0},
                "audio": {"clap": 1.0, "desync": 1.0},
            }
        ),
    )
    from_reward_cfg = DiffusionARTLoss.prepare_actor_batch(
        make_av_rollout_batch(uid), torch.zeros(6), make_config(reward=reward)
    )
    torch.testing.assert_close(explicit.batch["art_guidance"], from_reward_cfg.batch["art_guidance"])
    with pytest.raises(ValueError, match="per-component scores"):
        DiffusionARTLoss.prepare_actor_batch(
            make_av_rollout_batch(uid),
            torch.zeros(6),
            make_config(art_modality_reward_weights={"video": {"missing": 1.0}, "audio": {"clap": 1.0}}),
        )


def test_prepare_actor_batch_rejects_missing_or_truncated_trajectories():
    batch = make_rollout_batch(["a", "a"])
    with pytest.raises(ValueError, match="all_latents"):
        DiffusionARTLoss.prepare_actor_batch(
            DataProto.from_dict(
                tensors={"all_timesteps": batch.batch["all_timesteps"]},
                non_tensors={"uid": np.array(["a", "a"], dtype=object)},
            ),
            torch.tensor([0.0, 1.0]),
            make_config(),
        )
    truncated = make_rollout_batch(["a", "a"])
    truncated.batch["all_latents"] = truncated.batch["all_latents"][:, 1:]
    with pytest.raises(ValueError, match="full ODE trajectory"):
        DiffusionARTLoss.prepare_actor_batch(truncated, torch.tensor([0.0, 1.0]), make_config())


def compose_trainer_config(overrides):
    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


ART_OVERRIDES = [
    "data.train_batch_size=24",
    "algorithm.trainer_type=direct_preference",
    "algorithm.sample_source=online",
    "actor_rollout_ref.model.algorithm=diffusion_art",
    "actor_rollout_ref.model.model_type=diffusion_art_model",
    "actor_rollout_ref.model.lora_rank=64",
    'actor_rollout_ref.model.policy_state_adapters=["default","old"]',
    "actor_rollout_ref.actor.ppo_mini_batch_size=12",
    "actor_rollout_ref.rollout.rollout_adapter=old",
    "actor_rollout_ref.rollout.algo.noise_level=0.0",
    "actor_rollout_ref.rollout.n=8",
    "algorithm.old_policy_decay_schedule=copy",
    "algorithm.old_policy_update_interval=1",
]


def test_validate_trainer_config_accepts_the_recipe_and_cascades_the_algorithm_name(caplog):
    config = compose_trainer_config(ART_OVERRIDES)
    assert config.actor_rollout_ref.actor.diffusion_loss.loss_mode == "diffusion_art"
    with caplog.at_level(logging.WARNING):
        DiffusionARTLoss.validate_trainer_config(config)
    assert "two disjoint updates" not in caplog.text


def test_validate_trainer_config_warns_on_a_single_update_per_rollout(caplog):
    config = compose_trainer_config(ART_OVERRIDES + ["actor_rollout_ref.actor.ppo_mini_batch_size=24"])
    with caplog.at_level(logging.WARNING):
        DiffusionARTLoss.validate_trainer_config(config)
    assert "two disjoint updates" in caplog.text


@pytest.mark.parametrize(
    "override, match",
    [
        ("actor_rollout_ref.rollout.rollout_adapter=default", "rollout_adapter must be old"),
        ("actor_rollout_ref.rollout.algo.noise_level=0.7", "noise_level must be 0.0"),
        ("actor_rollout_ref.model.lora_rank=0", "lora_rank must be > 0"),
        ("actor_rollout_ref.actor.use_kl_loss=true", "use_kl_loss must be False"),
        ("actor_rollout_ref.actor.enable_timestep_staging=true", "enable_timestep_staging must be False"),
        ("algorithm.train_timestep_count=3", "only used by the dgpo loss"),
        ("algorithm.old_policy_decay_schedule=linear_to_0_5", "old_policy_decay_schedule must be copy"),
        ("algorithm.old_policy_decay=0.5", "old_policy_decay must be null or 0.0"),
        ("algorithm.old_policy_update_interval=2", "old_policy_update_interval must be 1"),
        ("actor_rollout_ref.actor.ppo_epochs=2", "ppo_epochs must be 1"),
        ("actor_rollout_ref.rollout.n=1", "rollout.n must be >= 2"),
        ("actor_rollout_ref.model.pipeline.true_cfg_scale=2.0", "true_cfg_scale=2.0 must equal"),
        ("+actor_rollout_ref.rollout.pipeline.video_modality_scale=3.0", "video_modality_scale=3.0 must be 1.0"),
        ("+actor_rollout_ref.rollout.pipeline.audio_rescale_scale=0.7", "audio_rescale_scale=0.7 must be 0.0"),
    ],
)
def test_validate_trainer_config_rejects_unsupported_settings(override, match):
    config = compose_trainer_config(ART_OVERRIDES + [override])
    with pytest.raises(ValueError, match=match):
        DiffusionARTLoss.validate_trainer_config(config)


@pytest.mark.parametrize(
    "n, mini, micro, gpus, ok",
    [
        (8, 16, 16, 8, True),  # LTX K=8 arm: 16 samples per rank, one micro batch of 16
        (2, 16, 16, 8, False),  # LTX K=2 arm as first launched: 4 samples per rank, micro 16 -> engine assertion
        (2, 16, 2, 8, True),  # K=2 fix: 4 samples per rank = 2 micro batches of 2
        (2, 16, 4, 8, True),  # also valid: one micro batch of 4
        (2, 16, 3, 8, False),  # 4 samples per rank not divisible by 3
        (4, 3, 1, 8, False),  # 12 samples not divisible by dp 8
    ],
)
def test_validate_trainer_config_checks_per_rank_micro_batching(n, mini, micro, gpus, ok):
    overrides = [
        o
        for o in ART_OVERRIDES
        if not o.startswith(("actor_rollout_ref.rollout.n=", "actor_rollout_ref.actor.ppo_mini_batch_size="))
    ]
    overrides += [
        f"actor_rollout_ref.rollout.n={n}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro}",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.nnodes=1",
        f"data.train_batch_size={2 * mini}",
    ]
    config = compose_trainer_config(overrides)
    if ok:
        DiffusionARTLoss.validate_trainer_config(config)
    else:
        with pytest.raises(ValueError, match="divisible by"):
            DiffusionARTLoss.validate_trainer_config(config)


def test_validate_trainer_config_allows_guidance_extras_in_validation_only():
    config = compose_trainer_config(
        ART_OVERRIDES
        + [
            "+actor_rollout_ref.rollout.val_kwargs.pipeline.video_modality_scale=3.0",
            "+actor_rollout_ref.rollout.val_kwargs.pipeline.audio_rescale_scale=0.7",
        ]
    )
    DiffusionARTLoss.validate_trainer_config(config)


def test_loss_uses_per_segment_flow_times_when_given():
    """``segment_sigma`` lets each modality use its own flow time (MiniMax H3 video/audio sigma grids)."""
    inputs = make_inputs()
    video_seq_len = torch.full((4,), 2, dtype=torch.long)
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    sigma = inputs["t_expanded"].flatten()
    audio_sigma = sigma * 0.5
    audio_sigma_next = inputs["sigma_next"] * 0.5
    data = TensorDict(
        {
            "art_guidance": inputs["art_guidance"],
            "sigma_next": inputs["sigma_next"],
            "video_seq_len": video_seq_len,
            "segment_sigma": torch.stack([sigma, audio_sigma], dim=1),
            "segment_sigma_next": torch.stack([inputs["sigma_next"], audio_sigma_next], dim=1),
        },
        batch_size=[4],
    )
    config = make_actor(art_mask_metric="fixed_mse", art_mask_threshold=1e9)
    result = get_diffusion_loss_fn("diffusion_art")(config=config, model_output=model_output, data=data)
    expected = torch.zeros(4)
    for rows, seg_sigma, seg_next in (
        (slice(0, 2), sigma, inputs["sigma_next"]),
        (slice(2, 4), audio_sigma, audio_sigma_next),
    ):
        terms = art_loss(
            inputs["forward_prediction"][:, rows],
            inputs["old_prediction"][:, rows],
            inputs["art_guidance"][:, rows],
            seg_sigma,
            seg_next,
            mask_metric="fixed_mse",
            mask_threshold=1e9,
        )
        expected = expected + 0.5 * terms["per_item_loss"]
    torch.testing.assert_close(result.loss, expected.mean())
    # The audio segment's loss weight is its own (halved) flow time, so it differs from the shared-sigma result.
    del data["segment_sigma"], data["segment_sigma_next"]
    shared = get_diffusion_loss_fn("diffusion_art")(config=config, model_output=model_output, data=data)
    assert not torch.allclose(result.loss, shared.loss)
    with pytest.raises(ValueError, match="segment_sigma"):
        data["segment_sigma"] = torch.stack([sigma, audio_sigma, audio_sigma], dim=1)
        data["segment_sigma_next"] = torch.stack([inputs["sigma_next"], audio_sigma_next], dim=1)
        get_diffusion_loss_fn("diffusion_art")(config=config, model_output=model_output, data=data)


def test_prepare_actor_batch_gathers_per_segment_timesteps():
    uid = ["a", "a", "b", "b"]
    batch = make_rollout_batch(uid, num_steps=5)
    all_timesteps = batch.batch["all_timesteps"]
    audio = all_timesteps * 0.5
    batch.batch["all_segment_timesteps"] = torch.stack([all_timesteps, audio], dim=-1)
    batch.batch["video_seq_len"] = torch.full((4,), 2, dtype=torch.long)
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, torch.tensor([0.1, 0.9, 0.3, 0.5]), make_config(art_exclude_final_step=True)
    )
    seg = result.batch["train_segment_timesteps"]
    seg_next = result.batch["train_segment_timesteps_next"]
    assert seg.shape == (4, 4, 2) and seg_next.shape == (4, 4, 2)
    torch.testing.assert_close(seg[..., 0], result.batch["train_timesteps"])
    torch.testing.assert_close(seg[..., 1], result.batch["train_timesteps"] * 0.5)
    torch.testing.assert_close(seg_next[..., 0], result.batch["train_timesteps_next"])
    torch.testing.assert_close(seg_next[..., 1], result.batch["train_timesteps_next"] * 0.5)
    assert "all_segment_timesteps" not in result.batch.keys()
    with pytest.raises(ValueError, match="all_segment_timesteps"):
        bad = make_rollout_batch(uid, num_steps=5)
        bad.batch["all_segment_timesteps"] = torch.zeros(4, 5, 3)
        bad.batch["video_seq_len"] = torch.full((4,), 2, dtype=torch.long)
        DiffusionARTLoss.prepare_actor_batch(bad, torch.tensor([0.1, 0.9, 0.3, 0.5]), make_config())


def test_validate_trainer_config_rejects_paired_preference():
    """The trainers double the mini batch for algorithm.paired_preference, which the per-rank micro-batching check
    does not model; DiffusionART's pairwise variant is algorithm.art_pairwise_reward."""
    config = compose_trainer_config(ART_OVERRIDES + ["algorithm.paired_preference=true"])
    with pytest.raises(ValueError, match="paired_preference"):
        DiffusionARTLoss.validate_trainer_config(config)
