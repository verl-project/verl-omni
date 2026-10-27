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
"""CPU regression: validation must denoise with the *validation* diffusion kernel."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.agent_loop import bagel_corl_tq as tq_mod
from verl_omni.agent_loop.bagel_corl import validation_artifact_relpath
from verl_omni.agent_loop.bagel_corl_gen_serve import stash_gen_row_from_diffusion_output

# The recipe's real numbers: training is deliberately stochastic (``noise_level=0.7``,
# 15 steps) while validation pins the deterministic reference (``noise_level=0.0``, 50
# steps).  A test that used 0.7 on both sides could not tell the two kernels apart.
_TRAIN_STEPS = 15
_TRAIN_NOISE = 0.7
_VAL_STEPS = 50
_VAL_NOISE = 0.0


def _rollout_cfg():
    """A training kernel plus a validation overlay, shaped like ``DiffusionRolloutConfig``."""
    return OmegaConf.create(
        {
            "pipeline": {
                "num_inference_steps": _TRAIN_STEPS,
                "noise_level": _TRAIN_NOISE,
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.5,
            },
            "algo": {"noise_level": _TRAIN_NOISE, "sde_type": "sde", "sde_window_size": 2},
            "val_kwargs": {
                "seed": 123,
                # Val overrides only what it needs; ``cfg_*`` and ``sde_type`` must
                # survive from the training config.
                "pipeline": {"num_inference_steps": _VAL_STEPS},
                "algo": {"noise_level": _VAL_NOISE},
            },
        }
    )


def _train_sampling_params() -> dict:
    """What the base worker hands down for a training batch."""
    return {
        "num_inference_steps": _TRAIN_STEPS,
        "noise_level": _TRAIN_NOISE,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.5,
        "sde_type": "sde",
        "sde_window_size": 2,
        "global_steps": 30,
        "logprobs": True,
    }


def test_val_overlay_swaps_in_the_validation_kernel():
    """The two knobs the recipe sets on ``val_kwargs`` must actually win."""
    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), _rollout_cfg())

    assert out["num_inference_steps"] == _VAL_STEPS, "VAL_GEN_STEPS never reached the request"
    assert out["noise_level"] == _VAL_NOISE, "VAL_NOISE_LEVEL never reached the request"
    assert out["seed"] == 123


def test_val_overlay_keeps_train_values_for_keys_val_does_not_set():
    """The overlay is a merge, not a replacement: unset val keys keep the train value."""
    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), _rollout_cfg())

    assert out["cfg_text_scale"] == 4.0
    assert out["cfg_img_scale"] == 1.5
    assert out["sde_type"] == "sde"
    assert out["sde_window_size"] == 2
    assert out["global_steps"] == 30, "non-AR worker keys must survive for the trainer"


def test_val_overlay_clears_logprobs():
    """``logprobs=False`` is required: ``build_gen_sampling_params`` refuses a request that wants log-probs *and*
    carries ``noise_level <= 0``."""
    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), _rollout_cfg())

    assert out["logprobs"] is False


def test_val_overlay_accepts_a_flowgrpo_refused_combination():
    """The overlay's own output must not trip ``build_gen_sampling_params``'s guard."""
    from verl_omni.agent_loop.bagel_corl_gen_serve import build_gen_sampling_params

    rollout = _rollout_cfg()
    rollout["calculate_log_probs"] = True
    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), rollout)

    # ``base["logprobs"]=False`` is the per-phase answer, so the guard must pass.
    built = build_gen_sampling_params(rollout, base=out, seed=7)
    assert built["noise_level"] == _VAL_NOISE
    assert built["logprobs"] is False


def test_val_overlay_does_not_mutate_its_input():
    """The worker reuses one ``sampling_params`` dict across sibling rollouts."""
    params = _train_sampling_params()
    before = dict(params)

    tq_mod.val_overlay_sampling_params(params, _rollout_cfg())

    assert params == before, "training siblings would inherit the val kernel"


def _worker(monkeypatch):
    """Instantiate the real worker class without Ray, capturing what the base receives."""
    captured: dict = {}

    async def fake_run_prompt(_self, prompt, sampling_params, trajectory, trace=False):
        captured["params"] = dict(sampling_params)
        captured["trajectory"] = trajectory

    base = tq_mod._AgentLoopWorkerTQBase
    monkeypatch.setattr(base, "_run_prompt", fake_run_prompt)

    # ``@ray.remote`` replaced the class object, so reach the real one to bypass __init__.
    worker_cls = tq_mod.BagelCorlAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.config = OmegaConf.create({"actor_rollout_ref": {"rollout": _rollout_cfg()}})
    return worker, captured


def test_run_prompt_overlays_the_val_kernel_only_when_validating(monkeypatch):
    """Training rollouts must stay stochastic; only validation gets the ODE reference."""
    worker, captured = _worker(monkeypatch)

    asyncio.run(worker._run_prompt({"uid": "u"}, _train_sampling_params(), trajectory={"validate": True}))
    assert captured["params"]["noise_level"] == _VAL_NOISE
    assert captured["params"]["num_inference_steps"] == _VAL_STEPS
    assert captured["params"]["logprobs"] is False

    asyncio.run(worker._run_prompt({"uid": "u"}, _train_sampling_params(), trajectory={"validate": False}))
    assert captured["params"]["noise_level"] == _TRAIN_NOISE, "training must stay SDE"
    assert captured["params"]["num_inference_steps"] == _TRAIN_STEPS
    assert captured["params"]["logprobs"] is True, "GEN traj stash needs log-probs"


def test_val_overlay_tolerates_a_struct_val_kwargs_without_seed():
    """Regression: the worker's ``val_kwargs`` is a *struct* node that need not carry ``seed``."""
    val_kwargs = OmegaConf.create(
        {
            "pipeline": {"num_inference_steps": _VAL_STEPS},
            "algo": {"noise_level": _VAL_NOISE},
        }
    )
    OmegaConf.set_struct(val_kwargs, True)  # the real node is struct; missing keys raise
    assert "seed" not in val_kwargs

    rollout = OmegaConf.create({"val_kwargs": val_kwargs})
    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), rollout)

    assert out["num_inference_steps"] == _VAL_STEPS, "the val kernel must still be applied"
    assert out["noise_level"] == _VAL_NOISE
    assert out["logprobs"] is False
    assert "seed" not in out, "must not invent a seed the recipe never set"


def test_val_overlay_applies_to_the_runtime_dataclass_shape():
    """The worker sees a *dataclass* ``DiffusionSamplingConfig``, not an OmegaConf node."""
    from verl_omni.workers.config.diffusion.rollout import (
        DiffusionPipelineConfig,
        DiffusionRolloutAlgoConfig,
        DiffusionSamplingConfig,
    )

    val_kwargs = DiffusionSamplingConfig(
        seed=123,
        pipeline=DiffusionPipelineConfig(num_inference_steps=_VAL_STEPS),
        algo=DiffusionRolloutAlgoConfig(noise_level=_VAL_NOISE),
    )
    rollout = types.SimpleNamespace(val_kwargs=val_kwargs)

    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), rollout)

    assert out["num_inference_steps"] == _VAL_STEPS
    assert out["noise_level"] == _VAL_NOISE
    assert out["seed"] == 123
    assert out["logprobs"] is False
    assert out["global_steps"] == 30


def test_train_rollout_cfg_is_not_overlaid_when_val_kwargs_lack_the_subtree():
    """A val node with no ``pipeline``/``algo`` must not fabricate an ODE kernel."""
    rollout = OmegaConf.create({"val_kwargs": {"seed": 7}})

    out = tq_mod.val_overlay_sampling_params(_train_sampling_params(), rollout)

    assert out["noise_level"] == _TRAIN_NOISE, "must not invent noise_level=0"
    assert out["num_inference_steps"] == _TRAIN_STEPS


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        ({"logprobs": False}, "val/step_000030/sample_1.01"),
        ({"logprobs": True}, "step_000030/sample_1.01"),
        ({}, "step_000030/sample_1.01"),
        (None, "step_000030/sample_1.01"),
    ],
)
def test_validation_artifact_relpath_tags_only_validation(sampling_params, expected):
    """Val PNGs move under ``val/``; the training tree stays byte-identical."""
    assert validation_artifact_relpath("step_000030/sample_1.01", sampling_params) == expected


# The GEN traj stash is the *saving* path for the rollout image, and it used to
# require the SDE trajectory unconditionally.  Validation runs ODE
# (``noise_level=0.0``), which by construction has no SDE log-probs, so the stash
# raised *before* ``save_diffusion_image`` and validation produced no image at all.
# Measured 2026-09-30 on ``bagel_corl_rm1_0123_20260930_095143``: the init
# validation at ``step=0`` died with
#   "GEN traj stash incomplete from vLLM-Omni (all_latents=ok, timesteps=ok,
#    log_probs=missing)"
# and ``find /tmp/bagel_corl_gen -name '*.png' -newermt ... `` was empty.


def _noise_free_output(**extra):
    """A diffusion output for an ODE decode: an image but no SDE trajectory."""
    return types.SimpleNamespace(
        diffusion_output=torch.zeros(3, 8, 8),
        extra_fields=dict(extra),
        log_probs=None,
        stop_reason="stop",
    )


def test_validation_row_keeps_its_image_without_a_trajectory(tmp_path):
    """``require_traj=False`` must yield a ``valid`` row with an image and no traj."""
    row = stash_gen_row_from_diffusion_output(
        _noise_free_output(), seed=1, image_root=str(tmp_path), require_traj=False
    )

    assert row["valid"] is True, "val images are dumped/scored only for valid rows"
    assert row["all_latents"] is None
    assert row["timesteps"] is None
    assert row["rollout_log_probs"] is None
    assert row["image_path"] and Path(row["image_path"]).exists(), "image must be written"


def test_training_still_fails_closed_without_a_trajectory(tmp_path):
    """The default is unchanged: a training GEN call with no traj is a hard error."""
    with pytest.raises(RuntimeError, match="traj stash incomplete"):
        stash_gen_row_from_diffusion_output(_noise_free_output(), seed=1, image_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == [], "fail-closed must not write an image"


def test_training_row_still_carries_the_full_trajectory(tmp_path):
    """With traj present the row is byte-for-byte the FlowGRPO seed it always was."""
    output = types.SimpleNamespace(
        diffusion_output=torch.zeros(3, 8, 8),
        extra_fields={
            "all_latents": torch.zeros(2, 4),
            "all_timesteps": torch.tensor([1.0, 2.0]),
        },
        log_probs=torch.tensor([0.1, 0.2]),
        stop_reason="stop",
    )

    row = stash_gen_row_from_diffusion_output(output, seed=3, image_root=str(tmp_path))

    assert row["valid"] is True
    assert row["all_latents"] is not None
    assert row["timesteps"] is not None
    assert row["rollout_log_probs"] is not None
    assert Path(row["image_path"]).exists()


def test_a_valid_row_without_traj_is_dropped_from_the_gen_training_lane():
    """The row shape validation produces must be one the packer already understands."""
    from verl_omni.agent_loop.bagel_corl_lib import GenSample

    sample = GenSample(
        gen_sample_uid="g.01",
        gen_group_uid="g",
        seed_index=0,
        valid=True,
        prompt_token_ids=[1, 2, 3],
        image_path="/tmp/whatever.png",
    )

    assert tq_mod._sample_has_gen_traj(sample) is False, "no traj -> not a training seed"
    assert sample.valid is True, "but still valid, so the image is dumped and scored"
