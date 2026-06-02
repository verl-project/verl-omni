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

"""CPU tests for Wan2.2 DanceGRPO pipeline adapters.

This suite covers the three integration points described in
``docs/contributing/integrating_a_diffusion_model.md``:

1. **Shared utilities** (``wan22_dance_grpo/common.py``): time-shift, CFG,
   flatten, and deterministic seeding.
2. **Training adapter** (``wan22_dance_grpo/diffusers_training_adapter.py``):
   input preparation and the forward + reverse-SDE sampling step.
3. **Registry dispatch** — verifying that both adapters are discoverable via
   ``DiffusionModelBase`` and ``VllmOmniPipelineBase`` under the
   ``("WanPipeline", "dance_grpo")`` key.

All tests run on CPU without loading model weights.
"""

# =============================================================================
# NOTE: ``verl_omni/__init__.py`` eagerly imports ``verl_omni.pipelines``,
# which in turn imports ``vllm_omni`` sub-modules that depend on ``vllm``.
# Because ``vllm`` is not available locally, we import the shared-utility
# module ``common.py`` via ``importlib`` directly, *without* going through
# the ``verl_omni`` package init chain.
#
# The adapter and registry tests are guarded by a ``_VLLM_AVAILABLE`` flag
# that is set only when all required imports succeed.
# =============================================================================

import importlib.util
import os
import sys

import pytest
import torch

# ── Pure-Python utility tests (no vllm required) ─────────────────────────────

_common_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "verl_omni",
    "pipelines",
    "wan22_dance_grpo",
    "common.py",
)
_common_spec = importlib.util.spec_from_file_location("wan22_common", _common_path)
_common_mod = importlib.util.module_from_spec(_common_spec)
sys.modules["wan22_common"] = _common_mod
_common_spec.loader.exec_module(_common_mod)

sd3_time_shift = _common_mod.sd3_time_shift
apply_cfg = _common_mod.apply_cfg
flatten = _common_mod.flatten
seed_from_prompt_ids = _common_mod.seed_from_prompt_ids


class TestSD3TimeShift:
    """``sd3_time_shift`` — the Wan2.2 timestep rescaling function."""

    def test_shape_preserved(self):
        t = torch.linspace(0, 1, 50)
        out = sd3_time_shift(5.0, t)
        assert out.shape == t.shape
        assert out.dtype == t.dtype

    def test_monotonic(self):
        t = torch.linspace(0, 1, 100)
        out = sd3_time_shift(5.0, t)
        diffs = out[1:] - out[:-1]
        assert torch.all(diffs >= 0), "output must be monotonically non-decreasing"

    def test_shift_one_is_identity(self):
        t = torch.linspace(0, 1, 50)
        out = sd3_time_shift(1.0, t)
        torch.testing.assert_close(out, t)

    def test_edge_cases(self):
        assert sd3_time_shift(3.0, torch.tensor(0.0)).item() == 0.0
        assert sd3_time_shift(3.0, torch.tensor(1.0)).item() == 1.0

    def test_larger_shift_increases_curvature(self):
        t = torch.tensor(0.5)
        low = sd3_time_shift(2.0, t)
        high = sd3_time_shift(10.0, t)
        assert high > low


class TestApplyCFG:
    """``apply_cfg`` — standard classifier-free guidance."""

    def test_scale_one_is_positive_only(self):
        pos = torch.randn(2, 16, 8, 8)
        neg = torch.randn(2, 16, 8, 8)
        out = apply_cfg(pos, neg, 1.0)
        torch.testing.assert_close(out, pos)

    def test_scale_two_formula(self):
        pos = torch.ones(2, 4)
        neg = torch.zeros(2, 4)
        out = apply_cfg(pos, neg, 2.0)
        expected = neg + 2.0 * (pos - neg)
        torch.testing.assert_close(out, expected)

    def test_scale_zero_returns_negative_only(self):
        pos = torch.ones(2, 4)
        neg = torch.zeros(2, 4)
        out = apply_cfg(pos, neg, 0.0)
        torch.testing.assert_close(out, neg)


class TestFlatten:
    """``flatten`` — nested-list flattening generator."""

    def test_flat_list(self):
        assert list(flatten([1, 2, 3])) == [1, 2, 3]

    def test_nested_list(self):
        assert list(flatten([[1, 2], [3, [4, 5]]])) == [1, 2, 3, 4, 5]

    def test_empty_nested(self):
        assert list(flatten([[], [[[]]], [1]])) == [1]

    def test_single_element(self):
        assert list(flatten([42])) == [42]


class TestSeedFromPromptIDs:
    """``seed_from_prompt_ids`` — deterministic seed derivation."""

    def test_tensor_input(self):
        ids = torch.randint(0, 50000, (77,))
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**64

    def test_flat_list_input(self):
        ids = [49406, 320, 1000]
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)

    def test_nested_list_input(self):
        ids = [[49406, 320], [1000, 5001]]
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)

    def test_deterministic(self):
        ids = torch.randint(0, 50000, (77,))
        s1 = seed_from_prompt_ids(ids)
        s2 = seed_from_prompt_ids(ids.clone())
        assert s1 == s2

    def test_different_ids_different_seeds(self):
        ids_a = torch.zeros(10, dtype=torch.long)
        ids_b = torch.ones(10, dtype=torch.long)
        assert seed_from_prompt_ids(ids_a) != seed_from_prompt_ids(ids_b)

    def test_2d_tensor(self):
        ids = torch.randint(0, 50000, (1, 77))
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            seed_from_prompt_ids(42)


# ── Adapter tests (require vllm-omni / vllm) ─────────────────────────────────

try:
    from unittest.mock import MagicMock

    import torch
    from tensordict import TensorDict

    import verl_omni  # noqa: F401 — triggers the full package init chain
    from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
    from verl_omni.pipelines.wan22_dance_grpo.diffusers_training_adapter import (
        Wan22DanceGRPO,
        _configure_wan_scheduler,
    )
    from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
    from verl_omni.workers.config.diffusion.rollout import (
        DiffusionPipelineConfig,
        DiffusionRolloutAlgoConfig,
    )

    _VLLM_AVAILABLE = True
except Exception:
    _VLLM_AVAILABLE = False


# ===========================================================================
# Helpers
# ===========================================================================


def _make_model_config(
    *,
    guidance_scale: float | None = 1.0,
    noise_level: float = 1.0,
    sde_type: str = "dance_sde",
) -> "DiffusionModelConfig":
    """Build a minimal ``DiffusionModelConfig`` without hitting ``__post_init__``."""
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "WanPipeline")
    object.__setattr__(cfg, "algorithm", "dance_grpo")
    object.__setattr__(cfg, "external_lib", None)
    object.__setattr__(
        cfg,
        "pipeline",
        DiffusionPipelineConfig(
            guidance_scale=guidance_scale,
            num_inference_steps=10,
            num_frames=5,
            height=64,
            width=64,
        ),
    )
    object.__setattr__(
        cfg,
        "algo",
        DiffusionRolloutAlgoConfig(
            noise_level=noise_level,
            sde_type=sde_type,
        ),
    )
    return cfg


def _wan22_latent_shape(batch_size: int = 2, num_steps: int = 5) -> tuple:
    """Produce the 6-D latent shape used by the Wan2.2 training adapter:
    ``(B, T, C, F, H, W)``."""
    return (batch_size, num_steps, 16, 5, 8, 12)


def _batch_tensors(batch_size: int = 2, num_steps: int = 5):
    L, D = 16, 64
    return {
        "latents": torch.randn(*_wan22_latent_shape(batch_size, num_steps)),
        "timesteps": torch.linspace(999, 0, num_steps).unsqueeze(0).expand(batch_size, -1),
        "prompt_embeds": torch.randn(batch_size, L, D),
        "prompt_embeds_mask": torch.ones(batch_size, L, dtype=torch.int32),
        "negative_prompt_embeds": torch.randn(batch_size, L, D),
        "negative_prompt_embeds_mask": torch.ones(batch_size, L, dtype=torch.int32),
    }


# ===========================================================================
# Config helpers
# ===========================================================================


def test_configure_wan_scheduler_timesteps_are_set():
    """``_configure_wan_scheduler`` sets correct number of timesteps."""
    if not _VLLM_AVAILABLE:
        pytest.skip("vllm not available")
    scheduler = FlowMatchSDEDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "shift": 5.0,
            "use_dynamic_shifting": False,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_image_seq_len": 0,
            "max_image_seq_len": 0,
            "flip_sin_to_cos": True,
            "scaling_factor": 1.0,
        }
    )
    _configure_wan_scheduler(
        scheduler,
        num_inference_steps=10,
        shift=5.0,
        device="cpu",
    )
    assert scheduler.timesteps is not None
    assert len(scheduler.timesteps) == 10
    assert scheduler.sigmas is not None
    assert len(scheduler.sigmas) == 11


# ===========================================================================
# Training adapter — registry
# ===========================================================================


class TestWan22DanceGRPORegistry:
    """``Wan22DanceGRPO`` registration under ``("WanPipeline", "dance_grpo")``."""

    def test_class_is_registered(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        cfg = _make_model_config()
        cls = DiffusionModelBase.get_class(cfg)
        assert cls is Wan22DanceGRPO

    def test_unknown_algorithm_raises(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        cfg = _make_model_config()
        object.__setattr__(cfg, "algorithm", "unknown_algo")
        with pytest.raises(NotImplementedError, match="No diffusion model registered"):
            DiffusionModelBase.get_class(cfg)

    def test_rollout_adapter_registered(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        cls = VllmOmniPipelineBase.get_class("WanPipeline", "dance_grpo")
        assert cls is not None
        assert cls.__name__ == "Wan22DanceGRPOPipelineWithLogProb"


# ===========================================================================
# Training adapter — prepare_model_inputs
# ===========================================================================


class TestWan22DanceGRPOPrepareModelInputs:
    """``Wan22DanceGRPO.prepare_model_inputs`` — per-step input construction."""

    def test_basic_slicing_and_shapes(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        tensors = _batch_tensors()
        micro_batch = TensorDict({}, batch_size=2)
        model_config = _make_model_config(guidance_scale=1.0)

        model_inputs, negative_model_inputs = Wan22DanceGRPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=2,
        )

        assert model_inputs["hidden_states"].shape == (2, 16, 5, 8, 12)
        assert model_inputs["timestep"].shape == (2,)
        assert model_inputs["timestep"].dtype == torch.long
        assert negative_model_inputs == {}

    def test_cfg_enabled_returns_negative_inputs(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        tensors = _batch_tensors()
        micro_batch = TensorDict({}, batch_size=2)
        model_config = _make_model_config(guidance_scale=5.0)

        model_inputs, negative_model_inputs = Wan22DanceGRPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=2,
        )

        assert negative_model_inputs != {}
        assert negative_model_inputs["hidden_states"].shape == model_inputs["hidden_states"].shape
        torch.testing.assert_close(
            negative_model_inputs["encoder_hidden_states"],
            tensors["negative_prompt_embeds"],
        )
        assert "encoder_hidden_states_mask" in model_inputs
        assert "encoder_attention_mask" in negative_model_inputs

    def test_prompt_embed_masking(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        tensors = _batch_tensors()
        mask = torch.zeros(2, 16, dtype=torch.int32)
        mask[:, :8] = 1
        tensors["prompt_embeds_mask"] = mask
        micro_batch = TensorDict({}, batch_size=2)
        model_config = _make_model_config(guidance_scale=1.0)

        model_inputs, _ = Wan22DanceGRPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=mask,
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        encoder_hidden = model_inputs["encoder_hidden_states"]
        assert encoder_hidden.shape == tensors["prompt_embeds"].shape
        assert encoder_hidden[:, 8:].abs().sum().item() == 0.0
        assert encoder_hidden[:, :8].abs().sum().item() > 0.0

    def test_missing_mask_does_not_mutate(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        tensors = _batch_tensors()
        micro_batch = TensorDict({}, batch_size=2)
        model_config = _make_model_config(guidance_scale=1.0)
        original_embeds = tensors["prompt_embeds"].clone()

        model_inputs, _ = Wan22DanceGRPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=None,
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=None,
            micro_batch=micro_batch,
            step=0,
        )

        torch.testing.assert_close(model_inputs["encoder_hidden_states"], original_embeds)

    def test_transformer_kwargs(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        tensors = _batch_tensors()
        micro_batch = TensorDict({}, batch_size=2)
        model_config = _make_model_config(guidance_scale=1.0)

        model_inputs, _ = Wan22DanceGRPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        assert "hidden_states" in model_inputs
        assert "timestep" in model_inputs
        assert "encoder_hidden_states" in model_inputs
        assert "encoder_hidden_states_mask" in model_inputs
        assert "encoder_hidden_states_image" in model_inputs
        assert "return_dict" in model_inputs
        assert model_inputs["encoder_hidden_states_image"] is None
        assert model_inputs["return_dict"] is False


# ===========================================================================
# Training adapter — forward_and_sample_previous_step
# ===========================================================================


class TestWan22DanceGRPOForwardAndSample:
    """``Wan22DanceGRPO.forward_and_sample_previous_step``."""

    def test_no_cfg_calls_module_once(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        batch_size, num_steps = 2, 5
        noise_pred = torch.randn(batch_size, 16, 5, 8, 12)
        module = MagicMock(return_value=(noise_pred,))

        latents = torch.randn(batch_size, num_steps + 1, 16, 5, 8, 12)
        timesteps = torch.linspace(999, 0, num_steps).unsqueeze(0).expand(batch_size, -1)

        model_inputs = {
            "hidden_states": latents[:, 0],
            "timestep": timesteps[:, 0].long(),
            "encoder_hidden_states": torch.randn(batch_size, 16, 64),
            "encoder_hidden_states_mask": torch.ones(batch_size, 16, dtype=torch.int32),
            "encoder_hidden_states_image": None,
            "return_dict": False,
        }

        scheduler = MagicMock(spec=FlowMatchSDEDiscreteScheduler)
        scheduler.sample_previous_step.return_value = (
            torch.randn(batch_size, 16, 5, 8, 12),
            torch.randn(batch_size),
            torch.randn(batch_size, 16, 5, 8, 12),
            torch.randn(batch_size, 1, 1, 1, 1),
            torch.randn(batch_size),
        )

        scheduler_inputs = {"all_latents": latents, "all_timesteps": timesteps}

        result = Wan22DanceGRPO.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(guidance_scale=1.0),
            model_inputs=model_inputs,
            negative_model_inputs={},
            scheduler_inputs=scheduler_inputs,
            step=0,
        )

        module.assert_called_once()
        assert len(result) == 4

    def test_cfg_calls_module_twice_and_combines(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        batch_size, num_steps = 2, 5
        pos_pred = torch.ones(batch_size, 16, 5, 8, 12)
        neg_pred = torch.zeros(batch_size, 16, 5, 8, 12)
        module = MagicMock(side_effect=[(pos_pred,), (neg_pred,)])

        latents = torch.randn(batch_size, num_steps + 1, 16, 5, 8, 12)
        timesteps = torch.linspace(999, 0, num_steps).unsqueeze(0).expand(batch_size, -1)
        guidance_scale = 5.0

        model_inputs = {
            "hidden_states": latents[:, 0],
            "timestep": timesteps[:, 0].long(),
            "encoder_hidden_states": torch.randn(batch_size, 16, 64),
            "encoder_hidden_states_mask": torch.ones(batch_size, 16, dtype=torch.int32),
            "encoder_hidden_states_image": None,
            "return_dict": False,
        }
        negative_model_inputs = {
            "hidden_states": latents[:, 0],
            "timestep": timesteps[:, 0].long(),
            "encoder_hidden_states": torch.randn(batch_size, 16, 64),
            "encoder_attention_mask": torch.ones(batch_size, 16, dtype=torch.int32),
            "encoder_hidden_states_image": None,
            "return_dict": False,
        }

        scheduler = MagicMock(spec=FlowMatchSDEDiscreteScheduler)
        scheduler.sample_previous_step.return_value = (
            torch.randn(batch_size, 16, 5, 8, 12),
            torch.randn(batch_size),
            torch.randn(batch_size, 16, 5, 8, 12),
            torch.randn(batch_size, 1, 1, 1, 1),
            torch.randn(batch_size),
        )

        scheduler_inputs = {"all_latents": latents, "all_timesteps": timesteps}

        result = Wan22DanceGRPO.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(guidance_scale=guidance_scale),
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
            scheduler_inputs=scheduler_inputs,
            step=0,
        )

        assert module.call_count == 2
        assert len(result) == 4

    def test_scheduler_receives_correct_args(self):
        if not _VLLM_AVAILABLE:
            pytest.skip("vllm not available")
        batch_size, num_steps = 2, 5
        noise_pred = torch.randn(batch_size, 16, 5, 8, 12)
        module = MagicMock(return_value=(noise_pred,))

        latents = torch.randn(batch_size, num_steps + 1, 16, 5, 8, 12)
        timesteps = torch.linspace(999, 0, num_steps).unsqueeze(0).expand(batch_size, -1)

        model_inputs = {
            "hidden_states": latents[:, 0],
            "timestep": timesteps[:, 0].long(),
            "encoder_hidden_states": torch.randn(batch_size, 16, 64),
            "encoder_hidden_states_mask": torch.ones(batch_size, 16, dtype=torch.int32),
            "encoder_hidden_states_image": None,
            "return_dict": False,
        }

        scheduler = MagicMock(spec=FlowMatchSDEDiscreteScheduler)
        scheduler.sample_previous_step.return_value = (
            torch.randn(batch_size, 16, 5, 8, 12),
            torch.randn(batch_size),
            torch.randn(batch_size, 16, 5, 8, 12),
            torch.randn(batch_size, 1, 1, 1, 1),
            torch.randn(batch_size),
        )

        scheduler_inputs = {"all_latents": latents, "all_timesteps": timesteps}
        model_config = _make_model_config(guidance_scale=1.0, noise_level=0.8, sde_type="dance_sde")

        Wan22DanceGRPO.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=model_config,
            model_inputs=model_inputs,
            negative_model_inputs={},
            scheduler_inputs=scheduler_inputs,
            step=1,
        )

        scheduler.sample_previous_step.assert_called_once()
        call_kwargs = scheduler.sample_previous_step.call_args.kwargs

        assert torch.equal(call_kwargs["sample"], latents[:, 1].float())
        assert torch.equal(call_kwargs["prev_sample"], latents[:, 2].float())
        assert torch.equal(call_kwargs["timestep"], timesteps[:, 1])
        assert call_kwargs["noise_level"] == 0.8
        assert call_kwargs["sde_type"] == "dance_sde"
        assert call_kwargs["return_logprobs"] is True
        assert call_kwargs["return_sqrt_dt"] is True
