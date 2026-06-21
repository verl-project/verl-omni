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
"""CPU tests for QwenImageEditPlus FlowGRPO training adapter
and the rollout-side ``_get_qwen_prompt_embeds`` visual-feature wiring.

Necessity: These adapters are the boundary between rollout-collected tensors and
the transformer forward. Tests cover the I2I-specific condition-image latent
concatenation, prompt input routing, CFG branching, and the noise-pred slicing
that isolates target tokens from condition tokens. The rollout-adapter case
asserts that ``pixel_values`` and ``image_grid_thw`` derived from
``condition_images`` are forwarded to the Qwen2.5-VL text encoder — without
this, ``<|image_pad|>`` tokens in the rollout's pre-tokenized prompt collapse
to empty word embeddings and the diffusion transformer denoises into noise.
All tests run on CPU without loading real weights.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_edit_flow_grpo.diffusers_training_adapter import QwenImageEditPlus
from verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter import (
    QwenImageEditPlusPipelineWithLogProb,
)
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig, DiffusionRolloutAlgoConfig

# ---------------------------------------------------------------------------
# Tensor dimensions used throughout the tests
#   batch_size         = 2
#   n_steps            = 3   (SDE window steps stored in all_latents)
#   latent_seq_len     = 16  (target image tokens in packed latent space)
#   cond_seq_len       = 8   (condition image tokens — only for I2I)
#   latent_channels    = 8
#   text_seq_len       = 12
#   text_channels      = 64
# ---------------------------------------------------------------------------

_B = 2
_N = 3
_LS = 16  # latent seq len (target)
_CS = 8  # condition seq len
_LC = 8  # latent channels
_TS = 12  # text seq len
_TC = 64  # text channels


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    *,
    algorithm: str = "flow_grpo",
    true_cfg_scale: float = 1.0,
    noise_level: float = 0.7,
    sde_type: str = "sde",
) -> DiffusionModelConfig:
    """Build a minimal DiffusionModelConfig without triggering __post_init__."""
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "QwenImageEditPlusPipeline")
    object.__setattr__(cfg, "algorithm", algorithm)
    object.__setattr__(cfg, "external_lib", None)
    object.__setattr__(cfg, "pipeline", DiffusionPipelineConfig(true_cfg_scale=true_cfg_scale))
    object.__setattr__(cfg, "algo", DiffusionRolloutAlgoConfig(noise_level=noise_level, sde_type=sde_type))
    return cfg


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _make_module_mock(dtype: torch.dtype = torch.float32) -> MagicMock:
    """Build a module mock whose ``img_in.weight.dtype`` returns a real dtype.

    The training adapter casts ``hidden_states`` / ``image_latents`` to
    ``module.img_in.weight.dtype`` before concat. A bare ``MagicMock()``
    returns a MagicMock for that attribute and ``.to(MagicMock())`` raises.
    """
    module = MagicMock()
    module.img_in.weight.dtype = dtype
    return module


def _batch_tensors():
    """Return a dict of realistic test tensors."""
    return {
        # all_latents from rollout: [B, N+1, seq_len, C]
        "all_latents": torch.randn(_B, _N + 1, _LS, _LC),
        # all_timesteps from rollout: [B, N]
        "all_timesteps": torch.rand(_B, _N) * 1000,
        # condition image latents (I2I): [B, cond_seq_len, C]
        "image_latents": torch.randn(_B, _CS, _LC),
        "prompt_embeds": torch.randn(_B, _TS, _TC),
        "prompt_embeds_mask": torch.ones(_B, _TS, dtype=torch.bool),
        "negative_prompt_embeds": torch.randn(_B, _TS, _TC),
        "negative_prompt_embeds_mask": torch.ones(_B, _TS, dtype=torch.bool),
    }


def _make_micro_batch(
    tensors: dict, *, include_image_latents: bool = True, include_img_shapes: bool = True
) -> TensorDict:
    """Build a TensorDict micro_batch as the training worker would produce it."""
    td_data = {}
    if include_image_latents:
        td_data["image_latents"] = tensors["image_latents"]

    td = TensorDict(td_data, batch_size=_B)

    if include_img_shapes:
        # img_shapes for edit model: two image regions (target + condition)
        img_shapes = [[(1, _LS // 4, _LS // 4), (1, _CS // 4, _CS // 4)]] * _B
        td["img_shapes"] = NonTensorData(img_shapes)

    return td


def _make_scheduler_inputs(tensors: dict) -> dict:
    return {
        "all_latents": tensors["all_latents"],
        "all_timesteps": tensors["all_timesteps"],
    }


# ===========================================================================
# 1. Registry
# ===========================================================================


class TestQwenImageEditPlusRegistry:
    def test_registered_for_flow_grpo(self):
        cfg = _make_model_config(algorithm="flow_grpo")
        assert DiffusionModelBase.get_class(cfg) is QwenImageEditPlus


# ===========================================================================
# 2. prepare_model_inputs — input construction
# ===========================================================================


class TestQwenImageEditPlusPrepareModelInputs:
    def test_without_image_latents_hidden_states_equals_target_latent(self):
        """When no condition image, hidden_states must be exactly the target latent."""
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch(tensors, include_image_latents=False)
        step = 0

        model_inputs, negative_model_inputs = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=step,
        )

        expected_hidden = tensors["all_latents"][:, step]  # [B, LS, LC]
        torch.testing.assert_close(model_inputs["hidden_states"], expected_hidden)

    def test_image_latents_concatenated_along_seq_dim(self):
        """Condition image latents must be concatenated with target latents on dim-1."""
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch(tensors, include_image_latents=True)
        step = 0

        model_inputs, _ = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=step,
        )

        # hidden_states should be [B, LS + CS, LC]
        assert model_inputs["hidden_states"].shape == (_B, _LS + _CS, _LC)

    def test_positive_and_negative_share_hidden_states(self):
        """Both positive and negative inputs use the same noised latent (only prompts differ)."""
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch(tensors)
        step = 1

        model_inputs, negative_model_inputs = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=step,
        )

        torch.testing.assert_close(
            model_inputs["hidden_states"],
            negative_model_inputs["hidden_states"],
        )

    def test_timestep_is_divided_by_1000(self):
        """Timestep must be scaled to [0, 1] range before being passed to the transformer."""
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch(tensors)
        step = 2

        model_inputs, _ = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=step,
        )

        expected_timestep = tensors["all_timesteps"][:, step] / 1000.0
        torch.testing.assert_close(model_inputs["timestep"], expected_timestep)

    def test_positive_prompt_embeds_routed_correctly(self):
        """Positive prompt embeds must end up in model_inputs, not negative_model_inputs."""
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch(tensors)

        model_inputs, negative_model_inputs = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        torch.testing.assert_close(model_inputs["encoder_hidden_states"], tensors["prompt_embeds"])
        torch.testing.assert_close(negative_model_inputs["encoder_hidden_states"], tensors["negative_prompt_embeds"])

    def test_guidance_embed_is_none(self):
        """QwenImageEditPlus does not use guidance embeds (always None)."""
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch(tensors)

        model_inputs, _ = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        assert model_inputs["guidance"] is None

    def test_img_shapes_from_micro_batch_used_when_present(self):
        """img_shapes stored in micro_batch should pass through unchanged."""
        tensors = _batch_tensors()
        custom_shapes = [[(1, 4, 4), (1, 2, 2)]] * _B
        micro_batch = TensorDict({}, batch_size=_B)
        micro_batch["img_shapes"] = NonTensorData(custom_shapes)

        model_inputs, _ = QwenImageEditPlus.prepare_model_inputs(
            module=_make_module_mock(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        assert model_inputs["img_shapes"] == custom_shapes


# ===========================================================================
# 3. forward_and_sample_previous_step — noise-pred and SDE step
# ===========================================================================


class TestQwenImageEditPlusForwardAndSamplePreviousStep:
    def _make_scheduler_mock(self):
        """Return a scheduler mock whose sample_previous_step returns plausible values."""
        log_prob = torch.randn(_B)
        prev_sample_mean = torch.randn(_B, _LS, _LC)
        std_dev_t = torch.tensor(0.1)
        sqrt_dt = torch.tensor(0.01)
        scheduler = MagicMock()
        scheduler.sample_previous_step.return_value = (
            MagicMock(),  # unused prev_sample
            log_prob,
            prev_sample_mean,
            std_dev_t,
            sqrt_dt,
        )
        return scheduler

    def test_noise_pred_sliced_to_target_seq_len(self):
        """Transformer output (target + condition tokens) must be sliced to target tokens only."""
        tensors = _batch_tensors()
        step = 0
        target_seq_len = tensors["all_latents"][:, step].shape[1]  # _LS

        # Simulate transformer returning target + condition tokens concatenated
        full_pred = torch.randn(_B, _LS + _CS, _LC)
        module = MagicMock(return_value=(full_pred,))
        scheduler = self._make_scheduler_mock()

        model_inputs = {
            "hidden_states": torch.randn(_B, _LS + _CS, _LC),
            "return_dict": False,
        }

        QwenImageEditPlus.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(true_cfg_scale=1.0),
            model_inputs=model_inputs,
            negative_model_inputs=None,
            scheduler_inputs=_make_scheduler_inputs(tensors),
            step=step,
        )

        # Verify sample_previous_step received noise_pred sliced to [B, LS, LC]
        call_kwargs = scheduler.sample_previous_step.call_args
        passed_noise_pred = call_kwargs.kwargs.get("model_output")
        if passed_noise_pred is None:
            passed_noise_pred = call_kwargs.args[1]
        assert passed_noise_pred.shape == (_B, target_seq_len, _LC)

    def test_no_cfg_single_forward_pass(self):
        """When true_cfg_scale <= 1.0, the transformer must be called exactly once."""
        tensors = _batch_tensors()
        step = 0
        noise_pred = torch.randn(_B, _LS, _LC)
        module = MagicMock(return_value=(noise_pred,))
        scheduler = self._make_scheduler_mock()

        model_inputs = {"hidden_states": torch.randn(_B, _LS, _LC), "return_dict": False}

        QwenImageEditPlus.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(true_cfg_scale=1.0),
            model_inputs=model_inputs,
            negative_model_inputs=None,
            scheduler_inputs=_make_scheduler_inputs(tensors),
            step=step,
        )

        module.assert_called_once()

    def test_cfg_requires_two_forward_passes(self):
        """When true_cfg_scale > 1.0, the transformer is called for both positive and negative."""
        tensors = _batch_tensors()
        step = 0
        pos_pred = torch.randn(_B, _LS, _LC)
        neg_pred = torch.randn(_B, _LS, _LC)
        module = MagicMock(side_effect=[(pos_pred,), (neg_pred,)])
        scheduler = self._make_scheduler_mock()

        model_inputs = {"hidden_states": torch.randn(_B, _LS, _LC), "return_dict": False}
        negative_model_inputs = {"hidden_states": torch.randn(_B, _LS, _LC), "return_dict": False}

        QwenImageEditPlus.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(true_cfg_scale=3.0),
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
            scheduler_inputs=_make_scheduler_inputs(tensors),
            step=step,
        )

        assert module.call_count == 2

    def test_cfg_applies_rescaled_norm_formula(self):
        """Verify the norm-preserving CFG: combined pred is rescaled to positive pred's norm."""
        tensors = _batch_tensors()
        step = 0
        true_cfg_scale = 3.0

        pos_pred = torch.ones(_B, _LS, _LC)
        neg_pred = torch.zeros(_B, _LS, _LC)
        module = MagicMock(side_effect=[(pos_pred,), (neg_pred,)])
        scheduler = self._make_scheduler_mock()

        model_inputs = {"hidden_states": torch.randn(_B, _LS, _LC), "return_dict": False}
        negative_model_inputs = {"hidden_states": torch.randn(_B, _LS, _LC), "return_dict": False}

        QwenImageEditPlus.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(true_cfg_scale=true_cfg_scale),
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
            scheduler_inputs=_make_scheduler_inputs(tensors),
            step=step,
        )

        call_kwargs = scheduler.sample_previous_step.call_args
        passed_noise_pred = call_kwargs.kwargs.get("model_output")
        if passed_noise_pred is None:
            passed_noise_pred = call_kwargs.args[1]

        comb = neg_pred + true_cfg_scale * (pos_pred - neg_pred)
        cond_norm = torch.norm(pos_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb, dim=-1, keepdim=True)
        expected = comb * (cond_norm / noise_norm)
        torch.testing.assert_close(passed_noise_pred, expected)

    def test_requires_scheduler_inputs(self):
        """scheduler_inputs=None must raise AssertionError (latent history is mandatory)."""
        with pytest.raises(AssertionError):
            QwenImageEditPlus.forward_and_sample_previous_step(
                module=MagicMock(),
                scheduler=MagicMock(),
                model_config=_make_model_config(),
                model_inputs={"hidden_states": torch.randn(_B, _LS, _LC), "return_dict": False},
                negative_model_inputs=None,
                scheduler_inputs=None,
                step=0,
            )


# ===========================================================================
# 4. Rollout adapter — _get_qwen_prompt_embeds visual-feature wiring
# ===========================================================================


class _SyntheticImage:
    """Lightweight stand-in for PIL.Image — enough for downstream mocks."""

    def __init__(self, size=(64, 64)):
        self.size = size
        self.mode = "RGB"


class _RecordingTextEncoder:
    """Captures the kwargs the rollout adapter passes into text_encoder()."""

    dtype = torch.float32

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.last_kwargs = None

    def __call__(self, *, input_ids, attention_mask, output_hidden_states, pixel_values=None, image_grid_thw=None):
        self.last_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        batch, seq_len = input_ids.shape
        return SimpleNamespace(hidden_states=[torch.randn(batch, seq_len, self.hidden_dim, dtype=torch.float32)])


def _make_image_processor_mock(num_features_per_image: int = 4, hw: int = 14):
    """Mimic Qwen2VLImageProcessor outputs without loading transformers weights."""

    def _impl(*, images, return_tensors):
        n = len(images)
        return {
            "pixel_values": torch.randn(n, num_features_per_image, hw * hw, dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, hw, hw]] * n, dtype=torch.long),
        }

    proc = MagicMock(side_effect=lambda **kw: _impl(**kw))
    return proc


def _make_rollout_self(text_encoder, image_processor):
    """Build a SimpleNamespace exposing every attribute that
    QwenImageEditPlusPipelineWithLogProb._get_qwen_prompt_embeds reads.
    Bypasses heavy QwenImageEditPlusPipeline construction.
    """
    return SimpleNamespace(
        device=torch.device("cpu"),
        text_encoder=text_encoder,
        processor=SimpleNamespace(image_processor=image_processor),
        prompt_template_encode_start_idx=0,
        # Real impl splits hidden_states via the attention mask. For the test we
        # just emit one row per batch item; that's all the surrounding code needs.
        _extract_masked_hidden=lambda hidden_states, mask: list(hidden_states),
    )


class TestQwenImageEditRolloutVisualFeaturesWiring:
    """Regression tests for the rollout fix that injects pixel_values /
    image_grid_thw into the Qwen2.5-VL text_encoder. Without this wiring the
    transformer denoises into noise — empirical symptom: noisy / garbled rollout
    images.
    """

    def _run(self, condition_images):
        text_encoder = _RecordingTextEncoder(hidden_dim=16)
        image_processor = _make_image_processor_mock()
        fake_self = _make_rollout_self(text_encoder, image_processor)
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

        prompt_embeds, mask = QwenImageEditPlusPipelineWithLogProb._get_qwen_prompt_embeds(
            fake_self,
            prompt_ids,
            condition_images=condition_images,
        )
        return text_encoder, image_processor, prompt_embeds, mask

    def test_pixel_values_and_image_grid_thw_injected_when_condition_images_present(self):
        """The fix: condition_images must produce non-None pixel_values and
        image_grid_thw on the text_encoder call.
        """
        text_encoder, image_processor, _, _ = self._run([_SyntheticImage(), _SyntheticImage()])

        image_processor.assert_called_once()
        kw = text_encoder.last_kwargs
        assert kw is not None, "text_encoder was never invoked"
        assert kw["pixel_values"] is not None, "pixel_values must be forwarded to text_encoder"
        assert kw["image_grid_thw"] is not None, "image_grid_thw must be forwarded to text_encoder"
        # N condition images -> N feature rows so each <|image_pad|> aligns to a real image
        assert kw["pixel_values"].shape[0] == 2
        assert kw["image_grid_thw"].shape[0] == 2

    def test_pixel_values_dtype_matches_text_encoder(self):
        """pixel_values must be cast to text_encoder.dtype to avoid silent fp mismatches."""
        text_encoder, _, _, _ = self._run([_SyntheticImage()])
        assert text_encoder.last_kwargs["pixel_values"].dtype == _RecordingTextEncoder.dtype

    def test_no_image_processing_when_condition_images_none(self):
        """The pre-fix path (no condition images) must keep image-feature kwargs as None."""
        text_encoder, image_processor, _, _ = self._run(None)
        assert image_processor.called is False
        assert text_encoder.last_kwargs["pixel_values"] is None
        assert text_encoder.last_kwargs["image_grid_thw"] is None

    def test_no_image_processing_when_condition_images_empty(self):
        """Empty list short-circuits image processing — guards against zero-image edge case."""
        text_encoder, image_processor, _, _ = self._run([])
        assert image_processor.called is False
        assert text_encoder.last_kwargs["pixel_values"] is None
        assert text_encoder.last_kwargs["image_grid_thw"] is None


# ===========================================================================
# 6. Rollout adapter — _pick_condition_images selection contract
# ===========================================================================


class TestPickConditionImagesSelection:
    """``_pick_condition_images`` must prefer the *raw* PIL list that the agent
    loop used to expand ``<|image_pad|>`` placeholders. Picking the resized
    list from ``additional_information["condition_images"]`` would mismatch
    the placeholder count and trigger a ``ValueError`` in the Qwen2.5-VL
    text encoder. These tests pin the selection contract.
    """

    def _pick(self, custom_prompt):
        from verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter import (
            _pick_condition_images,
        )

        return _pick_condition_images(custom_prompt)

    def test_prefers_raw_multi_modal_data_image_over_resized_additional(self):
        raw = [_SyntheticImage(size=(512, 512)), _SyntheticImage(size=(640, 480))]
        resized = [_SyntheticImage(size=(384, 384))]
        custom_prompt = {
            "multi_modal_data": {"image": raw},
            "additional_information": {"condition_images": resized},
        }
        # The selection must return the RAW list (length 2, the actual images),
        # not the resized one (length 1).
        assert self._pick(custom_prompt) is raw

    def test_picks_raw_even_when_shorter_than_resized(self):
        """The contract is "raw beats resized when raw is non-empty" — not
        "longer beats shorter". A regression that adds a length comparison
        (e.g. ``if len(raw) >= len(resized)``) would flip the wrong way for
        single-image edits and silently route the resized list through.
        """
        raw = [_SyntheticImage(size=(512, 512))]
        resized = [
            _SyntheticImage(size=(384, 384)),
            _SyntheticImage(size=(384, 384)),
            _SyntheticImage(size=(384, 384)),
        ]
        custom_prompt = {
            "multi_modal_data": {"image": raw},
            "additional_information": {"condition_images": resized},
        }
        assert self._pick(custom_prompt) is raw

    def test_falls_back_to_additional_information_when_no_multi_modal(self):
        # When the agent loop did not populate multi_modal_data (e.g. e2e
        # smoke), ``additional_information["condition_images"]`` is the only
        # available list. We accept it as a best-effort fallback.
        resized = [_SyntheticImage(size=(384, 384))]
        custom_prompt = {"additional_information": {"condition_images": resized}}
        assert self._pick(custom_prompt) is resized

    def test_falls_back_when_multi_modal_image_is_empty_list(self):
        resized = [_SyntheticImage()]
        custom_prompt = {
            "multi_modal_data": {"image": []},
            "additional_information": {"condition_images": resized},
        }
        # An empty raw list means no actual images were sent — fall through.
        assert self._pick(custom_prompt) is resized

    def test_falls_back_when_multi_modal_image_is_none(self):
        resized = [_SyntheticImage()]
        custom_prompt = {
            "multi_modal_data": {"image": None},
            "additional_information": {"condition_images": resized},
        }
        assert self._pick(custom_prompt) is resized

    def test_returns_none_when_neither_source_has_images(self):
        # Text-only prompt with no condition images at all.
        custom_prompt = {"multi_modal_data": {"image": []}, "additional_information": {}}
        assert self._pick(custom_prompt) is None

    def test_returns_none_for_empty_dict(self):
        assert self._pick({}) is None

    def test_returns_none_for_non_dict_custom_prompt(self):
        # Defensive: vllm-omni may pass a non-dict prompt under exotic configs.
        assert self._pick("plain string prompt") is None
        assert self._pick(None) is None

    def test_handles_missing_multi_modal_data_key(self):
        # additional_information present, multi_modal_data key absent.
        resized = [_SyntheticImage()]
        assert self._pick({"additional_information": {"condition_images": resized}}) is resized

    def test_does_not_pick_non_list_image_field(self):
        # If multi_modal_data["image"] is not a list (e.g. a single PIL by
        # mistake), we do NOT take it — fall through to the resized list.
        resized = [_SyntheticImage()]
        custom_prompt = {
            "multi_modal_data": {"image": _SyntheticImage()},
            "additional_information": {"condition_images": resized},
        }
        # Selection contract requires a list to consume the raw branch.
        assert self._pick(custom_prompt) is resized
