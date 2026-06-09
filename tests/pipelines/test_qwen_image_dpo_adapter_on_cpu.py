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
"""CPU tests for QwenImageDPO training adapter.

Necessity: The Qwen-Image DPO adapter is the boundary between offline parquet
tensors and the Qwen-Image transformer forward. These tests cover input
preparation, guidance embeddings, True-CFG branching, and noise prediction
without loading Qwen-Image weights or running on GPU.
"""

from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_dpo.diffusers_training_adapter import QwenImageDPO
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


def _make_model_config(
    *,
    guidance_scale: float = 1.0,
    true_cfg_scale: float = 1.0,
    height: int = 512,
    width: int = 512,
) -> DiffusionModelConfig:
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "QwenImagePipeline")
    object.__setattr__(cfg, "algorithm", "dpo")
    object.__setattr__(cfg, "external_lib", None)
    object.__setattr__(
        cfg,
        "pipeline",
        DiffusionPipelineConfig(
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
        ),
    )
    return cfg


def _batch_tensors(batch_size: int = 2):
    latents = torch.randn(batch_size, 1024, 64)
    timesteps = torch.tensor([1000.0, 500.0][:batch_size])
    prompt_embeds = torch.randn(batch_size, 12, 64)
    prompt_embeds_mask = torch.ones(batch_size, 12, dtype=torch.int32)
    negative_prompt_embeds = torch.randn(batch_size, 12, 64)
    negative_prompt_embeds_mask = torch.ones(batch_size, 12, dtype=torch.int32)
    return {
        "latents": latents,
        "timesteps": timesteps,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
    }


def _make_module(*, guidance_embeds: bool = False) -> MagicMock:
    module = MagicMock()
    module.config.guidance_embeds = guidance_embeds
    return module


class TestQwenImageDPORegistry:
    def test_registered_for_qwen_image_dpo_algorithm(self):
        cfg = _make_model_config()
        assert DiffusionModelBase.get_class(cfg) is QwenImageDPO


class TestQwenImageDPOBuildTransformerInputs:
    def test_includes_qwen_image_keys(self):
        tensors = _batch_tensors()
        img_shapes = [[(1, 32, 32)]] * 2
        guidance = torch.tensor([1.0, 1.0])

        inputs = QwenImageDPO.build_transformer_inputs(
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            img_shapes=img_shapes,
            guidance=guidance,
        )

        assert inputs["hidden_states"].shape == tensors["latents"].shape
        torch.testing.assert_close(inputs["timestep"], tensors["timesteps"] / 1000.0)
        assert inputs["encoder_hidden_states"].shape == tensors["prompt_embeds"].shape
        assert inputs["encoder_hidden_states_mask"].shape == tensors["prompt_embeds_mask"].shape
        assert inputs["img_shapes"] == img_shapes
        torch.testing.assert_close(inputs["guidance"], guidance)
        assert inputs["return_dict"] is False


class TestQwenImageDPOPrepareModelInputs:
    def test_no_true_cfg_returns_positive_inputs_only(self):
        tensors = _batch_tensors()
        model_config = _make_model_config(true_cfg_scale=1.0)

        model_inputs, negative_model_inputs = QwenImageDPO.prepare_model_inputs(
            module=_make_module(guidance_embeds=False),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=TensorDict({}, batch_size=2),
            step=0,
        )

        assert negative_model_inputs is None
        assert model_inputs["hidden_states"].shape == tensors["latents"].shape
        assert model_inputs["guidance"] is None
        assert model_inputs["img_shapes"] == [[(1, 32, 32)]] * 2

    def test_guidance_embeds_adds_guidance_tensor(self):
        tensors = _batch_tensors()
        guidance_scale = 4.0
        model_config = _make_model_config(guidance_scale=guidance_scale)

        model_inputs, _ = QwenImageDPO.prepare_model_inputs(
            module=_make_module(guidance_embeds=True),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=TensorDict({}, batch_size=2),
            step=0,
        )

        torch.testing.assert_close(model_inputs["guidance"], torch.full((2,), guidance_scale))

    def test_true_cfg_returns_negative_inputs(self):
        tensors = _batch_tensors()
        model_config = _make_model_config(true_cfg_scale=4.0)

        model_inputs, negative_model_inputs = QwenImageDPO.prepare_model_inputs(
            module=_make_module(guidance_embeds=False),
            model_config=model_config,
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=TensorDict({}, batch_size=2),
            step=0,
        )

        assert negative_model_inputs is not None
        torch.testing.assert_close(
            negative_model_inputs["encoder_hidden_states"],
            tensors["negative_prompt_embeds"],
        )
        torch.testing.assert_close(
            negative_model_inputs["encoder_hidden_states_mask"],
            tensors["negative_prompt_embeds_mask"],
        )
        assert model_inputs["hidden_states"].shape == negative_model_inputs["hidden_states"].shape

    def test_rejects_missing_prompt_embeds_mask(self):
        tensors = _batch_tensors()
        with pytest.raises(ValueError, match="prompt_embeds_mask is required"):
            QwenImageDPO.prepare_model_inputs(
                module=_make_module(),
                model_config=_make_model_config(),
                latents=tensors["latents"],
                timesteps=tensors["timesteps"],
                prompt_embeds=tensors["prompt_embeds"],
                prompt_embeds_mask=None,
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=TensorDict({}, batch_size=2),
                step=0,
            )

    def test_true_cfg_rejects_missing_negative_prompt_inputs(self):
        tensors = _batch_tensors()
        with pytest.raises(ValueError, match="True-CFG requires negative prompt inputs"):
            QwenImageDPO.prepare_model_inputs(
                module=_make_module(),
                model_config=_make_model_config(true_cfg_scale=4.0),
                latents=tensors["latents"],
                timesteps=tensors["timesteps"],
                prompt_embeds=tensors["prompt_embeds"],
                prompt_embeds_mask=tensors["prompt_embeds_mask"],
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=TensorDict({}, batch_size=2),
                step=0,
            )


class TestQwenImageDPOForwardAndSamplePreviousStep:
    def test_no_true_cfg_returns_positive_noise_pred(self):
        pos_pred = torch.randn(2, 1024, 64)
        module = MagicMock(return_value=(pos_pred,))
        model_inputs = {"hidden_states": pos_pred}

        result = QwenImageDPO.forward_and_sample_previous_step(
            module=module,
            scheduler=MagicMock(),
            model_config=_make_model_config(true_cfg_scale=1.0),
            model_inputs=model_inputs,
            negative_model_inputs=None,
            scheduler_inputs=None,
            step=0,
        )

        module.assert_called_once_with(**model_inputs)
        torch.testing.assert_close(result, pos_pred)

    def test_true_cfg_combines_and_renormalizes_predictions(self):
        pos_pred = torch.ones(2, 1024, 64) * 2.0
        neg_pred = torch.ones(2, 1024, 64)
        module = MagicMock(side_effect=[(pos_pred,), (neg_pred,)])
        true_cfg_scale = 3.0
        model_inputs = {"hidden_states": pos_pred}
        negative_model_inputs = {"hidden_states": neg_pred}

        result = QwenImageDPO.forward_and_sample_previous_step(
            module=module,
            scheduler=MagicMock(),
            model_config=_make_model_config(true_cfg_scale=true_cfg_scale),
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
            scheduler_inputs=None,
            step=0,
        )

        assert module.call_count == 2
        comb_pred = neg_pred + true_cfg_scale * (pos_pred - neg_pred)
        expected = comb_pred * (
            torch.norm(pos_pred, dim=-1, keepdim=True) / torch.norm(comb_pred, dim=-1, keepdim=True)
        )
        torch.testing.assert_close(result, expected)

    def test_true_cfg_requires_negative_inputs_when_guidance_enabled(self):
        with pytest.raises(ValueError, match="True-CFG requires negative prompt inputs"):
            QwenImageDPO.forward_and_sample_previous_step(
                module=MagicMock(),
                scheduler=MagicMock(),
                model_config=_make_model_config(true_cfg_scale=4.0),
                model_inputs={"hidden_states": torch.randn(2, 1024, 64)},
                negative_model_inputs=None,
                scheduler_inputs=None,
                step=0,
            )
