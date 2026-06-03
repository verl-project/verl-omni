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
"""CPU tests for StableDiffusion3DPO training adapter.

Necessity: The SD3 DPO adapter is the boundary between parquet tensors and the
transformer forward. These tests cover input preparation, CFG branching, and
noise prediction without loading SD3 weights or running on GPU.
"""

from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.sd3_dpo.diffusers_training_adapter import StableDiffusion3DPO
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


def _make_model_config(*, guidance_scale: float | None = 1.0) -> DiffusionModelConfig:
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "StableDiffusion3Pipeline")
    object.__setattr__(cfg, "algorithm", "dpo")
    object.__setattr__(cfg, "external_lib", None)
    object.__setattr__(cfg, "pipeline", DiffusionPipelineConfig(guidance_scale=guidance_scale))
    return cfg


def _batch_tensors(batch_size: int = 2):
    latents = torch.randn(batch_size, 16, 8, 8)
    timesteps = torch.tensor([100.0, 50.0][:batch_size])
    prompt_embeds = torch.randn(batch_size, 12, 64)
    prompt_embeds_mask = torch.ones(batch_size, 12, dtype=torch.int32)
    pooled = torch.randn(batch_size, 32)
    negative_prompt_embeds = torch.randn(batch_size, 12, 64)
    negative_prompt_embeds_mask = torch.ones(batch_size, 12, dtype=torch.int32)
    negative_pooled = torch.randn(batch_size, 32)
    return {
        "latents": latents,
        "timesteps": timesteps,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "negative_pooled_prompt_embeds": negative_pooled,
    }


class TestStableDiffusion3DPORegistry:
    def test_registered_for_sd3_dpo_algorithm(self):
        cfg = _make_model_config()
        assert DiffusionModelBase.get_class(cfg) is StableDiffusion3DPO


class TestStableDiffusion3DPOBuildTransformerInputs:
    def test_includes_sd3_keys(self):
        tensors = _batch_tensors()
        inputs = StableDiffusion3DPO.build_transformer_inputs(
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            pooled_prompt_embeds=tensors["pooled_prompt_embeds"],
        )

        assert inputs["hidden_states"].shape == tensors["latents"].shape
        assert inputs["encoder_hidden_states"].shape == tensors["prompt_embeds"].shape
        assert inputs["pooled_projections"].shape == tensors["pooled_prompt_embeds"].shape
        assert inputs["timestep"].shape == tensors["timesteps"].shape
        assert inputs["joint_attention_kwargs"]["attention_mask"].shape == tensors["prompt_embeds_mask"].shape


class TestStableDiffusion3DPOPrepareModelInputs:
    def test_no_cfg_returns_positive_inputs_only(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict({"pooled_prompt_embeds": tensors["pooled_prompt_embeds"]}, batch_size=2)
        model_config = _make_model_config(guidance_scale=1.0)

        model_inputs, negative_model_inputs = StableDiffusion3DPO.prepare_model_inputs(
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

        assert negative_model_inputs is None
        assert model_inputs["hidden_states"].shape == tensors["latents"].shape
        torch.testing.assert_close(
            model_inputs["pooled_projections"],
            tensors["pooled_prompt_embeds"],
        )

    def test_null_guidance_scale_defaults_to_no_cfg(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict({"pooled_prompt_embeds": tensors["pooled_prompt_embeds"]}, batch_size=2)

        _, negative_model_inputs = StableDiffusion3DPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=_make_model_config(guidance_scale=None),
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        assert negative_model_inputs is None

    def test_cfg_returns_negative_inputs(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict(
            {
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"],
                "negative_pooled_prompt_embeds": tensors["negative_pooled_prompt_embeds"],
            },
            batch_size=2,
        )
        model_config = _make_model_config(guidance_scale=4.0)

        model_inputs, negative_model_inputs = StableDiffusion3DPO.prepare_model_inputs(
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

        assert negative_model_inputs is not None
        torch.testing.assert_close(
            negative_model_inputs["encoder_hidden_states"],
            tensors["negative_prompt_embeds"],
        )
        torch.testing.assert_close(
            negative_model_inputs["pooled_projections"],
            tensors["negative_pooled_prompt_embeds"],
        )
        assert model_inputs["hidden_states"].shape == negative_model_inputs["hidden_states"].shape

    def test_rejects_missing_prompt_embeds_mask(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict({"pooled_prompt_embeds": tensors["pooled_prompt_embeds"]}, batch_size=2)
        with pytest.raises(ValueError, match="prompt_embeds_mask is required"):
            StableDiffusion3DPO.prepare_model_inputs(
                module=MagicMock(),
                model_config=_make_model_config(),
                latents=tensors["latents"],
                timesteps=tensors["timesteps"],
                prompt_embeds=tensors["prompt_embeds"],
                prompt_embeds_mask=None,
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=micro_batch,
                step=0,
            )

    def test_rejects_missing_pooled_prompt_embeds(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict({}, batch_size=2)
        with pytest.raises(KeyError, match="pooled_projections"):
            StableDiffusion3DPO.prepare_model_inputs(
                module=MagicMock(),
                model_config=_make_model_config(),
                latents=tensors["latents"],
                timesteps=tensors["timesteps"],
                prompt_embeds=tensors["prompt_embeds"],
                prompt_embeds_mask=tensors["prompt_embeds_mask"],
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=micro_batch,
                step=0,
            )


class TestStableDiffusion3DPOForward:
    def test_no_cfg_returns_positive_noise_pred(self):
        pos_pred = torch.randn(2, 16, 8, 8)
        module = MagicMock(return_value=(pos_pred,))
        model_inputs = {"hidden_states": pos_pred}

        result = StableDiffusion3DPO.forward(
            module=module,
            model_config=_make_model_config(guidance_scale=1.0),
            model_inputs=model_inputs,
            negative_model_inputs=None,
        )

        module.assert_called_once_with(**model_inputs)
        torch.testing.assert_close(result, pos_pred)

    def test_cfg_combines_positive_and_negative_predictions(self):
        pos_pred = torch.ones(2, 16, 8, 8)
        neg_pred = torch.zeros(2, 16, 8, 8)
        module = MagicMock(side_effect=[(pos_pred,), (neg_pred,)])
        guidance_scale = 3.0
        model_inputs = {"hidden_states": pos_pred}
        negative_model_inputs = {"hidden_states": neg_pred}

        result = StableDiffusion3DPO.forward(
            module=module,
            model_config=_make_model_config(guidance_scale=guidance_scale),
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
        )

        assert module.call_count == 2
        expected = neg_pred + guidance_scale * (pos_pred - neg_pred)
        torch.testing.assert_close(result, expected)

    def test_cfg_requires_negative_inputs_when_guidance_enabled(self):
        with pytest.raises(ValueError, match="CFG requires negative prompt inputs"):
            StableDiffusion3DPO.forward(
                module=MagicMock(),
                model_config=_make_model_config(guidance_scale=4.0),
                model_inputs={"hidden_states": torch.randn(2, 16, 8, 8)},
                negative_model_inputs=None,
            )
