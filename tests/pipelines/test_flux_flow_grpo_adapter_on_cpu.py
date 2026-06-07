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
"""CPU tests for the FLUX FlowGRPO training adapter.

These tests keep the adapter boundary covered without loading FLUX weights.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.flux_flow_grpo.common import getattr_not_none
from verl_omni.pipelines.flux_flow_grpo.diffusers_training_adapter import Flux
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig, DiffusionRolloutAlgoConfig


def _make_model_config(
    *,
    true_cfg_scale: float = 1.0,
    guidance_scale: float | None = 3.5,
) -> DiffusionModelConfig:
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "FluxPipeline")
    object.__setattr__(cfg, "algorithm", "flow_grpo")
    object.__setattr__(cfg, "external_lib", None)
    object.__setattr__(cfg, "local_path", "dummy-flux")
    object.__setattr__(
        cfg,
        "pipeline",
        DiffusionPipelineConfig(
            height=512,
            width=512,
            num_inference_steps=4,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
        ),
    )
    object.__setattr__(cfg, "algo", DiffusionRolloutAlgoConfig(noise_level=0.7, sde_type="sde"))
    return cfg


def _batch_tensors(batch_size: int = 2):
    seq_len = 64
    text_len = 16
    hidden_size = 32
    pooled_size = 8
    latent_width = 12
    return {
        "latents": torch.randn(batch_size, 3, seq_len, latent_width),
        "timesteps": torch.tensor([[1000.0, 500.0], [900.0, 400.0]]),
        "prompt_embeds": torch.randn(batch_size, text_len, hidden_size),
        "negative_prompt_embeds": torch.randn(batch_size, text_len, hidden_size),
        "prompt_embeds_mask": torch.ones(batch_size, text_len, dtype=torch.int32),
        "negative_prompt_embeds_mask": torch.ones(batch_size, text_len, dtype=torch.int32),
        "pooled_prompt_embeds": torch.randn(batch_size, pooled_size),
        "negative_pooled_prompt_embeds": torch.randn(batch_size, pooled_size),
        "text_ids": torch.zeros(batch_size, text_len, 3),
        "negative_text_ids": torch.zeros(batch_size, text_len, 3),
        "latent_image_ids": torch.zeros(batch_size, seq_len, 3),
    }


class _DummyModule:
    def __init__(
        self,
        outputs: list[torch.Tensor] | None = None,
        *,
        guidance_embeds: bool = False,
        direct_guidance_embeds: bool | None = None,
    ):
        self.config = SimpleNamespace(guidance_embeds=guidance_embeds)
        if direct_guidance_embeds is not None:
            self.guidance_embeds = direct_guidance_embeds
        self.outputs = list(outputs or [])
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (self.outputs.pop(0),)


class _DummyScheduler:
    def __init__(self):
        self.kwargs = None

    def sample_previous_step(self, **kwargs):
        self.kwargs = kwargs
        batch_size = kwargs["sample"].shape[0]
        return (
            kwargs["prev_sample"],
            torch.ones(batch_size),
            torch.zeros_like(kwargs["sample"]),
            torch.ones(batch_size),
            torch.full((batch_size,), 0.5),
        )


class TestFluxFlowGRPORegistry:
    def test_registered_for_flux_flow_grpo(self):
        assert DiffusionModelBase.get_class(_make_model_config()) is Flux


class TestFluxFlowGRPORolloutParamCompat:
    def test_getattr_not_none_reads_optional_sampling_params(self):
        assert getattr_not_none(SimpleNamespace(), "guidance_scale", 3.5) == 3.5
        assert getattr_not_none(SimpleNamespace(guidance_scale=None), "guidance_scale", 3.5) == 3.5
        assert getattr_not_none(SimpleNamespace(guidance_scale=2.25), "guidance_scale", 3.5) == 2.25

    def test_vllm_omni_custom_pipeline_uses_dummy_initial_load(self):
        pytest.importorskip("vllm_omni")

        from verl_omni.pipelines.flux_flow_grpo.vllm_omni_rollout_adapter import FluxPipelineWithLogProb
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

        server = object.__new__(vLLMOmniHttpServer)
        server.model_config = SimpleNamespace(architecture="FluxPipeline", algorithm="flow_grpo")
        engine_args = {"diffusion_load_format": "safetensors"}

        server._configure_custom_pipeline(engine_args)

        assert engine_args["enable_dummy_pipeline"] is True
        assert engine_args["diffusion_load_format"] == "dummy"
        assert engine_args["custom_pipeline_args"] == {
            "pipeline_class": f"{FluxPipelineWithLogProb.__module__}.{FluxPipelineWithLogProb.__qualname__}"
        }

    def test_custom_pipeline_fills_missing_transformer_config_from_checkpoint(self, tmp_path):
        pytest.importorskip("vllm_omni")

        from vllm_omni.diffusion.data import TransformerConfig

        from verl_omni.pipelines.flux_flow_grpo.vllm_omni_rollout_adapter import (
            _ensure_flux_transformer_config,
            _get_flux_transformer_config_kwargs,
        )

        model_dir = tmp_path / "flux"
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir(parents=True)
        (transformer_dir / "config.json").write_text(
            json.dumps(
                {
                    "_class_name": "FluxTransformer2DModel",
                    "guidance_embeds": False,
                    "num_layers": 19,
                    "num_single_layers": 38,
                }
            ),
            encoding="utf-8",
        )

        od_config = SimpleNamespace(
            model=str(model_dir),
            tf_model_config=TransformerConfig.from_dict({"num_layers": 2}),
        )

        _ensure_flux_transformer_config(od_config)

        merged_config = od_config.tf_model_config.to_dict()
        assert merged_config["guidance_embeds"] is False
        assert merged_config["num_layers"] == 2
        assert merged_config["num_single_layers"] == 38

        class DummyFluxTransformer:
            def __init__(
                self,
                od_config,
                num_layers: int = 19,
                guidance_embeds: bool = True,
                ignored_default: str = "default",
            ):
                pass

        config_kwargs = _get_flux_transformer_config_kwargs(DummyFluxTransformer, od_config.tf_model_config)
        assert config_kwargs["guidance_embeds"] is False
        assert config_kwargs["num_layers"] == 2
        assert "ignored_default" not in config_kwargs

    def test_sde_window_is_clamped_to_available_timesteps(self):
        pytest.importorskip("vllm_omni")

        from verl_omni.pipelines.flux_flow_grpo.vllm_omni_rollout_adapter import _normalize_sde_window

        assert _normalize_sde_window((0, 4), num_timesteps=4) == (0, 4)
        assert _normalize_sde_window((3, 5), num_timesteps=2) == (1, 2)
        assert _normalize_sde_window((0, 4), num_timesteps=1) == (0, 1)

    def test_module_parameter_dtype_reads_first_parameter_dtype(self):
        pytest.importorskip("vllm_omni")

        from verl_omni.pipelines.flux_flow_grpo.vllm_omni_rollout_adapter import _module_parameter_dtype

        assert _module_parameter_dtype(torch.nn.Linear(2, 2).to(dtype=torch.bfloat16), torch.float32) == torch.bfloat16
        assert _module_parameter_dtype(torch.nn.Identity(), torch.float32) == torch.float32

    def test_guidance_embeds_prefers_diffusers_config(self):
        pytest.importorskip("vllm_omni")

        from verl_omni.pipelines.flux_flow_grpo.vllm_omni_rollout_adapter import _has_guidance_embeds

        assert _has_guidance_embeds(SimpleNamespace(config=SimpleNamespace(guidance_embeds=True))) is True
        assert _has_guidance_embeds(SimpleNamespace(config=SimpleNamespace(guidance_embeds=False))) is False
        assert _has_guidance_embeds(SimpleNamespace(guidance_embeds=True)) is True
        assert _has_guidance_embeds(SimpleNamespace()) is False

    def test_extract_prompt_batch_preserves_batched_dict_prompts(self):
        pytest.importorskip("vllm_omni")

        from verl_omni.pipelines.flux_flow_grpo.vllm_omni_rollout_adapter import _extract_prompt_batch

        prompts = [
            {"prompt": "a red cabin", "prompt_2": "clip cabin", "negative_prompt": "blurry"},
            {"prompt": "a blue lake", "prompt_2": "clip lake", "negative_prompt": "low quality"},
        ]

        prompt, prompt_2, negative_prompt, negative_prompt_2 = _extract_prompt_batch(prompts)

        assert prompt == ["a red cabin", "a blue lake"]
        assert prompt_2 == ["clip cabin", "clip lake"]
        assert negative_prompt == ["blurry", "low quality"]
        assert negative_prompt_2 is None


class TestFluxFlowGRPOBuildTransformerInputs:
    def test_squeezes_position_ids_and_scales_timestep(self):
        tensors = _batch_tensors()
        guidance = torch.full((2,), 3.5)

        inputs = Flux.build_transformer_inputs(
            latents=tensors["latents"][:, 0],
            timesteps=tensors["timesteps"][:, 0],
            prompt_embeds=tensors["prompt_embeds"],
            pooled_prompt_embeds=tensors["pooled_prompt_embeds"],
            text_ids=tensors["text_ids"],
            latent_image_ids=tensors["latent_image_ids"],
            guidance=guidance,
        )

        torch.testing.assert_close(inputs["hidden_states"], tensors["latents"][:, 0])
        torch.testing.assert_close(inputs["timestep"], tensors["timesteps"][:, 0] / 1000.0)
        torch.testing.assert_close(inputs["guidance"], guidance)
        assert inputs["txt_ids"].shape == tensors["text_ids"].shape[1:]
        assert inputs["img_ids"].shape == tensors["latent_image_ids"].shape[1:]
        assert inputs["return_dict"] is False


class TestFluxFlowGRPOPrepareModelInputs:
    def test_no_cfg_returns_positive_inputs_only(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict(
            {
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"],
                "text_ids": tensors["text_ids"],
                "latent_image_ids": tensors["latent_image_ids"],
            },
            batch_size=2,
        )

        model_inputs, negative_model_inputs = Flux.prepare_model_inputs(
            module=_DummyModule(guidance_embeds=True),
            model_config=_make_model_config(true_cfg_scale=1.0, guidance_scale=2.5),
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=1,
        )

        assert negative_model_inputs is None
        torch.testing.assert_close(model_inputs["hidden_states"], tensors["latents"][:, 1])
        torch.testing.assert_close(model_inputs["timestep"], tensors["timesteps"][:, 1] / 1000.0)
        torch.testing.assert_close(model_inputs["guidance"], torch.full((2,), 2.5))
        torch.testing.assert_close(model_inputs["pooled_projections"], tensors["pooled_prompt_embeds"])

    def test_guidance_tensor_accepts_direct_module_flag(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict(
            {
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"],
                "text_ids": tensors["text_ids"],
                "latent_image_ids": tensors["latent_image_ids"],
            },
            batch_size=2,
        )

        model_inputs, _ = Flux.prepare_model_inputs(
            module=_DummyModule(direct_guidance_embeds=True),
            model_config=_make_model_config(guidance_scale=2.75),
            latents=tensors["latents"],
            timesteps=tensors["timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        torch.testing.assert_close(model_inputs["guidance"], torch.full((2,), 2.75))

    def test_true_cfg_returns_negative_inputs(self):
        tensors = _batch_tensors()
        micro_batch = TensorDict(
            {
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"],
                "negative_pooled_prompt_embeds": tensors["negative_pooled_prompt_embeds"],
                "text_ids": tensors["text_ids"],
                "negative_text_ids": tensors["negative_text_ids"],
                "latent_image_ids": tensors["latent_image_ids"],
            },
            batch_size=2,
        )

        _, negative_model_inputs = Flux.prepare_model_inputs(
            module=_DummyModule(),
            model_config=_make_model_config(true_cfg_scale=3.0),
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
        torch.testing.assert_close(negative_model_inputs["encoder_hidden_states"], tensors["negative_prompt_embeds"])
        torch.testing.assert_close(
            negative_model_inputs["pooled_projections"],
            tensors["negative_pooled_prompt_embeds"],
        )

    def test_rejects_missing_flux_rollout_fields(self):
        tensors = _batch_tensors()
        with pytest.raises(KeyError, match="pooled_prompt_embeds"):
            Flux.prepare_model_inputs(
                module=_DummyModule(),
                model_config=_make_model_config(),
                latents=tensors["latents"],
                timesteps=tensors["timesteps"],
                prompt_embeds=tensors["prompt_embeds"],
                prompt_embeds_mask=tensors["prompt_embeds_mask"],
                negative_prompt_embeds=tensors["negative_prompt_embeds"],
                negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
                micro_batch=TensorDict({}, batch_size=2),
                step=0,
            )


class TestFluxFlowGRPOForwardAndSample:
    def test_true_cfg_combines_predictions_without_norm_rescale(self):
        pos_pred = torch.ones(2, 64, 12)
        neg_pred = torch.full((2, 64, 12), 0.25)
        module = _DummyModule(outputs=[pos_pred, neg_pred])
        scheduler = _DummyScheduler()
        scheduler_inputs = {
            "all_latents": torch.randn(2, 3, 64, 12),
            "all_timesteps": torch.tensor([[1000.0, 500.0], [900.0, 400.0]]),
        }

        log_prob, _, _, _ = Flux.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_make_model_config(true_cfg_scale=3.0),
            model_inputs={"hidden_states": scheduler_inputs["all_latents"][:, 0]},
            negative_model_inputs={"hidden_states": scheduler_inputs["all_latents"][:, 0]},
            scheduler_inputs=scheduler_inputs,
            step=0,
        )

        expected = neg_pred + 3.0 * (pos_pred - neg_pred)
        assert scheduler.kwargs is not None
        torch.testing.assert_close(scheduler.kwargs["model_output"], expected.float())
        torch.testing.assert_close(log_prob, torch.ones(2))
