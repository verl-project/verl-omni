from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl_omni.pipelines.boogu_image_diffusion_nft.diffusers_training_adapter import BooguImageDiffusionNFT
from verl_omni.pipelines.boogu_image_flow_grpo.common import build_boogu_sigmas
from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import BooguImageFlowGRPO
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import prepare_model_inputs
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


def _model_config(architecture: str, algorithm: str = "flow_grpo", true_cfg_scale: float = 1.5) -> DiffusionModelConfig:
    config = object.__new__(DiffusionModelConfig)
    object.__setattr__(config, "architecture", architecture)
    object.__setattr__(config, "external_lib", None)
    object.__setattr__(config, "algorithm", algorithm)
    object.__setattr__(
        config,
        "pipeline",
        SimpleNamespace(
            height=1024,
            width=1024,
            num_inference_steps=4,
            true_cfg_scale=true_cfg_scale,
        ),
    )
    object.__setattr__(config, "algo", SimpleNamespace(noise_level=0.0, sde_type="sde"))
    object.__setattr__(config, "local_path", "Boogu/Boogu-Image-0.1-Base")
    return config


class _BooguModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            axes_dim_rope=(4, 4, 4),
            axes_lens=(2, 2, 2),
        )
        self.calls = []

    def forward(
        self,
        hidden_states,
        timestep,
        instruction_hidden_states,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states=None,
        return_dict=False,
    ):
        self.calls.append(
            {
                "timestep": timestep,
                "instruction_hidden_states": instruction_hidden_states,
                "freqs_cis": freqs_cis,
                "instruction_attention_mask": instruction_attention_mask,
                "ref_image_hidden_states": ref_image_hidden_states,
                "return_dict": return_dict,
            }
        )
        bias = instruction_hidden_states.mean(dim=(1, 2), keepdim=True).view(-1, 1, 1, 1)
        return hidden_states + bias


class TestBooguFlowGRPO:
    def test_build_module_uses_diffusers_pipeline_with_remote_code(self):
        with patch(
            "verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter.DiffusionPipeline.from_pretrained"
        ) as mock_from_pretrained:
            mock_pipeline = SimpleNamespace(transformer=torch.nn.Linear(1, 1))
            mock_from_pretrained.return_value = mock_pipeline

            module = BooguImageFlowGRPO.build_module(_model_config("BooguImagePipeline"), torch.float16)

        assert module is mock_pipeline.transformer
        mock_from_pretrained.assert_called_once_with(
            "Boogu/Boogu-Image-0.1-Base",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    def test_prepare_model_inputs_supports_t2i_and_i2i(self):
        module = _BooguModule()
        model_config = _model_config("BooguImagePipeline", true_cfg_scale=2.0)
        latents = torch.randn(1, 3, 4, 8, 8)
        timesteps = torch.tensor([[900.0, 500.0, 100.0]])
        prompt_embeds = torch.ones(1, 2, 6)
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        negative_prompt_embeds = torch.zeros(1, 2, 6)
        negative_prompt_mask = torch.ones(1, 2, dtype=torch.long)

        model_inputs, negative_model_inputs = prepare_model_inputs(
            module=module,
            model_config=model_config,
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_mask,
            micro_batch={"condition_image_latents": torch.randn(1, 1, 4, 8, 8)},
            step=0,
        )

        assert model_inputs["hidden_states"].shape == (1, 4, 8, 8)
        torch.testing.assert_close(model_inputs["timestep"], torch.tensor([0.9]))
        assert len(model_inputs["freqs_cis"]) == 3
        assert model_inputs["ref_image_hidden_states"][0][0].shape == (4, 8, 8)
        assert negative_model_inputs is not None
        assert negative_model_inputs["ref_image_hidden_states"][0][0].shape == (4, 8, 8)

    def test_forward_applies_boogu_cfg(self):
        module = _BooguModule()
        model_config = _model_config("BooguImagePipeline", true_cfg_scale=2.0)
        model_inputs = {
            "hidden_states": torch.zeros(1, 4, 8, 8),
            "timestep": torch.tensor([0.9]),
            "instruction_hidden_states": torch.ones(1, 2, 6),
            "freqs_cis": [torch.zeros(2, 2) for _ in range(3)],
            "instruction_attention_mask": torch.ones(1, 2, dtype=torch.long),
            "ref_image_hidden_states": None,
            "return_dict": False,
        }
        negative_model_inputs = {
            **model_inputs,
            "instruction_hidden_states": torch.zeros(1, 2, 6),
        }

        prediction = BooguImageFlowGRPO.forward(module, model_config, model_inputs, negative_model_inputs)

        assert prediction.shape == (1, 4, 8, 8)
        torch.testing.assert_close(prediction, torch.full((1, 4, 8, 8), 2.0))

    def test_set_timesteps_uses_boogu_shifted_sigmas(self):
        class _Scheduler:
            def __init__(self):
                self.config = {
                    "do_shift": True,
                    "dynamic_time_shift": True,
                    "time_shift_version": "v1",
                    "base_shift": 0.5,
                    "max_shift": 1.15,
                    "time_shift_v2_half_scaling_factor": 60.0,
                }
                self.kwargs = None

            def set_timesteps(self, **kwargs):
                self.kwargs = kwargs

        scheduler = _Scheduler()
        model_config = _model_config("BooguImagePipeline")

        expected_sigmas = build_boogu_sigmas(
            num_inference_steps=4,
            num_tokens=(1024 // 8) * (1024 // 8),
            scheduler_config=scheduler.config,
        )
        BooguImageFlowGRPO.set_timesteps(scheduler, model_config, "cpu")

        assert scheduler.kwargs is not None
        assert scheduler.kwargs["num_inference_steps"] == 4
        assert scheduler.kwargs["device"] == "cpu"
        torch.testing.assert_close(torch.tensor(scheduler.kwargs["sigmas"]), torch.tensor(expected_sigmas))


class TestBooguDiffusionNFT:
    def test_prepare_model_inputs_and_forward(self):
        module = _BooguModule()
        model_config = _model_config("BooguImageTurboPipeline", algorithm="diffusion_nft", true_cfg_scale=2.0)
        latents = torch.zeros(1, 4, 8, 8)
        timesteps = torch.tensor([900.0])
        prompt_embeds = torch.ones(1, 2, 6)
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        negative_prompt_embeds = torch.zeros(1, 2, 6)
        negative_prompt_mask = torch.ones(1, 2, dtype=torch.long)

        model_inputs, negative_model_inputs = BooguImageDiffusionNFT.prepare_model_inputs(
            module=module,
            model_config=model_config,
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_mask,
            micro_batch={},
            step=0,
        )

        prediction = BooguImageDiffusionNFT.forward(module, model_config, model_inputs, negative_model_inputs)

        assert model_inputs["hidden_states"].shape == (1, 4, 8, 8)
        torch.testing.assert_close(model_inputs["timestep"], torch.tensor([0.9]))
        torch.testing.assert_close(prediction, torch.full((1, 4, 8, 8), 2.0))
