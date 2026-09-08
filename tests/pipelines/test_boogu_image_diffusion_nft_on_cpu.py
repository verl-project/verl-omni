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
"""CPU tests for the Boogu-Image DiffusionNFT training adapter."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture(scope="module")
def adapters():
    """Load real training modules while isolating unrelated rollout imports."""
    root = Path(__file__).resolve().parents[2] / "verl_omni"
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "verl_omni" or name.startswith("verl_omni.pipelines"):
                del sys.modules[name]
        package = ModuleType("verl_omni")
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        for directory in (root / "pipelines").iterdir():
            if not (directory / "__init__.py").is_file():
                continue
            if directory.name in {"boogu_image_diffusion_nft", "schedulers"}:
                continue
            name = f"verl_omni.pipelines.{directory.name}"
            package = ModuleType(name)
            package.__path__ = [str(directory)]
            package.__all__ = []
            sys.modules[name] = package
        pipelines = importlib.import_module("verl_omni.pipelines")
        yield SimpleNamespace(
            pipelines=pipelines,
            adapter=pipelines.BooguImageDiffusionNFT,
            base=importlib.import_module("verl_omni.pipelines.model_base").DiffusionModelBase,
            common=importlib.import_module("verl_omni.pipelines.boogu_image_flow_grpo.common"),
            flow=importlib.import_module("verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter"),
            nft=importlib.import_module("verl_omni.pipelines.boogu_image_diffusion_nft.diffusers_training_adapter"),
            config=importlib.import_module("verl_omni.workers.config"),
        )


def _model_config(adapters, guidance_scale=1.0):
    config = object.__new__(adapters.config.DiffusionModelConfig)
    object.__setattr__(config, "architecture", "BooguImagePipeline")
    object.__setattr__(config, "algorithm", "diffusion_nft")
    object.__setattr__(config, "external_lib", None)
    object.__setattr__(config, "pipeline", adapters.config.DiffusionPipelineConfig(guidance_scale=guidance_scale))
    return config


def test_registered_for_boogu_image_diffusion_nft(adapters):
    assert adapters.base.get_class_by_name("BooguImagePipeline", "diffusion_nft") is adapters.adapter
    assert adapters.base.get_class(_model_config(adapters)) is adapters.adapter
    assert "BooguImageDiffusionNFT" in adapters.pipelines.__all__


@pytest.mark.parametrize("has_condition", [False, True])
@pytest.mark.parametrize("has_negative", [False, True])
def test_prepare_model_inputs_accepts_single_step_tensors(adapters, tmp_path, has_condition, has_negative):
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text('{"num_train_timesteps": 2000}', encoding="utf-8")
    config = _model_config(adapters)
    object.__setattr__(config, "local_path", str(tmp_path))
    module = SimpleNamespace(config=SimpleNamespace(axes_dim_rope=[16, 24, 24], axes_lens=[1, 8, 8]))
    latents = torch.arange(96, dtype=torch.float16).reshape(2, 3, 4, 4)
    timesteps = torch.tensor([500.0, 1500.0])
    micro_batch = TensorDict({}, batch_size=[2])
    if has_condition:
        micro_batch["condition_image_latents"] = latents.float() + 1
    kwargs = dict(
        module=module,
        model_config=config,
        prompt_embeds=torch.ones(2, 5, 8),
        prompt_embeds_mask=torch.ones(2, 5, dtype=torch.bool),
        negative_prompt_embeds=torch.zeros(2, 5, 8) if has_negative else None,
        negative_prompt_embeds_mask=torch.zeros(2, 5, dtype=torch.bool) if has_negative else None,
        micro_batch=micro_batch,
    )
    freqs_cis = torch.ones(1, dtype=torch.complex64)
    with (
        patch.object(adapters.nft, "get_boogu_freqs_cis", return_value=freqs_cis) as nft_freqs,
        patch.object(adapters.flow, "get_boogu_freqs_cis", return_value=freqs_cis),
    ):
        actual = adapters.adapter.prepare_model_inputs(latents=latents, timesteps=timesteps, step=7, **kwargs)
        expected = adapters.flow.BooguImage.prepare_model_inputs(
            latents=latents.unsqueeze(1), timesteps=timesteps.unsqueeze(1), step=0, **kwargs
        )

    nft_freqs.assert_called_once_with(module.config.axes_dim_rope, module.config.axes_lens)
    torch.testing.assert_close(actual, expected)
    model_inputs, negative_model_inputs = actual
    assert model_inputs["hidden_states"] is latents
    assert model_inputs["hidden_states"].shape == (2, 3, 4, 4)
    torch.testing.assert_close(model_inputs["timestep"], torch.tensor([0.75, 0.25], dtype=latents.dtype))
    assert (negative_model_inputs is not None) == has_negative
    if has_condition:
        references = model_inputs["ref_image_hidden_states"]
        assert len(references) == 2
        for index, images in enumerate(references):
            assert len(images) == 1
            torch.testing.assert_close(images[0], (latents.float() + 1)[index].to(latents.dtype))
    else:
        assert model_inputs["ref_image_hidden_states"] is None


@pytest.mark.parametrize("output_form", ["tuple", "sample", "tensor"])
def test_forward_inverts_velocity_and_reconstructs_x0(adapters, output_form):
    generator = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 3, 2, 2, generator=generator)
    epsilon = torch.randn(4, 3, 2, 2, generator=generator)
    t = torch.tensor([0.0, 0.25, 0.75, 1.0]).reshape(4, 1, 1, 1)
    xt = (1.0 - t) * x0 + t * epsilon
    velocity = x0 - epsilon
    outputs = {"tuple": (velocity,), "sample": SimpleNamespace(sample=velocity), "tensor": velocity}
    module = MagicMock(return_value=outputs[output_form])
    model_inputs = {"hidden_states": xt, "timestep": 1.0 - t.flatten(), "return_dict": False}

    prediction = adapters.adapter.forward(module, _model_config(adapters), model_inputs)

    module.assert_called_once_with(**model_inputs)
    torch.testing.assert_close(prediction, -velocity)
    torch.testing.assert_close(xt - t * prediction, x0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("guidance_scale", "expected_scale"), [(2.5, 2.5), (None, 4.0)])
def test_forward_applies_text_cfg_before_negation(adapters, guidance_scale, expected_scale):
    positive = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    negative = positive.flip(-1) * 0.5
    module = MagicMock(side_effect=[(positive,), (negative,)])
    model_inputs = {"instruction_hidden_states": torch.ones(2, 5, 8)}
    negative_model_inputs = {"instruction_hidden_states": torch.zeros(2, 5, 8)}

    prediction = adapters.adapter.forward(
        module, _model_config(adapters, guidance_scale), model_inputs, negative_model_inputs
    )

    assert module.call_args_list == [call(**model_inputs), call(**negative_model_inputs)]
    expected = adapters.common.apply_boogu_text_cfg(positive, negative, expected_scale).neg()
    torch.testing.assert_close(prediction, expected)


@pytest.mark.parametrize(("guidance_scale", "has_negative"), [(0.5, True), (1.0, True), (4.0, False)])
def test_forward_skips_cfg_without_active_guidance_or_negative_inputs(adapters, guidance_scale, has_negative):
    velocity = torch.ones(2, 3, 2, 2)
    module = MagicMock(return_value=(velocity,))
    model_inputs = {"instruction_hidden_states": torch.ones(2, 5, 8)}
    negative_model_inputs = {"instruction_hidden_states": torch.zeros(2, 5, 8)} if has_negative else None

    prediction = adapters.adapter.forward(
        module, _model_config(adapters, guidance_scale), model_inputs, negative_model_inputs
    )

    module.assert_called_once_with(**model_inputs)
    torch.testing.assert_close(prediction, -velocity)
