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
"""CPU coverage for repo-native BAGEL atomic SFT primitives."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl_omni.pipelines.bagel_flow_grpo import bagel_sft_model
from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelForTraining, BagelTrainingConfig
from verl_omni.pipelines.bagel_flow_grpo.bagel_sft_model import (
    BagelForSFT,
    BagelMLPConnector,
    BagelSFTConfig,
    BagelSiglipVisionTower,
    _apply_marker_token_ids,
    _checkpoint_tensor_for_target,
    _configure_position_grids,
    _load_model_checkpoint,
    _load_vae_checkpoint,
)


class TinyVisionTower(nn.Module):
    def __init__(self, *, patch_size: int = 2, hidden_size: int = 8, max_grid: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.max_grid = max_grid
        self.vision_model = nn.Module()
        self.vision_model.proj = nn.Linear(3 * patch_size**2, hidden_size)

    def forward(self, pixel_values, patch_mask=None):
        del patch_mask
        batch_size, channels, height, width = pixel_values.shape
        patch_size = self.patch_size
        patches = pixel_values.reshape(
            batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size
        )
        patches = torch.einsum("bchpwq->bhwcpq", patches).reshape(batch_size, -1, 3 * patch_size**2)
        position_ids = (
            torch.arange(height // patch_size, device=pixel_values.device)[:, None] * self.max_grid
            + torch.arange(width // patch_size, device=pixel_values.device)[None, :]
        ).reshape(-1)
        return self.vision_model.proj(patches), position_ids.unsqueeze(0).expand(batch_size, -1)


class IdentityVisionEncoder(nn.Module):
    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class MaskedMeanVisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_attention_mask = None

    def forward(self, *, inputs_embeds, attention_mask):
        self.last_attention_mask = attention_mask.detach().clone()
        allowed = attention_mask[:, 0].eq(0).to(inputs_embeds.dtype)
        weights = allowed / allowed.sum(dim=-1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.matmul(weights, inputs_embeds))


class TinySiglipModel(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 2,
        hidden_size: int = 8,
        max_grid: int = 4,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.embeddings = nn.Module()
        self.vision_model.embeddings.patch_embedding = nn.Conv2d(
            3,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.vision_model.embeddings.position_embedding = nn.Embedding(max_grid**2, hidden_size)
        nn.init.zeros_(self.vision_model.embeddings.position_embedding.weight)
        self.vision_model.encoder = encoder or IdentityVisionEncoder()
        self.vision_model.post_layernorm = nn.LayerNorm(hidden_size)


class TinyVAE(nn.Module):
    def __init__(self, latent_channels: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, latent_channels, kernel_size=1)

    def encode(self, images):
        return self.proj(images)


def tiny_config(**overrides) -> BagelSFTConfig:
    values = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "latent_patch_size": 2,
        "max_latent_size": 4,
        "latent_channel": 2,
        "vae_downsample": 1,
        "timestep_shift": 1.0,
        "vit_patch_size": 2,
        "vit_hidden_size": 8,
        "vit_max_num_patch_per_side": 4,
        "text_start_id": 62,
        "text_end_id": 63,
        "start_of_image_id": 60,
        "end_of_image_id": 61,
    }
    values.update(overrides)
    return BagelSFTConfig(**values)


def tiny_model(**config_overrides) -> BagelForSFT:
    config = tiny_config(**config_overrides)
    return BagelForSFT(config, vision_tower=TinyVisionTower(), vae_encoder=TinyVAE())


def text_batch():
    ids = torch.tensor([[62, 1, 2, 63], [62, 4, 63, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    return ids, mask


def response_batch(*, padded: bool = False):
    if padded:
        inputs = torch.tensor([[62, 10, 11, 12], [62, 14, 15, 0]])
        labels = torch.tensor([[10, 11, 12, 63], [14, 15, 63, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    else:
        inputs = torch.tensor([[62, 10, 11, 12], [62, 14, 15, 16]])
        labels = torch.tensor([[10, 11, 12, 63], [14, 15, 16, 63]])
        mask = torch.ones_like(inputs, dtype=torch.bool)
    return inputs, labels, mask


def image_batch():
    values = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    return values / values.max()


def latent_batch():
    return torch.linspace(-1, 1, 2 * 2 * 4 * 4).reshape(2, 2, 4, 4)


def test_sft_model_public_api_is_intentionally_narrow():
    assert set(bagel_sft_model.__all__) == {
        "BagelForSFT",
        "BagelSFTConfig",
        "BagelSFTOutput",
    }


def test_bagel_flow_grpo_package_public_api_is_intentionally_narrow(monkeypatch):
    package_name = "verl_omni.pipelines._bagel_flow_grpo_public_api_test"
    package_path = Path(bagel_sft_model.__file__).with_name("__init__.py")

    sft_stub = ModuleType(f"{package_name}.bagel_sft_model")
    for name in (
        "BagelCheckpointLoadReport",
        "BagelForSFT",
        "BagelFrozenVAEEncoder",
        "BagelSFTConfig",
        "BagelSFTOutput",
        "BagelSiglipVisionTower",
    ):
        setattr(sft_stub, name, object())
    diffusion_stub = ModuleType(f"{package_name}.diffusers_training_adapter")
    diffusion_stub.BagelDiffusion = object()
    rollout_stub = ModuleType(f"{package_name}.vllm_omni_rollout_adapter")
    rollout_stub.BagelPipelineWithLogProb = object()
    monkeypatch.setitem(sys.modules, sft_stub.__name__, sft_stub)
    monkeypatch.setitem(sys.modules, diffusion_stub.__name__, diffusion_stub)
    monkeypatch.setitem(sys.modules, rollout_stub.__name__, rollout_stub)

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_path,
        submodule_search_locations=[str(package_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    spec.loader.exec_module(package)

    expected = {"BagelDiffusion", "BagelPipelineWithLogProb"}
    assert set(package.__all__) == expected
    assert {name for name in vars(package) if name[:1].isupper()} == expected


def test_t2i_loss_is_normalized_per_sample_and_valid_patch():
    torch.manual_seed(0)
    model = tiny_model()
    text_ids, text_mask = text_batch()
    latents = latent_batch()
    noise = torch.linspace(1, -1, 2 * 4 * 8).reshape(2, 4, 8)
    patch_mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)

    output = model.forward_t2i(
        prompt_input_ids=text_ids,
        prompt_attention_mask=text_mask,
        timestep_logits=torch.tensor([0.25, 0.75]),
        target_latents=latents,
        noise=noise,
        target_patch_mask=patch_mask,
    )

    manual_tokens = (output.velocity.float() - output.target.float()).square().mean(dim=-1)
    manual_per_sample = (manual_tokens * patch_mask).sum(dim=-1) / patch_mask.sum(dim=-1)
    assert torch.allclose(output.loss_per_sample, manual_per_sample)
    assert torch.allclose(output.loss, manual_per_sample.mean())


def test_t2i_masked_patch_cannot_affect_valid_predictions():
    torch.manual_seed(22)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    target_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    first_noise = torch.zeros(2, 4, 8)
    second_noise = first_noise.clone()
    second_noise[:, 2:] = 100

    common = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "timestep_logits": torch.tensor([0.2, 0.8]),
        "target_latents": latent_batch(),
        "target_patch_mask": target_mask,
    }
    first = model.forward_t2i(**common, noise=first_noise)
    second = model.forward_t2i(**common, noise=second_noise)

    assert torch.equal(first.velocity[target_mask], second.velocity[target_mask])
    assert torch.equal(first.loss, second.loss)


def test_t2i_original_inputs_match_flowgrpo_forward_and_backward(monkeypatch):
    torch.manual_seed(1)
    config = tiny_config()
    flow_model = BagelForTraining(config)
    sft_model = BagelForSFT(config, vision_tower=TinyVisionTower(), vae_encoder=TinyVAE())
    flow_model.load_state_dict(
        {name: tensor for name, tensor in sft_model.state_dict().items() if name in flow_model.state_dict()},
        strict=True,
    )
    text_ids = torch.tensor([[62, 1, 2, 63], [62, 4, 5, 63]])
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    target_patches, position_ids = sft_model._patchify_latents(latent_batch())
    noise = torch.randn_like(target_patches)
    noisy, _, shifted = sft_model._prepare_flow_target(
        target_patches, timestep_logits=torch.tensor([0.2, 0.8]), noise=noise
    )

    original_forward = BagelForTraining.forward
    expected = original_forward(
        flow_model,
        hidden_states=noisy,
        timestep=shifted,
        text_token_ids=text_ids,
        text_attention_mask=text_mask,
        latent_pos_ids=position_ids,
    )[0]
    captured_kwargs = {}

    def capture_original_call(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(BagelForTraining, "forward", capture_original_call)
    actual = sft_model.forward_t2i(
        prompt_input_ids=text_ids,
        prompt_attention_mask=text_mask,
        timestep_logits=torch.tensor([0.2, 0.8]),
        target_latents=latent_batch(),
        noise=noise,
    ).velocity

    assert "image_position_ids" not in captured_kwargs
    assert "latent_attention_mask" not in captured_kwargs
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    probe = torch.linspace(-1, 1, expected.numel()).reshape_as(expected)
    (expected.float() * probe).sum().backward()
    (actual.float() * probe).sum().backward()
    flow_parameters = dict(flow_model.named_parameters())
    sft_parameters = dict(sft_model.named_parameters())
    assert flow_parameters.keys() <= sft_parameters.keys()
    for name, flow_parameter in flow_parameters.items():
        sft_gradient = sft_parameters[name].grad
        if flow_parameter.grad is None:
            assert sft_gradient is None, name
        else:
            assert sft_gradient is not None, name
            torch.testing.assert_close(sft_gradient, flow_parameter.grad, rtol=0, atol=0, msg=name)


def test_t2i_is_invariant_to_longer_right_padded_batch_neighbors():
    torch.manual_seed(11)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    latents = latent_batch()
    target_patches, _ = model._patchify_latents(latents)
    noise = torch.randn_like(target_patches)
    logits = torch.tensor([0.2, 0.8])

    batched = model.forward_t2i(
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        timestep_logits=logits,
        target_latents=latents,
        noise=noise,
    )
    single = model.forward_t2i(
        prompt_input_ids=prompt_ids[1:2, :3],
        prompt_attention_mask=prompt_mask[1:2, :3],
        timestep_logits=logits[1:2],
        target_latents=latents[1:2],
        noise=noise[1:2],
    )

    torch.testing.assert_close(batched.velocity[1], single.velocity[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched.loss_per_sample[1], single.loss_per_sample[0], rtol=1e-5, atol=1e-6)


def test_flow_target_uses_sigmoid_logits_and_timestep_shift():
    model = tiny_model(timestep_shift=3.0)
    clean = torch.tensor([[[1.0, -1.0]]])
    noise = torch.tensor([[[5.0, 3.0]]])

    noisy, target, shifted = model._prepare_flow_target(
        clean,
        timestep_logits=torch.zeros(1),
        noise=noise,
    )

    expected_shifted = torch.tensor([0.75])
    assert torch.equal(shifted, expected_shifted)
    assert torch.equal(noisy, 0.25 * clean + 0.75 * noise)
    assert torch.equal(target, noise - clean)

    with pytest.raises(ValueError, match="finite"):
        model._prepare_flow_target(clean, timestep_logits=torch.tensor([float("nan")]), noise=noise)


def test_sft_preserves_flow_generation_state_key_boundary():
    flow_keys = set(BagelForTraining(tiny_config()).state_dict())
    model = tiny_model()
    sft_keys = set(model.state_dict())

    assert flow_keys <= sft_keys
    assert type(model.layers[0]).__name__ == "_BagelSFTMoTLayer"
    assert model._no_split_modules == [type(model.layers[0]).__name__]
    assert all(not name.startswith(("core.", "model.")) for name in sft_keys)
    assert all(
        name.startswith(("lm_head.", "vit_model.", "connector.", "vit_pos_embed.")) for name in sft_keys - flow_keys
    )


def test_reflect_supervises_only_response_and_is_causal():
    torch.manual_seed(2)
    model = tiny_model().eval()
    prefix_ids, prefix_mask = text_batch()
    response_inputs, response_labels, response_mask = response_batch()
    changed_inputs = response_inputs.clone()
    changed_labels = response_labels.clone()
    changed_inputs[:, 2] = torch.tensor([20, 21])
    changed_labels[:, 1] = torch.tensor([20, 21])

    baseline = model.forward_reflect(
        prefix_input_ids=prefix_ids,
        prefix_attention_mask=prefix_mask,
        response_input_ids=response_inputs,
        response_labels=response_labels,
        response_loss_mask=response_mask,
        current_vit_pixel_values=image_batch(),
    )
    modified = model.forward_reflect(
        prefix_input_ids=prefix_ids,
        prefix_attention_mask=prefix_mask,
        response_input_ids=changed_inputs,
        response_labels=changed_labels,
        response_loss_mask=response_mask,
        current_vit_pixel_values=image_batch(),
    )

    # A changed response input is first visible at its own causal position.
    assert torch.equal(baseline.logits[:, :2], modified.logits[:, :2])
    assert not torch.equal(baseline.logits[:, 2], modified.logits[:, 2])
    manual = torch.nn.functional.cross_entropy(
        baseline.logits.float().reshape(-1, baseline.logits.shape[-1]),
        response_labels.reshape(-1),
        reduction="none",
    ).reshape(response_labels.shape)
    assert torch.allclose(baseline.loss_per_sample, manual.mean(dim=-1))


def test_reflect_uses_causal_full_causal_attention_segments():
    torch.manual_seed(21)
    model = tiny_model().eval()
    prefix_ids, prefix_mask = text_batch()
    response_inputs, response_labels, response_mask = response_batch()
    captured = {}
    original = model._run_sft_sequence

    def capture(sequence, **kwargs):
        captured["mask"] = kwargs["attention_mask"].detach().clone()
        return original(sequence, **kwargs)

    model._run_sft_sequence = capture
    output = model.forward_reflect(
        prefix_input_ids=prefix_ids,
        prefix_attention_mask=prefix_mask,
        response_input_ids=response_inputs,
        response_labels=response_labels,
        response_loss_mask=response_mask,
        current_vit_pixel_values=image_batch(),
    )

    mask = captured["mask"][0, 0]
    prefix_length = prefix_ids.shape[1]
    response_start = output.context_length
    image_block = mask[prefix_length:response_start, prefix_length:response_start]
    response_block = mask[response_start:, response_start:]
    assert image_block.all()
    assert torch.equal(mask[:prefix_length, :prefix_length], torch.ones(prefix_length, prefix_length).tril().bool())
    assert torch.equal(response_block, torch.ones_like(response_block).tril())


def test_reflect_padding_is_excluded_from_ce_normalization():
    torch.manual_seed(3)
    model = tiny_model()
    prefix_ids, prefix_mask = text_batch()
    response_inputs, response_labels, response_mask = response_batch(padded=True)
    response_inputs = response_inputs.to(dtype=torch.int32)
    response_labels = response_labels.to(dtype=torch.int32)

    output = model.forward_reflect(
        prefix_input_ids=prefix_ids,
        prefix_attention_mask=prefix_mask,
        response_input_ids=response_inputs,
        response_labels=response_labels,
        response_loss_mask=response_mask,
        current_vit_pixel_values=image_batch(),
    )

    token_loss = torch.nn.functional.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        response_labels.reshape(-1).long(),
        reduction="none",
    ).reshape(response_labels.shape)
    expected = (token_loss * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
    assert torch.allclose(output.loss_per_sample, expected)


def test_reflect_masked_vision_patch_cannot_affect_logits():
    torch.manual_seed(31)
    model = tiny_model().eval()
    prefix_ids, prefix_mask = text_batch()
    response_inputs, response_labels, response_mask = response_batch()
    vision_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    baseline_pixels = image_batch()
    changed_pixels = baseline_pixels.clone()
    changed_pixels[:, :, 2:, :] += 100

    common = {
        "prefix_input_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "response_input_ids": response_inputs,
        "response_labels": response_labels,
        "response_loss_mask": response_mask,
        "current_vit_patch_mask": vision_mask,
    }
    baseline = model.forward_reflect(**common, current_vit_pixel_values=baseline_pixels)
    changed = model.forward_reflect(**common, current_vit_pixel_values=changed_pixels)

    assert torch.equal(baseline.logits, changed.logits)
    assert torch.equal(baseline.loss, changed.loss)


def test_reflect_vision_patch_mask_fails_closed():
    model = tiny_model().eval()
    prefix_ids, prefix_mask = text_batch()
    response_inputs, response_labels, response_mask = response_batch()
    common = {
        "prefix_input_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "response_input_ids": response_inputs,
        "response_labels": response_labels,
        "response_loss_mask": response_mask,
        "current_vit_pixel_values": image_batch(),
    }

    with pytest.raises(ValueError, match="patch_mask must have shape"):
        model.forward_reflect(**common, current_vit_patch_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one valid patch"):
        model.forward_reflect(
            **common,
            current_vit_patch_mask=torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool),
        )


def test_reflect_requires_shifted_response_and_supervised_eos():
    model = tiny_model()
    prefix_ids, prefix_mask = text_batch()
    response_inputs, response_labels, response_mask = response_batch()

    bad_labels = response_labels.clone()
    bad_labels[:, -1] = 17
    with pytest.raises(ValueError, match="final supervised response label"):
        model.forward_reflect(
            prefix_input_ids=prefix_ids,
            prefix_attention_mask=prefix_mask,
            response_input_ids=response_inputs,
            response_labels=bad_labels,
            response_loss_mask=response_mask,
            current_vit_pixel_values=image_batch(),
        )

    bad_inputs = response_inputs.clone()
    bad_inputs[:, 2] = 19
    with pytest.raises(ValueError, match="one-token shifted"):
        model.forward_reflect(
            prefix_input_ids=prefix_ids,
            prefix_attention_mask=prefix_mask,
            response_input_ids=bad_inputs,
            response_labels=response_labels,
            response_loss_mask=response_mask,
            current_vit_pixel_values=image_batch(),
        )


def test_edit_target_cannot_change_clean_conditioning_prefix():
    torch.manual_seed(4)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 63, 0]])
    edit_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    current = latent_batch()
    target_a = current + 0.1
    target_b = current - 0.7
    noise_a = torch.full((2, 4, 8), 0.25)
    noise_b = torch.full((2, 4, 8), -0.5)

    first = model.forward_edit(
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        edit_input_ids=edit_ids,
        edit_attention_mask=edit_mask,
        current_vit_pixel_values=image_batch(),
        current_latents=current,
        target_latents=target_a,
        timestep_logits=torch.tensor([0.3, 0.6]),
        noise=noise_a,
        return_hidden_states=True,
    )
    second = model.forward_edit(
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        edit_input_ids=edit_ids,
        edit_attention_mask=edit_mask,
        current_vit_pixel_values=image_batch(),
        current_latents=current,
        target_latents=target_b,
        timestep_logits=torch.tensor([0.3, 0.6]),
        noise=noise_b,
        return_hidden_states=True,
    )

    assert first.context_length == second.context_length
    assert torch.equal(first.hidden_states[:, : first.context_length], second.hidden_states[:, : second.context_length])
    assert not torch.equal(first.velocity, second.velocity)


def test_edit_clean_prefix_has_zero_gradient_to_target():
    torch.manual_seed(41)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 23, 63]])
    target = (latent_batch() + 0.2).requires_grad_()

    output = model.forward_edit(
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        edit_input_ids=edit_ids,
        edit_attention_mask=torch.ones_like(edit_ids, dtype=torch.bool),
        current_vit_pixel_values=image_batch(),
        current_latents=latent_batch(),
        target_latents=target,
        timestep_logits=torch.tensor([0.3, 0.6]),
        noise=torch.zeros(2, 4, 8),
        return_hidden_states=True,
    )
    gradient = torch.autograd.grad(
        output.hidden_states[:, : output.context_length].float().square().mean(),
        target,
        allow_unused=True,
    )[0]

    assert gradient is None or not torch.count_nonzero(gradient)


def test_edit_masked_target_patch_cannot_affect_valid_predictions():
    torch.manual_seed(42)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 23, 63]])
    target_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    first_noise = torch.zeros(2, 4, 8)
    second_noise = first_noise.clone()
    second_noise[:, 2:] = 100

    common = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "edit_input_ids": edit_ids,
        "edit_attention_mask": torch.ones_like(edit_ids, dtype=torch.bool),
        "current_vit_pixel_values": image_batch(),
        "current_latents": latent_batch(),
        "target_latents": latent_batch() + 0.2,
        "timestep_logits": torch.tensor([0.3, 0.6]),
        "target_patch_mask": target_mask,
    }
    first = model.forward_edit(**common, noise=first_noise)
    second = model.forward_edit(**common, noise=second_noise)

    assert torch.equal(first.velocity[target_mask], second.velocity[target_mask])
    assert torch.equal(first.loss, second.loss)


def test_edit_masked_clean_latent_patch_cannot_affect_valid_predictions():
    torch.manual_seed(43)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 23, 63]])
    current_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    baseline_current = latent_batch()
    changed_current = baseline_current.clone()
    changed_current[:, :, 2:, :] += 100
    common = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "edit_input_ids": edit_ids,
        "edit_attention_mask": torch.ones_like(edit_ids, dtype=torch.bool),
        "current_vit_pixel_values": image_batch(),
        "current_patch_mask": current_mask,
        "target_latents": latent_batch() + 0.2,
        "timestep_logits": torch.tensor([0.3, 0.6]),
        "noise": torch.zeros(2, 4, 8),
    }

    baseline = model.forward_edit(**common, current_latents=baseline_current)
    changed = model.forward_edit(**common, current_latents=changed_current)

    assert torch.equal(baseline.velocity, changed.velocity)
    assert torch.equal(baseline.loss, changed.loss)


def test_edit_masked_vision_patch_cannot_affect_valid_predictions():
    torch.manual_seed(44)
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 23, 63]])
    vision_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    baseline_pixels = image_batch()
    changed_pixels = baseline_pixels.clone()
    changed_pixels[:, :, 2:, :] += 100
    common = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "edit_input_ids": edit_ids,
        "edit_attention_mask": torch.ones_like(edit_ids, dtype=torch.bool),
        "current_vit_patch_mask": vision_mask,
        "current_latents": latent_batch(),
        "target_latents": latent_batch() + 0.2,
        "timestep_logits": torch.tensor([0.3, 0.6]),
        "noise": torch.zeros(2, 4, 8),
    }

    baseline = model.forward_edit(**common, current_vit_pixel_values=baseline_pixels)
    changed = model.forward_edit(**common, current_vit_pixel_values=changed_pixels)

    assert torch.equal(baseline.velocity, changed.velocity)
    assert torch.equal(baseline.loss, changed.loss)


def test_edit_clean_patch_mask_fails_closed():
    model = tiny_model().eval()
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 23, 63]])
    common = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "edit_input_ids": edit_ids,
        "edit_attention_mask": torch.ones_like(edit_ids, dtype=torch.bool),
        "current_vit_pixel_values": image_batch(),
        "current_latents": latent_batch(),
        "target_latents": latent_batch() + 0.2,
        "timestep_logits": torch.tensor([0.3, 0.6]),
        "noise": torch.zeros(2, 4, 8),
    }

    with pytest.raises(ValueError, match="current_patch_mask must have shape"):
        model.forward_edit(**common, current_patch_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one valid current patch"):
        model.forward_edit(
            **common,
            current_patch_mask=torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool),
        )


def test_frozen_vae_encodes_images_but_is_not_registered_or_differentiated():
    torch.manual_seed(5)
    model = tiny_model()
    vae_state_before_dtype_change = {name: tensor.clone() for name, tensor in model.vae_encoder.state_dict().items()}
    prompt_ids, prompt_mask = text_batch()
    edit_ids = torch.tensor([[62, 20, 21, 63], [62, 22, 23, 63]])
    output = model.forward_edit(
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        edit_input_ids=edit_ids,
        edit_attention_mask=torch.ones_like(edit_ids, dtype=torch.bool),
        current_vit_pixel_values=image_batch(),
        current_vae_images=image_batch(),
        target_images=image_batch() * 0.5,
        timestep_logits=torch.tensor([0.4, 0.7]),
        noise=torch.zeros(2, 4, 8),
    )
    output.loss.backward()

    vae_parameters = list(model.vae_encoder.parameters())
    registered_parameter_ids = {id(parameter) for parameter in model.parameters()}
    assert all(not parameter.requires_grad for parameter in vae_parameters)
    assert all(parameter.grad is None for parameter in vae_parameters)
    assert all(id(parameter) not in registered_parameter_ids for parameter in vae_parameters)
    assert all(not name.startswith("vae_encoder") for name in model.state_dict())
    assert model.connector.fc1.weight.grad is not None
    assert torch.isfinite(model.connector.fc1.weight.grad).all()

    model.train()
    assert not model.vae_encoder.module.training
    model.to(dtype=torch.bfloat16)
    assert model.vae2llm.weight.dtype == torch.bfloat16
    assert all(
        not tensor.is_floating_point() or tensor.dtype == torch.float32
        for tensor in model.vae_encoder.state_dict().values()
    )
    for name, tensor in model.vae_encoder.state_dict().items():
        assert torch.equal(tensor, vae_state_before_dtype_change[name])


def _raw_model_state(model):
    raw = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("lm_head."):
            source = "language_model.lm_head." + name[len("lm_head.") :]
        elif name.startswith("vit_model."):
            source = name
        elif name.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
            source = name
        elif name.startswith(("connector.", "vit_pos_embed.")):
            source = name
        else:
            source = "language_model.model." + name
        raw[source] = tensor.detach().clone()
    return raw


def test_checkpoint_mapping_is_shape_checked_and_complete():
    torch.manual_seed(6)
    source = tiny_model()
    target = tiny_model()
    raw_model = _raw_model_state(source)
    raw_vae = {name: tensor.detach().clone() for name, tensor in source.vae_encoder.state_dict().items()}

    loaded, ignored = _load_model_checkpoint(target, raw_model)
    loaded_vae = _load_vae_checkpoint(target.vae_encoder.module, raw_vae)

    assert ignored == {"latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"}
    assert len(loaded) == len(source.state_dict()) - len(ignored)
    assert len(loaded_vae) == len(source.vae_encoder.state_dict())
    for name, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], tensor)

    malformed = dict(raw_model)
    malformed["language_model.lm_head.weight"] = malformed["language_model.lm_head.weight"][:-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        _load_model_checkpoint(tiny_model(), malformed)

    missing = dict(raw_model)
    del missing["language_model.lm_head.weight"]
    with pytest.raises(ValueError, match="missing"):
        _load_model_checkpoint(tiny_model(), missing)

    unknown = dict(raw_model)
    unknown["unused.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unsupported"):
        _load_model_checkpoint(tiny_model(), unknown)

    malformed_vae = dict(raw_vae)
    malformed_vae["proj.weight"] = malformed_vae["proj.weight"][:-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        _load_vae_checkpoint(tiny_model().vae_encoder.module, malformed_vae)

    missing_vae = dict(raw_vae)
    del missing_vae["proj.bias"]
    with pytest.raises(ValueError, match="missing"):
        _load_vae_checkpoint(tiny_model().vae_encoder.module, missing_vae)

    unknown_vae = dict(raw_vae)
    unknown_vae["unknown.extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unsupported"):
        _load_vae_checkpoint(tiny_model().vae_encoder.module, unknown_vae)

    duplicate_vae = dict(raw_vae)
    duplicate_vae["module.proj.weight"] = duplicate_vae["proj.weight"].clone()
    with pytest.raises(ValueError, match="multiple"):
        _load_vae_checkpoint(tiny_model().vae_encoder.module, duplicate_vae)

    without_fixed_positions = {
        name: tensor
        for name, tensor in raw_model.items()
        if name not in {"latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"}
    }
    _load_model_checkpoint(tiny_model(), without_fixed_positions)


def test_published_siglip_linear_patch_weight_and_post_layernorm_are_numerically_correct():
    torch.manual_seed(7)
    source_siglip = TinySiglipModel()
    target_siglip = TinySiglipModel()
    target_siglip.load_state_dict(source_siglip.state_dict())
    source = BagelForSFT(
        tiny_config(),
        vision_tower=BagelSiglipVisionTower(source_siglip, patch_size=2, max_num_patch_per_side=4),
        vae_encoder=TinyVAE(),
    )
    target = BagelForSFT(
        tiny_config(),
        vision_tower=BagelSiglipVisionTower(target_siglip, patch_size=2, max_num_patch_per_side=4),
        vae_encoder=TinyVAE(),
    )
    weight_name = "vit_model.vision_model.embeddings.patch_embedding.weight"
    conv_weight = source.state_dict()[weight_name]
    published_weight = conv_weight.permute(0, 2, 3, 1).reshape(conv_weight.shape[0], -1)
    converted = _checkpoint_tensor_for_target(published_weight, conv_weight, weight_name)

    assert torch.equal(converted, conv_weight)

    raw_model = _raw_model_state(source)
    raw_model[weight_name] = published_weight
    _load_model_checkpoint(target, raw_model)
    assert torch.equal(target.state_dict()[weight_name], conv_weight)

    pixels = image_batch()
    actual, _ = source.vit_model(pixels)
    conv = source.vit_model.vision_model.embeddings.patch_embedding
    expected = conv(pixels).permute(0, 2, 3, 1).reshape_as(actual)
    expected = source.vit_model.vision_model.post_layernorm(expected)
    assert torch.allclose(actual, expected)


def test_siglip_masked_patches_cannot_leak_through_internal_attention():
    torch.manual_seed(45)
    encoder = MaskedMeanVisionEncoder()
    siglip = TinySiglipModel(encoder=encoder)
    tower = BagelSiglipVisionTower(siglip, patch_size=2, max_num_patch_per_side=4)
    patch_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    baseline_pixels = image_batch()
    changed_pixels = baseline_pixels.clone()
    changed_pixels[:, :, 2:, :] += 100

    baseline, _ = tower(baseline_pixels, patch_mask=patch_mask)
    changed, _ = tower(changed_pixels, patch_mask=patch_mask)

    assert torch.equal(baseline[patch_mask], changed[patch_mask])
    assert torch.isfinite(changed).all()
    allowed = encoder.last_attention_mask[0, 0].eq(0)
    expected_first_image = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(allowed[:4], expected_first_image)


def test_connector_and_marker_configuration_fail_closed():
    with pytest.raises(ValueError, match="unsupported BAGEL connector activation"):
        BagelMLPConnector(8, 16, "unknown")

    config = tiny_config()
    _apply_marker_token_ids(
        config,
        {
            "text_start_id": 52,
            "text_end_id": 53,
            "vision_start_id": 54,
            "vision_end_id": 55,
        },
    )
    assert (config.text_start_id, config.text_end_id) == (52, 53)
    assert (config.start_of_image_id, config.end_of_image_id) == (54, 55)

    with pytest.raises(ValueError, match="distinct"):
        _apply_marker_token_ids(
            config,
            {
                "text_start_id": 52,
                "text_end_id": 52,
                "vision_start_id": 54,
                "vision_end_id": 55,
            },
        )


def test_fixed_position_grids_are_rebuilt_from_checkpoint_shapes():
    config = tiny_config(max_latent_size=2, vit_max_num_patch_per_side=2)
    _configure_position_grids(
        config,
        {
            "latent_pos_embed.pos_embed": torch.zeros(25, config.hidden_size),
            "vit_pos_embed.pos_embed": torch.zeros(36, config.hidden_size),
        },
    )

    assert config.max_latent_size == 5
    assert config.vit_max_num_patch_per_side == 6

    with pytest.raises(ValueError, match="square grid"):
        _configure_position_grids(
            config,
            {"latent_pos_embed.pos_embed": torch.zeros(24, config.hidden_size)},
        )


def test_config_reads_separate_llm_and_vit_files(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "vae_config": {"z_channels": 4, "downsample": 16},
                "latent_patch_size": 4,
                "max_latent_size": 8,
                "timestep_shift": 3.0,
                "vit_max_num_patch_per_side": 12,
            }
        )
    )
    (tmp_path / "llm_config.json").write_text(
        json.dumps(
            {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "tie_word_embeddings": False,
            }
        )
    )
    (tmp_path / "vit_config.json").write_text(json.dumps({"patch_size": 8, "hidden_size": 24}))

    config = BagelSFTConfig.from_model_path(str(tmp_path))
    flow_config = BagelTrainingConfig.from_model_path(str(tmp_path))

    assert config.hidden_size == 32
    assert config.latent_patch_size == 4
    assert config.latent_channel == 4
    assert config.vae_downsample == 16
    assert config.timestep_shift == 3.0
    assert config.vit_patch_size == 8
    assert config.vit_hidden_size == 24
    assert config.vit_max_num_patch_per_side == 12
    assert not hasattr(flow_config, "vit_hidden_size")
