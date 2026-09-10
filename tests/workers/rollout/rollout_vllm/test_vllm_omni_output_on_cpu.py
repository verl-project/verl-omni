# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.diffusion_rollout_output import quantize_pixels
from verl_omni.pipelines.rollout_artifacts import MediaArtifact
from verl_omni.pipelines.rollout_media import MediaSpec
from verl_omni.pipelines.rollout_request import OmniRolloutRequest
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


@pytest.fixture
def diffusion_strategy():
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 0
    return DiffusionStrategy(server)


def _request_output(artifacts, primary="image_preview", audio=None):
    return SimpleNamespace(
        images=[{name: artifact.data for name, artifact in artifacts.items()}],
        multimodal_output={
            "metadata": {
                "media_artifacts": {
                    "primary": primary,
                    "audio": audio,
                    "preview": "image_preview" if "image_preview" in artifacts else None,
                    "specs": {name: asdict(artifact.spec) for name, artifact in artifacts.items()},
                }
            }
        },
        trajectory_latents=None,
        trajectory_log_probs=None,
        trajectory_timesteps=None,
    )


def _pixels_output(pixels):
    encoded = quantize_pixels(pixels, "zero_one", context="adapter")
    return _request_output(
        {"image_preview": MediaArtifact(MediaSpec("image", "decoded", "CHW"), encoded.reshape(1, 1, -1))}
    )


def test_diffusion_prompt_preserves_multimodal_processor_kwargs(diffusion_strategy):
    diffusion_strategy.server.engine = SimpleNamespace(
        default_sampling_params_list=[object()],
        engine=SimpleNamespace(get_stage_metadata=lambda stage_id: SimpleNamespace(stage_type="diffusion")),
    )
    mm_processor_kwargs = {"fps": 24, "sampling_rate": 32000}
    request = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=[1, 2, 3],
        image_data=["image"],
        audio_data=["audio"],
        mm_processor_kwargs=mm_processor_kwargs,
    )
    prompt, _ = diffusion_strategy.preprocess_input(request, {"task": "ref2va"}, None)
    assert prompt["multi_modal_data"] == {"image": ["image"], "audio": ["audio"]}
    assert prompt["mm_processor_kwargs"] == mm_processor_kwargs


def test_pixel_output_is_always_uint8(diffusion_strategy):
    pixels = torch.tensor([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])
    output = diffusion_strategy.process_output(_pixels_output(pixels), None, {})
    assert output.diffusion_output.dtype == torch.uint8
    assert output.diffusion_output.flatten().tolist() == [0, 0, 64, 128, 255, 255]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pixel_quantization_does_not_mutate_input(diffusion_strategy, dtype):
    pixels = torch.tensor([-1.0, 0.25, 0.5, 2.0], dtype=dtype)
    original = pixels.clone()
    output = diffusion_strategy.process_output(_pixels_output(pixels), None, {})
    torch.testing.assert_close(pixels, original)
    assert output.diffusion_output.flatten().tolist() == [0, 64, 128, 255]


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_pixel_output_rejects_nonfinite_values(nonfinite):
    with pytest.raises(ValueError, match="nonfinite decoded pixels"):
        quantize_pixels(torch.tensor([0.0, nonfinite, 1.0]), "zero_one", context="adapter")


def test_pixel_quantization_preserves_float_audio(diffusion_strategy):
    pixels = quantize_pixels(torch.tensor([0.0, 0.5, 1.0]), "zero_one", context="adapter")
    audio = torch.tensor([[0.125, -0.25, 0.5]], dtype=torch.float16)
    output = diffusion_strategy.process_output(
        _request_output(
            {
                "image_preview": MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels.reshape(3, 1, 1)),
                "audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=48000), audio),
            },
            audio="audio",
        ),
        None,
        {},
    )
    torch.testing.assert_close(output.extra_fields["audio"], audio)
    assert output.extra_fields["audio_sample_rate"] == 48000


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_latent_output_preserves_native_dtype_and_axes(diffusion_strategy, dtype):
    latents = torch.tensor([[-1.0, 0.5, 2.0]], dtype=dtype)
    output = diffusion_strategy.process_output(
        _request_output(
            {
                "image_latent": MediaArtifact(MediaSpec("image", "latent", "LC"), latents),
            },
            primary="image_latent",
        ),
        None,
        {"output_type": "latent"},
    )
    torch.testing.assert_close(output.diffusion_output, latents)
    assert output.diffusion_output.data_ptr() == latents.data_ptr()


@pytest.mark.parametrize("raw", [torch.zeros(1, 3, 2, 2), (torch.zeros(3, 2, 2), torch.zeros(2, 8))])
def test_legacy_output_is_not_interpreted_from_shape(diffusion_strategy, raw):
    with pytest.raises(ValueError, match="named media_artifacts declaration required"):
        diffusion_strategy.process_output(SimpleNamespace(images=[raw], multimodal_output=None), None, {})


@pytest.mark.parametrize(
    "sampling_params,expected_dtype",
    [
        ({}, torch.uint8),
        ({"output_type": "latent"}, torch.float32),
        ({"extra_args": {"output_type": "latent"}}, torch.float32),
    ],
)
def test_empty_output_uses_modality_dtype(diffusion_strategy, sampling_params, expected_dtype):
    output = diffusion_strategy.process_output(None, None, sampling_params)
    assert output.diffusion_output.dtype == expected_dtype
    assert output.diffusion_output.numel() == 0
