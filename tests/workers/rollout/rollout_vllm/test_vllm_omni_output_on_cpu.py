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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


@pytest.fixture
def diffusion_strategy():
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 0
    return DiffusionStrategy(server)


def _request_output(diffusion_output, multimodal_output=None):
    return SimpleNamespace(
        images=[diffusion_output],
        multimodal_output=multimodal_output or {},
        trajectory_latents=None,
        trajectory_log_probs=None,
        trajectory_timesteps=None,
    )


def test_diffusion_prompt_preserves_multimodal_processor_kwargs(diffusion_strategy):
    diffusion_strategy.server.engine = SimpleNamespace(default_sampling_params_list=[object()])
    multi_modal_data = {"image": ["image"], "audio": ["audio"]}
    mm_processor_kwargs = {"fps": 24, "sampling_rate": 32000}

    prompt, _ = diffusion_strategy.preprocess_input(
        [1, 2, 3],
        {"task": "ref2va"},
        multi_modal_data,
        None,
        None,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    assert prompt["multi_modal_data"] == multi_modal_data
    assert prompt["mm_processor_kwargs"] == mm_processor_kwargs


def test_pixel_output_is_always_uint8(diffusion_strategy):
    pixels = torch.tensor([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])

    output = diffusion_strategy.process_output(_request_output(pixels), params=None, sampling_params={})

    assert output.diffusion_output.dtype == torch.uint8
    assert output.diffusion_output.tolist() == [0, 0, 64, 128, 255, 255]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pixel_quantization_does_not_mutate_input(diffusion_strategy, dtype):
    pixels = torch.tensor([-1.0, 0.25, 0.5, 2.0], dtype=dtype)
    original = pixels.clone()

    output = diffusion_strategy.process_output(_request_output(pixels), params=None, sampling_params={})

    torch.testing.assert_close(pixels, original)
    assert output.diffusion_output.tolist() == [0, 64, 128, 255]


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_pixel_output_rejects_nonfinite_values(diffusion_strategy, nonfinite):
    pixels = torch.tensor([0.0, nonfinite, 1.0])

    with pytest.raises(ValueError, match="Pixel rollout output must contain only finite values"):
        diffusion_strategy.process_output(_request_output(pixels), params=None, sampling_params={})


def test_pixel_quantization_preserves_float_audio(diffusion_strategy):
    pixels = torch.tensor([0.0, 0.5, 1.0])
    audio = torch.tensor([[0.125, -0.25, 0.5]], dtype=torch.float32)

    output = diffusion_strategy.process_output(
        _request_output(
            pixels,
            {"metadata": {"rl": {"audio": audio, "audio_sample_rate": 48_000}}},
        ),
        params=None,
        sampling_params={},
    )

    assert output.diffusion_output.dtype == torch.uint8
    assert output.extra_fields["audio"].dtype == torch.float32
    torch.testing.assert_close(output.extra_fields["audio"], audio[0])
    assert output.extra_fields["audio_sample_rate"] == 48_000


@pytest.mark.parametrize(
    "latents",
    [
        torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float16),
        np.array([-1.0, 0.5, 2.0], dtype=np.float16),
    ],
)
def test_latent_output_remains_float(diffusion_strategy, latents):
    output = diffusion_strategy.process_output(
        _request_output(latents), params=None, sampling_params={"output_type": "latent"}
    )

    assert output.diffusion_output.dtype == torch.float32
    torch.testing.assert_close(output.diffusion_output, torch.as_tensor(latents).float())


@pytest.mark.parametrize(
    ("sampling_params", "expected_dtype"),
    [
        ({}, torch.uint8),
        ({"output_type": "latent"}, torch.float32),
        ({"extra_args": {"output_type": "latent"}}, torch.float32),
    ],
)
def test_empty_output_uses_modality_dtype(diffusion_strategy, sampling_params, expected_dtype):
    output = diffusion_strategy.process_output(None, params=None, sampling_params=sampling_params)

    assert output.diffusion_output.dtype == expected_dtype
    assert output.diffusion_output.numel() == 0
