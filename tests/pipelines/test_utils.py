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
"""CPU tests for ``verl_omni.pipelines.utils``.

Necessity: DPO training reuses shared noise/timesteps between policy and ref
forwards. These tests verify the ref path can inject precomputed tensors and
that partial inputs are rejected before silent misalignment.
"""

from dataclasses import dataclass

import pytest
import torch

from verl_omni.pipelines.utils import prepare_noisy_latents


@dataclass
class _MockScheduler:
    timesteps: torch.Tensor
    sigmas: torch.Tensor

    def scale_noise(self, latents, timesteps, noise):
        del timesteps
        return latents + noise


class TestPrepareNoisyLatents:
    def test_samples_noise_when_not_provided(self):
        latents = torch.randn(2, 4, 8, 8)
        scheduler = _MockScheduler(
            timesteps=torch.tensor([100.0, 50.0, 10.0]),
            sigmas=torch.tensor([1.0, 0.5, 0.1]),
        )
        noisy, noise, timesteps = prepare_noisy_latents(latents, scheduler)
        assert noisy.shape == latents.shape
        assert noise.shape == latents.shape
        assert timesteps.shape == (2,)

    def test_reuses_provided_noise_and_timesteps(self):
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([50.0, 10.0])
        scheduler = _MockScheduler(timesteps=torch.tensor([100.0, 50.0, 10.0]), sigmas=torch.tensor([1.0, 0.5, 0.1]))
        noisy, out_noise, out_timesteps = prepare_noisy_latents(latents, scheduler, noise=noise, timesteps=timesteps)
        torch.testing.assert_close(out_noise, noise)
        torch.testing.assert_close(out_timesteps, timesteps)
        torch.testing.assert_close(noisy, latents + noise)

    def test_rejects_partial_noise_or_timesteps(self):
        latents = torch.randn(1, 4, 8, 8)
        scheduler = _MockScheduler(timesteps=torch.tensor([1.0]), sigmas=torch.tensor([0.5]))
        with pytest.raises(KeyError, match="together"):
            prepare_noisy_latents(latents, scheduler, noise=torch.randn_like(latents), timesteps=None)
