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

"""CPU unit tests for the Boogu-Image convention-bridge helpers."""

import numpy as np
import torch

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    boogu_timestep_from_scheduler,
    configure_boogu_sde_timesteps,
)


class _FakeNativeScheduler:
    def set_timesteps(self, **kwargs):
        self.call = kwargs
        # Shifted Boogu t values; the bridge must not try to reproduce how
        # these were derived from the checkpoint config.
        self.timesteps = torch.tensor([0.0, 0.2, 0.75], device=kwargs["device"])


class _FakeSDEScheduler:
    def register_to_config(self, **kwargs):
        self.config_call = kwargs

    def set_timesteps(self, **kwargs):
        self.call = kwargs


def test_native_schedule_is_bridged_without_second_shift():
    native = _FakeNativeScheduler()
    sde = _FakeSDEScheduler()

    configure_boogu_sde_timesteps(
        sde,
        native_scheduler=native,
        num_inference_steps=3,
        num_tokens=16384,
        device="cpu",
    )

    assert native.call == {"num_inference_steps": 3, "device": "cpu", "num_tokens": 16384}
    assert sde.config_call == {"use_dynamic_shifting": False, "shift": 1.0}
    assert sde.call["num_inference_steps"] == 3
    assert sde.call["device"] == "cpu"
    np.testing.assert_allclose(sde.call["sigmas"], [1.0, 0.8, 0.25], atol=1e-6)


def test_timestep_mapping_inverts_sigma():
    # Scheduler-native timesteps are sigma * N (descending); Boogu t = 1 - sigma.
    n = 1000
    timesteps = torch.tensor([1000.0, 750.0, 500.0, 250.0, 0.0])
    boogu_t = boogu_timestep_from_scheduler(timesteps, n)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.allclose(boogu_t, expected, atol=1e-6)
