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
"""SD3 CRD velocity targets and reference-only CFG."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.sd3_crd.diffusers_training_adapter import StableDiffusion3CRD
from verl_omni.pipelines.sd3_crd.vllm_omni_rollout_adapter import StableDiffusion3CRDPipeline


def test_registered_crd_rollout_supports_request_batching():
    pipeline = VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "crd")
    assert pipeline is StableDiffusion3CRDPipeline
    assert pipeline.supports_request_batch is True
    assert pipeline.always_encode_negative_prompt is True


def test_anchor_cfg_does_not_change_policy_forward():
    config = SimpleNamespace(pipeline=SimpleNamespace(reference_guidance_scale=4.5))
    module = MagicMock(side_effect=lambda **kw: (kw["prediction"],))
    positive, negative = {"prediction": torch.tensor([2.0])}, {"prediction": torch.tensor([1.0])}
    torch.testing.assert_close(StableDiffusion3CRD.forward(module, config, positive, negative), torch.tensor([2.0]))
    module.assert_called_once_with(**positive)
    module.reset_mock()
    torch.testing.assert_close(
        StableDiffusion3CRD.forward_reference(module, config, positive, negative), torch.tensor([5.5])
    )
    assert module.call_args_list == [call(**positive), call(**negative)]
    assert StableDiffusion3CRD.old_policy_adapter == "moving_reference"


def test_training_target_uses_exact_noise_and_stops_gradient():
    clean = torch.randn(2, 2, 3, requires_grad=True)
    noise = torch.randn(2, 1, 2, 3, requires_grad=True)
    batch = TensorDict(
        {"latents_clean": clean, "forward_noise": noise, "train_timesteps": torch.tensor([[0.5], [999.5]])}, [2]
    )
    scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    latents, times, context = StableDiffusion3CRD.prepare_training_inputs(batch, 0, scheduler)
    torch.testing.assert_close(context["velocity_target"], noise[:, 0] - clean)
    assert not context["velocity_target"].requires_grad
    torch.testing.assert_close(times, torch.tensor([0.5, 999.5]))
    t = times[:, None, None] / 1000
    torch.testing.assert_close(latents, (1 - t) * clean + t * noise[:, 0])


@pytest.mark.parametrize("sigmas", [[1.0, 0.5], [1.0, 0.5, 0.0], [0.0, 0.0]])
def test_rollout_skips_terminal_zero_steps(sigmas):
    scheduler = FlowMatchSDEDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(sigmas=sigmas)
    pipeline = object.__new__(StableDiffusion3CRDPipeline)
    predict = MagicMock(side_effect=lambda _cfg, _scale, inputs, *_: torch.ones_like(inputs["hidden_states"]))
    for name, value in {
        "scheduler": scheduler,
        "device": torch.device("cpu"),
        "_interrupt": False,
        "od_config": SimpleNamespace(dtype=torch.float32),
        "predict_noise_maybe_with_cfg": predict,
    }.items():
        object.__setattr__(pipeline, name, value)
    start = torch.ones(2, 1, 2, 2)
    latents, trajectory, log_probs, timesteps = pipeline.diffuse(
        None,
        None,
        None,
        None,
        start.clone(),
        scheduler.timesteps,
        False,
        1.0,
        0.0,
        (0, len(sigmas)),
        "sde",
        None,
        False,
    )
    torch.testing.assert_close(latents, start - sigmas[0])
    assert torch.isfinite(latents).all()
    assert predict.call_count == sum(sigma > 0 for sigma in sigmas)
    assert trajectory is None and log_probs is None
    torch.testing.assert_close(timesteps, scheduler.timesteps.expand(2, -1))
