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
"""The Qwen-Image DiffusionART rollout must be the FlowGRPO rollout forced onto a full-window, noise-free ODE.

Necessity: DiffusionART regresses velocities at the states the rollout actually visited; a stochastic step or a
partial SDE window would make the cached states inconsistent with the recomputed ``v_old``. Output construction
(named artifacts, trajectory, prompt embeddings) is inherited unchanged from the FlowGRPO adapter.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_omni")

from verl_omni.pipelines.model_base import VllmOmniPipelineBase  # noqa: E402
from verl_omni.pipelines.qwen_image_diffusion_art import QwenImageDiffusionARTPipeline  # noqa: E402
from verl_omni.pipelines.qwen_image_flow_grpo import QwenImagePipelineWithLogProb  # noqa: E402


def test_registered_and_inherits_output_construction():
    assert VllmOmniPipelineBase.get_class("QwenImagePipeline", "diffusion_art") is QwenImageDiffusionARTPipeline
    assert issubclass(QwenImageDiffusionARTPipeline, QwenImagePipelineWithLogProb)
    for name in ("forward", "post_decode", "diffusion_io_spec"):
        assert name not in QwenImageDiffusionARTPipeline.__dict__, name
    assert QwenImageDiffusionARTPipeline.supports_request_batch == QwenImagePipelineWithLogProb.supports_request_batch


def test_diffuse_forces_the_full_capture_window_and_rejects_non_ode_requests(monkeypatch):
    captured = {}

    def fake_diffuse(self, *args):
        captured["args"] = args
        return "sentinel"

    monkeypatch.setattr(QwenImagePipelineWithLogProb, "diffuse", fake_diffuse)
    pipe = object.__new__(QwenImageDiffusionARTPipeline)
    timesteps = torch.tensor([1000.0, 750.0, 500.0, 250.0])
    common = ("pe", "pm", "npe", "npm", "latents", "shapes", "txt", "ntxt", timesteps, True, None, 4.0)
    result = pipe.diffuse(*common, 0.0, (1, 3), "cps", "gen", False)  # the caller's window is replaced
    assert result == "sentinel"
    args = captured["args"]
    assert args[:12] == common
    noise_level, sde_window, sde_type, generator, logprobs = args[12:]
    assert noise_level == 0.0 and sde_window == (0, len(timesteps)) and sde_type == "cps"
    assert generator == "gen" and logprobs is False
    for noise_level, logprobs in ((0.7, False), (0.0, True)):
        with pytest.raises(ValueError, match="deterministic ODE"):
            pipe.diffuse(*common, noise_level, (0, 4), "cps", "gen", logprobs)


def test_step_execution_path_is_forced_onto_the_same_ode(monkeypatch):
    monkeypatch.setattr(QwenImagePipelineWithLogProb, "prepare_encode", lambda self, state, **kw: state)
    pipe = object.__new__(QwenImageDiffusionARTPipeline)
    state = SimpleNamespace(noise_level=0.0, logprobs=False, sde_window=(1, 2), timesteps=[1.0, 0.5, 0.25])
    out = pipe.prepare_encode(state)
    assert out.noise_level == 0.0 and out.logprobs is False and out.sde_window == (0, 3)
    with pytest.raises(ValueError, match="deterministic ODE"):
        pipe.prepare_encode(SimpleNamespace(noise_level=0.8, logprobs=False, sde_window=(1, 2), timesteps=[1.0]))
