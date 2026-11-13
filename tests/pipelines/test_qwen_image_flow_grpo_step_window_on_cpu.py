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

"""CPU tests for the Qwen-Image FlowGRPO step-execution SDE window.

The shipped scripts set step_execution=true, so the window is drawn in
prepare_encode rather than forward. These tests call that method with the
encode and latent stages stubbed out.
"""

from types import SimpleNamespace

import torch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import (
    QwenImagePipelineWithLogProb,
)


class _Scheduler:
    def set_begin_index(self, index: int) -> None:
        self.begin = index


def _pipeline() -> QwenImagePipelineWithLogProb:
    pipeline = object.__new__(QwenImagePipelineWithLogProb)
    pipeline.device = "cpu"
    pipeline.default_sample_size = 8
    pipeline.vae_scale_factor = 8
    pipeline.tokenizer_max_length = 16
    pipeline.transformer = SimpleNamespace(in_channels=16, guidance_embeds=False)
    pipeline.scheduler = _Scheduler()
    pipeline._extract_prompt_ids = lambda prompts: (torch.tensor([[1, 2, 3]]), torch.ones(1, 3), None, None)
    pipeline.check_cfg_parallel_validity = lambda *args, **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(1, 4, 8), torch.ones(1, 4))
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 4, 2)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.arange(6), 6)
    return pipeline


def _state(seed: int, extra: dict) -> tuple[StepRequestState, torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    sampling = OmniDiffusionSamplingParams(
        height=64,
        width=64,
        num_inference_steps=6,
        true_cfg_scale=1.0,
        max_sequence_length=16,
        generator=generator,
        seed=seed,
        extra_args=extra,
    )
    return StepRequestState(request_id=f"r{seed}", sampling=sampling, prompt="x"), generator


def test_prepare_encode_keeps_a_constant_window_when_the_default_seed_is_forwarded() -> None:
    pipeline = _pipeline()
    windows = []
    for step in (1, 7):
        extra = {"sde_window_seed": 0, "global_steps": step, "sde_window_size": 2, "sde_window_range": [0, 5]}
        state, generator = _state(step, extra)
        before = generator.get_state().clone()
        pipeline.prepare_encode(state)
        windows.append(state.sde_window)
        assert state.sampling.generator is generator
        assert torch.equal(generator.get_state(), before)

    expected = int(torch.randint(0, 4, (1,), generator=torch.Generator().manual_seed(0)).item())
    assert windows == [(expected, expected + 2), (expected, expected + 2)]


def test_prepare_encode_without_a_shared_seed_still_uses_the_request_generator() -> None:
    extra = {"sde_window_size": 2, "sde_window_range": [0, 5]}
    pipeline = _pipeline()
    state, generator = _state(3, extra)
    twin = torch.Generator().manual_seed(3)
    pipeline.prepare_encode(state)
    expected = int(torch.randint(0, 4, (1,), generator=twin).item())
    assert state.sde_window == (expected, expected + 2)
    assert torch.equal(generator.get_state(), twin.get_state())
