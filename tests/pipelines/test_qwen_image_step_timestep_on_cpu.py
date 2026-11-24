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
"""CPU test for the step-mode timestep passed to the Qwen-Image transformer.

Necessity: in step mode (``step_execution=True``) ``denoise_step`` must feed the
transformer the same fp32 ``t / 1000`` that ``diffuse()`` and the trainer use.
A bf16 timestep changes the model output, so the trainer's recomputed log-probs
no longer match the rollout's.
"""

from types import SimpleNamespace

import torch

from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb


def test_denoise_step_passes_fp32_timestep():
    pipeline = object.__new__(QwenImagePipelineWithLogProb)
    pipeline._interrupt = False
    pipeline._attention_kwargs = None
    pipeline.transformer = SimpleNamespace(img_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16)))
    seen = {}

    def predict_noise_maybe_with_cfg(do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, *args):
        seen["positive"], seen["negative"] = positive_kwargs, negative_kwargs
        return positive_kwargs["hidden_states"]

    pipeline.predict_noise_maybe_with_cfg = predict_noise_maybe_with_cfg

    timesteps = torch.tensor([926.7904663085938, 1000.0])
    input_batch = SimpleNamespace(
        timesteps=timesteps,
        latents=torch.zeros(2, 4, 8),
        guidance=None,
        prompt_embeds=torch.zeros(2, 3, 8),
        prompt_embeds_mask=torch.ones(2, 3),
        img_shapes=[[(1, 2, 2)]] * 2,
        txt_seq_lens=[3, 3],
        do_true_cfg=True,
        true_cfg_scale=4.0,
        negative_prompt_embeds=torch.zeros(2, 3, 8),
        negative_prompt_embeds_mask=torch.ones(2, 3),
        negative_txt_seq_lens=[3, 3],
        cfg_normalize=False,
    )

    pipeline.denoise_step(input_batch)

    for branch in ("positive", "negative"):
        timestep = seen[branch]["timestep"]
        assert timestep.dtype == torch.float32
        assert torch.equal(timestep, timesteps / 1000)
