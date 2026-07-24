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

import os

os.environ.setdefault("VERL_OMNI_SKIP_AUTO_IMPORTS", "1")

import torch
from tensordict import TensorDict

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn


class _LossConfig:
    ce_weight = 1.0
    mse_weight = 0.5
    ignore_index = -100


class _ActorConfig:
    diffusion_loss = _LossConfig()
    use_kl_loss = False


def test_bagel_sft_loss_combines_text_and_image_terms():
    loss_fn = get_diffusion_loss_fn("bagel_sft")
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, -100], [2, 3, 4, 5]])
    image_velocity = torch.zeros(2, 1, 3, 4, requires_grad=True)
    image_velocity_target = torch.ones(2, 1, 3, 4)
    image_loss_mask = torch.ones(2, 1, 3)

    result = loss_fn(
        config=_ActorConfig(),
        model_output={"logits": logits, "image_velocity": image_velocity},
        data=TensorDict(
            {
                "labels": labels,
                "image_velocity_target": image_velocity_target,
                "image_loss_mask": image_loss_mask,
            },
            batch_size=[2],
        ),
    )

    assert result.loss.requires_grad
    assert "bagel_sft/ce_loss" in result.metrics
    assert "bagel_sft/mse_loss" in result.metrics
    result.loss.backward()
    assert logits.grad is not None
    assert image_velocity.grad is not None
