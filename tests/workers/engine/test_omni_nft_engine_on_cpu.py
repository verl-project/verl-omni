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
"""CPU regression coverage for the default NFT model-adapter contract."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp import diffusers_impl


class _Policies(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.tensor(value))
                for name, value in (("old", 1.0), ("current", 2.0), ("reference", 3.0))
            }
        )
        self.active = "current"

    @contextmanager
    def select(self, name):
        previous = self.active
        self.active = name
        try:
            yield
        finally:
            self.active = previous

    def forward(self, latents):
        return latents * self.weights[self.active]


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "dgpo"])
@pytest.mark.parametrize("forward_only", [False, True])
def test_default_adapter_keeps_nft_forward_and_loss_inputs(monkeypatch, algorithm, forward_only):
    class Adapter(DiffusionModelBase):
        @classmethod
        def prepare_model_inputs(cls, module, model_config, latents, *args, **kwargs):
            return {"latents": latents}, None

        @classmethod
        def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
            return module(**model_inputs)

    monkeypatch.setitem(DiffusionModelBase._registry, ("_NFTTest", algorithm), Adapter)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(diffusers_impl.DiffusersFSDPEngine, "_init_device_mesh", lambda self: None)
    model_config = SimpleNamespace(architecture="_NFTTest", algorithm=algorithm, external_lib=None, lora_rank=0)
    engine_config = SimpleNamespace(
        strategy="fsdp2", full_determinism=False, param_offload=False, optimizer_offload=False, forward_only=False
    )
    engine = diffusers_impl.NFTDiffusersFSDPEngine(model_config, engine_config, None, None)
    engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    engine.use_ulysses_sp = False
    engine.module = _Policies()
    engine.use_adapter = lambda name: engine.module.select("old" if name == "old" else "current")
    engine.disable_adapter = lambda: engine.module.select("reference")
    engine._set_adapter = Mock()
    engine.get_data_parallel_group = lambda: None
    batch = TensorDict(
        {
            "latents_clean": torch.full((2, 3, 4), 2.0),
            "forward_noise": torch.full((2, 3, 4), -1.0),
            "train_timesteps": torch.tensor([[500.0, 250.0], [500.0, 750.0]]),
            "prompt_embeds": torch.zeros(2, 3, 4),
            "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.bool),
            "reward_prob": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "advantages": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "group_index": torch.zeros(2, dtype=torch.long),
            "group_size": torch.full((2,), 2, dtype=torch.long),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(batch, gradient_accumulation_steps=3, sp_size=1)

    def loss_function(*, model_output, data, dp_group):
        for key in ("reward_prob", "advantages"):
            torch.testing.assert_close(data[key], batch[key][:, 1])
        for key in ("group_index", "group_size"):
            torch.testing.assert_close(data[key], batch[key])
        assert tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None) == 3
        assert not model_output["old_prediction"].requires_grad
        assert not model_output["ref_forward_prediction"].requires_grad
        torch.testing.assert_close(model_output["old_prediction"], model_output["xt"])
        torch.testing.assert_close(model_output["ref_forward_prediction"], model_output["xt"] * 3)
        return model_output["forward_prediction"].square().mean(), {}

    loss, output = engine.forward_step(batch, None if forward_only else loss_function, forward_only, step=1)
    if forward_only:
        assert loss.item() == 1.0
        assert loss.device == batch["latents_clean"].device
    else:
        loss.backward()
        assert engine.module.weights["current"].grad.item() > 0
    assert engine.module.weights["old"].grad is None
    assert engine.module.weights["reference"].grad is None
    assert output["loss"] == pytest.approx(loss.item())
    engine._set_adapter.assert_called_once_with("default")
