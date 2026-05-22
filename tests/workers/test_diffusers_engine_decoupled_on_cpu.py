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
"""CPU contract tests for the decoupled Diffusers FSDP engine path."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.workers.engine.base import EngineRegistry

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp.diffusers_impl import (
    DiffusersFSDPEngineRouter,
    DiffusionFSDPEngineAlgorithmRegistry,
    NFTDiffusersFSDPEngine,
    PPODiffusersFSDPEngine,
)


@DiffusionModelBase.register("FakeDecoupledPipeline", algorithm="diffusion_nft")
class FakeDecoupledModel(DiffusionModelBase):
    @classmethod
    def build_scheduler(cls, model_config):
        raise NotImplementedError

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device: str):
        raise NotImplementedError

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    ):
        raise NotImplementedError

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler,
        model_config,
        model_inputs,
        negative_model_inputs,
        scheduler_inputs,
        step: int,
    ):
        raise NotImplementedError


class FakeAdapterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.active_adapter = "default"
        self.calls = []

    def set_adapter(self, name: str):
        self.active_adapter = name

    def disable_adapters(self):
        self.active_adapter = "reference"

    def enable_adapters(self):
        self.active_adapter = "default"

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, return_dict=False):
        self.calls.append(
            {
                "adapter": self.active_adapter,
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
                "prompt_embeds": encoder_hidden_states.detach().clone(),
                "prompt_embeds_mask": encoder_attention_mask.detach().clone(),
            }
        )
        offset = {"old": 1.0, "default": 2.0, "reference": 3.0}[self.active_adapter]
        return (hidden_states * self.scale + offset,)


def _engine() -> NFTDiffusersFSDPEngine:
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.module = FakeAdapterModule()
    engine.model_config = SimpleNamespace(architecture="FakeDecoupledPipeline", algorithm="diffusion_nft")
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    return engine


def _registered_engine() -> NFTDiffusersFSDPEngine:
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.module = FakeAdapterModule()
    engine.model_config = SimpleNamespace(architecture="FakeDecoupledPipeline", algorithm="diffusion_nft")
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    return engine


def _batch() -> TensorDict:
    batch_size = 2
    steps = 2
    latent_shape = (3, 4, 4)
    return TensorDict(
        {
            "latents_clean": torch.randn(batch_size, *latent_shape),
            "train_timesteps": torch.tensor([[100, 200], [300, 400]], dtype=torch.long),
            "forward_noise": torch.randn(batch_size, steps, *latent_shape),
            "prompt_embeds": torch.randn(batch_size, 5, 8),
            "prompt_embeds_mask": torch.ones(batch_size, 5, dtype=torch.bool),
            "reward_prob": torch.full((batch_size, steps), 0.5),
        },
        batch_size=batch_size,
    )


def test_forward_decoupled_step_reuses_same_xt_for_old_default_and_reference() -> None:
    engine = _engine()
    micro_batch = _batch()

    def loss_function(model_output, data, dp_group=None):
        assert "reward_prob" in data
        assert "old_log_probs" not in data
        assert "old_log_probs" not in micro_batch
        return model_output["forward_prediction"].mean(), {}

    loss, output = engine.forward_step(
        micro_batch=micro_batch,
        loss_function=loss_function,
        forward_only=False,
        step=0,
    )

    assert loss.requires_grad
    assert [call["adapter"] for call in engine.module.calls] == ["old", "default", "reference"]
    torch.testing.assert_close(engine.module.calls[0]["hidden_states"], engine.module.calls[1]["hidden_states"])
    torch.testing.assert_close(engine.module.calls[0]["hidden_states"], engine.module.calls[2]["hidden_states"])
    torch.testing.assert_close(engine.module.calls[0]["timestep"], engine.module.calls[1]["timestep"])
    torch.testing.assert_close(engine.module.calls[0]["prompt_embeds"], engine.module.calls[2]["prompt_embeds"])
    assert set(output["model_output"]) == {
        "old_prediction",
        "forward_prediction",
        "ref_forward_prediction",
        "x0",
        "xt",
        "t_expanded",
    }


def test_forward_decoupled_step_selects_step_noise_from_forward_noise_sequence() -> None:
    engine = _engine()
    micro_batch = _batch()
    micro_batch["latents_clean"].zero_()
    micro_batch["forward_noise"].zero_()
    micro_batch["forward_noise"][:, 1].fill_(4.0)

    def loss_function(model_output, data, dp_group=None):
        return model_output["forward_prediction"].mean(), {}

    _, output = engine.forward_step(
        micro_batch=micro_batch,
        loss_function=loss_function,
        forward_only=False,
        step=1,
    )

    t = micro_batch["train_timesteps"][:, 1].float() / 1000.0
    expected_xt = t.view(-1, 1, 1, 1) * micro_batch["forward_noise"][:, 1]
    torch.testing.assert_close(output["model_output"]["xt"], expected_xt)


def test_diffusion_fsdp_engine_registry_selects_by_algorithm() -> None:
    assert DiffusionFSDPEngineAlgorithmRegistry.get_engine_cls(SimpleNamespace(algorithm="diffusion_nft")) is (
        NFTDiffusersFSDPEngine
    )
    assert DiffusionFSDPEngineAlgorithmRegistry.get_engine_cls(SimpleNamespace(algorithm="flow_grpo")) is (
        PPODiffusersFSDPEngine
    )
    assert DiffusionFSDPEngineAlgorithmRegistry.get_engine_cls(SimpleNamespace(algorithm="mix_grpo")) is (
        PPODiffusersFSDPEngine
    )
    assert EngineRegistry._engines["diffusion_model"]["fsdp"]["cuda"] is DiffusersFSDPEngineRouter


def test_diffusion_fsdp_engine_registry_requires_algorithm() -> None:
    with pytest.raises(ValueError, match="algorithm must be set"):
        DiffusionFSDPEngineAlgorithmRegistry.get_engine_cls(SimpleNamespace())


def test_diffusion_fsdp_engine_registry_rejects_unknown_algorithm() -> None:
    with pytest.raises(NotImplementedError, match="unknown_algo"):
        DiffusionFSDPEngineAlgorithmRegistry.get_engine_cls(SimpleNamespace(algorithm="unknown_algo"))


def test_nft_engine_forward_backward_batch_without_old_log_probs(monkeypatch) -> None:
    engine = _registered_engine()
    monkeypatch.setattr(
        "verl_omni.workers.engine.fsdp.diffusers_impl.prepare_micro_batches",
        lambda data, dp_group, same_micro_num_in_dp: ([data], None),
    )

    def loss_function(model_output, data, dp_group=None):
        assert "reward_prob" in data
        assert "old_log_probs" not in data
        return model_output["forward_prediction"].mean(), {}

    output = engine.forward_backward_batch(_batch(), loss_function=loss_function, forward_only=True)

    assert output["model_output"]["forward_prediction"].shape[1] == 2
    assert [call["adapter"] for call in engine.module.calls[:3]] == ["old", "default", "reference"]
