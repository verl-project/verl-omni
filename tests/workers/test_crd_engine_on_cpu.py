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
"""CPU coverage of the real NFT engine dispatch into the SD3 CRD adapter/loss."""

from functools import partial
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss


class TinyPolicy(torch.nn.Module):
    """Use scalar policy states to expose reference selection and CFG errors."""

    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.tensor(value))
                for name, value in [("default", 0.2), ("old", 9.0), ("moving_reference", 0.4)]
            }
        )
        self.enabled = True
        self.calls = []
        self.set_adapter("default")

    def set_adapter(self, name):
        self.active = name
        for key, param in self.lora_A.items():
            param.requires_grad_(key == name)

    def disable_adapters(self):
        self.enabled = False

    def enable_adapters(self):
        self.enabled = True

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        self.calls.append((self.active if self.enabled else "reference", torch.is_grad_enabled()))
        weight = self.lora_A[self.active] if self.enabled else 0.1
        return (hidden_states * weight + encoder_hidden_states[:, :1, :1],)


def test_engine_uses_moving_reference_and_only_updates_current_policy():
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.module = TinyPolicy()
    engine.model_config = SimpleNamespace(
        architecture="StableDiffusion3Pipeline",
        algorithm="crd",
        external_lib=None,
        pipeline=SimpleNamespace(guidance_scale=1.0, reference_guidance_scale=4.5),
    )
    engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    engine.use_ulysses_sp = False
    engine.get_data_parallel_group = lambda: None
    batch = TensorDict(
        {
            "latents_clean": torch.tensor([[[0.2, 0.5]], [[0.8, 0.4]]]),
            "forward_noise": torch.tensor([[[[0.7, 0.1]]], [[[0.2, 0.9]]]]),
            "train_timesteps": torch.tensor([[300], [700]]),
            "prompt_embeds": torch.ones(2, 1, 1),
            "prompt_embeds_mask": torch.ones(2, 1),
            "negative_prompt_embeds": torch.zeros(2, 1, 1),
            "negative_prompt_embeds_mask": torch.ones(2, 1),
            "pooled_prompt_embeds": torch.ones(2, 1),
            "negative_pooled_prompt_embeds": torch.zeros(2, 1),
            "advantages": torch.tensor([[-1.0], [1.0]]),
            "raw_rewards": torch.tensor([0.2, 0.8]),
            "group_index": torch.zeros(2, dtype=torch.long),
            "group_size": torch.full((2,), 2),
        },
        [2],
    )
    actor = DiffusionActorConfig(
        strategy="fsdp2", rollout_n=2, diffusion_loss=DiffusionLossConfig(loss_mode="crd", ref_kl_coef=0.1)
    )

    def loss_function(model_output, data, dp_group):
        result = get_diffusion_loss_fn("crd")(config=actor, model_output=model_output, data=data)
        return result.loss, result.metrics

    optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.01)
    before = {k: v.detach().clone() for k, v in engine.module.lora_A.items()}
    loss, output = engine.forward_step(batch, loss_function, forward_only=False, step=0)
    loss.backward()
    optimizer.step()
    assert engine.module.calls == [
        ("moving_reference", False),
        ("default", True),
        ("reference", False),
        ("reference", False),
    ]
    assert torch.isfinite(loss)
    assert "actor/crd_loss" in output["metrics"]
    assert not torch.equal(engine.module.lora_A["default"], before["default"])
    for name in ("old", "moving_reference"):
        torch.testing.assert_close(engine.module.lora_A[name], before[name])
        assert engine.module.lora_A[name].grad is None


def test_real_sd3_peft_gradients_match_with_gradient_checkpointing():
    """Catch adapter/gradient-checkpoint switching bugs that a scalar fake model cannot expose."""
    from diffusers import SD3Transformer2DModel
    from peft import LoraConfig
    from verl.utils import tensordict_utils as tu

    from tests.workers.crd_fsdp_checkpoint import make_batch, make_reference_engine

    results = []
    for checkpointing in (False, True):
        torch.manual_seed(42)
        module = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            caption_projection_dim=16,
            pooled_projection_dim=16,
            pos_embed_max_size=8,
        )
        for name in ("default", "old", "moving_reference"):
            module.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_v"]), adapter_name=name)
        module.set_adapter("default")
        module.train()
        if checkpointing:
            module.enable_gradient_checkpointing()
        engine = object.__new__(NFTDiffusersFSDPEngine)
        engine.module = module
        engine.model_config = SimpleNamespace(
            architecture="StableDiffusion3Pipeline",
            algorithm="crd",
            external_lib=None,
            transformer_config=dict(module.config),
            pipeline=SimpleNamespace(guidance_scale=1.0, reference_guidance_scale=4.5),
        )
        engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        engine.use_ulysses_sp = False
        engine.get_data_parallel_group = lambda: None
        actor = DiffusionActorConfig(
            strategy="fsdp2", rollout_n=2, diffusion_loss=DiffusionLossConfig(loss_mode="crd", ref_kl_coef=0.1)
        )
        batch = make_batch(engine.model_config, "cpu")
        tu.assign_non_tensor(batch, gradient_accumulation_steps=1, sp_size=1)
        loss, _ = engine.forward_step(batch, partial(diffusion_loss, config=actor), False, 0)
        loss.backward()
        gradients = {name: value.grad.clone() for name, value in module.named_parameters() if value.grad is not None}
        assert gradients and all(".default." in name for name in gradients)
        assert all(torch.isfinite(grad).all() for grad in gradients.values())
        assert any(grad.abs().sum() > 0 for grad in gradients.values())
        results.append((loss.detach(), gradients))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    from verl.workers.config import FSDPOptimizerConfig

    engine.model_config.attn_backend = "native"
    engine.optimizer_config = FSDPOptimizerConfig(lr=1e-4, total_training_steps=4)
    reference = make_reference_engine(engine, "cpu")
    engine.module.zero_grad(set_to_none=True)
    rank_batches = [make_batch(engine.model_config, "cpu", rank=rank) for rank in range(2)]
    assert not torch.equal(rank_batches[0]["latents_clean"], rank_batches[1]["latents_clean"])
    assert rank_batches[0]["group_index"].tolist() == [0, 0]
    assert rank_batches[1]["group_index"].tolist() == [1, 1]
    global_batch = torch.cat(rank_batches)
    tu.assign_non_tensor(global_batch, gradient_accumulation_steps=1, sp_size=1)
    loss_function = partial(diffusion_loss, config=actor)
    global_loss, _ = reference.forward_step(global_batch, loss_function, False, 0)
    global_loss.backward()
    for batch in rank_batches:
        tu.assign_non_tensor(batch, gradient_accumulation_steps=2, sp_size=1)
        local_loss, _ = engine.forward_step(batch, loss_function, False, 0)
        local_loss.backward()
    gradients = {
        name: parameter.grad for name, parameter in engine.module.named_parameters() if parameter.grad is not None
    }
    expected = {
        name: parameter.grad for name, parameter in reference.module.named_parameters() if parameter.grad is not None
    }
    torch.testing.assert_close(gradients, expected, rtol=1e-4, atol=1e-6)
