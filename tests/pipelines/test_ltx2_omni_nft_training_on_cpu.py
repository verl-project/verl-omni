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
"""CPU coverage for LTX OmniNFT training through the shared NFT engine."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT, _gradient_checkpointing_with_cast
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp import diffusers_impl
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


@pytest.mark.parametrize("frame_rate", [24.0, 23.976])
def test_ltx_recipe_config_composes_and_preserves_fps_and_actor_guidance(tmp_path, frame_rate):
    config_dir = Path(__file__).resolve().parents[2] / "examples/omninft_trainer/ltx2"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="ltx2_omninft",
            overrides=[
                f"actor_rollout_ref.rollout.pipeline.frame_rate={frame_rate}",
                f"actor_rollout_ref.model.path={tmp_path}",
                "+actor_rollout_ref.model.architecture=LTX2Pipeline",
                "++actor_rollout_ref.model.load_tokenizer=False",
            ],
        )

    assert cfg.reward.reward_functions.desync.fps == frame_rate
    assert cfg.reward.reward_functions.video_align.fps == frame_rate
    assert cfg.actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate == frame_rate
    loss_config = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.diffusion_loss)
    assert loss_config.loss_mode == "omni_nft"
    model_config = DiffusionStrategy(SimpleNamespace()).init_model_config(cfg.actor_rollout_ref.model)
    assert LTX23OmniNFT._guidance_scales(model_config) == (1.0, 1.0)


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
        self.calls = []

    @contextmanager
    def select(self, name):
        previous = self.active
        self.active = name
        try:
            yield
        finally:
            self.active = previous

    def forward(self, latents):
        self.calls.append((self.active, torch.is_grad_enabled()))
        prediction = latents * self.weights[self.active]
        return prediction[:, :3], prediction[:, 3:]


def _register_adapter(monkeypatch):
    class Adapter(LTX23OmniNFT):
        @classmethod
        def prepare_model_inputs(cls, module, model_config, latents, *args, **kwargs):
            return {"latents": latents}, None

        @classmethod
        def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
            return module(**model_inputs)

    algorithm = "omni_nft"
    monkeypatch.setitem(DiffusionModelBase._registry, ("_NFTHooksTest", algorithm), Adapter)
    return SimpleNamespace(
        architecture="_NFTHooksTest",
        algorithm=algorithm,
        external_lib=None,
        enable_gradient_checkpointing=True,
        lora_rank=0,
    )


def _engine(monkeypatch, num_train_timesteps=1000):
    model_config = _register_adapter(monkeypatch)
    engine_config = SimpleNamespace(
        strategy="fsdp2",
        ulysses_sequence_parallel_size=1,
        mixed_precision=None,
        full_determinism=False,
        param_offload=False,
        optimizer_offload=False,
        forward_only=False,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(diffusers_impl.DiffusersFSDPEngine, "_init_device_mesh", lambda self: None)
    engine = diffusers_impl.NFTDiffusersFSDPEngine(model_config, engine_config, None, None)
    engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=num_train_timesteps))
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.module = _Policies()
    engine.use_adapter = lambda name: engine.module.select("old" if name == "old" else "current")
    engine.disable_adapter = lambda: engine.module.select("reference")
    engine._set_adapter = Mock()
    engine.get_data_parallel_group = lambda: None
    return engine


def _batch(per_step_noise=False, scale=1000):
    batch = TensorDict(
        {
            "train_timesteps": torch.tensor([[0.5, 0.25], [0.5, 0.75]]) * scale,
            "prompt_embeds": torch.zeros(2, 3, 4),
            "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.bool),
            "reward_prob": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "advantages": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "group_index": torch.zeros(2, dtype=torch.long),
            "group_size": torch.full((2,), 2, dtype=torch.long),
        },
        batch_size=2,
    )
    branches = (("video_", 3, 2.0, -1.0), ("audio_", 2, 3.0, -2.0))
    for prefix, length, clean_value, noise_value in branches:
        clean = torch.full((2, length, 4), clean_value)
        noise = torch.full_like(clean, noise_value)
        batch[f"{prefix}latents_clean"] = clean
        batch[f"{prefix}forward_noise"] = (
            torch.stack((torch.full_like(noise, 99.0), noise), dim=1) if per_step_noise else noise
        )
    tu.assign_non_tensor(batch, gradient_accumulation_steps=3, sp_size=1)
    return batch


@pytest.mark.parametrize(
    "strategy,sp_size,message",
    [("fsdp", 1, "only actor.strategy=fsdp2"), ("fsdp2", 2, "does not implement Ulysses")],
)
def test_omni_nft_rejects_unsupported_engine_config_before_init(monkeypatch, strategy, sp_size, message):
    model_config = _register_adapter(monkeypatch)
    get_rank = Mock()
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    config = SimpleNamespace(strategy=strategy, ulysses_sequence_parallel_size=sp_size)
    with pytest.raises(NotImplementedError, match=message):
        diffusers_impl.NFTDiffusersFSDPEngine(model_config, config, None, None)
    get_rank.assert_not_called()


@pytest.mark.parametrize("scale", [1000, 2000])
@pytest.mark.parametrize("per_step_noise", [False, True])
def test_ltx_training_noises_each_branch_with_scheduler_time(monkeypatch, scale, per_step_noise):
    engine = _engine(monkeypatch, scale)
    batch = _batch(per_step_noise, scale)
    positive, negative, context = engine.prepare_model_inputs(batch, step=1)
    clean, noised, times = (
        tuple(context[modality][name] for modality in ("video", "audio")) for name in ("x0", "xt", "t_expanded")
    )
    time = torch.tensor([0.25, 0.75]).view(2, 1, 1)
    expected_video = (1 - time) * 2.0 + time * -1.0
    torch.testing.assert_close(noised[0], expected_video.expand(2, 3, 4))
    torch.testing.assert_close(times[0], time)
    assert tu.get_non_tensor_data(batch, "num_train_timesteps", default=None) == scale
    assert negative is None
    expected_audio = ((1 - time) * 3.0 + time * -2.0).expand(2, 2, 4)
    torch.testing.assert_close(noised[1], expected_audio)
    torch.testing.assert_close(positive["latents"], torch.cat(noised, dim=1))
    assert clean[0] is batch["video_latents_clean"] and clean[1] is batch["audio_latents_clean"]


def test_ltx_forward_keeps_policy_gradients_and_loss_fields(monkeypatch):
    engine = _engine(monkeypatch)
    batch = _batch()
    prefixes = ("video_", "audio_")

    def loss_function(*, model_output, data, dp_group):
        torch.testing.assert_close(data["reward_prob"], batch["reward_prob"][:, 1])
        assert tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None) == 3
        assert tu.get_non_tensor_data(data, "sp_size", default=None) == 1
        for prefix in prefixes:
            assert not model_output[f"{prefix}old_prediction"].requires_grad
            assert not model_output[f"{prefix}ref_forward_prediction"].requires_grad
            assert model_output[f"{prefix}forward_prediction"].requires_grad
            torch.testing.assert_close(model_output[f"{prefix}old_prediction"], model_output[f"{prefix}xt"])
            torch.testing.assert_close(model_output[f"{prefix}ref_forward_prediction"], model_output[f"{prefix}xt"] * 3)
        loss = sum(model_output[f"{prefix}forward_prediction"].square().mean() for prefix in prefixes)
        return loss, {"checked": 1.0}

    loss, output = engine.forward_step(batch, loss_function, forward_only=False, step=1)
    loss.backward()
    assert engine.module.calls == [("old", False), ("current", True), ("reference", False)]
    assert engine.module.weights["old"].grad is None
    assert engine.module.weights["reference"].grad is None
    assert engine.module.weights["current"].grad.item() > 0
    assert output["loss"] == pytest.approx(loss.item())
    assert output["metrics"] == {"checked": 1.0}
    engine._set_adapter.assert_called_once_with("default")


def test_ltx_forward_only_uses_prediction_device(monkeypatch):
    engine = _engine(monkeypatch)
    batch = _batch()
    prepare_inputs = engine.prepare_model_inputs

    def prepare_with_separate_timestep_device(micro_batch, step):
        inputs = prepare_inputs(micro_batch, step)
        # After input preparation, metadata must not determine the output loss device.
        micro_batch["train_timesteps"] = micro_batch["train_timesteps"].to("meta")
        return inputs

    monkeypatch.setattr(engine, "prepare_model_inputs", prepare_with_separate_timestep_device)
    loss, output = engine.forward_step(batch, None, forward_only=True, step=1)
    key = "video_forward_prediction"
    assert loss.device == output["model_output"][key].device == torch.device("cpu")
    assert batch["train_timesteps"].device.type == "meta"
    assert loss.item() == 1.0 and output["metrics"] == {}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("custom_loader", [False, True])
def test_ltx_checkpoint_precedes_lora_and_fsdp(monkeypatch, enabled, custom_loader):
    engine = _engine(monkeypatch)
    engine.model_config.enable_gradient_checkpointing = enabled
    engine._is_lora = True
    module = engine.module
    events = []
    module.enable_gradient_checkpointing = Mock(side_effect=lambda **kwargs: events.append("checkpoint"))
    from contextlib import nullcontext

    from diffusers import AutoModel

    engine.engine_config.model_dtype = "fp32"
    engine.model_config.config_path = None
    engine.model_config.local_path = "unused"
    engine.model_config.trust_remote_code = False
    engine.model_config.transformer_subfolder = "transformer"
    engine.model_config.attn_backend = "native"
    engine.device_mesh = None
    module.config = SimpleNamespace()
    module.set_attention_backend = Mock()
    adapter = DiffusionModelBase.get_class(engine.model_config)
    monkeypatch.setattr(adapter, "build_module", classmethod(lambda cls, *args: module if custom_loader else None))
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: module)
    monkeypatch.setattr(diffusers_impl, "get_init_weight_context_manager", lambda **kwargs: nullcontext)
    engine._build_lora_module = lambda loaded: events.append("lora") or loaded
    monkeypatch.setattr(diffusers_impl, "_maybe_compile_repeated_blocks", lambda *args: events.append("compile"))
    engine._build_scheduler = lambda: engine.scheduler
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    engine._build_fsdp_module = lambda loaded: events.append("fsdp") or loaded
    engine._build_optimizer = lambda loaded: None
    engine._build_lr_scheduler = lambda optimizer: None

    engine._build_model_optimizer()
    expected = (["checkpoint"] if enabled else []) + ["lora"] + ["compile", "fsdp"]
    assert events == expected
    assert engine.module is module
    if enabled:
        module.enable_gradient_checkpointing.assert_called_once()
        assert callable(module.enable_gradient_checkpointing.call_args.kwargs["gradient_checkpointing_func"])
    else:
        module.enable_gradient_checkpointing.assert_not_called()


def test_checkpoint_wrapper_casts_inputs_and_preserves_gradients():
    module = torch.nn.Linear(4, 2, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        module.weight.fill_(1.0)
    inputs = torch.ones(2, 4, requires_grad=True)
    checkpoint = _gradient_checkpointing_with_cast(torch.bfloat16)
    result = checkpoint(module, inputs)
    result.float().sum().backward()
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.full((2, 2), 4.0, dtype=torch.bfloat16))
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.0))
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 2.0))


def test_ltx_adapter_dispatches_joint_training_inputs():
    from verl_omni.pipelines.model_base import DiffusionModelBase

    config = SimpleNamespace(architecture="LTX2Pipeline", algorithm="omni_nft", external_lib=None)
    adapter = DiffusionModelBase.get_class(config)
    batch = TensorDict(
        {
            "video_latents_clean": torch.ones(1, 3, 4),
            "audio_latents_clean": torch.full((1, 2, 4), 2.0),
            "video_forward_noise": torch.zeros(1, 3, 4),
            "audio_forward_noise": torch.zeros(1, 2, 4),
            "train_timesteps": torch.tensor([[500.0]]),
        },
        batch_size=[1],
    )
    latents, timesteps, context = adapter.prepare_training_inputs(
        batch, step=0, scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    )
    assert adapter is LTX23OmniNFT
    torch.testing.assert_close(latents[:, :3], torch.full((1, 3, 4), 0.5))
    torch.testing.assert_close(latents[:, 3:], torch.ones(1, 2, 4))
    torch.testing.assert_close(timesteps, torch.tensor([500.0]))
    outputs = adapter.prepare_model_outputs({"forward_prediction": (latents[:, :3], latents[:, 3:])}, context)
    assert outputs["video_x0"] is batch["video_latents_clean"]
    assert outputs["audio_x0"] is batch["audio_latents_clean"]
