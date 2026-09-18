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
"""Check timestep selection, transfer ownership and public engine/config wiring."""

import weakref
from contextlib import nullcontext
from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import FSDPEngineConfig

import verl_omni
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.utils.config import validate_config
from verl_omni.workers.config import FSDPDiffusionActorConfig
from verl_omni.workers.engine.fsdp import diffusers_impl
from verl_omni.workers.engine_workers import ActorRolloutRefWorker

PPO_OPTIONAL = (
    "ref_log_prob",
    "ref_prev_sample_mean",
    "teacher_prev_sample_mean",
    "old_prev_sample_mean",
    "rollout_is_weights",
)


def _batch(algorithm, steps=4, noise_mode="per_step"):
    generator = torch.Generator().manual_seed(100)

    def values(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    tensors = {
        "prompt_embeds": values(4, 3, 2),
        "prompt_embeds_mask": torch.ones(4, 3, dtype=torch.int64),
        "negative_prompt_embeds": values(4, 3, 2),
        "negative_prompt_embeds_mask": torch.ones(4, 3, dtype=torch.int64),
        "unused_trajectory": values(4, steps * 2, 3, 2),
    }
    timesteps = torch.arange(steps, dtype=torch.float64).flip(0)[None].expand(4, -1)
    if algorithm == "ppo":
        tensors.update(
            all_latents=values(4, steps + 1, 3, 2),
            all_timesteps=timesteps,
            old_log_probs=values(4, steps),
            advantages=values(4, steps),
        )
        for key in PPO_OPTIONAL:
            tensors[key] = values(4, steps, 3, 2) if key.endswith("sample_mean") else values(4, steps)
    else:
        timesteps = timesteps + torch.arange(4, dtype=torch.float64)[:, None] * 100
        tensors.update(latents_clean=values(4, 3, 2), train_timesteps=timesteps, reward_prob=values(4, steps))
        if noise_mode != "generated":
            tensors["forward_noise"] = values(4, steps, 3, 2) if noise_mode == "per_step" else values(4, 3, 2)
    batch = TensorDict(tensors, batch_size=[4])
    tu.assign_non_tensor(batch, micro_batch_size_per_gpu=2, height=16, width=24, vae_scale_factor=8)
    return batch


@pytest.fixture
def transfer_spy(monkeypatch):
    """Model device-copy ownership with CPU clones, without claiming hardware coverage."""
    observed = SimpleNamespace(shared=[], steps=[])
    original_to = torch.Tensor.to

    def tensor_to(tensor, *args, **kwargs):
        if args and args[0] == "cpu":
            copied = tensor.clone()
            observed.steps.append((tuple(tensor.shape), tensor.dtype, weakref.ref(copied)))
            return copied
        return original_to(tensor, *args, **kwargs)

    def tensordict_to(data, device, **kwargs):
        assert device == "cpu"
        observed.shared.append(set(data.keys()))
        return data.clone()

    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setattr(TensorDict, "to", tensordict_to)
    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: "cpu")
    return observed


def _engine(algorithm, staging):
    cls = diffusers_impl.PPODiffusersFSDPEngine if algorithm == "ppo" else diffusers_impl.NFTDiffusersFSDPEngine
    engine = object.__new__(cls)
    train_batch = engine.train_batch

    def train_with_staging(data, loss_function):
        tu.assign_non_tensor(data, enable_timestep_staging=staging)
        return train_batch(data, loss_function)

    engine.train_batch = train_with_staging
    engine.ulysses_sequence_parallel_size = 1
    engine.ulysses_device_mesh = None
    engine.get_data_parallel_group = lambda: None
    engine.module = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        engine.module.weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64))
    engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.01)
    engine.optimizer_config = SimpleNamespace(clip_grad=1000.0)
    observed = SimpleNamespace(steps=[], prompts=[], prior_inputs=[], previous=[], backward=0)

    def forward_step(data, loss_function, forward_only, step):
        observed.prior_inputs.append(sum(ref() is not None for ref in observed.previous))
        observed.prompts.append(id(data["prompt_embeds"]))
        if algorithm == "ppo":
            latent = data["all_latents"]
            timestep = data["all_timesteps"][:, step]
            features = latent[:, step, 0] + 0.3 * latent[:, step + 1, 0]
            for key in ("old_log_probs", "advantages", *PPO_OPTIONAL):
                if key in data:
                    features = features + data[key][:, step].reshape(len(data), -1).mean(-1, keepdim=True)
            observed.previous = [weakref.ref(latent)]
        else:
            x0 = data["latents_clean"]
            noise = data.get("forward_noise", None)
            if noise is None:
                noise = torch.randn_like(x0)
            elif noise.ndim == x0.ndim + 1:
                noise = noise[:, step]
            timestep = data["train_timesteps"][:, step]
            features = x0[:, 0] + noise[:, 0] + data["reward_prob"][:, step, None]
            observed.previous = [weakref.ref(data["train_timesteps"])]
        observed.steps.append(timestep.tolist())
        features = features + data["prompt_embeds"][:, 0] - 0.2 * data["negative_prompt_embeds"][:, 0]
        prediction = engine.module(features + timestep[:, None] / 1000)
        outputs = {"prediction": prediction}
        loss = prediction.square().mean() / tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=1)
        if loss.requires_grad:

            def count_backward(gradient):
                assert all(ref() is not None for ref in observed.previous)
                observed.backward += 1
                return gradient

            loss.register_hook(count_backward)
        return loss, {"model_output": outputs, "loss": loss.detach().item(), "metrics": {}}

    engine.forward_step = forward_step
    return engine, observed


@pytest.mark.parametrize(
    "algorithm,noise_mode", [("ppo", "per_step"), ("nft", "per_step"), ("nft", "static"), ("nft", "generated")]
)
@pytest.mark.parametrize("steps", [1, 4, 12])
@pytest.mark.parametrize("retain", [False, True])
def test_staging_matches_updates_and_preserves_inputs(transfer_spy, algorithm, noise_mode, steps, retain):
    candidate, observed = _engine(algorithm, staging=True)
    reference, _ = _engine(algorithm, staging=False)
    source = _batch(algorithm, steps, noise_mode)
    frozen = source.clone()
    for _ in range(2):
        batch = source.clone()
        tu.assign_non_tensor(batch, return_model_output=retain)
        torch.manual_seed(99)
        actual = candidate.train_batch(batch, loss_function=lambda **kwargs: None)
        torch.manual_seed(99)
        expected = reference.train_batch(batch.clone(), loss_function=lambda **kwargs: None)
        assert actual["loss"] == expected["loss"]
        assert actual["metrics"] == expected["metrics"]
        assert bool(actual["model_output"]) is retain
        if retain:
            torch.testing.assert_close(
                actual["model_output"]["prediction"], expected["model_output"]["prediction"], rtol=0, atol=0
            )
        torch.testing.assert_close(candidate.module.weight, reference.module.weight, rtol=0, atol=0)
        torch.testing.assert_close(candidate.module.weight.grad, reference.module.weight.grad, rtol=0, atol=0)
        for key, value in frozen.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(batch[key], value, rtol=0, atol=0)
    assert observed.backward == 4 * steps
    expected_steps = [
        [float(step + (row * 100 if algorithm == "nft" else 0)) for row in range(start, start + 2)]
        for _ in range(2)
        for start in (0, 2)
        for step in reversed(range(steps))
    ]
    assert observed.steps == expected_steps
    assert observed.prior_inputs == [0] * (4 * steps)
    assert all(
        observed.prompts[start : start + steps] == [observed.prompts[start]] * steps
        for start in range(0, 4 * steps, steps)
    )
    staged_copies = [keys for keys in transfer_spy.shared if "unused_trajectory" not in keys]
    assert len(staged_copies) == 4
    assert all("prompt_embeds" in keys for keys in staged_copies)
    assert all("all_latents" not in keys and "train_timesteps" not in keys for keys in staged_copies)
    assert all(ref() is None for _, _, ref in transfer_spy.steps)
    assert all(shape[1] <= 2 for shape, _, _ in transfer_spy.steps)


@pytest.mark.parametrize("algorithm", ["ppo", "nft"])
def test_inference_bypasses_staging(transfer_spy, algorithm):
    candidate, observed = _engine(algorithm, staging=True)
    reference, _ = _engine(algorithm, staging=False)
    batch = _batch(algorithm)
    tu.assign_non_tensor(batch, return_model_output=False, enable_timestep_staging=True)
    actual = candidate.infer_batch(batch)
    expected = reference.infer_batch(batch.clone())
    torch.testing.assert_close(
        actual["model_output"]["prediction"], expected["model_output"]["prediction"], rtol=0, atol=0
    )
    assert observed.backward == 0
    assert transfer_spy.steps == []
    assert all("unused_trajectory" in keys for keys in transfer_spy.shared)


@pytest.mark.parametrize("algorithm", ["ppo", "nft"])
def test_staging_recreated_after_failed_forward(transfer_spy, algorithm):
    engine, observed = _engine(algorithm, staging=True)
    original_forward = engine.forward_step

    def failing_forward(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    engine.forward_step = failing_forward
    with pytest.raises(RuntimeError, match="injected forward failure"):
        engine.train_batch(_batch(algorithm), loss_function=lambda **kwargs: None)
    assert all(ref() is None for _, _, ref in transfer_spy.steps)
    engine.forward_step = original_forward
    result = engine.train_batch(_batch(algorithm), loss_function=lambda **kwargs: None)
    assert result["model_output"] == {}
    assert observed.backward == 8


def test_ppo_optional_fields_can_be_absent(transfer_spy):
    engine, observed = _engine("ppo", staging=True)
    batch = _batch("ppo")
    for key in PPO_OPTIONAL:
        del batch[key]
    assert engine.train_batch(batch, loss_function=lambda **kwargs: None)["model_output"] == {}
    assert observed.backward == 8


@pytest.mark.parametrize(
    "case", ["empty", "latent_length", "loss_length", "noise_shape", "missing_mask", "input_grad", "device"]
)
def test_invalid_staging_input_fails_before_forward(transfer_spy, case):
    algorithm = "nft" if case in {"noise_shape", "missing_mask"} else "ppo"
    engine, observed = _engine(algorithm, staging=True)
    batch = _batch(algorithm)
    if case == "empty":
        batch["all_timesteps"] = torch.empty(4, 0)
    elif case == "latent_length":
        batch["all_latents"] = batch["all_latents"][:, :3]
    elif case == "loss_length":
        batch["old_log_probs"] = batch["old_log_probs"][:, :3]
    elif case == "noise_shape":
        batch["forward_noise"] = batch["forward_noise"][:, :3]
    elif case == "missing_mask":
        del batch["prompt_embeds_mask"]
    elif case == "input_grad":
        batch["all_latents"].requires_grad_()
    else:
        batch["all_latents"] = torch.empty(4, 5, 3, 2, device="meta")
    with pytest.raises(ValueError):
        engine.train_batch(batch, loss_function=lambda **kwargs: None)
    assert observed.steps == []
    assert observed.backward == 0
    assert not transfer_spy.shared and not transfer_spy.steps


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
@pytest.mark.parametrize("enabled", [False, True])
def test_hydra_actor_forwards_timestep_staging(strategy, enabled):
    config_dir = Path(verl_omni.__file__).parent / "trainer/config/diffusion/actor"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                f"strategy={strategy}",
                "ppo_micro_batch_size_per_gpu=2",
                f"enable_timestep_staging={str(enabled).lower()}",
            ],
        )
    actor = omega_conf_to_dataclass(cfg)
    assert isinstance(actor, FSDPDiffusionActorConfig)
    assert type(actor.engine) is FSDPEngineConfig
    assert actor.engine is actor.fsdp_config
    assert actor.enable_timestep_staging is enabled
    assert not hasattr(actor.engine, "enable_timestep_staging")
    assert actor.engine.strategy == strategy


@pytest.mark.parametrize("enabled", [False, True])
def test_public_trainer_override_reaches_actor_worker(enabled):
    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    overrides = ["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"]
    if enabled:
        overrides.append("actor_rollout_ref.actor.enable_timestep_staging=true")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="diffusion_trainer", overrides=overrides)
    validate_config(cfg)
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert actor.enable_timestep_staging is enabled
    assert type(actor.engine) is FSDPEngineConfig
    ref = omega_conf_to_dataclass(cfg.actor_rollout_ref.ref)
    assert not ref.enable_timestep_staging
    assert type(ref.engine) is FSDPEngineConfig

    received = []

    def train_mini_batch(data):
        received.append(tu.get_non_tensor_data(data, "enable_timestep_staging", default=None))
        return None

    worker = SimpleNamespace(config=cfg.actor_rollout_ref, actor=SimpleNamespace(train_mini_batch=train_mini_batch))
    batch = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(batch, enable_timestep_staging=not enabled)
    unwrap(ActorRolloutRefWorker.update_actor)(worker, batch)
    assert received == [enabled]


@pytest.mark.parametrize("entrypoint", ["main_diffusion", "main_diffusion_v1"])
def test_public_config_rejects_staging_with_sequence_parallel(monkeypatch, entrypoint):
    import importlib

    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                "actor_rollout_ref.actor.enable_timestep_staging=true",
                "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2",
            ],
        )
    with pytest.raises(ValueError, match="sequence_parallel_size=1"):
        validate_config(cfg)
    module = importlib.import_module(f"verl_omni.trainer.{entrypoint}")
    monkeypatch.setattr(module, "auto_set_device", lambda config: None)
    with pytest.raises(ValueError, match="sequence_parallel_size=1"):
        module.main.__wrapped__(cfg)


def test_worker_without_staging_config_resets_batch_flag():
    received = []
    worker = SimpleNamespace(
        config=OmegaConf.create({"actor": {}}),
        actor=SimpleNamespace(
            train_mini_batch=lambda data: received.append(
                tu.get_non_tensor_data(data, "enable_timestep_staging", default=None)
            )
        ),
    )
    batch = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(batch, enable_timestep_staging=True)
    unwrap(ActorRolloutRefWorker.update_actor)(worker, batch)
    assert received == [False]


@pytest.mark.parametrize(
    "algorithm,noise_mode", [("ppo", "per_step"), ("nft", "per_step"), ("nft", "static"), ("nft", "generated")]
)
def test_real_qwen_adapter_and_scheduler_contract(transfer_spy, algorithm, noise_mode):
    """Exercise real forward_step/adapters with a tiny CPU projection, not a transformer or FSDP."""
    engines = []
    for staging in (False, True):
        engine, _ = _engine(algorithm, staging)
        del engine.forward_step  # Use the real algorithm-specific class method.
        engine.use_ulysses_sp = False
        engine.model_config = SimpleNamespace(
            architecture="QwenImagePipeline",
            algorithm="flow_grpo" if algorithm == "ppo" else "diffusion_nft",
            external_lib=None,
            pipeline=SimpleNamespace(guidance_scale=1.0, true_cfg_scale=2.0),
            algo=SimpleNamespace(noise_level=0.8, sde_type="sde"),
        )
        engine.module.config = SimpleNamespace(guidance_embeds=False)
        projection = engine.module.forward

        def project(hidden_states, encoder_hidden_states, _projection=projection, **kwargs):
            return (_projection(hidden_states) + encoder_hidden_states[:, :1] * 0.1,)

        engine.module.forward = project
        engine.scheduler = FlowMatchSDEDiscreteScheduler()
        engine.scheduler.set_timesteps(5, device="cpu")
        engine.use_adapter = lambda name: nullcontext()
        engine.disable_adapter = nullcontext
        engine._set_adapter = lambda name: None
        engines.append(engine)

    batch = _batch(algorithm, noise_mode=noise_mode)
    if algorithm == "ppo":
        batch["all_timesteps"] = engines[0].scheduler.timesteps[:4].expand(4, -1).clone()
    tu.assign_non_tensor(batch, return_model_output=True)
    seen_loss_data = []

    def loss_function(model_output, data, dp_group):
        seen_loss_data.append(data.clone())
        prediction = model_output["prev_sample_mean" if algorithm == "ppo" else "forward_prediction"]
        weight = sum(value.float().mean() for value in data.values() if isinstance(value, torch.Tensor))
        loss = prediction.square().mean() * (1 + weight.square())
        return loss / tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None), {}

    results = []
    for engine in engines:
        torch.manual_seed(99)
        results.append(engine.train_batch(batch.clone(), loss_function))
    reference, candidate = results
    torch.testing.assert_close(torch.tensor(candidate["loss"]), torch.tensor(reference["loss"]), rtol=0, atol=0)
    for key in reference["model_output"]:
        torch.testing.assert_close(candidate["model_output"][key], reference["model_output"][key], rtol=0, atol=0)
    torch.testing.assert_close(engines[1].module.weight, engines[0].module.weight, rtol=0, atol=0)
    torch.testing.assert_close(engines[1].module.weight.grad, engines[0].module.weight.grad, rtol=0, atol=0)
    assert len(seen_loss_data) == 16
    for expected, actual in zip(seen_loss_data[:8], seen_loss_data[8:], strict=True):
        assert set(expected.keys()) == set(actual.keys())
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            else:
                assert tu.get_non_tensor_data(actual, key, default=None) == tu.get_non_tensor_data(
                    expected, key, default=None
                )
