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

"""DMD computation on independent CPU models, without a production engine."""

import copy
from functools import partial

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.dmd2.program import DMD2Program, backward_simulation, re_noise, shifted_sigmas
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig
from verl_omni.workers.utils.losses import diffusion_loss


def make_program(modalities=("image",), *, device="cpu", **overrides):
    config = DiffusionDMDConfig(
        **{
            "modality_shifts": {name: float(index + 2) for index, name in enumerate(modalities)},
            "modality_weights": {name: 0.7 + index for index, name in enumerate(modalities)},
            "score_discrete_steps": 0,
            "cfg_norm": "none",
            "teacher_guidance_scale": 1.0,
            **overrides,
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        # Teacher parameters also require grad so a missing no_grad is observable.
        models = {
            role: torch.nn.ModuleDict({name: torch.nn.Linear(3, 3) for name in modalities})
            for role in ("student", "fake_score", "teacher")
        }
    models = {role: model.to(device) for role, model in models.items()}
    calls = []

    def predict(role, latents, sigmas, negative):
        calls.append((role, negative, torch.is_grad_enabled()))
        return {
            name: models[role][name](value) + (0.2 if negative else 0.4) * sigmas[name]
            for name, value in latents.items()
        }

    initial = {
        name: torch.linspace(-1, 1, 2 * (index + 2) * 3).reshape(2, index + 2, 3).to(device)
        for index, name in enumerate(modalities)
    }
    return DMD2Program(config, predict, seed=42), models, initial, calls


def loss_inputs(accumulation=1, sp_size=1):
    data = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(data, gradient_accumulation_steps=accumulation, sp_size=sp_size)
    actor = DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2"))
    return data, partial(diffusion_loss, config=actor)


@pytest.mark.parametrize("stage", ["student", "fake_score"])
@pytest.mark.parametrize("modalities", [("image",), ("image", "audio")])
@pytest.mark.parametrize("accumulation,sp_size", [(1, 1), (3, 1), (2, 4)])
def test_loss_gradient_and_update_match_independent_formula(stage, modalities, accumulation, sp_size):
    actual, models, initial, _ = make_program(modalities, teacher_guidance_scale=4.0)
    reference, expected_models, _, _ = make_program(modalities, teacher_guidance_scale=4.0)
    before = {role: copy.deepcopy(model.state_dict()) for role, model in models.items()}
    data, loss_function = loss_inputs(accumulation, sp_size)
    torch.manual_seed(107)
    actual_loss, metrics, invalid = actual.compute_loss(initial, stage, data, loss_function)
    torch.manual_seed(107)
    outputs = reference.stage_outputs(initial, stage)
    expected_loss = 0
    for name, output in outputs.items():
        generated = output["generated_x0"]
        if stage == "fake_score":
            target = output["noise"] - generated.detach()
            part = (output["noise_pred"] - target).square().mean()
        else:
            axes = tuple(range(1, generated.ndim))
            normalizer = (generated.detach() - output["teacher_x0"]).abs().mean(axes, keepdim=True).clamp_min(1e-6)
            gradient = (output["fake_x0"] - output["teacher_x0"]) / normalizer
            target = (generated - gradient).detach()
            part = 0.5 * (generated - target).square().mean()
        expected_loss = expected_loss + reference.config.modality_weights[name] * (part / accumulation * sp_size)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert not invalid and all(any(key.startswith(name + "/") for key in metrics) for name in modalities)
    assert tu.get_non_tensor_data(data, "sp_size", default=None) == sp_size
    assert tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None) == accumulation
    actual_loss.backward()
    expected_loss.backward()
    for role in models:
        for actual_param, expected_param in zip(
            models[role].parameters(), expected_models[role].parameters(), strict=True
        ):
            if role == stage:
                torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=0, atol=0)
            else:
                assert actual_param.grad is None and expected_param.grad is None
    torch.optim.SGD(models[stage].parameters(), lr=0.01).step()
    torch.optim.SGD(expected_models[stage].parameters(), lr=0.01).step()
    assert any(not torch.equal(value, before[stage][key]) for key, value in models[stage].state_dict().items())
    for role in models:
        for key, value in models[role].state_dict().items():
            torch.testing.assert_close(value, expected_models[role].state_dict()[key], rtol=0, atol=0)
            if role != stage:
                torch.testing.assert_close(value, before[role][key], rtol=0, atol=0)


@pytest.mark.parametrize("exit_step", range(4))
def test_sampler_retains_only_selected_exit_and_matches_explicit_transitions(exit_step):
    parameter = torch.tensor(0.3, requires_grad=True)
    initial = {"image": torch.tensor([[0.2, -0.5]], requires_grad=True)}
    calls = []

    def predict(latents, sigmas):
        calls.append((torch.is_grad_enabled(), latents["image"].detach().clone()))
        return {"image": parameter * latents["image"]}

    torch.manual_seed(71)
    result = backward_simulation(predict, initial, {"image": 2.0}, 4, exit_step, require_grad=True)["image"]
    torch.manual_seed(71)
    latent = initial["image"].detach().clone()
    for index in range(exit_step + 1):
        time = torch.tensor(1 - index / 4)
        sigma = 2 * time / (1 + time)
        clean = latent - sigma * parameter.detach() * latent
        if index != exit_step:
            next_time = torch.tensor(1 - (index + 1) / 4)
            next_sigma = 2 * next_time / (1 + next_time)
            latent = (1 - next_sigma) * clean + next_sigma * torch.randn_like(clean)
    torch.testing.assert_close(result.detach(), clean, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(parameter.grad, (-sigma * latent).sum(), rtol=0, atol=0)
    assert initial["image"].grad is None
    assert [enabled for enabled, _ in calls] == [False] * exit_step + [True]


@pytest.mark.parametrize("guidance,norm", [(1.0, "none"), (4.0, "none"), (4.0, "layer_norm"), (4.0, "scalar")])
def test_teacher_cfg_uses_only_detached_teacher_predictions(guidance, norm):
    program, models, initial, calls = make_program(teacher_guidance_scale=guidance, cfg_norm=norm)
    outputs = program.stage_outputs(initial, "student")
    teacher_calls = [call for call in calls if call[0] == "teacher"]
    assert teacher_calls == [("teacher", False, False)] + ([("teacher", True, False)] if guidance != 1 else [])
    assert outputs["image"]["generated_x0"].requires_grad
    assert not outputs["image"]["teacher_x0"].requires_grad
    assert not outputs["image"]["fake_x0"].requires_grad
    assert all(parameter.grad is None for parameter in models["teacher"].parameters())


def test_noise_is_fp32_and_detached_from_generated_samples():
    clean = {"image": torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)}
    noisy, noise = re_noise(clean, {"image": torch.tensor(0.25)})
    assert noisy["image"].dtype == noise["image"].dtype == torch.float32
    assert not noisy["image"].requires_grad
    torch.testing.assert_close(noisy["image"], 0.75 * clean["image"].float() + 0.25 * noise["image"])


@pytest.mark.parametrize("stage", ["student", "fake_score"])
def test_control_rng_resume_with_caller_cpu_rng_reproduces_next_gradient(stage):
    program, models, initial, _ = make_program(("image", "audio"))
    program.stage_outputs(initial, "fake_score")
    saved = program.state_dict()
    cpu_rng = torch.get_rng_state()
    data, loss_function = loss_inputs()
    loss, _, _ = program.compute_loss(initial, stage, data, loss_function)
    loss.backward()
    expected_grads = [parameter.grad.clone() for parameter in models[stage].parameters()]
    resumed, resumed_models, _, _ = make_program(("image", "audio"))
    before_load = torch.get_rng_state().clone()
    resumed.load_state_dict(saved)
    assert torch.equal(before_load, torch.get_rng_state())
    torch.set_rng_state(cpu_rng)
    actual_loss, _, _ = resumed.compute_loss(initial, stage, data, loss_function)
    actual_loss.backward()
    torch.testing.assert_close(actual_loss, loss, rtol=0, atol=0)
    for parameter, gradient in zip(resumed_models[stage].parameters(), expected_grads, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


def test_inference_uses_final_exit_without_graph_or_control_rng_change():
    program, _, initial, calls = make_program()
    initial = {name: value.bfloat16() for name, value in initial.items()}
    state = program.state_dict()
    output = program.sample(initial)
    assert len(calls) == 4 and all(not enabled for _, _, enabled in calls)
    assert output["image"].dtype == torch.float32 and not output["image"].requires_grad
    assert torch.equal(state["control_rng"], program.state_dict()["control_rng"])
    assert program.training_stages() == ("fake_score", "fake_score", "student")


@pytest.mark.parametrize("stage", ["student", "fake_score"])
def test_nonfinite_output_is_reported_to_the_update_owner(stage):
    program, models, initial, _ = make_program()
    with torch.no_grad():
        models["fake_score"]["image"].weight.fill_(float("nan"))
    data, loss_function = loss_inputs()
    _, _, invalid = program.compute_loss(initial, stage, data, loss_function)
    assert invalid


@pytest.mark.parametrize("name", ["gradient_accumulation_steps", "sp_size"])
@pytest.mark.parametrize("value", [None, 0, True, 1.5])
def test_loss_scaling_must_be_explicit_before_sampling(name, value):
    program, _, initial, calls = make_program()
    data, loss_function = loss_inputs()
    tu.assign_non_tensor(data, **{name: value})
    with pytest.raises(ValueError, match=name):
        program.compute_loss(initial, "student", data, loss_function)
    assert not calls


@pytest.mark.parametrize(
    "kwargs", [{"score_discrete_steps": 1000}, {"rollout_timestep_shift": 4}, {"score_timestep_shift": 4}]
)
def test_unsupported_sampling_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_program(**kwargs)


@pytest.mark.parametrize("name,value", [("score_sigma_min", 0.1), ("score_sigma_max", 0.9)])
def test_sigma_bounds_cannot_be_mistaken_for_base_time_bounds(name, value):
    with pytest.raises(ValueError, match="score_time_min/max"):
        make_program(**{name: value})


@pytest.mark.parametrize("stage", ["student", "fake_score"])
@pytest.mark.parametrize("bounds", [(0.02, 0.98), (0.15, 0.6)])
def test_score_time_bounds_are_pre_shift_and_shared_across_the_microbatch(stage, bounds, monkeypatch):
    program, _, initial, _ = make_program(
        ("image", "audio"), score_time_min=bounds[0], score_time_max=bounds[1], teacher_guidance_scale=4.0
    )
    monkeypatch.setattr(program, "_sample", lambda *args, **kwargs: initial)
    predictions = program.predict
    observed = []

    def predict(role, latents, sigmas, negative):
        observed.append(sigmas)
        return predictions(role, latents, sigmas, negative)

    program.predict = predict
    generator = torch.Generator().manual_seed(107)
    base_time = torch.empty(()).uniform_(*bounds, generator=generator)
    torch.manual_seed(107)
    program.stage_outputs(initial, stage)
    for sigmas in observed:
        for name, shift in program.config.modality_shifts.items():
            assert sigmas[name].ndim == 0
            assert sigmas[name].dtype == torch.float32
            expected = shift * base_time / (1 + (shift - 1) * base_time)
            torch.testing.assert_close(sigmas[name], expected, rtol=0, atol=0)
        assert not torch.equal(sigmas["image"], sigmas["audio"])
    assert len(observed) == (3 if stage == "student" else 1)


def test_documented_shifted_score_endpoints():
    sigmas = shifted_sigmas(torch.tensor([0.02, 0.98]), {"image": 3.0}, torch.device("cpu"))["image"]
    torch.testing.assert_close(sigmas, torch.tensor([0.05769230769, 0.99324324324]))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4, 5), (2, 3, 2, 4, 5)])
def test_layer_norm_cfg_rejects_unpacked_layout_before_sampling(shape):
    program, _, _, calls = make_program(teacher_guidance_scale=4.0, cfg_norm="layer_norm")
    state, cpu_rng = program.state_dict(), torch.get_rng_state()
    with pytest.raises(ValueError, match="packed"):
        program.stage_outputs({"image": torch.ones(shape)}, "student")
    assert not calls
    assert torch.equal(state["control_rng"], program.state_dict()["control_rng"])
    assert torch.equal(cpu_rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "stage,guidance,norm",
    [
        ("student", 4.0, "none"),
        ("student", 4.0, "scalar"),
        ("student", 1.0, "layer_norm"),
        ("fake_score", 4.0, "layer_norm"),
    ],
)
def test_unpacked_latents_remain_supported_without_layer_norm_cfg(stage, guidance, norm):
    program, _, _, _ = make_program(teacher_guidance_scale=guidance, cfg_norm=norm)
    initial = {"image": torch.ones(2, 3, 4, 3)}
    outputs = program.stage_outputs(initial, stage)
    assert outputs["image"]["generated_x0"].shape == initial["image"].shape
    assert program.sample(initial)["image"].shape == initial["image"].shape


def test_layer_norm_cfg_matches_packed_feature_axis_formula(monkeypatch):
    program, _, initial, _ = make_program(teacher_guidance_scale=4.0, cfg_norm="layer_norm")
    monkeypatch.setattr(program, "_sample", lambda *args, **kwargs: initial)
    predictions = program.predict
    observed = {}

    def predict(role, latents, sigmas, negative):
        result = predictions(role, latents, sigmas, negative)
        if role == "teacher":
            observed[negative] = result["image"]
            observed["noisy"], observed["sigma"] = latents["image"], sigmas["image"]
        return result

    program.predict = predict
    output = program.stage_outputs(initial, "student")["image"]
    cond, uncond = observed[False], observed[True]
    guided = uncond + 4 * (cond - uncond)
    guided = guided * (cond.square().sum(-1, keepdim=True).sqrt() / guided.square().sum(-1, keepdim=True).sqrt())
    expected = observed["noisy"] - observed["sigma"] * guided
    torch.testing.assert_close(output["teacher_x0"], expected)


def test_invalid_stage_and_layout_fail_before_rng_is_consumed():
    program, _, initial, _ = make_program()
    state = program.state_dict()
    for invalid in ({"other": torch.ones(2, 3)}, {"image": torch.ones(2)}, {"image": torch.empty(0, 3)}):
        with pytest.raises(ValueError):
            program.stage_outputs(invalid, "student")
    with pytest.raises(ValueError, match="Unknown DMD stage"):
        program.stage_outputs(initial, "teacher")
    assert torch.equal(state["control_rng"], program.state_dict()["control_rng"])


@pytest.mark.parametrize("prediction", [{}, {"image": torch.zeros(2, 1, 3)}])
def test_prediction_layout_cannot_silently_broadcast(prediction):
    program, _, initial, _ = make_program()
    program.predict = lambda *args: prediction
    with pytest.raises(ValueError, match="Velocity predictions"):
        program.stage_outputs(initial, "fake_score")
