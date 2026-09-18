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
"""Exercise PPO/NFT output lifetime through the real engine batch entry points."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.workers.engine.fsdp import diffusers_impl


@pytest.fixture(params=["ppo", "nft"])
def engine_case(request, monkeypatch):
    engine_cls, timesteps_key = {
        "ppo": (diffusers_impl.PPODiffusersFSDPEngine, "all_timesteps"),
        "nft": (diffusers_impl.NFTDiffusersFSDPEngine, "train_timesteps"),
    }[request.param]

    def make_engine():
        # FSDP initialization requires devices; the batch loop and optimizer are real.
        engine = object.__new__(engine_cls)
        engine.ulysses_sequence_parallel_size = 1
        engine.ulysses_device_mesh = None
        engine.module = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            engine.module.weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64))
        engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.01)
        engine.optimizer_config = SimpleNamespace(clip_grad=1000.0)
        engine.get_data_parallel_group = lambda: None
        observed = SimpleNamespace(refs=[], live_before_step=[], calls=[], backward_calls=0)

        def forward_step(micro_batch, loss_function, forward_only, step):
            observed.live_before_step.append(sum(ref() is not None for ref in observed.refs))
            observed.calls.append((micro_batch["sample_id"].tolist(), step, torch.is_grad_enabled()))
            prediction = engine.module(micro_batch["features"] + step)
            model_output = {"prediction": prediction, "auxiliary": prediction.detach().square()}
            output_refs = tuple(weakref.ref(value) for value in model_output.values())
            observed.refs.extend(output_refs)

            if loss_function is None:
                assert forward_only
                loss = prediction.new_tensor(1.0)
                metrics = {}
            else:
                loss, metrics = loss_function(model_output, micro_batch)

            if loss.requires_grad:

                def check_backward(gradient):
                    assert all(ref() is not None for ref in output_refs)
                    observed.backward_calls += 1
                    return gradient

                loss.register_hook(check_backward)

            return loss, {"model_output": model_output, "loss": loss.detach().item(), "metrics": metrics}

        engine.forward_step = forward_step
        return engine, observed

    def prepare_micro_batches(data, dp_group, same_micro_num_in_dp):
        assert dp_group is None
        assert same_micro_num_in_dp
        assert tu.get_non_tensor_data(data, "use_dynamic_bsz", default=None) is False
        assert tu.get_non_tensor_data(data, "sp_size", default=None) == 1
        return list(data.split(tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", default=None))), None

    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(diffusers_impl, "prepare_micro_batches", prepare_micro_batches)
    return make_engine, timesteps_key


def _batch(timesteps_key, num_steps, micro_batch_size, return_model_output=None):
    batch = TensorDict(
        {
            "features": torch.arange(12, dtype=torch.float64).reshape(6, 2) / 10,
            "sample_id": torch.arange(6),
            timesteps_key: torch.arange(num_steps).expand(6, -1),
        },
        batch_size=[6],
    )
    tu.assign_non_tensor(batch, micro_batch_size_per_gpu=micro_batch_size)
    if return_model_output is not None:
        tu.assign_non_tensor(batch, return_model_output=return_model_output)
    return batch


def _loss(model_output, data):
    loss = (model_output["prediction"] - 1).square().mean()
    loss = loss / tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    return loss, {"objective": loss.detach().item()}


@pytest.mark.parametrize("num_steps", [1, 4])
@pytest.mark.parametrize("micro_batch_size", [2, 6])
@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("return_model_output", [None, False, True])
def test_output_retention_contract(engine_case, num_steps, micro_batch_size, forward_only, return_model_output):
    make_engine, timesteps_key = engine_case
    engine, observed = make_engine()
    batch = _batch(timesteps_key, num_steps, micro_batch_size, return_model_output)
    initial_weight = engine.module.weight.detach().clone()

    if forward_only:
        output = engine.infer_batch(batch, loss_function=_loss)
    else:
        output = engine.train_batch(batch, loss_function=_loss)

    num_micro_batches = 6 // micro_batch_size
    num_calls = num_micro_batches * num_steps
    assert observed.calls == [
        (list(range(start, start + micro_batch_size)), step, not forward_only)
        for start in range(0, 6, micro_batch_size)
        for step in range(num_steps)
    ]
    assert observed.backward_calls == (0 if forward_only else num_calls)
    assert len(output["loss"]) == num_micro_batches
    assert all(len(losses) == num_steps for losses in output["loss"])
    assert output["metrics"]["objective"] == [loss for losses in output["loss"] for loss in losses]

    keep_outputs = forward_only or return_model_output is True
    expected_live_counts = [2 * index if keep_outputs else 0 for index in range(num_calls)]
    assert observed.live_before_step == expected_live_counts
    # Postprocessing creates new stacked tensors; no per-step tensor should survive it.
    assert all(ref() is None for ref in observed.refs)

    if keep_outputs:
        expected = torch.cat(
            [
                torch.stack(
                    [torch.nn.functional.linear(features + step, initial_weight) for step in range(num_steps)],
                    dim=1,
                )
                for features in batch["features"].split(micro_batch_size)
            ],
            dim=0,
        )
        torch.testing.assert_close(output["model_output"]["prediction"], expected, rtol=0, atol=0)
        torch.testing.assert_close(output["model_output"]["auxiliary"], expected.square(), rtol=0, atol=0)
    else:
        assert output["model_output"] == {}
    if forward_only:
        assert engine.module.weight.grad is None
        torch.testing.assert_close(engine.module.weight, initial_weight, rtol=0, atol=0)


def test_training_update_matches_retained_outputs_across_repeated_calls(engine_case):
    make_engine, timesteps_key = engine_case
    optimized, _ = make_engine()
    reference, _ = make_engine()

    for _ in range(2):
        actual = optimized.train_batch(_batch(timesteps_key, 4, 2), loss_function=_loss)
        expected = reference.train_batch(_batch(timesteps_key, 4, 2, True), loss_function=_loss)

        assert actual["model_output"] == {}
        assert expected["model_output"]
        assert actual["loss"] == expected["loss"]
        assert actual["metrics"] == expected["metrics"]
        torch.testing.assert_close(optimized.module.weight.grad, reference.module.weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(optimized.module.weight, reference.module.weight, rtol=0, atol=0)


def test_inference_without_loss_preserves_outputs(engine_case):
    make_engine, timesteps_key = engine_case
    engine, observed = make_engine()
    output = engine.infer_batch(_batch(timesteps_key, 3, 2))

    assert output["model_output"]["prediction"].shape == (6, 3, 2)
    assert output["loss"] == [[1.0] * 3] * 3
    assert output["metrics"] == {}
    assert observed.backward_calls == 0
    assert engine.module.weight.grad is None
