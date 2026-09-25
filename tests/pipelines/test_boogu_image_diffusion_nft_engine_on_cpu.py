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
"""CPU tests for the DiffusionNFT engine's timestep-scale contract.

``train_timesteps`` are emitted as ``sigma * num_train_timesteps`` by the per-pipeline
rollout adapters, so the engine has to divide by *that* number to recover flow time in
``[0, 1]``. It previously divided by the literal ``1000.0``, which silently built a wrong
``xt`` for any checkpoint whose scheduler config disagrees -- invisible only because every
in-tree checkpoint happens to declare ``num_train_timesteps: 1000``.

The engine also publishes the integer it divides by on the micro-batch, so the adapter
conditioning the DiT reads the same value rather than deriving a second one.

The existing adapter-level test (``test_prepare_model_inputs_accepts_single_step_tensors``)
runs at ``N=2000`` and therefore *could* have caught this, but it only exercises the
adapter's own conversion, never the engine's divisor.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine

_PREPARE_MODEL_INPUTS = "verl_omni.workers.engine.fsdp.diffusers_impl.prepare_model_inputs"


def _stub_engine(num_train_timesteps: int | None) -> NFTDiffusersFSDPEngine:
    """An engine exposing only what ``prepare_model_inputs`` reads.

    ``None`` installs a scheduler whose config has no ``num_train_timesteps`` at all.
    """
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.scheduler = SimpleNamespace(config=SimpleNamespace())
    if num_train_timesteps is not None:
        engine.scheduler.config.num_train_timesteps = num_train_timesteps
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    engine.module = SimpleNamespace()
    engine.model_config = SimpleNamespace()
    return engine


def _micro_batch(timesteps: torch.Tensor) -> TensorDict:
    return TensorDict(
        {
            "latents_clean": torch.zeros(2, 3, 4, 4),
            "train_timesteps": timesteps,
            "prompt_embeds": torch.ones(2, 5, 8),
            "prompt_embeds_mask": torch.ones(2, 5, dtype=torch.bool),
        },
        batch_size=2,
    )


@pytest.mark.parametrize("num_train_timesteps", [1000, 2000])
def test_engine_divides_train_timesteps_by_the_scheduler_scale(num_train_timesteps: int):
    """Flow time must be ``timestep / N`` using the scheduler's own ``N``.

    ``N=2000`` is the regression case: with the old literal ``1000.0`` the engine would
    report twice the intended flow time, so ``xt`` would carry the wrong noise level for
    the timestep the model is conditioned on.
    """
    engine = _stub_engine(num_train_timesteps)
    timesteps = torch.tensor([[750.0, 250.0], [500.0, 100.0]])

    micro_batch = _micro_batch(timesteps)
    with patch(_PREPARE_MODEL_INPUTS, return_value=({}, {})):
        *_, t_expanded = engine.prepare_model_inputs(micro_batch=micro_batch, step=0)

    expected = (timesteps[:, 0] / num_train_timesteps).view(-1, 1, 1, 1)
    torch.testing.assert_close(t_expanded, expected)
    # The adapter conditions the DiT with this integer, so the engine hands it over.
    assert tu.get_non_tensor_data(micro_batch, "num_train_timesteps", default=None) == num_train_timesteps

    # Span the whole schedule to confirm the divisor is uniform across steps.
    with patch(_PREPARE_MODEL_INPUTS, return_value=({}, {})):
        *_, t_expanded_step1 = engine.prepare_model_inputs(micro_batch=_micro_batch(timesteps), step=1)
    torch.testing.assert_close(t_expanded_step1, (timesteps[:, 1] / num_train_timesteps).view(-1, 1, 1, 1))


def test_engine_flow_time_is_complementary_to_the_boogu_timestep():
    """B2's cross-module invariant, without importing the pipeline.

    The engine builds ``xt`` from ``t = ts / N`` and publishes that ``N`` on the
    micro-batch; the Boogu adapter conditions the transformer on ``1 - ts / N`` read from
    the same entry. Both derive from one integer, so the two must sum to one.
    """
    timesteps = torch.tensor([[750.0, 250.0], [500.0, 100.0]])
    for num_train_timesteps in (1000, 2000):
        engine = _stub_engine(num_train_timesteps)
        micro_batch = _micro_batch(timesteps)
        with patch(_PREPARE_MODEL_INPUTS, return_value=({}, {})):
            *_, t_expanded = engine.prepare_model_inputs(micro_batch=micro_batch, step=0)

        published = tu.get_non_tensor_data(micro_batch, "num_train_timesteps", default=None)
        assert published == num_train_timesteps
        boogu_time = 1.0 - timesteps[:, 0] / published
        torch.testing.assert_close(t_expanded.flatten() + boogu_time, torch.ones(2))


def test_engine_rejects_a_scheduler_without_a_timestep_scale():
    """A scheduler that cannot report ``N`` must fail loudly instead of guessing ``1000``."""
    engine = _stub_engine(num_train_timesteps=None)

    with patch(_PREPARE_MODEL_INPUTS, return_value=({}, {})):
        with pytest.raises(ValueError, match="num_train_timesteps"):
            engine.prepare_model_inputs(
                micro_batch=_micro_batch(torch.tensor([[750.0, 250.0], [500.0, 100.0]])), step=0
            )
