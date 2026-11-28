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
"""CPU tests for vLLM-Omni rollout determinism propagation."""

import os
from types import SimpleNamespace

import pytest
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

from verl_omni.workers.rollout.vllm_rollout import utils as worker_utils
from verl_omni.workers.rollout.vllm_rollout.npu_utils import vLLMOmniNPUColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


def _stub_worker_extension_hooks(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(worker_utils, "set_death_signal", lambda: None)
    monkeypatch.setattr(worker_utils.VLLMOmniHijack, "hijack", lambda: None)
    monkeypatch.setattr(
        "verl_omni.workers.rollout.vllm_rollout.process_hidden_states.apply_hidden_states_patches",
        lambda: None,
    )
    monkeypatch.setattr(
        "verl.workers.engine.utils.enable_full_determinism",
        lambda *, seed: calls.append(seed),
    )
    return calls


@pytest.mark.parametrize(
    "extension_cls",
    [vLLMOmniColocateWorkerExtension, vLLMOmniNPUColocateWorkerExtension],
)
def test_worker_extension_enables_full_determinism_from_verl_seed(monkeypatch, extension_cls):
    calls = _stub_worker_extension_hooks(monkeypatch)
    monkeypatch.setenv("VERL_FULL_DETERMINISM", "1")
    monkeypatch.setenv("VERL_SEED", "123")

    instance = extension_cls.__new__(extension_cls)

    assert isinstance(instance, extension_cls)
    assert calls == [123]


@pytest.mark.parametrize(
    ("full_det", "verl_seed"),
    [
        (None, None),
        ("0", "42"),
        ("1", None),
    ],
)
def test_worker_extension_skips_determinism_without_flag_and_seed(monkeypatch, full_det, verl_seed):
    calls = _stub_worker_extension_hooks(monkeypatch)
    monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    monkeypatch.delenv("VERL_SEED", raising=False)
    if full_det is not None:
        monkeypatch.setenv("VERL_FULL_DETERMINISM", full_det)
    if verl_seed is not None:
        monkeypatch.setenv("VERL_SEED", verl_seed)

    vLLMOmniColocateWorkerExtension.__new__(vLLMOmniColocateWorkerExtension)

    assert calls == []


@pytest.mark.parametrize(
    ("configured_seed", "expected_seed"),
    [
        (123, 123),
        (0, 0),
        (None, 42),
    ],
)
def test_server_post_init_exports_determinism_env_for_workers(monkeypatch, configured_seed, expected_seed):
    calls: list[int] = []
    monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    monkeypatch.delenv("VERL_SEED", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(
        "verl.workers.engine.utils.enable_full_determinism",
        lambda *, seed: calls.append(seed),
    )
    monkeypatch.setattr(vLLMHttpServer, "_post_init", lambda self, _devices: None)

    server = object.__new__(vLLMOmniHttpServer)
    server.config = SimpleNamespace(
        full_determinism=True,
        seed=configured_seed,
        ulysses_degree=1,
        ring_degree=1,
    )
    server._generate_strategy = SimpleNamespace(post_init=lambda _devices: None)

    server._post_init("0,1")

    assert calls == [expected_seed]
    assert os.environ["VERL_FULL_DETERMINISM"] == "1"
    assert os.environ["VERL_SEED"] == str(expected_seed)
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"


def test_server_post_init_leaves_determinism_env_untouched_when_disabled(monkeypatch):
    calls: list[int] = []
    monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    monkeypatch.delenv("VERL_SEED", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(
        "verl.workers.engine.utils.enable_full_determinism",
        lambda *, seed: calls.append(seed),
    )
    monkeypatch.setattr(vLLMHttpServer, "_post_init", lambda self, _devices: None)

    server = object.__new__(vLLMOmniHttpServer)
    server.config = SimpleNamespace(
        full_determinism=False,
        seed=7,
        ulysses_degree=1,
        ring_degree=1,
    )
    server._generate_strategy = SimpleNamespace(post_init=lambda _devices: None)

    server._post_init("0")

    assert calls == []
    assert "VERL_FULL_DETERMINISM" not in os.environ
    assert "VERL_SEED" not in os.environ
    assert "VLLM_BATCH_INVARIANT" not in os.environ


@pytest.mark.parametrize(
    ("full_determinism", "server_seed", "request_seed", "expected_seed"),
    [
        (True, 99, None, 99),
        (True, 0, None, 0),
        (True, None, None, 42),
        (True, 99, 7, 7),
        (True, 99, 0, 0),
        (False, 99, None, None),
    ],
)
def test_diffusion_strategy_defaults_missing_seed_under_full_determinism(
    full_determinism,
    server_seed,
    request_seed,
    expected_seed,
):
    server = SimpleNamespace(
        config=SimpleNamespace(full_determinism=full_determinism, seed=server_seed),
        model_config=None,
        engine=SimpleNamespace(
            default_sampling_params_list=[object()],
            engine=SimpleNamespace(
                get_stage_metadata=lambda _idx: SimpleNamespace(stage_type="diffusion"),
            ),
        ),
    )
    strategy = DiffusionStrategy(server)
    request = SimpleNamespace(to_diffusion_prompt=lambda: {"prompt_ids": [1, 2]})
    sampling_params = {} if request_seed is None else {"seed": request_seed}

    _prompt, params = strategy.preprocess_input(request, sampling_params, lora_request=None)

    assert params[-1].seed == expected_seed


@pytest.mark.parametrize(
    ("full_determinism", "server_seed", "sampling_params", "expected_seed"),
    [
        (True, 99, {}, 99),
        (True, 99, {"seed": None}, 99),
        (True, 0, {"seed": None}, 0),
        (True, None, {"seed": None}, 42),
        (True, 99, {"seed": 7}, 7),
        (True, 99, {"seed": 0}, 0),
        (False, 99, {"seed": None}, None),
    ],
)
def test_ar_strategy_defaults_missing_or_null_seed_under_full_determinism(
    full_determinism,
    server_seed,
    sampling_params,
    expected_seed,
):
    server = SimpleNamespace(
        config=SimpleNamespace(
            full_determinism=full_determinism,
            seed=server_seed,
            max_model_len=64,
            prompt_length=32,
            response_length=16,
            repetition_penalty=1.0,
        ),
        model_config=SimpleNamespace(processor=None),
    )
    strategy = ARStrategy(server)
    request = SimpleNamespace(
        prompt=SimpleNamespace(
            token_ids=[1, 2, 3],
            mask=None,
            negative_token_ids=None,
            extra_token_ids=None,
            negative_extra_token_ids=None,
            mm_processor_kwargs=None,
        ),
        multi_modal_data=lambda: None,
    )

    _prompt, params = strategy.preprocess_input(request, dict(sampling_params), lora_request=None)

    assert params.seed == expected_seed
