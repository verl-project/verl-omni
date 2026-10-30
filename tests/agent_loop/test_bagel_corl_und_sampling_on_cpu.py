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
"""CPU regression: the UND AR lane must never decode with the engine's raw defaults."""

from __future__ import annotations

import asyncio
import types

import pytest
from omegaconf import OmegaConf

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import (
    UND_AR_SAMPLING_FALLBACK,
    UNTRAINED_AR_SAMPLING,
    config_sampling_value,
    fill_und_ar_sampling_params,
)

# The exact values that collapse this checkpoint: vLLM's untruncated decoding defaults,
# which is also what ``verl``'s generic AR worker sends for ``repetition_penalty``.
AS_TRAINED = dict(UNTRAINED_AR_SAMPLING)

AR_KNOBS = tuple(UND_AR_SAMPLING_FALLBACK)

# What the recipe's ``ROLLOUT_*`` knobs resolve to, and what
# ``AgentLoopWorkerTQ.generate_sequences`` actually hands the UND route: the three AR knobs
# from the config, plus the hardcoded ``repetition_penalty=1.0``.
GENERIC_WORKER_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "logprobs": 0,
}


@pytest.mark.parametrize("config", [None, OmegaConf.create({})], ids=["none", "empty"])
def test_omitted_knobs_get_the_mitigated_fallback(config):
    """A caller that sends nothing must still not land on the as-trained defaults."""
    filled = fill_und_ar_sampling_params({}, config)
    for name, expected in UND_AR_SAMPLING_FALLBACK.items():
        assert filled[name] == expected, f"{name} not filled from the fallback"
    assert filled != AS_TRAINED, "the fallback must not be the engine's untruncated defaults"


def test_the_generic_ar_worker_repetition_penalty_is_replaced():
    """The regression, exactly: the base worker's hardcoded ``repetition_penalty=1.0``."""
    filled = fill_und_ar_sampling_params(dict(GENERIC_WORKER_PARAMS), OmegaConf.create({"repetition_penalty": 1.05}))
    assert filled["repetition_penalty"] == 1.05, "the hardcoded 1.0 must not survive onto the UND route"
    assert filled["temperature"] == 0.7, "the config's own AR values must be left alone"
    assert filled["logprobs"] == 0, "non-AR keys must pass through untouched"


@pytest.mark.parametrize("name", sorted(UNTRAINED_AR_SAMPLING))
def test_untruncated_values_are_treated_as_unset(name):
    """Each untruncated default is the collapse, not a preference, so all four are replaced."""
    filled = fill_und_ar_sampling_params({name: UNTRAINED_AR_SAMPLING[name]}, OmegaConf.create({}))
    assert filled[name] == UND_AR_SAMPLING_FALLBACK[name]
    assert filled[name] != UNTRAINED_AR_SAMPLING[name]


def test_config_values_win_over_the_fallback():
    """``ROLLOUT_*`` / deploy-config values are the source of truth when present."""
    config = OmegaConf.create({"temperature": 0.9, "top_p": 0.95, "top_k": 8, "repetition_penalty": 1.02})
    filled = fill_und_ar_sampling_params(
        {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}, config
    )
    assert filled == {"temperature": 0.9, "top_p": 0.95, "top_k": 8, "repetition_penalty": 1.02}


def test_deliberate_caller_values_are_preserved():
    """A value that is *not* an untruncated default was chosen by someone; keep it."""
    sent = {
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 8,
        "repetition_penalty": 1.2,
        "max_tokens": 64,
        "logprobs": None,
    }
    filled = fill_und_ar_sampling_params(dict(sent), OmegaConf.create({"temperature": 0.9}))
    assert filled["temperature"] == 0.2, "an explicit temperature must not be overwritten by config"
    assert filled["top_p"] == 0.85
    assert filled["top_k"] == 8
    assert filled["repetition_penalty"] == 1.2
    assert filled["max_tokens"] == 64, "non-AR keys must pass through untouched"
    assert filled["logprobs"] is None


def test_partial_caller_values_only_fill_the_gaps():
    """The realistic failure: the caller sent the diffusion knobs and no AR ones at all."""
    gen_only = {"num_inference_steps": 50, "noise_level": 0.7, "sde_window_size": 2}
    filled = fill_und_ar_sampling_params(dict(gen_only), OmegaConf.create({"top_k": 50}))
    assert all(name in filled for name in AR_KNOBS), "every AR knob must be present on the UND route"
    assert filled["num_inference_steps"] == 50, "GEN keys are stripped elsewhere, not here"


@pytest.mark.parametrize(
    "config",
    [
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"temperature": 0.7}}}),
        OmegaConf.create({"rollout": {"temperature": 0.7}}),
        {"temperature": 0.7},
    ],
    ids=["trainer-config", "rollout-section", "plain-dict"],
)
def test_config_probe_reads_every_config_shape(config):
    """The probe must not assume one shape: the call site passes the *trainer* config."""
    assert config_sampling_value(config, "temperature") == 0.7


def test_config_probe_prefers_the_rollout_section_the_ar_engine_was_launched_with():
    """``actor_rollout_ref.rollout`` is what decides the decode, so it outranks the root."""
    config = OmegaConf.create(
        {
            "temperature": 1.0,
            "actor_rollout_ref": {"rollout": {"temperature": 0.7}},
            "rollout": {"temperature": 0.9},
        }
    )
    assert config_sampling_value(config, "temperature") == 0.7


def test_config_probe_returns_none_when_absent():
    """An absent knob must fall back, not raise, so a config probe can never fail a rollout."""
    assert config_sampling_value(OmegaConf.create({"temperature": 0.7}), "top_k") is None
    assert config_sampling_value(OmegaConf.create({"actor_rollout_ref": {"rollout": {}}}), "top_k") is None
    assert config_sampling_value(None, "temperature") is None


# Call-site guards. The helper tests pin the contract; these drive the real ``generate``
# with recording child clients, so deleting the fill from the routing path fails the
# suite instead of silently restoring the as-trained collapse.


def _dual_role_client_cls():
    """Import the routing client, or skip where its vLLM import chain needs CUDA."""
    try:
        from verl_omni.workers.rollout.bagel_dual_role_llm_server import BagelDualRoleLLMServerClient
    except (ImportError, RuntimeError) as exc:  # pragma: no cover -- environment-dependent
        pytest.skip(f"dual-role server needs a CUDA-capable import chain: {exc}")
    return BagelDualRoleLLMServerClient


class _RecordingClient:
    """Stand-in for an ``LLMServerClient`` child: records what the lane was handed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append(dict(sampling_params))
        return object()


def _route(sampling_params: dict, config=None) -> tuple[_RecordingClient, _RecordingClient]:
    client_cls = _dual_role_client_cls()
    und, gen = _RecordingClient(), _RecordingClient()
    fake_self = types.SimpleNamespace(
        und_client=und,
        gen_client=gen,
        config=OmegaConf.create({} if config is None else config),
    )
    asyncio.run(client_cls.generate(fake_self, "req-1", prompt_ids=[1, 2, 3], sampling_params=sampling_params))
    return und, gen


def test_und_route_is_filled_when_the_caller_sent_no_ar_knobs():
    """The exact regression: non-gen-shaped params route to UND and must get the AR knobs."""
    und, gen = _route({"max_tokens": 64}, config={"rollout": {"top_k": 50}})
    assert not gen.calls, "a request with no diffusion keys must not go to the GEN lane"
    sent = und.calls[0]
    for name in AR_KNOBS:
        assert name in sent, f"{name} must be present on the UND route, not left to the engine"
    assert sent["top_k"] == 50, "a configured value must win over the fallback"
    assert sent["temperature"] != AS_TRAINED["temperature"], "the UND route must not decode as-trained"
    assert sent["repetition_penalty"] != AS_TRAINED["repetition_penalty"]


def test_debug_route_does_not_require_the_episode_collector(monkeypatch):
    import verl_omni.workers.rollout.bagel_dual_role_llm_server as serving_client

    monkeypatch.setattr(serving_client, "_BAGEL_DUAL_ROLE_DEBUG", True)
    und, gen = _route({"max_tokens": 64})
    assert len(und.calls) == 1 and not gen.calls


def test_und_route_replaces_the_generic_workers_hardcoded_repetition_penalty():
    """End to end on the shape the real caller sends, which is what the fix is about."""
    und, gen = _route(
        dict(GENERIC_WORKER_PARAMS),
        config={"actor_rollout_ref": {"rollout": {"repetition_penalty": 1.05}}},
    )
    assert not gen.calls
    sent = und.calls[0]
    assert sent["repetition_penalty"] == 1.05
    assert sent["temperature"] == 0.7 and sent["top_p"] == 0.9 and sent["top_k"] == 50
    assert sent["logprobs"] == 0


def test_und_route_reads_the_same_section_the_recipe_pins():
    """The real call site passes the *trainer* config, so the fill must find the recipe keys."""
    config = {
        "actor_rollout_ref": {"rollout": {"temperature": 0.55, "top_p": 0.85, "top_k": 7, "repetition_penalty": 1.11}}
    }
    und, _ = _route({"max_tokens": 64}, config=config)
    sent = und.calls[0]
    assert sent["temperature"] == 0.55
    assert sent["top_p"] == 0.85
    assert sent["top_k"] == 7
    assert sent["repetition_penalty"] == 1.11


def test_gen_route_still_strips_the_ar_knobs():
    """The GEN half of the strip must be untouched by the UND-side fill."""
    und, gen = _route({"num_inference_steps": 50, "noise_level": 0.7, "temperature": 0.7, "top_k": 50})
    assert not und.calls, "diffusion-shaped params must go to the GEN lane"
    sent = gen.calls[0]
    assert sent["num_inference_steps"] == 50
    assert not (set(AR_KNOBS) & set(sent)), f"AR knobs must not reach the diffusion lane: {sorted(sent)}"
