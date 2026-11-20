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
"""CPU tests for the duplex agent loop: episode → trajectory → AgentLoopOutput.

Everything below the Ray boundary is real (run-spec lifting, trajectory
assembly, replay build, adapter mapping); the rollout server and its session
are fakes with the engine contract's shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import listening_then_speaking_wires
from tests.utils.dataset.duplex.test_replay_on_cpu import FakeStreamingProcessor, FakeTokenizer
from verl_omni.agent_loop.omni_duplex_agent_loop import OmniDuplexAgentLoop
from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY


@dataclass
class _Router:
    server: _FakeRolloutServer
    acquired: list[str] = field(default_factory=list)
    released_count: int = 0

    async def _acquire(self, request_id: str):
        self.acquired.append(request_id)
        return "server-0", self.server

    async def _release(self, server_id: str, request_id: str | None = None):
        self.released_count += 1

    def __post_init__(self) -> None:
        self.acquire_server = _Remote(self._acquire)
        self.release_server = _Remote(self._release)


class _Remote:
    """Ray actor-method stand-in: the loop calls ``method.remote(...)``."""

    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return await self._fn(*args, **kwargs)


class _FakeRolloutServer:
    def __init__(self, result: dict[str, Any]):
        self._result = result
        self.episodes: list[dict[str, Any]] = []
        self.released = 0
        self.run_duplex_episode = _Remote(self._run_duplex_episode)

    async def _run_duplex_episode(self, run_spec):
        self.episodes.append(run_spec)
        return self._result


class _Client:
    def __init__(self, router: _Router):
        self._load_balancer = router


def _loop(server_result: dict[str, Any]) -> tuple[OmniDuplexAgentLoop, _FakeRolloutServer]:
    server = _FakeRolloutServer(server_result)
    client = _Client(_Router(server))
    config = SimpleNamespace(
        config={
            "actor_rollout_ref": {
                "rollout": {
                    "engine_kwargs": {"vllm_omni": {"pipeline_name": "minicpmo_4_5_duplex"}},
                    "response_length": 256,
                }
            }
        }
    )
    loop = OmniDuplexAgentLoop.__new__(OmniDuplexAgentLoop)
    # Bypass AgentLoopBase.__init__ (its continuous-token builder needs a real
    # tokenizer); bind only the attributes the duplex loop consumes.
    loop.config = config.config
    loop.rollout_config = SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "minicpmo_4_5_duplex"}})
    loop.server_manager = client
    loop.tokenizer = FakeTokenizer()
    loop.processor = FakeStreamingProcessor()
    loop.response_length = 256
    loop.rollout_adapter = OmniDuplexAgentLoop._resolve_rollout_adapter(loop.rollout_config)
    return loop, server


def _server_result(session_id: str = "duplex-abc") -> dict[str, Any]:
    wires = listening_then_speaking_wires()
    return {
        "episode_id": "ep-1",
        "session_id": session_id,
        "status": "completed",
        "record_sets": wires,
        "event": {"axis": "turn", "user_turn_end_s": 2.0},
        "client_chunks": [np.zeros(16000, dtype=np.float32) for _ in range(5)],
    }


def _episode(axis: str = "turn") -> dict[str, Any]:
    return {
        "episode_id": "ep-1",
        "axis": axis,
        "event": {"axis": axis, "user_turn_end_s": 2.0},
        "chunks": [{"pcm": np.zeros(16000, dtype=np.float32), "is_speech": True} for _ in range(5)],
        "instructions": None,
    }


@pytest.mark.asyncio
async def test_run_maps_one_episode_onto_the_policy_sequence():
    loop, server = _loop(_server_result())
    output = await loop.run({"temperature": 0.7}, duplex_episode=_episode(), duplex_seed=11)
    assert output.prompt_ids  # the session context prefix
    assert len(output.response_ids) == 5 * 13
    assert len(output.response_mask) == len(output.response_ids)
    assert sum(output.response_mask) == 5  # five sampled decisions
    assert output.response_logprobs is not None
    assert len(output.response_logprobs) == len(output.response_ids)
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert payload["unit_mels"] and len(payload["unit_mels"]) == 5
    assert "unit_pcm" not in payload
    # The server saw the run spec with the seed and the chunks.
    (run_spec,) = server.episodes
    assert 0 <= run_spec["seed"] < 2**63  # uuid-mixed from the dataset seed (group diversity)
    assert len(run_spec["chunks"]) == 5
    assert run_spec["temperature"] == 0.7
    assert server._result is not None
    assert loop.server_manager._load_balancer.released_count == 1  # the lease returned


@pytest.mark.asyncio
async def test_run_lifts_interruption_injection_into_the_run_spec():
    loop, server = _loop(_server_result())
    episode = _episode("interruption")
    episode["inject_delay_s"] = 0.4
    await loop.run({}, duplex_episode=episode, duplex_seed=3)
    (run_spec,) = server.episodes
    assert run_spec["inject"] == {"delay_s": 0.4, "tail_silence_chunks": 2}
    assert run_spec["speak_wait_horizon_s"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_run_rejects_episodes_without_chunks():
    loop, _ = _loop(_server_result())
    with pytest.raises(ValueError, match="no audio chunks"):
        await loop.run({}, duplex_episode={**_episode(), "chunks": []}, duplex_seed=1)


@pytest.mark.asyncio
async def test_run_rejects_samples_without_the_episode_column():
    loop, _ = _loop(_server_result())
    with pytest.raises(TypeError, match="duplex_episode"):
        await loop.run({})


@pytest.mark.asyncio
async def test_misaligned_trajectory_fails_the_episode():
    result = _server_result()
    result["client_chunks"] = [np.zeros(16000, dtype=np.float32) for _ in range(4)]  # one short
    loop, _ = _loop(result)
    with pytest.raises(Exception, match="client chunks"):
        await loop.run({}, duplex_episode=_episode(), duplex_seed=7)


def test_resolve_adapter_rejects_non_duplex_pipelines():
    config = SimpleNamespace(
        engine_kwargs={"vllm_omni": {"pipeline_name": "minicpmo_4_5"}}  # thinker-only adapter
    )
    with pytest.raises(ValueError, match="duplex-capable"):
        OmniDuplexAgentLoop._resolve_rollout_adapter(config)


def test_loop_is_registered_under_its_agent_name():
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

    from verl_omni.agent_loop import OmniDuplexAgentLoop  # noqa: F401 - registration import

    assert "omni_duplex_agent" in _agent_loop_registry


@pytest.mark.asyncio
async def test_interruption_anchor_is_rewritten_onto_the_unit_clock():
    # speech_chunk_end_index=2: the second fed chunk closed the interrupting
    # speech; the yield deadline must anchor to that client unit's decision
    # time, not the corpus-relative mining value.
    result = _server_result()
    result["event"] = {"axis": "interruption", "interrupt_speech_end_s": 99.0}
    result["speech_chunk_end_index"] = 2
    loop, _ = _loop(result)
    output = await loop.run({}, duplex_episode=_episode("interruption"), duplex_seed=3)
    event = output.extra_fields["duplex_replay"]["event"]
    # The listening scope's units are at t=0,1 and the response speaks from
    # t=2; the second client unit's decision time is 1.0 s.
    assert event["interrupt_speech_end_s"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_out_of_range_interruption_anchor_fails():
    result = _server_result()
    result["event"] = {"axis": "interruption", "interrupt_speech_end_s": 1.0}
    result["speech_chunk_end_index"] = 99
    loop, _ = _loop(result)
    with pytest.raises(ValueError, match="fed chunks"):
        await loop.run({}, duplex_episode=_episode("interruption"), duplex_seed=3)


@pytest.mark.asyncio
async def test_group_members_get_distinct_session_seeds():
    # rollout.n copies share one dataset row; their sessions must not share
    # the seed or the group collapses to identical rollouts.
    loop, server = _loop(_server_result())
    await loop.run({}, duplex_episode=_episode(), duplex_seed=42)
    await loop.run({}, duplex_episode=_episode(), duplex_seed=42)
    first, second = server.episodes
    assert first["seed"] != second["seed"]
    assert (first["seed"] - 42 + 2**63) % 2**63 < 2**63  # derived from the dataset seed
