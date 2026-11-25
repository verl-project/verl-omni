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

from tests.utils.dataset.duplex._duplex_fixtures import interruption_event, listening_then_speaking_wires
from tests.utils.dataset.duplex.test_replay_on_cpu import FakeStreamingProcessor, FakeTokenizer
from verl_omni.agent_loop.omni_duplex_agent_loop import OmniDuplexAgentLoop
from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_duplex_strategy import DuplexEpisodeError


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
    def __init__(self, result: dict[str, Any], *, failures: int = 0):
        self._result = result
        self._failures = failures
        self.episodes: list[dict[str, Any]] = []
        self.released = 0
        self.run_duplex_episode = _Remote(self._run_duplex_episode)

    async def _run_duplex_episode(self, run_spec):
        self.episodes.append(run_spec)
        if len(self.episodes) <= self._failures:
            raise DuplexEpisodeError("fake engine failure")
        return self._result


class _Client:
    def __init__(self, router: _Router):
        self._load_balancer = router


def _loop(
    server_result: dict[str, Any],
    *,
    response_length: int = 256,
    failures: int = 0,
    model_path: str | None = None,
) -> tuple[OmniDuplexAgentLoop, _FakeRolloutServer]:
    server = _FakeRolloutServer(server_result, failures=failures)
    client = _Client(_Router(server))
    loop = OmniDuplexAgentLoop.__new__(OmniDuplexAgentLoop)
    # Bypass AgentLoopBase.__init__ (its continuous-token builder needs a real
    # tokenizer); bind only the attributes the duplex loop consumes — the fake
    # mirrors the real base, whose config is a DictConfig (attribute access)
    # and which sets self.rollout_config with NO response_length attribute of
    # its own.
    loop.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                engine_kwargs={"vllm_omni": {"pipeline_name": "minicpmo_4_5_duplex"}},
            ),
            model=SimpleNamespace(path=model_path),
        )
    )
    loop.rollout_config = SimpleNamespace(
        engine_kwargs={"vllm_omni": {"pipeline_name": "minicpmo_4_5_duplex"}},
        response_length=response_length,
    )
    loop.server_manager = client
    loop.tokenizer = FakeTokenizer()
    loop.processor = FakeStreamingProcessor()
    loop.rollout_adapter = OmniDuplexAgentLoop._resolve_rollout_adapter(loop.rollout_config)
    return loop, server


def _server_result(session_id: str = "duplex-abc", *, engine_silence_units: int = 0) -> dict[str, Any]:
    wires = listening_then_speaking_wires(engine_silence_units=engine_silence_units)
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
    output = await loop.run({"temperature": 0.7}, duplex_episode=_episode(), duplex_seed=11, global_steps=4)
    assert output.prompt_ids  # the session context prefix
    assert len(output.response_ids) == 5 * 13
    assert len(output.response_mask) == len(output.response_ids)
    assert sum(output.response_mask) == 5  # five sampled decisions
    # Engine-captured logprobs stay inside the replay payload — the output
    # column stays unset so verl cannot mix them with old_log_probs.
    assert output.response_logprobs is None
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert payload["response_logprobs"]  # the captured channel rides the payload (M0 gate)
    assert payload["unit_mels"] and len(payload["unit_mels"]) == 5
    assert "unit_pcm" not in payload
    # Off-policy staleness tags verl's _compute_metrics int-casts.
    assert output.extra_fields["min_global_steps"] == 4
    assert output.extra_fields["max_global_steps"] == 4
    # A clean episode enters the drop-rate column at 0.
    assert output.extra_fields["reward_extra_info"] == {"duplex_pacing_corrupted": 0.0}
    # The server saw the run spec with the seed and the chunks, and without
    # the instructions key (an explicit value silences the engine pipeline).
    (run_spec,) = server.episodes
    assert 0 <= run_spec["seed"] < 2**63  # uuid-mixed from the dataset seed (group diversity)
    assert len(run_spec["chunks"]) == 5
    assert run_spec["temperature"] == 0.7
    assert "instructions" not in run_spec
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


@pytest.mark.asyncio
async def test_overlong_episode_fails_loudly_not_with_attributeerror():
    # AgentLoopBase has no response_length attribute; the overlong check must
    # read the rollout config and raise the length error, not AttributeError.
    loop, _ = _loop(_server_result(), response_length=5)
    with pytest.raises(ValueError, match="exceeding response_length=5"):
        await loop.run({}, duplex_episode=_episode(), duplex_seed=1, global_steps=2)


@pytest.mark.asyncio
async def test_episode_failure_is_re_driven_with_a_fresh_seed():
    loop, server = _loop(_server_result(), failures=1)
    output = await loop.run({}, duplex_episode=_episode(), duplex_seed=11)
    assert output.prompt_ids  # the retry succeeded and mapped normally
    assert len(server.episodes) == 2
    first, second = server.episodes
    assert first["seed"] != second["seed"]  # the re-drive mixes a fresh uuid
    assert loop.server_manager._load_balancer.released_count == 1  # one lease, held across the retry


@pytest.mark.asyncio
async def test_double_failure_is_a_degenerate_mask_zero_episode():
    loop, server = _loop(_server_result(), failures=2)
    output = await loop.run({}, duplex_episode=_episode(), duplex_seed=5, global_steps=9)
    # The rollout batch is not poisoned: a valid, empty, mask-0 episode.
    assert output.response_ids == []
    assert output.response_mask == []
    assert output.reward_score == 0.0  # keeps the scorer off the broken payload
    assert output.response_logprobs is None
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert payload["unit_layouts"] == [] and payload["audio_bounds"] == []
    assert payload["status"] == "failed"
    assert output.extra_fields["reward_extra_info"] == {"duplex_episode_failed": 1.0}
    assert output.extra_fields["min_global_steps"] == 9
    assert output.extra_fields["max_global_steps"] == 9
    assert len(server.episodes) == 2  # the original drive plus the retry
    assert loop.server_manager._load_balancer.released_count == 1


def test_global_steps_normalization():
    import torch
    from tensordict import NonTensorData

    from verl_omni.agent_loop.omni_duplex_agent_loop import _episode_global_steps

    assert _episode_global_steps({"global_steps": 4}) == 4
    assert _episode_global_steps({"global_steps": torch.tensor(5)}) == 5
    assert _episode_global_steps({"global_steps": NonTensorData(6)}) == 6
    assert _episode_global_steps({}) is None


@pytest.mark.asyncio
async def test_actor_model_path_rides_into_the_replay(tmp_path):
    # The engine embeds the snapshot's bundled assets/HT_ref_audio.wav in
    # every session; the replay must resolve it from the actor config's path
    # (70 context ids, first span the 60-row reference block).
    import soundfile as sf

    snapshot = tmp_path / "snapshot"
    (snapshot / "assets").mkdir(parents=True)
    sf.write(str(snapshot / "assets" / "HT_ref_audio.wav"), np.zeros(96256, dtype=np.float32), 16000)
    loop, _ = _loop(_server_result(), model_path=str(snapshot))
    output = await loop.run({}, duplex_episode=_episode(), duplex_seed=3)
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert len(payload["prompt_ids"]) == 70
    assert payload["ref_audio"]["num_audio_tokens"] == 60
    assert payload["audio_bounds"][0] == [8, 68]


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
async def test_interruption_anchors_are_rewritten_onto_the_unit_clock():
    # speech_chunk_start_index=2 / end_index=3: the second fed chunk opened the
    # interrupting speech and the third closed it; both anchors must follow
    # those client units' decision times, not the corpus-relative mining values.
    result = _server_result()
    result["event"] = interruption_event(onset_s=99.0, end_s=99.0)
    result["speech_chunk_start_index"] = 2
    result["speech_chunk_end_index"] = 3
    loop, _ = _loop(result)
    output = await loop.run({}, duplex_episode=_episode("interruption"), duplex_seed=3)
    event = output.extra_fields["duplex_replay"]["event"]
    # The listening scope's units are at t=0,1 and the response speaks from
    # t=2; the second and third client units decided at 1.0 s and 2.0 s.
    assert event["interrupt_speech_onset_s"] == pytest.approx(1.0)
    assert event["interrupt_speech_end_s"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_out_of_range_interruption_anchor_fails():
    result = _server_result()
    result["event"] = interruption_event(onset_s=1.0, end_s=1.0)
    result["speech_chunk_start_index"] = 1
    result["speech_chunk_end_index"] = 99
    loop, _ = _loop(result)
    with pytest.raises(ValueError, match="fed chunks"):
        await loop.run({}, duplex_episode=_episode("interruption"), duplex_seed=3)
    result["speech_chunk_start_index"] = 0
    result["speech_chunk_end_index"] = 1
    loop, _ = _loop(result)
    with pytest.raises(ValueError, match="fed chunks"):
        await loop.run({}, duplex_episode=_episode("interruption"), duplex_seed=3)


@pytest.mark.asyncio
async def test_engine_silence_inside_the_scored_window_drops_the_episode():
    # The runtime injected a silent unit at t=2.0, inside the turn window
    # [2.0, 5.0): its timing is the runtime's, so the episode degrades to the
    # mask-0 output flagged as pacing-corrupted — the trainer's per-step mean
    # of the column is the drop rate.
    loop, _ = _loop(_server_result(engine_silence_units=1))
    output = await loop.run({}, duplex_episode=_episode(), duplex_seed=3, global_steps=9)
    assert output.response_ids == []
    assert output.response_mask == []
    assert output.reward_score == 0.0
    assert output.extra_fields["reward_extra_info"] == {"duplex_episode_failed": 1.0, "duplex_pacing_corrupted": 1.0}
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert payload["unit_layouts"] == [] and payload["status"] == "failed"
    assert output.extra_fields["min_global_steps"] == 9


@pytest.mark.asyncio
async def test_engine_silence_outside_the_scored_window_keeps_the_episode():
    # The same injected unit at t=2.0 with the turn ending at 4.0 (window
    # [4.0, 7.0)) leaves the episode scoreable: it replays, flagged 0.
    result = _server_result(engine_silence_units=1)
    result["event"] = {"axis": "turn", "user_turn_end_s": 4.0}
    loop, _ = _loop(result)
    output = await loop.run({}, duplex_episode=_episode(), duplex_seed=3)
    assert output.response_ids and output.reward_score is None
    assert output.extra_fields["reward_extra_info"] == {"duplex_pacing_corrupted": 0.0}
    engine_layouts = [
        layout for layout in output.extra_fields[DUPLEX_REPLAY_KEY]["unit_layouts"] if layout["origin"] == "engine"
    ]
    assert [layout["time_s"] for layout in engine_layouts] == [pytest.approx(2.0)]


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
