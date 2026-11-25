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
"""One ``<unit>`` audio block per ingested 1 s unit, whatever its token count."""

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    CHUNK_EOS_TOKEN_ID,
    LISTEN_TOKEN_ID,
    SPEAK_CONTENT_TOKEN_ID,
    TURN_EOS_TOKEN_ID,
    listen_unit,
    record_set_wire,
    spoken_unit,
)
from tests.utils.dataset.duplex.test_replay_on_cpu import _vocabulary
from verl_omni.utils.dataset.duplex.replay import build_replay
from verl_omni.utils.dataset.duplex.trajectory import AXIS_TURN, DuplexEpisodeEvent, DuplexTrajectory

#: The token a mid-turn listen draw is rewritten to (``<|tts_bos|>``).
TTS_BOS_TOKEN_ID = 7
_PLACEHOLDERS = 10

_LISTEN, _SPEAK, _CHUNK_EOS, _TURN_EOS = (
    LISTEN_TOKEN_ID,
    SPEAK_CONTENT_TOKEN_ID,
    CHUNK_EOS_TOKEN_ID,
    TURN_EOS_TOKEN_ID,
)


def _units() -> list[dict]:
    """Five ingested units: listen, engine silence, a 5-token spoken unit, a turn_eos unit, listen."""
    five_token_spoken = spoken_unit(
        token_ids=[_SPEAK, _SPEAK, _SPEAK, _SPEAK],
        emission_ns=3_000_000_000,
    )
    # The third step draws listen mid-turn: the engine emits tts_bos instead —
    # a forced row carrying the raw draw's logprob.
    five_token_spoken["actions"][2] = (TTS_BOS_TOKEN_ID, "forced", -0.7, "content", "mid_turn_listen_rewrite")
    turn_eos_unit = spoken_unit(token_ids=[_SPEAK, _TURN_EOS], emission_ns=4_000_000_000)
    return [
        listen_unit(emission_ns=1_000_000_000),
        listen_unit(origin="engine", kind="forced", reason="force_listen", emission_ns=2_000_000_000),
        five_token_spoken,
        turn_eos_unit,
        listen_unit(emission_ns=5_000_000_000),
    ]


#: Each unit's non-``reemit`` tokens (what its ``<unit>`` block must hold) and their kinds.
_UNIT_TOKENS = [
    ([_LISTEN], ["sampled"]),
    ([_LISTEN], ["forced"]),
    ([_SPEAK, _SPEAK, TTS_BOS_TOKEN_ID, _SPEAK, _CHUNK_EOS], ["sampled", "sampled", "forced", "sampled", "sampled"]),
    ([_SPEAK, _TURN_EOS, _CHUNK_EOS], ["sampled", "sampled", "sampled"]),
    ([_LISTEN], ["sampled"]),
]


def test_spoken_unit_replay_grouping():
    units = _units()
    rng = np.random.default_rng(3)
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=2.0),
        record_set_wires=[record_set_wire(units=units)],
        client_audio_chunks=[rng.standard_normal(16000).astype(np.float32) for _ in range(4)],
    )
    vocabulary = _vocabulary()
    replay = build_replay(trajectory, vocabulary)
    num_units = len(units)
    prompt_length = len(replay.prompt_ids)
    response_ids, response_mask = replay.response_ids, replay.response_mask

    # Exactly one <unit> per ingested unit: intra-unit content tokens, the
    # boundary chunk_eos and the reemit rows add none.
    assert response_ids.count(vocabulary.unit_token_id) == num_units
    assert response_ids.count(vocabulary.unit_end_token_id) == num_units
    assert len(replay.unit_layouts) == num_units
    assert len(replay.audio_bounds) == num_units
    assert response_ids.count(_CHUNK_EOS) == 2  # the two boundaries, each replayed once

    cursor = 0
    for layout, (tokens, kinds) in zip(replay.unit_layouts, _UNIT_TOKENS, strict=True):
        window_length = 1 + _PLACEHOLDERS + len(tokens) + 1
        block = response_ids[cursor : cursor + window_length]
        # <unit>, its audio placeholders, the unit's tokens in order, </unit>.
        assert block == [
            vocabulary.unit_token_id,
            *[vocabulary.audio_placeholder_token_id] * _PLACEHOLDERS,
            *tokens,
            vocabulary.unit_end_token_id,
        ]
        first_token = cursor + 1 + _PLACEHOLDERS
        assert layout.action_position == first_token
        assert layout.action_positions == tuple(range(first_token, first_token + len(tokens)))
        assert layout.action_token_ids == tuple(tokens)
        audio_start = prompt_length + cursor + 1
        assert replay.audio_bounds[layout.unit_index] == [audio_start, audio_start + _PLACEHOLDERS]
        assert layout.audio_span == (audio_start, audio_start + _PLACEHOLDERS)
        # Mask 1 only at sampled action tokens; 0 at forced tokens and framing.
        assert response_mask[cursor : cursor + window_length] == [0] * (1 + _PLACEHOLDERS) + [
            int(kind == "sampled") for kind in kinds
        ] + [0]
        cursor += window_length
    # Across a spoken unit -> reemit -> next unit the restated terminator adds
    # nothing between </unit> and the next <unit>: the engine's scheduler drops a
    # segment's final sampled token and re-injects it at the next prefill, so its
    # KV holds the terminator once, and the replayed input holds it once too.
    spoken_end = sum(1 + _PLACEHOLDERS + len(tokens) + 1 for tokens, _ in _UNIT_TOKENS[:3])
    assert response_ids[spoken_end - 2 : spoken_end + 1 + _PLACEHOLDERS] == [
        _CHUNK_EOS,
        vocabulary.unit_end_token_id,
        vocabulary.unit_token_id,
        *[vocabulary.audio_placeholder_token_id] * _PLACEHOLDERS,
    ]
    assert cursor == len(response_ids) == len(response_mask) == len(replay.response_logprobs)

    # A forced token inside a spoken unit is masked while its sampled siblings are not.
    spoken = replay.unit_layouts[2]
    forced_position = spoken.action_positions[2]
    assert response_ids[forced_position] == TTS_BOS_TOKEN_ID and response_mask[forced_position] == 0
    assert [response_mask[position] for position in spoken.action_positions] == [1, 1, 0, 1, 1]
    # Captured logprobs ride the sampled tokens only.
    assert replay.response_logprobs[spoken.action_positions[0]] == pytest.approx(-1.3)
    assert replay.response_logprobs[spoken.action_positions[1]] == pytest.approx(-1.2)
    assert replay.response_logprobs[forced_position] == 0.0
    assert sum(response_mask) == 9

    # One PCM stream position per unit; the engine unit is one second of digital silence.
    assert [pcm.size for pcm in replay.unit_pcm] == [16000] * num_units
    assert np.all(replay.unit_pcm[1] == 0.0) and replay.unit_layouts[1].origin == "engine"
    assert all(np.any(replay.unit_pcm[index] != 0.0) for index in (0, 2, 3, 4))
