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
"""CPU tests for the unit-window replay builder."""

import re

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    CHUNK_EOS_TOKEN_ID,
    LISTEN_TOKEN_ID,
    SPEAK_CONTENT_TOKEN_ID,
    listening_then_speaking_wires,
    reemit_variant_wires,
)
from verl_omni.utils.dataset.duplex.records import RecordSetError
from verl_omni.utils.dataset.duplex.replay import (
    DEFAULT_SESSION_INSTRUCTIONS,
    DUPLEX_REPLAY_KEY,
    DuplexReplay,
    DuplexTokenVocabulary,
    audio_tokens_for_samples,
    build_actor_media,
    build_replay,
    extract_unit_mels,
    resolve_duplex_vocabulary,
)
from verl_omni.utils.dataset.duplex.trajectory import AXIS_TURN, DuplexEpisodeEvent, DuplexTrajectory

# A deterministic id table: every framing token is one id, none equal to unk.
_TOKEN_TABLE = {
    "<unit>": 9001,
    "</unit>": 9002,
    "<|listen|>": LISTEN_TOKEN_ID,
    "<|audio|>": 9004,
    "<|im_start|>": 9101,
    "system": 9102,
    "\n": 9103,
    "Streaming": 9104,
    "Omni": 9105,
    "Conversation.": 9106,
    "<|im_end|>": 9107,
    "You": 9201,
    "are": 9202,
    "terse.": 9203,
}


class FakeTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return _TOKEN_TABLE.get(token, None)

    def encode(self, text, add_special_tokens=False):
        # Split special markers off first (they glue to words in the rendered
        # context, e.g. "<|im_start|>system"), then split words on spaces.
        ids: list[int] = []
        for part in re.split(r"(<\|[^|]+\|>|</unit>|<unit>)", text):
            if not part:
                continue
            if part in _TOKEN_TABLE:
                ids.append(_TOKEN_TABLE[part])
                continue
            for word in part.replace("\n", " \n ").split(" "):
                if word:
                    ids.append(_TOKEN_TABLE[word])
        return ids


class FakeStreamingProcessor:
    """Streaming mel fake: one feature row per 100 ms of PCM, 16 kHz."""

    sampling_rate = 16000
    tokenizer = None  # bound lazily by tests that also need the vocabulary

    def __init__(self):
        self.reset_calls = []
        self.chunk_count = 0
        self.tokenizer = FakeTokenizer()

    def process_audio_streaming(self, chunk, reset=False, return_batch_feature=False):
        self.reset_calls.append(reset)
        self.chunk_count += 1
        rows = audio_tokens_for_samples(len(chunk))
        return {"audio_features": [np.ones((rows, 80), dtype=np.float32)]}


def _trajectory(engine_silence_units: int = 0) -> DuplexTrajectory:
    wires = listening_then_speaking_wires(engine_silence_units=engine_silence_units)
    rng = np.random.default_rng(1)
    client_units = 5
    return DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=2.0),
        record_set_wires=wires,
        client_audio_chunks=[rng.standard_normal(16000).astype(np.float32) for _ in range(client_units)],
    )


def _vocabulary() -> DuplexTokenVocabulary:
    return resolve_duplex_vocabulary(FakeTokenizer())


def test_audio_token_count_matches_the_unit_cadence():
    assert audio_tokens_for_samples(16000) == 10
    assert audio_tokens_for_samples(0) == 0
    assert audio_tokens_for_samples(160) == 0  # mel frame, cnn rounds to 1, pooling floors


def test_replay_rebuilds_the_engine_unit_layout():
    replay = build_replay(_trajectory(), _vocabulary())
    vocabulary = _vocabulary()
    # Context prefix: the MiniCPMODuplex.prepare() system prompt.
    assert replay.prompt_ids == [
        _TOKEN_TABLE["<|im_start|>"],
        _TOKEN_TABLE["system"],
        _TOKEN_TABLE["\n"],
        _TOKEN_TABLE["Streaming"],
        _TOKEN_TABLE["Omni"],
        _TOKEN_TABLE["Conversation."],
        _TOKEN_TABLE["<|im_end|>"],
    ]
    prompt_length = len(replay.prompt_ids)
    # First unit: <unit> + 10 placeholders + action; later units close with </unit> first.
    assert replay.response_ids[:12] == [
        vocabulary.unit_token_id,
        *[vocabulary.audio_placeholder_token_id] * 10,
        LISTEN_TOKEN_ID,
    ]
    # Every unit is 13 tokens: <unit>, 10 audio, action, </unit>.
    assert len(replay.response_ids) == 5 * 13
    assert len(replay.response_mask) == len(replay.response_ids)
    assert len(replay.response_logprobs) == len(replay.response_ids)
    # Actions sit at unit-relative offset 11 of each 13-token unit window.
    action_ids = replay.action_token_ids()
    assert action_ids == [
        LISTEN_TOKEN_ID,
        LISTEN_TOKEN_ID,
        SPEAK_CONTENT_TOKEN_ID,
        SPEAK_CONTENT_TOKEN_ID,
        CHUNK_EOS_TOKEN_ID,
    ]
    assert sum(replay.response_mask) == 5
    # Audio bounds cover exactly the bare placeholder runs, in full-row coordinates.
    assert replay.audio_bounds[0] == [prompt_length + 1, prompt_length + 11]
    assert replay.audio_bounds[4] == [prompt_length + 4 * 13 + 1, prompt_length + 4 * 13 + 11]
    assert all(
        replay.response_ids[start - prompt_length : end - prompt_length]
        == [vocabulary.audio_placeholder_token_id] * (end - start)
        for start, end in replay.audio_bounds
    )
    # Captured logprobs ride the action positions only.
    assert replay.response_logprobs[prompt_length - prompt_length + 11] == pytest.approx(-0.2)


def test_replay_masks_forced_engine_units_and_replays_silence():
    replay = build_replay(_trajectory(engine_silence_units=1), _vocabulary())
    engine_layout = next(layout for layout in replay.unit_layouts if layout.origin == "engine")
    assert engine_layout.num_audio_tokens == 10
    assert np.all(replay.unit_pcm[engine_layout.unit_index] == 0.0)
    assert replay.response_mask[engine_layout.action_position] == 0
    # Five sampled decisions of six; the forced engine listen unit is masked out.
    assert sum(replay.response_mask) == 5


def test_extra_fields_roundtrip():
    replay = build_replay(_trajectory(), _vocabulary())
    payload = {DUPLEX_REPLAY_KEY: replay.to_extra_fields()}
    restored = DuplexReplay.from_extra_fields(payload[DUPLEX_REPLAY_KEY])
    assert restored.prompt_ids == replay.prompt_ids
    assert restored.response_ids == replay.response_ids
    assert restored.response_mask == replay.response_mask
    assert restored.audio_bounds == replay.audio_bounds
    assert [layout.action_position for layout in restored.unit_layouts] == [
        layout.action_position for layout in replay.unit_layouts
    ]
    assert all(np.array_equal(left, right) for left, right in zip(restored.unit_pcm, replay.unit_pcm, strict=True))
    assert restored.status == "completed"
    assert restored.event["axis"] == AXIS_TURN


def test_streaming_mel_extraction_and_actor_media():
    replay = build_replay(_trajectory(), _vocabulary())
    processor = FakeStreamingProcessor()
    features, lens = extract_unit_mels(processor, replay)
    assert len(features) == len(replay.unit_layouts)
    assert lens == [10] * 5
    assert processor.reset_calls[0] is True and not any(processor.reset_calls[1:])
    media = build_actor_media(replay, features, lens)
    assert media["audio_feature_lens"] == [[10] * 5]
    assert media["audio_bounds"] == [[list(span) for span in replay.audio_bounds]]
    assert len(media["audio_features"][0]) == 5


def test_actor_media_rejects_length_mismatch():
    replay = build_replay(_trajectory(), _vocabulary())
    features, lens = extract_unit_mels(FakeStreamingProcessor(), replay)
    lens = [value + 1 for value in lens]
    with pytest.raises(RecordSetError, match="cannot splice one-to-one"):
        build_actor_media(replay, features, lens)


def test_processor_without_streaming_api_fails_loudly():
    replay = build_replay(_trajectory(), _vocabulary())

    class Offline:
        sampling_rate = 16000

    with pytest.raises(RecordSetError, match="process_audio_streaming"):
        extract_unit_mels(Offline(), replay)


def test_sample_rate_mismatch_is_rejected():
    replay = build_replay(_trajectory(), _vocabulary())

    class WrongRate(FakeStreamingProcessor):
        sampling_rate = 44100

    with pytest.raises(RecordSetError, match="44100"):
        extract_unit_mels(WrongRate(), replay)


def test_custom_instructions_reach_the_context_prefix():
    replay = build_replay(_trajectory(), _vocabulary(), instructions="You are terse.")
    assert replay.instructions == "You are terse."
    decoded_ids = set(replay.prompt_ids)
    assert decoded_ids  # prefix present and encoded through the tokenizer
    default = build_replay(_trajectory(), _vocabulary())
    assert default.instructions == DEFAULT_SESSION_INSTRUCTIONS
    assert replay.prompt_ids != default.prompt_ids


def test_replay_skips_reemit_windows_and_keeps_chunk_alignment():
    from verl_omni.utils.dataset.duplex.trajectory import AXIS_PAUSE, DuplexEpisodeEvent, DuplexTrajectory

    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(1.0, 2.0)),
        record_set_wires=reemit_variant_wires(),
        client_audio_chunks=[np.zeros(16000, dtype=np.float32) for _ in range(4)],
    )
    replay = build_replay(trajectory, _vocabulary())
    # Four decision windows of 13 tokens; the reemit restatements open none.
    assert len(replay.response_ids) == 4 * 13
    assert len(replay.unit_layouts) == 4
    assert sum(replay.response_mask) == 4
