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
    DEFAULT_REF_AUDIO_MODEL_PATH_ENV,
    DEFAULT_SESSION_INSTRUCTIONS,
    DUPLEX_REPLAY_KEY,
    DuplexRefAudio,
    DuplexReplay,
    DuplexTokenVocabulary,
    audio_tokens_for_mel_frames,
    audio_tokens_for_samples,
    build_actor_media,
    build_replay,
    extract_unit_mels,
    resolve_default_ref_audio,
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
    "<|audio_start|>": 9108,
    "<|audio_end|>": 9109,
    "You": 9201,
    "are": 9202,
    "terse.": 9203,
}

#: The bundled ``assets/HT_ref_audio.wav`` of MiniCPM-o 4.5: 6.016 s at 16 kHz.
_REF_SAMPLES = 96256
_HOP = 160


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


class FakeMelProcessor:
    """The exact streaming mel processor's chunk geometry (``StreamingMelProcessorExact``)."""

    sample_rate = 16000

    def __init__(self):
        self.chunk_samples = 1000 * 16000 // 1000
        self.first_chunk_samples = max(_HOP, (int(1035 * 16000 / 1000) // _HOP) * _HOP)
        self.is_first = True
        self.resets = 0

    def get_chunk_size(self) -> int:
        return self.first_chunk_samples if self.is_first else self.chunk_samples

    def get_config(self) -> dict:
        return {"effective_first_chunk_ms": self.first_chunk_samples / self.sample_rate * 1000.0}

    def reset(self):
        self.resets += 1


class FakeStreamingProcessor:
    """Mirrors the checkpoint processor's audio API shapes.

    ``process_audio_streaming(..., return_batch_feature=True)`` returns
    ``audio_features`` ``[1, 80, frames]`` and ``audio_feature_lens``
    ``[tensor([frames])]`` in mel frames; ``process_audio([clip])`` returns
    the offline ``[1, 80, frames]`` form; ``set_streaming_mode`` rebuilds the
    streaming mel processor on the processor's own extractor, as the remote
    code does. Every call is recorded.
    """

    sampling_rate = 16000

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.audio_processor = object()
        self._streaming_mel_processor = FakeMelProcessor()
        self.streaming_calls: list[tuple[int, bool, bool]] = []
        self.processed_chunks: list[np.ndarray] = []
        self.set_streaming_mode_calls: list[dict] = []
        self.reset_streaming_calls: list[int] = []
        self.process_audio_calls: list[np.ndarray] = []

    def set_streaming_mode(self, **kwargs):
        self.set_streaming_mode_calls.append(dict(kwargs))
        self._streaming_mel_processor = FakeMelProcessor()

    def reset_streaming(self):
        self.reset_streaming_calls.append(len(self.streaming_calls))
        self._streaming_mel_processor.reset()

    def get_streaming_chunk_size(self) -> int:
        return self._streaming_mel_processor.get_chunk_size()

    def process_audio_streaming(self, audio_chunk, reset=False, return_batch_feature=False, is_last_chunk=False):
        self.streaming_calls.append((len(audio_chunk), reset, return_batch_feature))
        self.processed_chunks.append(np.array(audio_chunk, dtype=np.float32, copy=True))
        frames = len(audio_chunk) // _HOP + 1
        self._streaming_mel_processor.is_first = False
        return {
            "audio_features": np.ones((1, 80, frames), dtype=np.float32) * len(audio_chunk),
            "audio_feature_lens": [np.array([frames])],
        }

    def process_audio(self, audios):
        (clip,) = audios
        self.process_audio_calls.append(np.asarray(clip))
        frames = len(clip) // _HOP + 1
        return {
            "audio_features": np.full((1, 80, frames), -1.0, dtype=np.float32),
            "audio_feature_lens": [np.array([frames])],
        }


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


def _ref_audio(samples: int = _REF_SAMPLES, rows: int | None = None) -> DuplexRefAudio:
    pcm = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    return DuplexRefAudio(
        pcm=pcm,
        sample_rate=16000,
        num_audio_tokens=rows if rows is not None else audio_tokens_for_samples(samples),
        source="test",
    )


def test_audio_token_count_matches_the_unit_cadence():
    assert audio_tokens_for_samples(16000) == 10
    assert audio_tokens_for_samples(0) == 0
    assert audio_tokens_for_samples(160) == 0  # mel frame, cnn rounds to 1, pooling floors
    # The bundled reference clip: 6.016 s → 602 mel → 301 CNN → 60 rows, never a constant.
    assert audio_tokens_for_samples(_REF_SAMPLES) == 60


def test_mel_frame_pooling_matches_the_checkpoint_tower():
    assert audio_tokens_for_mel_frames(602) == 60
    assert audio_tokens_for_mel_frames(104) == 10  # a 1 s unit with CNN redundancy frames
    assert audio_tokens_for_mel_frames(101) == 10
    assert audio_tokens_for_mel_frames(10) == 1
    assert audio_tokens_for_mel_frames(8) == 0
    assert audio_tokens_for_mel_frames(0) == 0


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
    assert replay.response_logprobs[11] == pytest.approx(-0.2)
    assert replay.ref_audio is None


def test_replay_masks_forced_engine_units_and_replays_silence():
    replay = build_replay(_trajectory(engine_silence_units=1), _vocabulary(), ref_audio=_ref_audio())
    engine_layout = next(layout for layout in replay.unit_layouts if layout.origin == "engine")
    assert engine_layout.num_audio_tokens == 10
    assert np.all(replay.unit_pcm[engine_layout.unit_index] == 0.0)
    assert replay.response_mask[engine_layout.action_position] == 0
    # Five sampled decisions of six; the forced engine listen unit is masked out.
    assert sum(replay.response_mask) == 5
    # The reference block leads the bounds; the engine unit's span shifts with the rest.
    assert replay.audio_bounds[0] == [8, 68]
    bare = build_replay(_trajectory(engine_silence_units=1), _vocabulary())
    shift = len(replay.prompt_ids) - len(bare.prompt_ids)
    assert replay.audio_bounds[1:] == [[start + shift, end + shift] for start, end in bare.audio_bounds]


def test_reference_audio_rides_the_context_and_the_first_span():
    vocabulary = _vocabulary()
    ref = _ref_audio()
    assert ref.num_audio_tokens == 60
    replay = build_replay(_trajectory(), vocabulary, ref_audio=ref)
    # session_context_texts: system text + <|audio_start|>, 60 <unit> rows, <|audio_end|><|im_end|>.
    assert replay.prompt_ids[:8] == [
        _TOKEN_TABLE["<|im_start|>"],
        _TOKEN_TABLE["system"],
        _TOKEN_TABLE["\n"],
        _TOKEN_TABLE["Streaming"],
        _TOKEN_TABLE["Omni"],
        _TOKEN_TABLE["Conversation."],
        _TOKEN_TABLE["\n"],
        _TOKEN_TABLE["<|audio_start|>"],
    ]
    assert replay.prompt_ids[8:68] == [vocabulary.unit_token_id] * 60
    assert replay.prompt_ids[68:] == [_TOKEN_TABLE["<|audio_end|>"], _TOKEN_TABLE["<|im_end|>"]]
    assert len(replay.prompt_ids) == 70  # the engine's context size
    assert replay.ref_audio is ref
    # The reference block is the first span; every unit span shifts by the whole prompt.
    assert replay.audio_bounds[0] == [8, 68]
    bare = build_replay(_trajectory(), vocabulary)
    shift = len(replay.prompt_ids) - len(bare.prompt_ids)
    assert shift == 63
    assert replay.audio_bounds[1:] == [[start + shift, end + shift] for start, end in bare.audio_bounds]
    assert replay.response_ids == bare.response_ids
    assert [layout.action_position for layout in replay.unit_layouts] == [
        layout.action_position for layout in bare.unit_layouts
    ]
    assert replay.prompt_ids == vocabulary.context_prefix_ids(None, ref_audio_rows=60)
    assert vocabulary.context_prefix_ids(None) == bare.prompt_ids


def test_reference_audio_mel_rides_the_offline_front_end_as_the_first_clip():
    replay = build_replay(_trajectory(), _vocabulary(), ref_audio=_ref_audio())
    processor = FakeStreamingProcessor()
    features, lens = extract_unit_mels(processor, replay)
    assert lens == [602, 104, 101, 101, 101, 101]
    assert len(processor.process_audio_calls) == 1
    np.testing.assert_array_equal(processor.process_audio_calls[0], replay.ref_audio.pcm)
    assert features[0].shape == (80, 602) and features[0].dtype == np.float32
    assert float(features[0][0, 0]) == -1.0  # the offline clip, not a unit's
    assert float(features[1][0, 0]) == 16480  # the first unit's chunk: 560-sample pre-roll + 15920 samples
    media = build_actor_media(replay, features, lens)
    assert media["audio_bounds"][0][0] == [8, 68]
    assert media["audio_feature_lens"][0][0] == 602  # mel frames; the tower pools them to the 60-row span
    assert audio_tokens_for_mel_frames(media["audio_feature_lens"][0][0]) == 60
    assert len(media["audio_features"][0]) == len(replay.audio_bounds) == 6


def test_resolve_default_ref_audio_mirrors_the_engine_asset_rule(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    model_dir = tmp_path / "snapshot"
    (model_dir / "assets").mkdir(parents=True)
    wav_path = model_dir / "assets" / "HT_ref_audio.wav"
    soundfile.write(str(wav_path), np.zeros((_REF_SAMPLES, 2), dtype=np.float32) + np.array([[0.25, 0.75]]), 16000)
    ref = resolve_default_ref_audio(str(model_dir))
    assert ref is not None
    assert ref.pcm.dtype == np.float32 and ref.pcm.shape == (_REF_SAMPLES,)  # channels averaged to mono
    assert np.allclose(ref.pcm, 0.5)
    assert ref.sample_rate == 16000
    assert ref.num_audio_tokens == 60
    assert ref.source == str(wav_path)
    # A snapshot without the asset resolves to None — the engine's ref_audio_required case.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert resolve_default_ref_audio(str(bare)) is None
    assert resolve_default_ref_audio(None) is None
    with pytest.raises(RecordSetError, match="bundles no assets/HT_ref_audio.wav"):
        build_replay(_trajectory(), _vocabulary(), model_path=str(bare))
    # The model-path parameter threads the asset into the context.
    replay = build_replay(_trajectory(), _vocabulary(), model_path=str(model_dir))
    assert len(replay.prompt_ids) == 70 and replay.audio_bounds[0] == [8, 68]


def test_resampled_reference_clip_follows_the_engine_normalization(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("torchaudio")
    model_dir = tmp_path / "snapshot"
    (model_dir / "assets").mkdir(parents=True)
    # 6.016 s at 8 kHz: half the samples, resampled to 16 kHz on resolution.
    soundfile.write(str(model_dir / "assets" / "HT_ref_audio.wav"), np.zeros(_REF_SAMPLES // 2, dtype=np.float32), 8000)
    ref = resolve_default_ref_audio(str(model_dir))
    assert ref is not None
    assert ref.sample_rate == 16000
    assert ref.pcm.size == pytest.approx(_REF_SAMPLES, abs=64)
    assert ref.num_audio_tokens == audio_tokens_for_samples(ref.pcm.size)
    assert ref.num_audio_tokens in (59, 60, 61)


def test_model_path_env_override_embeds_the_default_reference(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")
    model_dir = tmp_path / "snapshot"
    (model_dir / "assets").mkdir(parents=True)
    soundfile.write(str(model_dir / "assets" / "HT_ref_audio.wav"), np.zeros(_REF_SAMPLES, dtype=np.float32), 16000)
    monkeypatch.setenv(DEFAULT_REF_AUDIO_MODEL_PATH_ENV, str(model_dir))
    replay = build_replay(_trajectory(), _vocabulary())
    assert replay.ref_audio is not None and replay.ref_audio.num_audio_tokens == 60
    assert replay.audio_bounds[0] == [8, 68]
    monkeypatch.delenv(DEFAULT_REF_AUDIO_MODEL_PATH_ENV)
    assert build_replay(_trajectory(), _vocabulary()).ref_audio is None


def test_reference_clip_must_match_its_declared_rows():
    with pytest.raises(RecordSetError, match="pools to 10 tower rows"):
        build_replay(_trajectory(), _vocabulary(), ref_audio=_ref_audio(samples=16000, rows=60))
    with pytest.raises(RecordSetError, match="12000 Hz"):
        build_replay(_trajectory(), _vocabulary(), ref_audio=DuplexRefAudio(np.zeros(16000), 12000, 10))
    with pytest.raises(RecordSetError, match="at least one <unit> row"):
        _vocabulary().context_prefix_ids(None, ref_audio_rows=0)


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
    assert restored.ref_audio is None


def test_extra_fields_carry_the_reference_block():
    replay = build_replay(_trajectory(), _vocabulary(), ref_audio=_ref_audio())
    full = DuplexReplay.from_extra_fields(replay.to_extra_fields())
    assert full.ref_audio is not None and full.ref_audio.num_audio_tokens == 60
    np.testing.assert_array_equal(full.ref_audio.pcm, replay.ref_audio.pcm)
    assert full.ref_audio.source == "test"
    lean = DuplexReplay.from_extra_fields(replay.to_extra_fields(include_pcm=False))
    assert lean.ref_audio is not None and lean.ref_audio.pcm is None
    assert lean.audio_bounds[0] == [8, 68]
    with pytest.raises(RecordSetError, match="extract its mel features before dropping PCM"):
        extract_unit_mels(FakeStreamingProcessor(), lean)


def test_streaming_mel_extraction_and_actor_media():
    replay = build_replay(_trajectory(), _vocabulary())
    processor = FakeStreamingProcessor()
    features, lens = extract_unit_mels(processor, replay)
    assert len(features) == len(replay.unit_layouts)
    # Mel frames per clip: the first unit's chunk is 16480 samples, later ones 16000.
    assert lens == [104, 101, 101, 101, 101]
    assert all(clip.shape == (80, frames) for clip, frames in zip(features, lens, strict=True))
    media = build_actor_media(replay, features, lens)
    assert media["audio_feature_lens"] == [lens]
    assert media["audio_bounds"] == [[list(span) for span in replay.audio_bounds]]
    assert len(media["audio_features"][0]) == 5
    assert all(audio_tokens_for_mel_frames(frames) == 10 for frames in lens)


def test_streaming_call_sequence_mirrors_the_engine_session():
    replay = build_replay(_trajectory(), _vocabulary())
    shared = FakeStreamingProcessor()
    shared_mel = shared._streaming_mel_processor
    shared_extractor = shared.audio_processor
    extract_unit_mels(shared, replay)
    # Configured exactly once per episode, as the engine's per-session copy.
    assert shared.set_streaming_mode_calls == [
        {
            "mode": "exact",
            "chunk_ms": 1000,
            "first_chunk_ms": 1035,
            "cnn_redundancy_ms": 20,
            "enable_sliding_window": True,
            "slide_trigger_seconds": 30.0,
            "slide_stride_seconds": 10.0,
        }
    ]
    assert shared.reset_streaming_calls == [0]  # once, before any chunk
    # Every chunk: reset=False, return_batch_feature=True — the reset is the session's, never a unit's.
    assert [call[1:] for call in shared.streaming_calls] == [(False, True)] * 5
    # Ingestion geometry: the 1035 ms first chunk (hop-aligned 16480 samples, zero pre-roll),
    # then one 16000-sample chunk per unit off the concatenated stream.
    assert [call[0] for call in shared.streaming_calls] == [16480, 16000, 16000, 16000, 16000]
    first_unit = replay.unit_pcm[0]
    first_chunk = shared.processed_chunks[0]
    assert np.all(first_chunk[:560] == 0.0)
    np.testing.assert_array_equal(first_chunk[560:], first_unit[:15920])
    np.testing.assert_array_equal(shared.processed_chunks[1][:80], first_unit[15920:])
    # The shared processor keeps its own extractor and mel processor: the copy's are new objects.
    assert shared._streaming_mel_processor is shared_mel
    assert shared.audio_processor is shared_extractor


def test_signature_mismatch_fails_loudly():
    replay = build_replay(_trajectory(), _vocabulary())

    class Legacy(FakeStreamingProcessor):
        def process_audio_streaming(self, audio_chunk, chunk_idx=None):  # no reset=/return_batch_feature=
            return super().process_audio_streaming(audio_chunk)

    with pytest.raises(TypeError):
        extract_unit_mels(Legacy(), replay)


def test_short_stream_is_loud():
    replay = build_replay(_trajectory(), _vocabulary())
    replay.unit_pcm[-1] = replay.unit_pcm[-1][:8000]  # a unit shorter than its processor chunk
    with pytest.raises(RecordSetError, match="audio not enough"):
        extract_unit_mels(FakeStreamingProcessor(), replay)


def test_actor_media_rejects_length_mismatch():
    replay = build_replay(_trajectory(), _vocabulary())
    features, lens = extract_unit_mels(FakeStreamingProcessor(), replay)
    lens[1] -= 10  # 91 frames pool to 9 rows, not the span's 10
    with pytest.raises(RecordSetError, match="cannot splice one-to-one"):
        build_actor_media(replay, features, lens)
    with pytest.raises(RecordSetError, match="audio spans"):
        build_actor_media(replay, features[:-1], lens[:-1])


def test_reference_front_end_must_pool_to_the_placeholders():
    replay = build_replay(_trajectory(), _vocabulary(), ref_audio=_ref_audio())

    class ShortFrames(FakeStreamingProcessor):
        def process_audio(self, audios):
            return {"audio_features": np.ones((1, 80, 300), dtype=np.float32), "audio_feature_lens": [np.array([300])]}

    with pytest.raises(RecordSetError, match="60 <unit> placeholders"):
        extract_unit_mels(ShortFrames(), replay)

    class NoOffline(FakeStreamingProcessor):
        process_audio = None

    with pytest.raises(RecordSetError, match="process_audio"):
        extract_unit_mels(NoOffline(), replay)


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
    terse = build_replay(_trajectory(), _vocabulary(), instructions="You are terse.", ref_audio=_ref_audio())
    assert terse.prompt_ids[:3] == [_TOKEN_TABLE["<|im_start|>"], _TOKEN_TABLE["system"], _TOKEN_TABLE["\n"]]
    assert _TOKEN_TABLE["terse."] in terse.prompt_ids[:8]
    assert terse.prompt_ids[8:68] == [_TOKEN_TABLE["<unit>"]] * 60


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
