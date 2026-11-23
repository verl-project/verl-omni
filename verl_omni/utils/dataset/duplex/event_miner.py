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
"""Mine event-typed segments from DailyTalkContiguous for duplex GRPO training.

Implements the event-mining dataloader from verl-omni#905 §3.1: 3–5 s
pause/turn/interruption segments carved from the corpus's stereo two-track
audio and word-level alignments. The taxonomy follows Full-Duplex-Bench and
arXiv 2606.11167.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = [
    "EventSegment",
    "EventMiner",
    "SPEECH_SAMPLE_RATE",
]

SPEECH_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventSegment:
    """One event-typed training episode mined from a two-track dialogue.

    Attributes mirror the replay contract in verl-omni#905 §3.1: the user
    track is fed chunk-by-chunk at the model's 1 s cadence while the model
    generates concurrently; ``metadata`` carries per-axis timing ground truth
    for the rule rewards.
    """

    dialogue_id: str
    event_type: str  # "pause" | "turn" | "interruption"
    start_time: float  # event start, seconds from dialogue start
    end_time: float  # event end (exclusive), seconds
    user_audio: np.ndarray  # 16 kHz mono float32 the trainer feeds the model
    context_start: float  # beginning of the pre-event context window
    context_end: float  # end of the context window (= event start)
    context_audio: np.ndarray  # 16 kHz mono context fed before the event
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Alignment utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Word:
    text: str
    start: float
    end: float
    speaker: str


def _load_alignments(path: Path) -> list[_Word]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    words: list[_Word] = []
    for text, (start, end), speaker in payload["alignments"]:
        words.append(_Word(text=text, start=start, end=end, speaker=speaker))
    words.sort(key=lambda w: w.start)
    return words


def _track_to_mono(wav: np.ndarray, track: int) -> np.ndarray:
    """Extract one stereo track as mono float32."""
    if wav.ndim == 1:
        return wav.astype(np.float32)
    return wav[:, track].astype(np.float32)


def _slice_audio(audio: np.ndarray, start: float, end: float, sr: int = SPEECH_SAMPLE_RATE) -> np.ndarray:
    s = max(0, int(start * sr))
    e = min(audio.shape[-1], int(end * sr))
    return np.ascontiguousarray(audio[s:e], dtype=np.float32)


# ---------------------------------------------------------------------------
# Event mining
# ---------------------------------------------------------------------------


class EventMiner:
    """Carve pause / turn / interruption episodes from a two-track corpus.

    Parameters mirror verl-omni#905 §3.1:
      min_gap: minimum silence (s) that qualifies as a pause event.
      min_turn_gap: minimum inter-speaker gap (s) for a clean turn.
      overlap_threshold: minimum co-activity (s) for an interruption.
      segment_duration: target episode length (s); events are padded around.
      context_duration: pre-event audio (s) supplied as conversation history.
    """

    def __init__(
        self,
        min_gap: float = 1.0,
        min_turn_gap: float = 0.2,
        overlap_threshold: float = 0.2,
        segment_duration: float = 4.0,
        context_duration: float = 10.0,
        sample_rate: int = SPEECH_SAMPLE_RATE,
    ) -> None:
        self.min_gap = min_gap
        self.min_turn_gap = min_turn_gap
        self.overlap_threshold = overlap_threshold
        self.segment_duration = segment_duration
        self.context_duration = context_duration
        self.sample_rate = sample_rate

    # -- public API ---------------------------------------------------------

    def mine_dialogue(self, audio_path: Path, alignment_path: Path, dialogue_id: str = "") -> list[EventSegment]:
        """Mine all events from one dialogue's stereo WAV and alignments."""
        import soundfile as sf

        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sr} Hz in {audio_path}")
        words = _load_alignments(alignment_path)
        if not words:
            return []
        duration = audio_duration(wav, sr)

        # The "user" is the SPEAKER_MAIN track; the "partner" is the other.
        # In DailyTalkContiguous, SPEAKER_MAIN ↔ channel 0 by convention.
        user_channel = 0
        user_audio = _track_to_mono(wav, user_channel)

        segments: list[EventSegment] = []
        segments.extend(self._mine_pauses(words, user_audio, duration, dialogue_id))
        segments.extend(self._mine_turns(words, wav, duration, dialogue_id))
        segments.extend(self._mine_interruptions(words, wav, duration, dialogue_id))
        segments.sort(key=lambda s: s.start_time)
        return segments

    # -- pause mining -------------------------------------------------------

    def _mine_pauses(
        self, words: list[_Word], user_audio: np.ndarray, duration: float, dialogue_id: str
    ) -> list[EventSegment]:
        """Find gaps > min_gap between consecutive words from the user track."""
        user_words = [w for w in words if w.speaker == "SPEAKER_MAIN"]
        if len(user_words) < 2:
            return []
        out: list[EventSegment] = []
        for prev, nxt in zip(user_words, user_words[1:]):
            gap = nxt.start - prev.end
            if gap < self.min_gap:
                continue
            event_start = prev.end
            event_end = min(event_start + self.segment_duration, duration)
            if event_end - event_start < self.min_gap:
                continue
            seg = self._make_segment(
                dialogue_id, "pause", event_start, event_end, user_audio, duration,
                pause_duration=gap,
                preceding_word=prev.text,
                following_word=nxt.text,
            )
            out.append(seg)
        return out

    # -- turn mining --------------------------------------------------------

    def _mine_turns(
        self, words: list[_Word], wav: np.ndarray, duration: float, dialogue_id: str
    ) -> list[EventSegment]:
        """Find handoff points where the partner finishes and the user starts."""
        out: list[EventSegment] = []
        by_speaker: dict[str, list[_Word]] = {}
        for w in words:
            by_speaker.setdefault(w.speaker, []).append(w)
        speakers = list(by_speaker.keys())
        if len(speakers) < 2:
            return []
        partner_words = [w for w in words if w.speaker != "SPEAKER_MAIN"]
        if not partner_words:
            return []
        user_audio = _track_to_mono(wav, 0)
        for prev_partner in partner_words:
            for nxt_user in by_speaker.get("SPEAKER_MAIN", []):
                gap = nxt_user.start - prev_partner.end
                if gap < -self.overlap_threshold or gap > self.min_turn_gap + 1.0:
                    continue
                event_start = prev_partner.end - 0.5
                event_end = min(event_start + self.segment_duration, duration)
                if event_end <= event_start:
                    continue
                seg = self._make_segment(
                    dialogue_id, "turn", event_start, event_end, user_audio, duration,
                    turn_boundary=prev_partner.end,
                    user_speech_start=nxt_user.start,
                    gap=gap,
                )
                out.append(seg)
                break  # one event per partner word
        return out

    # -- interruption mining --------------------------------------------------

    def _mine_interruptions(
        self, words: list[_Word], wav: np.ndarray, duration: float, dialogue_id: str
    ) -> list[EventSegment]:
        """Find overlapping speech where the user starts while the partner talks.

        Interruptions are constructed by overlapping the two stereo tracks,
        with injection keyed to the model's would-be speak onset. Here we mine
        natural overlaps in the corpus; the training loop constructs synthetic
        ones by delaying the partner track (verl-omni#905 §3.1).
        """
        out: list[EventSegment] = []
        by_speaker: dict[str, list[_Word]] = {}
        for w in words:
            by_speaker.setdefault(w.speaker, []).append(w)
        speakers = list(by_speaker.keys())
        if len(speakers) < 2:
            return []
        # User = SPEAKER_MAIN (track 0); partner = track 1
        partner_audio = _track_to_mono(wav, 1)
        user_audio = _track_to_mono(wav, 0)
        for u in by_speaker.get("SPEAKER_MAIN", []):
            for p in by_speaker.get(speakers[1] if speakers[1] != "SPEAKER_MAIN" else speakers[0], []):
                overlap = min(u.end, p.end) - max(u.start, p.start)
                if overlap < self.overlap_threshold:
                    continue
                # Construct a mixed segment: both tracks playing
                mixed = user_audio + partner_audio  # both audible
                event_start = max(u.start, p.start) - 0.5
                event_end = min(event_start + self.segment_duration, duration)
                if event_end <= event_start:
                    continue
                seg = self._make_segment(
                    dialogue_id, "interruption", event_start, event_end, mixed, duration,
                    overlap_start=max(u.start, p.start),
                    overlap_end=min(u.end, p.end),
                    user_word=u.text,
                    partner_word=p.text,
                )
                out.append(seg)
                break  # one event per user word
        return out

    # -- helpers --------------------------------------------------------------

    def _make_segment(
        self,
        dialogue_id: str,
        event_type: str,
        event_start: float,
        event_end: float,
        audio: np.ndarray,
        duration: float,
        **metadata: Any,
    ) -> EventSegment:
        ctx_start = max(0.0, event_start - self.context_duration)
        ctx_end = event_start
        return EventSegment(
            dialogue_id=dialogue_id,
            event_type=event_type,
            start_time=event_start,
            end_time=event_end,
            user_audio=_slice_audio(audio, event_start, event_end, self.sample_rate),
            context_start=ctx_start,
            context_end=ctx_end,
            context_audio=_slice_audio(audio, ctx_start, ctx_end, self.sample_rate),
            metadata=metadata,
        )


def audio_duration(wav: np.ndarray, sample_rate: int) -> float:
    """Return duration in seconds."""
    return wav.shape[-1] / sample_rate
