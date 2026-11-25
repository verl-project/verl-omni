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
"""Event mining over a two-track corpus with word-level timestamps.

The training corpus (RFC verl-omni #905 §3.7: ``kyutai/DailyTalkContiguous``,
stereo two-track, word-level timestamps) is mined into short event-typed
segments — **pause** (user intra-utterance silence the model must sit out),
**turn** (user utterance ends; the model should take the floor) and
**interruption** (user speech starts while the model speaks; the model should
yield). Interruption segments are constructed by track overlap: the episode
feeds silence until the model starts speaking, then injects the user's
interrupting speech, so the injection point keys to the *policy's own* speak
onset and group members may see slightly different absolute timings — the
divergence the RFC accepts for replay construction.

Mining is pure timestamp arithmetic over a transcript manifest, so it runs on
CPU without the audio; the dataset resolves the audio at load time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from verl_omni.utils.dataset.duplex.trajectory import (
    AXIS_INTERRUPTION,
    AXIS_PAUSE,
    AXIS_TURN,
    DuplexEpisodeEvent,
)

__all__ = [
    "DuplexEpisodeSpec",
    "WordTimestamp",
    "episode_spec",
    "mine_episodes",
    "mine_interruption_candidates",
    "mine_pause_candidates",
    "mine_turn_candidates",
]


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """One word's time span on its speaker track."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class _SegmentCandidate:
    """One mined event window, in corpus-absolute seconds."""

    axis: str
    start_s: float
    end_s: float
    # Axis-specific anchors, corpus-absolute.
    pause_start_s: float | None = None
    pause_end_s: float | None = None
    turn_end_s: float | None = None
    # Interruption: the interrupting speech on the user track.
    interrupt_start_s: float | None = None
    interrupt_end_s: float | None = None
    # Delay from the live speak onset the driver injects at.
    inject_delay_s: float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DuplexEpisodeSpec:
    """One episode's dataset row: what to cut, what axis, how to drive.

    ``user_start_s`` / ``user_end_s`` bound the audio the rollout driver feeds
    (the user track never reacts to the policy); interruption episodes instead
    carry the interrupting speech span plus an injection delay relative to the
    observed speak onset, with pre-injection silence.
    """

    episode_id: str
    axis: str
    user_audio_ref: str
    start_s: float
    end_s: float
    event: DuplexEpisodeEvent
    interrupt_speech_s: tuple[float, float] | None = None
    inject_delay_s: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "episode_id": self.episode_id,
            "axis": self.axis,
            "user_audio_ref": self.user_audio_ref,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "event": {
                "axis": self.event.axis,
                "pause_window_s": self.event.pause_window_s,
                "user_turn_end_s": self.event.user_turn_end_s,
                "interrupt_speech_onset_s": self.event.interrupt_speech_onset_s,
                "interrupt_speech_end_s": self.event.interrupt_speech_end_s,
            },
            "provenance": dict(self.provenance),
        }
        if self.interrupt_speech_s is not None:
            payload["interrupt_speech_s"] = list(self.interrupt_speech_s)
        if self.inject_delay_s is not None:
            payload["inject_delay_s"] = self.inject_delay_s
        if not self.provenance:
            payload.pop("provenance", None)  # an empty struct breaks parquet round-trips
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DuplexEpisodeSpec:
        """Rebuild a spec from its parquet-stored dict (numpy-safe)."""

        def _pair(value: Any) -> tuple[float, float] | None:
            if value is None:
                return None
            values = list(value)
            return (float(values[0]), float(values[1])) if len(values) == 2 else None

        return cls(
            episode_id=str(payload["episode_id"]),
            axis=str(payload["axis"]),
            user_audio_ref=str(payload["user_audio_ref"]),
            start_s=float(payload["start_s"]),
            end_s=float(payload["end_s"]),
            event=DuplexEpisodeEvent.from_mapping(payload["event"]),
            interrupt_speech_s=_pair(payload.get("interrupt_speech_s")),
            inject_delay_s=(None if payload.get("inject_delay_s") is None else float(payload["inject_delay_s"])),
            provenance=dict(payload.get("provenance") or {}),
        )


def _utterances(words: Sequence[WordTimestamp], *, gap_s: float) -> list[list[WordTimestamp]]:
    """Split one track's words into utterances at silence gaps over ``gap_s``."""
    utterances: list[list[WordTimestamp]] = []
    for word in sorted(words, key=lambda item: item.start_s):
        if utterances and word.start_s - utterances[-1][-1].end_s <= gap_s:
            utterances[-1].append(word)
        else:
            utterances.append([word])
    return utterances


def mine_pause_candidates(
    user_words: Sequence[WordTimestamp],
    *,
    min_pause_s: float = 0.6,
    max_pause_s: float = 1.8,
    lead_s: float = 2.0,
    tail_s: float = 1.5,
    utterance_gap_s: float = 1.0,
) -> list[_SegmentCandidate]:
    """Pause windows: same-speaker silences of ``min_pause_s`` to ``max_pause_s``.

    Words are grouped into utterances at silences longer than
    ``max(utterance_gap_s, max_pause_s)``, so a pause always has the same
    speaker resuming after it; a longer silence is an utterance boundary
    instead, which the turn miner (splitting at ``utterance_gap_s``) labels a
    turn. A silence between ``utterance_gap_s`` and ``max_pause_s`` therefore
    mines both a pause (the speaker resumes) and a turn (the utterance ended).
    """
    candidates: list[_SegmentCandidate] = []
    for utterance in _utterances(user_words, gap_s=max(utterance_gap_s, max_pause_s)):
        for left, right in zip(utterance, utterance[1:], strict=False):
            silence = right.start_s - left.end_s
            if silence < min_pause_s or silence > max_pause_s:
                continue
            candidates.append(
                _SegmentCandidate(
                    axis=AXIS_PAUSE,
                    start_s=left.end_s - lead_s,
                    end_s=right.start_s + tail_s,
                    pause_start_s=left.end_s,
                    pause_end_s=right.start_s,
                )
            )
    return candidates


def mine_turn_candidates(
    user_words: Sequence[WordTimestamp],
    *,
    min_utterance_s: float = 1.0,
    lead_s: float = 3.0,
    tail_s: float = 2.0,
    utterance_gap_s: float = 1.0,
) -> list[_SegmentCandidate]:
    """Turn windows: the tail of each sufficiently long user utterance."""
    candidates: list[_SegmentCandidate] = []
    for utterance in _utterances(user_words, gap_s=utterance_gap_s):
        span = utterance[-1].end_s - utterance[0].start_s
        if span < min_utterance_s:
            continue
        turn_end = utterance[-1].end_s
        candidates.append(
            _SegmentCandidate(
                axis=AXIS_TURN,
                start_s=turn_end - lead_s,
                end_s=turn_end + tail_s,
                turn_end_s=turn_end,
            )
        )
    return candidates


def mine_interruption_candidates(
    user_words: Sequence[WordTimestamp],
    model_words: Sequence[WordTimestamp],
    *,
    min_interrupt_s: float = 1.5,
    inject_delay_s: float = 0.6,
    tail_s: float = 3.0,
    utterance_gap_s: float = 1.0,
) -> list[_SegmentCandidate]:
    """Interruption windows: user speech of at least ``min_interrupt_s`` timed
    against a model utterance.

    The corpus's model-track timing only decides *which* user utterance is a
    plausible interrupt (it overlaps a model utterance in the corpus); the
    live injection point is the policy's own speak onset plus
    ``inject_delay_s``, so the constructed overlap lands wherever the policy
    actually speaks. The clip must span ``min_interrupt_s`` so the policy has
    a measurable span to yield during.
    """
    candidates: list[_SegmentCandidate] = []
    model_utterances = _utterances(model_words, gap_s=utterance_gap_s)
    for utterance in _utterances(user_words, gap_s=utterance_gap_s):
        span = utterance[-1].end_s - utterance[0].start_s
        if span < min_interrupt_s:
            continue
        overlapping = any(
            utterance[0].start_s < model[-1].end_s and utterance[-1].end_s > model[0].start_s
            for model in model_utterances
        )
        if not overlapping:
            continue
        candidates.append(
            _SegmentCandidate(
                axis=AXIS_INTERRUPTION,
                start_s=utterance[0].start_s,
                end_s=utterance[-1].end_s + tail_s,
                interrupt_start_s=utterance[0].start_s,
                interrupt_end_s=utterance[-1].end_s,
                inject_delay_s=inject_delay_s,
            )
        )
    return candidates


def episode_spec(
    candidate: _SegmentCandidate,
    *,
    user_audio_ref: str,
    episode_id: str,
) -> DuplexEpisodeSpec:
    """Lift one candidate into a driver-facing episode spec (episode-relative windows)."""
    offset = candidate.start_s
    if candidate.axis == AXIS_PAUSE:
        event = DuplexEpisodeEvent(
            axis=AXIS_PAUSE,
            pause_window_s=(candidate.pause_start_s - offset, candidate.pause_end_s - offset),
        )
    elif candidate.axis == AXIS_TURN:
        event = DuplexEpisodeEvent(
            axis=AXIS_TURN,
            user_turn_end_s=candidate.turn_end_s - offset,
        )
    else:
        event = DuplexEpisodeEvent(
            axis=AXIS_INTERRUPTION,
            interrupt_speech_onset_s=_episode_relative(candidate.interrupt_start_s, offset),
            interrupt_speech_end_s=_episode_relative(candidate.interrupt_end_s, offset),
        )
    return DuplexEpisodeSpec(
        episode_id=episode_id,
        axis=candidate.axis,
        user_audio_ref=user_audio_ref,
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        event=event,
        interrupt_speech_s=(
            (candidate.interrupt_start_s, candidate.interrupt_end_s)
            if candidate.interrupt_start_s is not None
            else None
        ),
        inject_delay_s=candidate.inject_delay_s if candidate.axis == AXIS_INTERRUPTION else None,
        provenance=dict(candidate.provenance),
    )


def _episode_relative(anchor_s: float | None, offset_s: float) -> float | None:
    return None if anchor_s is None else anchor_s - offset_s


def mine_episodes(
    user_words: Sequence[WordTimestamp],
    model_words: Sequence[WordTimestamp],
    *,
    user_audio_ref: str,
    dialogue_id: str,
    axes: Sequence[str] = (AXIS_PAUSE, AXIS_TURN, AXIS_INTERRUPTION),
    max_window_s: float = 8.0,
    min_pause_s: float = 0.6,
    max_pause_s: float = 1.8,
    min_interrupt_s: float = 1.5,
) -> list[DuplexEpisodeSpec]:
    """Mine one dialogue's transcript into bounded episode specs.

    Args:
        user_words: The user track's word timestamps.
        model_words: The model track's word timestamps.
        user_audio_ref: Reference (path/URI) of the user track's audio.
        dialogue_id: Stable id prefix for the mined episodes.
        axes: Which axes to mine.
        max_window_s: Upper bound on an episode window — the RFC's bounded
            episodes (segment + up to 30 s context) are enforced at session
            admission; mining keeps the fed span itself short.
        min_pause_s: Shortest same-speaker silence mined as a pause.
        max_pause_s: Longest same-speaker silence mined as a pause.
        min_interrupt_s: Shortest user utterance mined as an interrupting clip.

    Returns:
        The dialogue's episode specs, ordered by start time.
    """
    candidates: list[_SegmentCandidate] = []
    if AXIS_PAUSE in axes:
        candidates.extend(mine_pause_candidates(user_words, min_pause_s=min_pause_s, max_pause_s=max_pause_s))
    if AXIS_TURN in axes:
        candidates.extend(mine_turn_candidates(user_words))
    if AXIS_INTERRUPTION in axes:
        candidates.extend(mine_interruption_candidates(user_words, model_words, min_interrupt_s=min_interrupt_s))
    specs: list[DuplexEpisodeSpec] = []
    for index, candidate in enumerate(sorted(candidates, key=lambda item: item.start_s)):
        if candidate.end_s - candidate.start_s > max_window_s:
            candidate = replace(candidate, end_s=candidate.start_s + max_window_s)
        specs.append(
            episode_spec(
                candidate,
                user_audio_ref=user_audio_ref,
                episode_id=f"{dialogue_id}-{candidate.axis}-{index:04d}",
            )
        )
    return specs
