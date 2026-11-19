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
"""Unit-window replay: rebuild the exact sequence the duplex engine ran.

The engine's MiniCPM-o 4.5 duplex data plane feeds, per 1 s unit,
``<unit>`` + ten bare audio-embedding placeholders, samples one action token,
and closes the unit (``terminator`` + ``</unit>``) when the next unit opens.
The trainer's actor must see that sequence token-for-token — the standard
processor rendering (``<|audio_start|> … <|audio_end|>`` marker spans) is a
*different* layout, so this module builds the engine layout directly and
carries explicit ``audio_bounds`` instead of marker-derived ones.

Engine-injected silent units (``origin=engine``) replay as one second of
digital silence: they are part of the input sequence the policy acted on.
Per-unit mel features are re-derived through the processor's own streaming
audio API (``process_audio_streaming``) — the same call the engine's data
plane makes — so parity holds by construction while the audio tower stays
frozen (RFC verl-omni #905 §3.2; the GPU parity gate closes the loop).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verl_omni.utils.dataset.duplex.records import (
    ORIGIN_ENGINE,
    RecordSetError,
)
from verl_omni.utils.dataset.duplex.trajectory import DuplexTrajectory

__all__ = [
    "DUPLEX_REPLAY_KEY",
    "DEFAULT_SESSION_INSTRUCTIONS",
    "DuplexReplay",
    "DuplexTokenVocabulary",
    "audio_tokens_for_samples",
    "build_replay",
    "extract_unit_mels",
]

#: Namespaced ``extra_fields`` key carrying one episode's replay payload
#: through the agent loop into the actor micro-batch (qwen3_tts pattern).
DUPLEX_REPLAY_KEY = "duplex_replay"

#: The duplex runtime's default system context (MiniCPMODuplex.prepare()).
DEFAULT_SESSION_INSTRUCTIONS = "Streaming Omni Conversation."

#: MiniCPM-o 4.5 duplex framing tokens (vLLM-Omni ``MiniCPMO45DuplexPolicy``).
_UNIT_TOKEN = "<unit>"
_UNIT_END_TOKEN = "</unit>"
_LISTEN_TOKEN = "<|listen|>"
_AUDIO_PLACEHOLDER_TOKEN = "<|audio|>"

#: 16 kHz stream, 160-sample mel hop, stride-2 CNN, stride-5 pooling.
_STREAM_SAMPLE_RATE = 16000
_MEL_HOP_SAMPLES = 160


def audio_tokens_for_samples(sample_count: int) -> int:
    """Audio embeddings per clip of ``sample_count`` PCM samples.

    Mirrors ``MiniCPMO45DuplexPolicy.audio_token_count``: a partial 160-sample
    mel frame counts, the stride-2 CNN rounds up, the stride-5 pooling rounds
    down. One exact duplex unit (16000 samples) is ten embeddings.
    """
    mel_frames = (max(0, int(sample_count)) + _MEL_HOP_SAMPLES - 1) // _MEL_HOP_SAMPLES
    cnn_frames = (mel_frames + 1) // 2
    return cnn_frames // 5


class DuplexTokenVocabulary:
    """The duplex framing token ids for one MiniCPM-o 4.5 tokenizer."""

    def __init__(self, tokenizer: Any):
        """Resolve framing token ids from ``tokenizer``.

        Args:
            tokenizer: The checkpoint's tokenizer (or the processor's).

        Raises:
            RecordSetError: When a framing token does not resolve to a single
                id — replaying with an unresolved token would silently build a
                mismatched context.
        """
        self._tokenizer = tokenizer
        self.unit_token_id = self._resolve(tokenizer, _UNIT_TOKEN)
        self.unit_end_token_id = self._resolve(tokenizer, _UNIT_END_TOKEN)
        self.listen_token_id = self._resolve(tokenizer, _LISTEN_TOKEN)
        self.audio_placeholder_token_id = self._resolve(tokenizer, _AUDIO_PLACEHOLDER_TOKEN)

    @staticmethod
    def _resolve(tokenizer: Any, token: str) -> int:
        value = tokenizer.convert_tokens_to_ids(token)
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else None
        unk_id = getattr(tokenizer, "unk_token_id", None)
        try:
            token_id = int(value) if value is not None else -1
        except (TypeError, ValueError):
            token_id = -1
        if token_id < 0 or token_id == unk_id:
            ids = tokenizer.encode(token, add_special_tokens=False)
            token_id = int(ids[0]) if len(ids) == 1 and int(ids[0]) != unk_id else -1
        if token_id < 0:
            raise RecordSetError(f"MiniCPM-o 4.5 tokenizer has no single id for duplex token {token!r}")
        return token_id

    def context_prefix_ids(self, instructions: str | None = None) -> list[int]:
        """The session's system-context prefix ids (MiniCPMODuplex.prepare())."""
        text = f"<|im_start|>system\n{instructions if instructions else DEFAULT_SESSION_INSTRUCTIONS}<|im_end|>"
        return list(self._tokenizer.encode(text, add_special_tokens=False))


def resolve_duplex_vocabulary(processor_or_tokenizer: Any) -> DuplexTokenVocabulary:
    """Resolve the framing vocabulary from a processor or a tokenizer."""
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    return DuplexTokenVocabulary(tokenizer)


@dataclass(frozen=True, slots=True)
class DuplexUnitLayout:
    """One unit's position map inside the replayed sequence."""

    unit_index: int
    origin: str
    #: Audio placeholder span as ``[start, end)`` over the full sequence row.
    audio_span: tuple[int, int]
    #: Absolute position of the unit's action token inside ``response_ids``
    #: (-1 for action-less units); audio spans stay in full-row coordinates.
    action_position: int
    num_audio_tokens: int
    speaking: bool
    time_s: float


@dataclass(slots=True)
class DuplexReplay:
    """The training-side replay of one duplex episode.

    ``prompt_ids`` is the session's system-context prefix; ``response_ids`` is
    the interleaved unit-window sequence after it. ``audio_bounds`` are
    ``[start, end)`` pairs over the concatenated row (prompt + response),
    covering the bare placeholder runs the audio tower outputs replace.
    """

    session_id: str
    seed: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    audio_bounds: list[list[int]]
    unit_layouts: list[DuplexUnitLayout]
    #: Per-unit PCM in unit order (client chunks and engine-injected silence).
    unit_pcm: list[np.ndarray]
    #: Captured rollout logprobs aligned with ``response_ids`` (0.0 at
    #: non-action and masked positions — the capture itself rides ``logprobs``).
    response_logprobs: list[float]
    event: Mapping[str, Any]
    status: str
    model_extension: Mapping[str, Any]
    sampling_params: tuple[Mapping[str, Any], ...]
    instructions: str = DEFAULT_SESSION_INSTRUCTIONS
    #: Per-unit mel features (re-derived through the processor's streaming
    #: API) and their row counts; attached by the rollout adapter before the
    #: actor consumes the replay.
    unit_mels: list | None = None
    unit_mel_lens: list[int] | None = None

    def action_token_ids(self) -> list[int]:
        return [
            layout_action
            for layout_action in (
                self.response_ids[layout.action_position] for layout in self.unit_layouts if layout.action_position >= 0
            )
        ]

    def to_extra_fields(self, *, include_pcm: bool = True) -> dict[str, Any]:
        """The agent-loop ``extra_fields`` payload for this replay.

        Args:
            include_pcm: Keep the per-unit PCM stream (debugging / parity
                dumps); the actor micro-batch payload drops it — the mel
                features and bounds are what training consumes.
        """
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "seed": self.seed,
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "audio_bounds": [list(span) for span in self.audio_bounds],
            "unit_layouts": [
                {
                    "unit_index": layout.unit_index,
                    "origin": layout.origin,
                    "audio_span": list(layout.audio_span),
                    "action_position": layout.action_position,
                    "num_audio_tokens": layout.num_audio_tokens,
                    "speaking": layout.speaking,
                    "time_s": layout.time_s,
                }
                for layout in self.unit_layouts
            ],
            "response_logprobs": list(self.response_logprobs),
            "event": dict(self.event),
            "status": self.status,
            "model": dict(self.model_extension),
            "sampling_params": [dict(params) for params in self.sampling_params],
            "instructions": self.instructions,
        }
        if include_pcm:
            payload["unit_pcm"] = [np.asarray(chunk, dtype=np.float32) for chunk in self.unit_pcm]
        if self.unit_mels is not None:
            payload["unit_mels"] = list(self.unit_mels)
            payload["unit_mel_lens"] = list(self.unit_mel_lens or [])
        return payload

    @classmethod
    def from_extra_fields(cls, payload: Mapping[str, Any]) -> DuplexReplay:
        return cls(
            session_id=payload["session_id"],
            seed=int(payload["seed"]),
            prompt_ids=list(payload["prompt_ids"]),
            response_ids=list(payload["response_ids"]),
            response_mask=list(payload["response_mask"]),
            audio_bounds=[list(span) for span in payload["audio_bounds"]],
            unit_layouts=[
                DuplexUnitLayout(
                    unit_index=int(layout["unit_index"]),
                    origin=str(layout["origin"]),
                    audio_span=(int(layout["audio_span"][0]), int(layout["audio_span"][1])),
                    action_position=int(layout["action_position"]),
                    num_audio_tokens=int(layout["num_audio_tokens"]),
                    speaking=bool(layout["speaking"]),
                    time_s=float(layout["time_s"]),
                )
                for layout in payload["unit_layouts"]
            ],
            unit_pcm=[np.asarray(chunk, dtype=np.float32) for chunk in payload.get("unit_pcm") or []],
            unit_mels=list(payload["unit_mels"]) if payload.get("unit_mels") is not None else None,
            unit_mel_lens=[int(value) for value in payload["unit_mel_lens"]] if payload.get("unit_mel_lens") else None,
            response_logprobs=[float(value) for value in payload["response_logprobs"]],
            event=dict(payload["event"]),
            status=str(payload["status"]),
            model_extension=dict(payload["model"]),
            sampling_params=tuple(dict(params) for params in payload["sampling_params"]),
            instructions=str(payload.get("instructions", DEFAULT_SESSION_INSTRUCTIONS)),
        )


def build_replay(
    trajectory: DuplexTrajectory,
    vocabulary: DuplexTokenVocabulary,
    *,
    instructions: str | None = None,
    sample_rate: int = _STREAM_SAMPLE_RATE,
) -> DuplexReplay:
    """Rebuild the engine's unit-window sequence for one trajectory.

    Args:
        trajectory: The assembled episode.
        vocabulary: The framing token vocabulary of the checkpoint.
        instructions: The session's system-prompt text (must equal the value
            the rollout driver opened the session with).
        sample_rate: The duplex stream's PCM sample rate.

    Returns:
        The replay with prompt/response ids, the validity mask over the
        response, explicit audio bounds, and the per-unit PCM stream.

    Raises:
        RecordSetError: On any layout disagreement between the trajectory's
            action stream and its unit windows.
    """
    prompt_ids = vocabulary.context_prefix_ids(instructions)
    prompt_length = len(prompt_ids)
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    audio_bounds: list[list[int]] = []
    unit_layouts: list[DuplexUnitLayout] = []
    unit_pcm: list[np.ndarray] = []
    decisions = trajectory.unit_decisions(listen_token_id=vocabulary.listen_token_id)
    client_chunks = iter(trajectory.client_audio_chunks)
    # The flat stream interleaves re-emitted restatements; decisions consume
    # only the non-reemitted actions, in order.
    flat_actions = iter(
        [
            (token_id, mask, logprob)
            for token_id, mask, logprob, kind in zip(
                trajectory.token_ids,
                trajectory.action_mask,
                trajectory.logprobs,
                trajectory.action_kinds,
                strict=True,
            )
            if kind != "reemit"
        ]
    )

    for decision in decisions:
        if decision.origin == ORIGIN_ENGINE:
            pcm = np.zeros(sample_rate, dtype=np.float32)
        else:
            pcm = next(client_chunks)
        num_audio_tokens = audio_tokens_for_samples(pcm.size)
        if num_audio_tokens <= 0:
            raise RecordSetError(
                f"duplex episode {trajectory.session_id}: unit {decision.unit_index} carries no audio embeddings"
            )
        response_ids.append(vocabulary.unit_token_id)
        audio_start = prompt_length + len(response_ids)
        response_ids.extend([vocabulary.audio_placeholder_token_id] * num_audio_tokens)
        audio_bounds.append([audio_start, audio_start + num_audio_tokens])
        action_position = len(response_ids)
        flat_token_id, action_mask_value, action_logprob = next(flat_actions)
        if flat_token_id != decision.action_token_id:
            raise RecordSetError(
                f"duplex episode {trajectory.session_id}: unit {decision.unit_index}'s decision token "
                f"{decision.action_token_id} disagrees with the flat stream token {flat_token_id}"
            )
        response_ids.append(decision.action_token_id)
        response_ids.append(vocabulary.unit_end_token_id)
        # Only the action-token position can carry mask 1; framing and media
        # positions are structural context, never loss positions.
        response_mask.extend([0] * (1 + num_audio_tokens))
        response_mask.append(action_mask_value)
        response_mask.append(0)
        response_logprobs.extend([0.0] * (1 + num_audio_tokens))
        response_logprobs.append(action_logprob or 0.0)
        response_logprobs.append(0.0)
        unit_pcm.append(pcm)
        unit_layouts.append(
            DuplexUnitLayout(
                unit_index=decision.unit_index,
                origin=decision.origin,
                audio_span=(audio_start, audio_start + num_audio_tokens),
                action_position=action_position,
                num_audio_tokens=num_audio_tokens,
                speaking=decision.speaking,
                time_s=decision.time_s,
            )
        )
    if any(True for _ in flat_actions):
        raise RecordSetError(
            f"duplex episode {trajectory.session_id}: the flat action stream carries more decisions than "
            "the unit windows; the trajectory columns disagree"
        )
    sampling_params = tuple(params for record_set in trajectory.record_sets for params in record_set.sampling_params)
    model_extension = trajectory.record_sets[-1].model if trajectory.record_sets else {}
    return DuplexReplay(
        session_id=trajectory.session_id,
        seed=trajectory.seed,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        audio_bounds=audio_bounds,
        unit_layouts=unit_layouts,
        unit_pcm=unit_pcm,
        response_logprobs=response_logprobs,
        event={
            "axis": trajectory.event.axis,
            "pause_window_s": trajectory.event.pause_window_s,
            "user_turn_end_s": trajectory.event.user_turn_end_s,
            "interrupt_speech_end_s": trajectory.event.interrupt_speech_end_s,
        },
        status=trajectory.status,
        model_extension=model_extension,
        sampling_params=sampling_params,
        instructions=instructions or DEFAULT_SESSION_INSTRUCTIONS,
    )


def extract_unit_mels(
    processor: Any,
    replay: DuplexReplay,
    *,
    sample_rate: int = _STREAM_SAMPLE_RATE,
) -> tuple[list[Any], list[int]]:
    """Re-derive per-unit mel features through the processor's streaming API.

    Drives the same ``process_audio_streaming(chunk, reset=...)`` call the
    engine's duplex data plane makes, one unit at a time in stream order, so
    the frozen audio tower sees the same features on both sides.

    Args:
        processor: The checkpoint's processor (must expose
            ``process_audio_streaming``).
        replay: The built replay (its ``unit_pcm`` is the stream).
        sample_rate: The stream's sample rate (validated against the
            processor's when it declares one).

    Returns:
        ``(features, lens)``: one mel feature tensor per unit and its length.

    Raises:
        RecordSetError: When the processor lacks the streaming audio API or a
            unit yields no features.
    """
    process = getattr(processor, "process_audio_streaming", None)
    if not callable(process):
        raise RecordSetError(
            "the MiniCPM-o 4.5 processor does not expose process_audio_streaming(); duplex replay "
            "cannot re-derive streaming mel features — check the checkpoint's remote code"
        )
    declared_rate = getattr(processor, "sampling_rate", None) or getattr(
        getattr(processor, "feature_extractor", None), "sampling_rate", None
    )
    if isinstance(declared_rate, int) and declared_rate != sample_rate:
        raise RecordSetError(f"duplex replay streams at {sample_rate} Hz but the processor expects {declared_rate} Hz")
    features: list[Any] = []
    lens: list[int] = []
    for unit_index, pcm in enumerate(replay.unit_pcm):
        try:
            batch_feature = process(pcm, reset=unit_index == 0, return_batch_feature=True)
        except TypeError:
            batch_feature = process(pcm, chunk_idx=unit_index)
        unit_features = getattr(batch_feature, "audio_features", None)
        if unit_features is None and isinstance(batch_feature, Mapping):
            unit_features = batch_feature.get("audio_features")
        if unit_features is None:
            raise RecordSetError(f"duplex replay unit {unit_index} of {replay.session_id} produced no audio features")
        feature = unit_features[0] if isinstance(unit_features, list | tuple) and len(unit_features) else unit_features
        feature = np.asarray(feature) if not hasattr(feature, "shape") else feature
        if getattr(feature, "size", 0) == 0:
            raise RecordSetError(f"duplex replay unit {unit_index} of {replay.session_id} produced empty features")
        features.append(feature)
        lens.append(int(feature.shape[0]))
    return features, lens


def build_actor_media(replay: DuplexReplay, features: Sequence[Any], lens: Sequence[int]) -> dict[str, Any]:
    """Assemble the actor's media inputs for one replayed episode.

    Returns the per-sample containers ``MiniCPMOThinkerAdapter``'s duplex
    branch merges into the model call: one audio feature clip per unit with
    explicit ``audio_bounds`` in full-row coordinates.
    """
    if len(features) != len(replay.unit_layouts) or len(lens) != len(replay.unit_layouts):
        raise RecordSetError(
            f"duplex replay of {replay.session_id} has {len(replay.unit_layouts)} units but "
            f"{len(features)} feature clips"
        )
    for layout, length in zip(replay.unit_layouts, lens, strict=True):
        if length != layout.num_audio_tokens:
            raise RecordSetError(
                f"duplex replay unit {layout.unit_index} of {replay.session_id}: {length} audio feature rows "
                f"but {layout.num_audio_tokens} placeholder tokens — the tower output cannot splice one-to-one"
            )
    return {
        "audio_features": [list(features)],
        "audio_feature_lens": [[int(value) for value in lens]],
        "audio_bounds": [[list(span) for span in replay.audio_bounds]],
    }
