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

The engine's MiniCPM-o 4.5 duplex data plane opens every audio session with
the system prompt and the model-bundled default reference audio (E15:
``assets/HT_ref_audio.wav`` between ``<|audio_start|>`` and ``<|audio_end|>``,
one ``<unit>`` placeholder per tower row), then feeds, per 1 s unit,
``<unit>`` + ten bare audio-embedding placeholders, samples one action token,
and closes the unit (``terminator`` + ``</unit>``) when the next unit opens.
The trainer's actor must see that sequence token-for-token — the standard
processor rendering (``<|audio_start|> … <|audio_end|>`` marker spans around
the units) is a *different* layout, so this module builds the engine layout
directly and carries explicit ``audio_bounds`` instead of marker-derived ones.

Engine-injected silent units (``origin=engine``) replay as one second of
digital silence: they are part of the input sequence the policy acted on.
Mel features are re-derived through the processor's own audio API — the
reference clip through the offline front end (``process_audio``), the units
through the streaming API driven with the engine's per-session processor
configuration and ingestion geometry — so parity holds by construction while
the audio tower stays frozen (RFC verl-omni #905 §3.2; the GPU parity gate
closes the loop).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verl_omni.utils.dataset.duplex.records import (
    ORIGIN_ENGINE,
    RecordSetError,
)
from verl_omni.utils.dataset.duplex.trajectory import DuplexTrajectory

__all__ = [
    "DEFAULT_REF_AUDIO_MODEL_PATH_ENV",
    "DEFAULT_SESSION_INSTRUCTIONS",
    "DUPLEX_REPLAY_KEY",
    "DuplexReplay",
    "DuplexRefAudio",
    "DuplexTokenVocabulary",
    "audio_tokens_for_mel_frames",
    "audio_tokens_for_samples",
    "build_actor_media",
    "build_replay",
    "extract_unit_mels",
    "resolve_default_ref_audio",
]

#: Namespaced ``extra_fields`` key carrying one episode's replay payload
#: through the agent loop into the actor micro-batch (qwen3_tts pattern).
DUPLEX_REPLAY_KEY = "duplex_replay"

#: The duplex runtime's default system context (MiniCPMODuplex.prepare()).
DEFAULT_SESSION_INSTRUCTIONS = "Streaming Omni Conversation."

#: Model snapshot the bundled reference audio resolves from when
#: :func:`build_replay` gets no ``model_path``; unset builds without a
#: reference block.
DEFAULT_REF_AUDIO_MODEL_PATH_ENV = "VERL_OMNI_DUPLEX_REF_AUDIO_MODEL_PATH"

#: MiniCPM-o 4.5 duplex framing tokens (vLLM-Omni ``MiniCPMO45DuplexPolicy``).
_UNIT_TOKEN = "<unit>"
_UNIT_END_TOKEN = "</unit>"
_LISTEN_TOKEN = "<|listen|>"
_AUDIO_PLACEHOLDER_TOKEN = "<|audio|>"

#: 16 kHz stream, 160-sample mel hop, stride-2 CNN, stride-5 pooling.
_STREAM_SAMPLE_RATE = 16000
_MEL_HOP_SAMPLES = 160
_AUDIO_POOL_STEP = 5

#: The prompt WAV the engine embeds in every audio duplex session
#: (``resolve_default_ref_audio`` → ``<model snapshot>/assets/HT_ref_audio.wav``).
_REF_AUDIO_RELPATH = ("assets", "HT_ref_audio.wav")

#: Streaming-tower configuration the engine applies to its per-session
#: processor copy (vLLM-Omni ``MiniCPMO45Stage0DuplexRuntime._configure_streaming_processor``).
_STREAMING_CHUNK_MS = 1000
_STREAMING_FIRST_CHUNK_MS = 1035
_STREAMING_CNN_REDUNDANCY_MS = 20
_STREAMING_SLIDE_TRIGGER_SECONDS = 30.0
_STREAMING_SLIDE_STRIDE_SECONDS = 10.0


def audio_tokens_for_samples(sample_count: int) -> int:
    """Audio embeddings per clip of ``sample_count`` PCM samples.

    Mirrors ``MiniCPMO45DuplexPolicy.audio_token_count``: a partial 160-sample
    mel frame counts, the stride-2 CNN rounds up, the stride-5 pooling rounds
    down. One exact duplex unit (16000 samples) is ten embeddings.
    """
    mel_frames = (max(0, int(sample_count)) + _MEL_HOP_SAMPLES - 1) // _MEL_HOP_SAMPLES
    cnn_frames = (mel_frames + 1) // 2
    return cnn_frames // _AUDIO_POOL_STEP


def audio_tokens_for_mel_frames(mel_frames: int) -> int:
    """Audio embeddings per clip of ``mel_frames`` mel frames.

    Mirrors the checkpoint's ``_get_feat_extract_output_lengths`` — the count
    the actor's frozen tower pools a mel clip to, i.e. the placeholder span it
    splices into.
    """
    cnn_frames = (max(0, int(mel_frames)) - 1) // 2 + 1
    return (cnn_frames - _AUDIO_POOL_STEP) // _AUDIO_POOL_STEP + 1


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

    def context_layout(
        self,
        instructions: str | None = None,
        *,
        ref_audio_rows: int | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """``(prefix_ids, reference_ids, suffix_ids)`` of the session context.

        Mirrors ``MiniCPMO45DuplexPolicy.session_context_texts`` as
        ``_prepare_session_context`` encodes it: the system text, then — when
        the session embeds reference audio — ``<|audio_start|>``, one ``<unit>``
        id per tower row of the clip, ``<|audio_end|>``. The prefix and suffix
        texts are encoded whole, as the engine encodes them.

        Args:
            instructions: The session's system-prompt text.
            ref_audio_rows: Tower rows of the embedded reference clip, or
                ``None`` for a reference-less context.

        Raises:
            RecordSetError: On a non-positive ``ref_audio_rows``.
        """
        if ref_audio_rows is not None and int(ref_audio_rows) <= 0:
            raise RecordSetError(
                f"the embedded reference audio must span at least one <unit> row, got {ref_audio_rows}"
            )
        prefix = f"<|im_start|>system\n{instructions if instructions else DEFAULT_SESSION_INSTRUCTIONS}"
        suffix = "<|im_end|>"
        if ref_audio_rows is not None:
            prefix += "\n<|audio_start|>"
            suffix = "<|audio_end|>" + suffix
        prefix_ids = list(self._tokenizer.encode(prefix, add_special_tokens=False))
        reference_ids = [self.unit_token_id] * int(ref_audio_rows) if ref_audio_rows is not None else []
        suffix_ids = list(self._tokenizer.encode(suffix, add_special_tokens=False))
        return prefix_ids, reference_ids, suffix_ids

    def context_prefix_ids(self, instructions: str | None = None, *, ref_audio_rows: int | None = None) -> list[int]:
        """The session's context ids (MiniCPMODuplex.prepare() layout)."""
        prefix_ids, reference_ids, suffix_ids = self.context_layout(instructions, ref_audio_rows=ref_audio_rows)
        return prefix_ids + reference_ids + suffix_ids


def resolve_duplex_vocabulary(processor_or_tokenizer: Any) -> DuplexTokenVocabulary:
    """Resolve the framing vocabulary from a processor or a tokenizer."""
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    return DuplexTokenVocabulary(tokenizer)


@dataclass(frozen=True, slots=True)
class DuplexRefAudio:
    """The model-bundled reference clip every audio duplex session embeds.

    ``num_audio_tokens`` is the clip's tower-row count — the ``<unit>`` placeholder
    count between ``<|audio_start|>`` and ``<|audio_end|>`` — computed through the
    engine's ``audio_token_count`` arithmetic, never hard-coded. ``pcm`` is
    ``None`` once a payload dropped it (``include_pcm=False``).
    """

    pcm: np.ndarray | None
    sample_rate: int
    num_audio_tokens: int
    #: Provenance: the resolved asset path (or the builder's label).
    source: str = ""


def resolve_default_ref_audio(
    model_path: str | None,
    *,
    sample_rate: int = _STREAM_SAMPLE_RATE,
) -> DuplexRefAudio | None:
    """The snapshot's bundled prompt clip, resolved as the engine resolves it.

    Mirrors vLLM-Omni's ``resolve_default_ref_audio``: ``<model>/assets/HT_ref_audio.wav``
    read as float32, channels averaged to mono (``_load_default_ref_audio``),
    resampled to the stream rate when the file's differs (``normalize_ref_audio``),
    with the tower-row count of ``_apply_first_append_context_tokens``.

    Args:
        model_path: The checkpoint directory, or a hub repo id resolved the way
            the engine does (assets only). Unlike the engine, a hub id resolves
            at the hub's current revision; pass a local snapshot path to pin one.
        sample_rate: The duplex stream rate.

    Returns:
        The clip, or ``None`` when the snapshot bundles no asset.

    Raises:
        RecordSetError: On an empty asset.
    """
    if not model_path:
        return None
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        try:
            model_dir = Path(snapshot_download(model_path, allow_patterns=["assets/*"]))
        except Exception as exc:
            raise RecordSetError(f"resolving the bundled reference audio for {model_path!r} failed: {exc!r}") from exc
    path = model_dir.joinpath(*_REF_AUDIO_RELPATH)
    if not path.is_file():
        return None
    import soundfile as sf

    waveform, file_rate = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        # soundfile is samples-first; the engine's reader hands the loader
        # (channels, samples) and averages axis 0.
        waveform = waveform.T.mean(axis=0)
    pcm = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)
    if pcm.size == 0:
        raise RecordSetError(f"bundled reference audio {path} is empty")
    if file_rate != sample_rate:
        import torch
        import torchaudio

        audio = torch.from_numpy(pcm).to(dtype=torch.float32).unsqueeze(0)
        pcm = torchaudio.functional.resample(audio, int(file_rate), int(sample_rate)).squeeze(0).cpu().numpy()
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    return DuplexRefAudio(
        pcm=pcm,
        sample_rate=int(sample_rate),
        num_audio_tokens=audio_tokens_for_samples(pcm.size),
        source=str(path),
    )


def _validate_ref_audio(ref_audio: DuplexRefAudio, *, sample_rate: int, session_id: str) -> None:
    if ref_audio.pcm is None or ref_audio.pcm.ndim != 1 or ref_audio.pcm.size == 0:
        raise RecordSetError(f"duplex episode {session_id}: the embedded reference clip carries no mono PCM")
    if int(ref_audio.sample_rate) != sample_rate:
        raise RecordSetError(
            f"duplex episode {session_id}: the reference clip is at {ref_audio.sample_rate} Hz "
            f"but the stream runs at {sample_rate} Hz"
        )
    rows = audio_tokens_for_samples(ref_audio.pcm.size)
    if rows != ref_audio.num_audio_tokens:
        raise RecordSetError(
            f"duplex episode {session_id}: the reference clip pools to {rows} tower rows for "
            f"{ref_audio.pcm.size} samples but declares {ref_audio.num_audio_tokens}; "
            "the context placeholders would not match the tower output"
        )


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

    ``prompt_ids`` is the session's context (system text, the embedded
    reference block when the session carried one); ``response_ids`` is the
    interleaved unit-window sequence after it. ``audio_bounds`` are
    ``[start, end)`` pairs over the concatenated row (prompt + response),
    covering the placeholder runs the audio tower outputs replace — the
    reference block first, then one per unit. ``unit_mels`` holds one mel clip
    per span in the same order.
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
    #: Per-span mel features (the reference clip, then one per unit; re-derived
    #: through the processor) and their mel-frame counts; attached by the
    #: rollout adapter before the actor consumes the replay.
    unit_mels: list | None = None
    unit_mel_lens: list[int] | None = None
    #: The embedded reference block; ``None`` for a reference-less build.
    ref_audio: DuplexRefAudio | None = None

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
            include_pcm: Keep the per-unit PCM stream and the reference clip's
                PCM (debugging / parity dumps); the actor micro-batch payload
                drops them — the mel features and bounds are what training
                consumes.
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
            "ref_audio": None,
        }
        if include_pcm:
            payload["unit_pcm"] = [np.asarray(chunk, dtype=np.float32) for chunk in self.unit_pcm]
        if self.unit_mels is not None:
            payload["unit_mels"] = list(self.unit_mels)
            payload["unit_mel_lens"] = list(self.unit_mel_lens or [])
        if self.ref_audio is not None:
            ref: dict[str, Any] = {
                "sample_rate": int(self.ref_audio.sample_rate),
                "num_audio_tokens": int(self.ref_audio.num_audio_tokens),
                "source": self.ref_audio.source,
            }
            if include_pcm and self.ref_audio.pcm is not None:
                ref["pcm"] = np.asarray(self.ref_audio.pcm, dtype=np.float32)
            payload["ref_audio"] = ref
        return payload

    @classmethod
    def from_extra_fields(cls, payload: Mapping[str, Any]) -> DuplexReplay:
        ref = payload.get("ref_audio")
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
            ref_audio=(
                DuplexRefAudio(
                    pcm=np.asarray(ref["pcm"], dtype=np.float32) if ref.get("pcm") is not None else None,
                    sample_rate=int(ref["sample_rate"]),
                    num_audio_tokens=int(ref["num_audio_tokens"]),
                    source=str(ref.get("source", "")),
                )
                if isinstance(ref, Mapping)
                else None
            ),
        )


def build_replay(
    trajectory: DuplexTrajectory,
    vocabulary: DuplexTokenVocabulary,
    *,
    instructions: str | None = None,
    sample_rate: int = _STREAM_SAMPLE_RATE,
    ref_audio: DuplexRefAudio | None = None,
    model_path: str | None = None,
) -> DuplexReplay:
    """Rebuild the engine's unit-window sequence for one trajectory.

    Args:
        trajectory: The assembled episode.
        vocabulary: The framing token vocabulary of the checkpoint.
        instructions: The session's system-prompt text (must equal the value
            the rollout driver opened the session with).
        sample_rate: The duplex stream's PCM sample rate.
        ref_audio: The reference block the engine embedded (every audio duplex
            session carries the model-bundled default). ``None`` defers to
            ``model_path``.
        model_path: The checkpoint the engine resolved the bundled
            ``assets/HT_ref_audio.wav`` from; ``None`` reads
            :data:`DEFAULT_REF_AUDIO_MODEL_PATH_ENV`. When neither is given the
            replay carries no reference block — faithful only to a session the
            engine opened without reference audio.

    Returns:
        The replay with prompt/response ids, the validity mask over the
        response, explicit audio bounds, and the per-unit PCM stream.

    Raises:
        RecordSetError: On any layout disagreement between the trajectory's
            action stream and its unit windows, a reference clip whose declared
            rows disagree with its PCM, or a ``model_path`` whose snapshot lacks
            the bundled asset the engine requires of audio sessions.
    """
    if ref_audio is None:
        model_path = model_path or os.environ.get(DEFAULT_REF_AUDIO_MODEL_PATH_ENV)
        if model_path:
            ref_audio = resolve_default_ref_audio(model_path, sample_rate=sample_rate)
            if ref_audio is None:
                raise RecordSetError(
                    f"model {model_path!r} bundles no {'/'.join(_REF_AUDIO_RELPATH)}; the engine embeds it in every "
                    "audio duplex session and refuses to open one without it — the replay cannot "
                    "match the engine context"
                )
    if ref_audio is not None:
        _validate_ref_audio(ref_audio, sample_rate=sample_rate, session_id=trajectory.session_id)
    prefix_ids, reference_ids, suffix_ids = vocabulary.context_layout(
        instructions,
        ref_audio_rows=ref_audio.num_audio_tokens if ref_audio is not None else None,
    )
    prompt_ids = prefix_ids + reference_ids + suffix_ids
    prompt_length = len(prompt_ids)
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    audio_bounds: list[list[int]] = []
    if reference_ids:
        # The reference block's placeholder run is the first span the tower splices into.
        audio_bounds.append([len(prefix_ids), len(prefix_ids) + len(reference_ids)])
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
            "interrupt_speech_onset_s": trajectory.event.interrupt_speech_onset_s,
            "interrupt_speech_end_s": trajectory.event.interrupt_speech_end_s,
        },
        status=trajectory.status,
        model_extension=model_extension,
        sampling_params=sampling_params,
        instructions=instructions or DEFAULT_SESSION_INSTRUCTIONS,
        ref_audio=ref_audio,
    )


def _episode_streaming_processor(processor: Any) -> Any:
    """A per-episode streaming processor, configured as the engine's per-session copy.

    Mirrors ``_configure_streaming_processor``: a shallow copy carrying deep
    copies of ``audio_processor`` and ``_streaming_mel_processor`` (one memo, so
    the mel processor keeps pointing at its own extractor — the streaming mel
    front end mutates the extractor's log-mel floor on every chunk, which must
    not leak into the shared processor's offline ``process_audio``), then
    ``set_streaming_mode(mode="exact", chunk_ms=1000, first_chunk_ms=1035,
    cnn_redundancy_ms=20, enable_sliding_window=True, slide_trigger_seconds=30.0,
    slide_stride_seconds=10.0)`` and ``reset_streaming()``; older remote code
    exposes ``configure_streaming`` instead.
    """
    episode = copy.copy(processor)
    memo: dict[int, Any] = {}
    shared_extractor = getattr(processor, "audio_processor", None)
    if shared_extractor is not None:
        episode.audio_processor = copy.deepcopy(shared_extractor, memo)
    shared_mel = getattr(processor, "_streaming_mel_processor", None)
    if shared_mel is not None:
        episode._streaming_mel_processor = copy.deepcopy(shared_mel, memo)
    set_streaming_mode = getattr(episode, "set_streaming_mode", None)
    if callable(set_streaming_mode):
        set_streaming_mode(
            mode="exact",
            chunk_ms=_STREAMING_CHUNK_MS,
            first_chunk_ms=_STREAMING_FIRST_CHUNK_MS,
            cnn_redundancy_ms=_STREAMING_CNN_REDUNDANCY_MS,
            enable_sliding_window=True,
            slide_trigger_seconds=_STREAMING_SLIDE_TRIGGER_SECONDS,
            slide_stride_seconds=_STREAMING_SLIDE_STRIDE_SECONDS,
        )
        reset_streaming = getattr(episode, "reset_streaming", None)
        if callable(reset_streaming):
            reset_streaming()
        return episode
    configure_streaming = getattr(episode, "configure_streaming", None)
    if callable(configure_streaming):
        configure_streaming(
            chunk_ms=_STREAMING_CHUNK_MS,
            enable_sliding_window=True,
            slide_trigger_seconds=_STREAMING_SLIDE_TRIGGER_SECONDS,
            slide_stride_seconds=_STREAMING_SLIDE_STRIDE_SECONDS,
        )
    return episode


def _mel_sample_rate(streaming: Any, default: int) -> int:
    """The streaming mel processor's rate (engine ``_sample_rate``)."""
    mel = getattr(streaming, "_streaming_mel_processor", None)
    return int(getattr(mel, "sample_rate", default) or default)


def _pad_first_chunk(first_chunk: np.ndarray, streaming: Any, sample_rate: int) -> np.ndarray:
    """Left-pad the stream's first chunk to the first-chunk window (engine ``_pad_first_audio_chunk_if_needed``).

    The hop-aligned first chunk sits below 1035 ms; the zero pre-roll gives the
    exact mel processor the right-context frames it needs to emit at the 1 s
    cadence from the first unit on.
    """
    mel = getattr(streaming, "_streaming_mel_processor", None)
    if mel is None or first_chunk.size == 0:
        return first_chunk
    rate = _mel_sample_rate(streaming, sample_rate)
    first_chunk_samples = int(_STREAMING_FIRST_CHUNK_MS * rate / 1000)
    if first_chunk.size >= first_chunk_samples:
        return first_chunk
    return np.concatenate([np.zeros(first_chunk_samples - first_chunk.size, dtype=np.float32), first_chunk])


def _streaming_chunk_size(streaming: Any) -> int:
    """Samples of the next processor chunk (engine ``_streaming_chunk_size``)."""
    get_chunk = getattr(streaming, "get_streaming_chunk_size", None)
    return int(get_chunk()) if callable(get_chunk) else _STREAMING_CHUNK_MS * _STREAM_SAMPLE_RATE // 1000


def _consumed_audio_samples(streaming: Any, chunk_idx: int, chunk_size: int, sample_rate: int) -> int:
    """Samples the encoder chunk ``chunk_idx`` consumes off the stream (engine ``_consumed_audio_samples``)."""
    rate = _mel_sample_rate(streaming, sample_rate)
    if chunk_idx != 0:
        return int(_STREAMING_CHUNK_MS * rate / 1000)
    get_config = getattr(getattr(streaming, "_streaming_mel_processor", None), "get_config", None)
    if callable(get_config):
        config = get_config()
        if isinstance(config, Mapping):
            consumed_ms = int(config.get("effective_first_chunk_ms", _STREAMING_FIRST_CHUNK_MS))
            return int(consumed_ms * rate / 1000)
    return chunk_size


def _declared_frames(lens: Any, what: str) -> int | None:
    """The clip's mel-frame count from a batch feature's ``audio_feature_lens``."""
    if lens is None:
        return None
    if hasattr(lens, "reshape") and not isinstance(lens, list | tuple):
        lens = lens.reshape(-1).tolist()
    if isinstance(lens, list | tuple):
        if not lens:
            return None
        if len(lens) != 1:
            raise RecordSetError(f"duplex replay {what} declares {len(lens)} audio clips; expected one")
        return _declared_frames(lens[0], what)
    if isinstance(lens, bool) or not isinstance(lens, int | float):
        raise RecordSetError(f"duplex replay {what} declares a non-numeric audio feature length {lens!r}")
    return int(lens)


def _mel_clip(batch_feature: Any, what: str) -> tuple[np.ndarray, int]:
    """The one ``(n_mels, frames)`` float32 clip of a processor batch feature and its frame count."""
    if isinstance(batch_feature, Mapping):
        features, lens = batch_feature.get("audio_features"), batch_feature.get("audio_feature_lens")
    else:
        features, lens = (
            getattr(batch_feature, "audio_features", None),
            getattr(batch_feature, "audio_feature_lens", None),
        )
    if features is None:
        raise RecordSetError(f"duplex replay {what} produced no audio features")
    if isinstance(features, list | tuple):
        if len(features) != 1:
            raise RecordSetError(f"duplex replay {what} produced {len(features)} audio clips; expected one")
        features = features[0]
    if hasattr(features, "detach"):  # the processor hands torch tensors
        features = features.detach().float().cpu().numpy()
    clip = np.asarray(features)
    if clip.ndim == 3 and clip.shape[0] == 1:
        clip = clip[0]
    if clip.ndim != 2:
        raise RecordSetError(
            f"duplex replay {what} produced audio features of shape {clip.shape}; "
            "expected [1, n_mels, frames] or [n_mels, frames]"
        )
    frames = _declared_frames(lens, what)
    if frames is None:
        frames = int(clip.shape[-1])
    if frames <= 0:
        raise RecordSetError(f"duplex replay {what} produced empty features")
    if frames > clip.shape[-1]:
        raise RecordSetError(f"duplex replay {what} declares {frames} mel frames but carries {clip.shape[-1]}")
    return np.ascontiguousarray(clip[:, :frames], dtype=np.float32), frames


def _ref_audio_clip(processor: Any, ref_audio: DuplexRefAudio, *, session_id: str) -> tuple[np.ndarray, int]:
    """Encode the reference clip through the OFFLINE front end (engine ``_stage_ref_audio_embeddings``).

    Runs on the shared processor, as the engine stages the reference at session init.
    """
    what = f"reference clip of {session_id}"
    if ref_audio.pcm is None:
        raise RecordSetError(f"duplex replay {what} carries no PCM; extract its mel features before dropping PCM")
    process_audio = getattr(processor, "process_audio", None)
    if not callable(process_audio):
        raise RecordSetError(
            "the MiniCPM-o 4.5 processor does not expose process_audio(); the embedded reference audio cannot be "
            "encoded through the offline front end — check the checkpoint's remote code"
        )
    clip, frames = _mel_clip(process_audio([ref_audio.pcm]), what)
    rows = audio_tokens_for_mel_frames(frames)
    if rows != ref_audio.num_audio_tokens:
        raise RecordSetError(
            f"duplex replay {what}: the offline front end yields {frames} mel frames ({rows} tower rows) but the "
            f"context carries {ref_audio.num_audio_tokens} <unit> placeholders — "
            "the tower output cannot splice one-to-one"
        )
    return clip, frames


def extract_unit_mels(
    processor: Any,
    replay: DuplexReplay,
    *,
    sample_rate: int = _STREAM_SAMPLE_RATE,
) -> tuple[list[Any], list[int]]:
    """Re-derive the replay's mel clips: the reference block, then every unit.

    The reference clip rides the offline front end (``process_audio([ref])``)
    on the shared processor, as the engine stages it. The units ride a
    per-episode processor copy configured exactly as the engine's per-session
    copy (:func:`_episode_streaming_processor`), driven with the engine's
    ingestion geometry over the concatenated unit stream: the first chunk
    left-padded to the first-chunk window, then the processor's chunk size
    consumed per unit, each fed once as
    ``process_audio_streaming(chunk, reset=False, return_batch_feature=True)``.

    Args:
        processor: The checkpoint's processor (streaming and offline audio APIs).
        replay: The built replay (its ``unit_pcm`` is the stream).
        sample_rate: The stream's sample rate (validated against the
            processor's when it declares one).

    Returns:
        ``(features, lens)``: one ``(n_mels, frames)`` float32 clip per
        ``audio_bounds`` span — the reference block first when the replay
        carries one — and its mel-frame count.

    Raises:
        RecordSetError: When the processor lacks an audio API, a unit yields
        no or malformed features, the stream runs short of a unit, or the
        reference clip's tower rows disagree with its placeholders.
    """
    declared_rate = getattr(processor, "sampling_rate", None) or getattr(
        getattr(processor, "feature_extractor", None), "sampling_rate", None
    )
    if isinstance(declared_rate, int) and declared_rate != sample_rate:
        raise RecordSetError(f"duplex replay streams at {sample_rate} Hz but the processor expects {declared_rate} Hz")
    features: list[Any] = []
    lens: list[int] = []
    if replay.ref_audio is not None:
        clip, frames = _ref_audio_clip(processor, replay.ref_audio, session_id=replay.session_id)
        features.append(clip)
        lens.append(frames)
    if not replay.unit_pcm:
        return features, lens
    streaming = _episode_streaming_processor(processor)
    process = getattr(streaming, "process_audio_streaming", None)
    if not callable(process):
        raise RecordSetError(
            "the MiniCPM-o 4.5 processor does not expose process_audio_streaming(); duplex replay "
            "cannot re-derive streaming mel features — check the checkpoint's remote code"
        )
    chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in replay.unit_pcm]
    buffer = np.concatenate([_pad_first_chunk(chunks[0], streaming, sample_rate), *chunks[1:]])
    for unit_index in range(len(chunks)):
        chunk_size = _streaming_chunk_size(streaming)
        if buffer.size < chunk_size:
            raise RecordSetError(
                f"duplex replay unit {unit_index} of {replay.session_id}: audio not enough: need {chunk_size} samples, "
                f"only {buffer.size}"
            )
        batch_feature = process(buffer[:chunk_size], reset=False, return_batch_feature=True)
        buffer = buffer[_consumed_audio_samples(streaming, unit_index, chunk_size, sample_rate) :]
        clip, frames = _mel_clip(batch_feature, f"unit {unit_index} of {replay.session_id}")
        features.append(clip)
        lens.append(frames)
    return features, lens


def build_actor_media(replay: DuplexReplay, features: Sequence[Any], lens: Sequence[int]) -> dict[str, Any]:
    """Assemble the actor's media inputs for one replayed episode.

    Returns the per-sample containers ``MiniCPMOThinkerAdapter``'s duplex
    branch merges into the model call: one audio feature clip per
    ``audio_bounds`` span (the reference block first, then one per unit), with
    explicit ``audio_bounds`` in full-row coordinates.

    Raises:
        RecordSetError: When the clips do not number as the spans, or a clip's
            mel frames do not pool to its span's placeholder count.
    """
    spans = replay.audio_bounds
    if len(features) != len(spans) or len(lens) != len(spans):
        raise RecordSetError(
            f"duplex replay of {replay.session_id} has {len(spans)} audio spans ({len(replay.unit_layouts)} units) but "
            f"{len(features)} feature clips"
        )
    for span, length in zip(spans, lens, strict=True):
        rows = audio_tokens_for_mel_frames(length)
        if rows != span[1] - span[0]:
            raise RecordSetError(
                f"duplex replay span {span} of {replay.session_id}: {length} mel frames pool to {rows} "
                f"audio feature rows but the span covers {span[1] - span[0]} placeholder tokens — "
                "the tower output cannot splice one-to-one"
            )
    return {
        "audio_features": [list(features)],
        "audio_feature_lens": [[int(value) for value in lens]],
        "audio_bounds": [[list(span) for span in spans]],
    }
