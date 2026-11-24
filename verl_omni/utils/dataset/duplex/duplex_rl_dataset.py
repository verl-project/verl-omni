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
"""The duplex episode dataset: parquet episode specs → fed PCM chunks.

Rows are mined episode specs (see :mod:`verl_omni.utils.dataset.duplex.mining`)
plus the user track's word timestamps inside the window. ``__getitem__`` loads
the user track once, cuts the episode window, and emits the native-cadence
1 s chunks with speech hints — everything ``omni_duplex_agent`` needs to drive
one real duplex session. The user track's content never reacts to the policy
(RFC verl-omni #905 §3.1: replay, not closed loop).

Selected from the run config::

    data.custom_cls.path=verl_omni/utils/dataset/duplex/duplex_rl_dataset.py
    data.custom_cls.name=DuplexEpisodeDataset
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: The duplex stream's native cadence.
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = _SAMPLE_RATE
#: Per-worker track cache bound: a corpus-scale run must not pin every track.
_WAVEFORM_CACHE_LIMIT = 8


def _load_user_track(reference: str, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """Load the user track as float32 mono at the stream rate."""
    from verl_omni.utils.dataset.omni_rl_datasets import _load_minicpm_audio

    return _load_minicpm_audio(reference, sample_rate)


def build_episode_chunks(
    waveform: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    words: list[Mapping[str, Any]] | None = None,
    sample_rate: int = _SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Cut one episode window into 1 s chunks with speech hints.

    Args:
        waveform: The full user track (mono float32).
        start_s: Episode window start, seconds into the track.
        end_s: Episode window end, seconds into the track.
        words: Word timestamps (``start_s`` / ``end_s``, track-absolute) for
            the speech hints; a chunk that overlaps any word carries
            ``is_speech=True``.
        sample_rate: The stream's sample rate.

    Returns:
        ``[{"pcm": float32[unit], "is_speech": bool}, ...]`` in feed order.
    """
    start_sample = int(round(start_s * sample_rate))
    end_sample = int(round(end_s * sample_rate))
    if end_sample <= start_sample:
        raise ValueError(f"duplex episode window [{start_s}, {end_s}) is empty")
    if start_sample >= waveform.shape[-1]:
        raise ValueError(
            f"duplex episode window starts at sample {start_sample} but the user track has "
            f"{waveform.shape[-1]}; the mined window lies past the audio"
        )
    # A corpus's last utterance can end at the file end, so the mined tail
    # may outrun the track — clip to what exists (the final chunk pads).
    end_sample = min(end_sample, waveform.shape[-1])
    segment = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
    words = list(words or [])
    chunks: list[dict[str, Any]] = []
    for offset in range(0, max(segment.size, 1), _CHUNK_SAMPLES):
        chunk = segment[offset : offset + _CHUNK_SAMPLES]
        if chunk.size < _CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES - chunk.size))
        chunk_start_s = start_s + offset / sample_rate
        chunk_end_s = chunk_start_s + _CHUNK_SAMPLES / sample_rate
        is_speech = any(
            float(word.get("start_s", 0.0)) < chunk_end_s and float(word.get("end_s", 0.0)) > chunk_start_s
            for word in words
        )
        chunks.append({"pcm": chunk, "is_speech": is_speech})
    return chunks


class DuplexEpisodeDataset(Dataset):
    """Loads mined duplex episodes and emits fed-chunk columns per sample."""

    def __init__(
        self,
        data_files: list[str],
        tokenizer=None,
        processor=None,
        config: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        """Read the episode parquet; media loads lazily per sample.

        Args:
            data_files: Parquet files of mined episode rows.
            tokenizer: Unused (duplex episodes carry no text prompt).
            processor: Unused (the replay drives the processor at rollout).
            config: The data config (``sample_rate`` override accepted).
            kwargs: Forwarded compatibility (ignored).
        """
        self.dataframe = pd.read_parquet(data_files)
        required = {"episode", "user_words"}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"duplex episode parquet is missing columns {sorted(missing)}; "
                "rows come from verl_omni.utils.dataset.duplex.mining"
            )
        config = dict(config or {})
        self.sample_rate = int(config.get("sample_rate", _SAMPLE_RATE))
        self._waveform_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.dataframe)

    def _waveform(self, reference: str) -> np.ndarray:
        if reference not in self._waveform_cache:
            if len(self._waveform_cache) >= _WAVEFORM_CACHE_LIMIT:
                self._waveform_cache.pop(next(iter(self._waveform_cache)))
            self._waveform_cache[reference] = _load_user_track(reference, self.sample_rate)
        return self._waveform_cache[reference]

    def __getitem__(self, index: int) -> dict[str, Any]:
        from verl_omni.utils.dataset.duplex.mining import DuplexEpisodeSpec

        row = self.dataframe.iloc[index].to_dict()
        spec = DuplexEpisodeSpec.from_dict(row["episode"] if isinstance(row["episode"], Mapping) else row)
        # Parquet hands back numpy containers; truth-testing an ndarray is
        # ambiguous, so normalize explicitly.
        words = row.get("user_words")
        words = [] if words is None else list(words)
        window_words = [
            {"start_s": float(word["start_s"]), "end_s": float(word["end_s"])}
            for word in words
            if float(word["end_s"]) > spec.start_s and float(word["start_s"]) < spec.end_s
        ]
        chunks = build_episode_chunks(
            self._waveform(spec.user_audio_ref),
            start_s=spec.start_s,
            end_s=spec.end_s,
            words=window_words,
            sample_rate=self.sample_rate,
        )
        episode = {
            "episode_id": spec.episode_id,
            "axis": spec.axis,
            "event": {
                "axis": spec.event.axis,
                "pause_window_s": list(spec.event.pause_window_s) if spec.event.pause_window_s else None,
                "user_turn_end_s": spec.event.user_turn_end_s,
                "interrupt_speech_end_s": spec.event.interrupt_speech_end_s,
            },
            "chunks": [{"pcm": chunk["pcm"], "is_speech": chunk["is_speech"]} for chunk in chunks],
            "instructions": row.get("instructions"),
            "inject_delay_s": spec.inject_delay_s,
            "tail_silence_chunks": int(row.get("tail_silence_chunks", 2) or 2),
            "speak_wait_horizon_s": float(row.get("speak_wait_horizon_s", 30.0) or 30.0),
            "tail_horizon_s": float(row.get("tail_horizon_s", 8.0) or 8.0),
            # stamp_commit marks auto-response episodes on the parquet row; the
            # agent loop forwards it so the driver commits the last speech chunk.
            "commit": bool(row.get("commit")),
        }
        return {
            "duplex_episode": episode,
            "duplex_seed": int(row.get("duplex_seed", 0) or 0),
            # verl v1 mints per-sample uids off raw_prompt; the episode id keeps
            # them stable across the parquet round-trip.
            "raw_prompt": str(spec.episode_id),
            # Reward managers read the per-sample ground truth before calling
            # the scorer; the duplex rule rewards ignore it, so it stays empty.
            "reward_model": {"ground_truth": ""},
            # Per-axis source tag: validation metrics split per axis.
            "data_source": f"dailytalk-duplex-{spec.axis}",
            "index": int(index),
            # TODO(data): drop once DataProto retires the non-empty-batch requirement.
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
