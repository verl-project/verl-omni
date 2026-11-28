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
"""CPU tests for the duplex episode dataset."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from verl_omni.utils.dataset.duplex.duplex_rl_dataset import DuplexEpisodeDataset, build_episode_chunks
from verl_omni.utils.dataset.duplex.mining import DuplexEpisodeSpec
from verl_omni.utils.dataset.duplex.trajectory import AXIS_INTERRUPTION, AXIS_PAUSE, AXIS_TURN, DuplexEpisodeEvent


@pytest.fixture(scope="module")
def user_track(tmp_path_factory):
    """10 s of user-track audio with an amplitude bump over the speech span."""
    rate = 16000
    samples = np.zeros(rate * 10, dtype=np.float32)
    samples[rate * 1 : rate * 4] = 0.5  # speech 1-4 s
    path = tmp_path_factory.mktemp("duplex") / "user.wav"
    sf.write(path, samples, rate)
    return str(path), samples


def test_build_episode_chunks_slices_windows_with_speech_hints(user_track):
    _, samples = user_track
    chunks = build_episode_chunks(
        samples,
        start_s=0.5,
        end_s=3.5,
        words=[{"start_s": 1.0, "end_s": 3.0}],
    )
    assert len(chunks) == 3
    assert [chunk["is_speech"] for chunk in chunks] == [True, True, True]  # speech spans 1-3 s
    silent = build_episode_chunks(samples, start_s=4.0, end_s=6.0, words=[{"start_s": 1.0, "end_s": 3.0}])
    assert [chunk["is_speech"] for chunk in silent] == [False, False]
    # Partial overlap marks the chunk as speech.
    edge = build_episode_chunks(samples, start_s=0.0, end_s=2.0, words=[{"start_s": 1.5, "end_s": 2.5}])
    assert [chunk["is_speech"] for chunk in edge] == [False, True]


def test_build_episode_chunks_pads_the_final_chunk_and_rejects_bad_windows(user_track):
    _, samples = user_track
    chunks = build_episode_chunks(samples, start_s=0.0, end_s=2.5)
    assert len(chunks) == 3
    assert all(chunk["pcm"].size == 16000 for chunk in chunks)
    assert np.all(chunks[-1]["pcm"][8000:] == 0.0)
    with pytest.raises(ValueError, match="empty"):
        build_episode_chunks(samples, start_s=2.0, end_s=2.0)
    with pytest.raises(ValueError, match="past the audio"):
        build_episode_chunks(samples, start_s=20.0, end_s=21.0)
    # A window whose tail outruns the track clips to the audio (the last
    # utterance of a corpus dialogue ends at the file end).
    clipped = build_episode_chunks(samples, start_s=0.0, end_s=20.0)
    assert len(clipped) == 10


def _row(
    spec: DuplexEpisodeSpec, words: list[dict], *, commit: bool = False, dialogue_id: str | None = "dd-0001"
) -> dict:
    row = {
        "episode": spec.to_dict(),
        "user_words": words,
        "duplex_seed": 5,
        "data_source": "dailytalk",
        "commit": commit,
    }
    if dialogue_id is not None:
        row["dialogue_id"] = dialogue_id
    return row


def _spec(user_track, axis: str) -> DuplexEpisodeSpec:
    return DuplexEpisodeSpec(
        episode_id=f"dd-{axis}-0001",
        axis=axis,
        user_audio_ref=user_track[0],
        start_s=0.5,
        end_s=3.5,
        event=DuplexEpisodeEvent(
            axis=axis,
            pause_window_s=(1.5, 2.5) if axis == AXIS_PAUSE else None,
            user_turn_end_s=2.5 if axis == AXIS_TURN else None,
            interrupt_speech_onset_s=0.5 if axis == AXIS_INTERRUPTION else None,
            interrupt_speech_end_s=2.0 if axis == AXIS_INTERRUPTION else None,
        ),
    )


def test_dataset_emits_episode_columns(user_track, tmp_path):
    rows = [
        _row(_spec(user_track, AXIS_TURN), [{"start_s": 1.0, "end_s": 3.0}], commit=True),
        _row(_spec(user_track, AXIS_PAUSE), []),
        _row(_spec(user_track, AXIS_INTERRUPTION), [], dialogue_id=None),  # a row built before the column existed
    ]
    parquet = tmp_path / "episodes.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    dataset = DuplexEpisodeDataset([str(parquet)])
    assert len(dataset) == 3
    sample = dataset[0]
    episode = sample["duplex_episode"]
    assert episode["axis"] == AXIS_TURN
    assert episode["event"]["user_turn_end_s"] == pytest.approx(2.5)  # episode-relative (3.0 s - 0.5 s window)
    assert len(episode["chunks"]) == 3
    assert sample["duplex_seed"] == 5
    assert sample["dialogue_id"] == "dd-0001"  # the build's source dialogue rides the sample
    # verl v1 mints uids off raw_prompt; it stays the stable episode id.
    assert sample["raw_prompt"] == "dd-turn-0001"
    # Rule-reward managers read ground_truth before the scorer runs.
    assert sample["reward_model"] == {"ground_truth": ""}
    # The source tag splits per axis so validation metrics do too.
    assert sample["data_source"] == f"dailytalk-duplex-{AXIS_TURN}"
    # The commit flag survives the spec round-trip into the run spec.
    assert episode["commit"] is True
    assert dataset[1]["data_source"] == f"dailytalk-duplex-{AXIS_PAUSE}"
    assert dataset[1]["duplex_episode"]["commit"] is False
    assert "dummy_tensor" in sample
    # The chunks carry real PCM from the track (speech span is non-zero).
    assert np.abs(episode["chunks"][1]["pcm"]).max() == pytest.approx(0.5)
    # Both interruption anchors survive the parquet round-trip; a missing
    # dialogue column reads as none instead of failing.
    interruption = dataset[2]
    assert interruption["duplex_episode"]["event"]["interrupt_speech_onset_s"] == pytest.approx(0.5)
    assert interruption["duplex_episode"]["event"]["interrupt_speech_end_s"] == pytest.approx(2.0)
    assert interruption["dialogue_id"] is None


def test_build_episode_chunks_rejects_a_start_before_the_track():
    waveform = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="before the track"):
        build_episode_chunks(waveform, start_s=-0.1, end_s=1.0)


def test_dataset_rejects_rows_without_the_mining_columns(tmp_path):
    parquet = tmp_path / "bad.parquet"
    pd.DataFrame([{"foo": 1}]).to_parquet(parquet)
    with pytest.raises(ValueError, match="missing columns"):
        DuplexEpisodeDataset([str(parquet)])
