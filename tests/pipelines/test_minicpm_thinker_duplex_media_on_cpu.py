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
"""CPU tests for the thinker adapter's duplex media branch (explicit bounds)."""

import numpy as np
import pytest
import torch

from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter
from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY

PLACEHOLDER = 9004  # the duplex tests' <|audio|> id
UNIT = 9001
UNIT_END = 9002


def _payload(bounds: list[list[int]], mels: int | None = None) -> dict:
    return {
        "audio_bounds": [list(span) for span in bounds],
        "unit_mels": [np.ones((span[1] - span[0], 80), dtype=np.float32) for span in bounds]
        if mels is None
        else [np.ones((mels, 80), dtype=np.float32) for _ in bounds],
        "unit_mel_lens": [span[1] - span[0] if mels is None else mels for span in bounds],
    }


def _padded_ids(num_samples: int, prompt_len: int, spans_per_sample: list[list[tuple[int, int]]]):
    rows = []
    for spans in spans_per_sample:
        ids = [1] * prompt_len
        for start, end in spans:
            ids.extend([UNIT] + [PLACEHOLDER] * (end - start) + [UNIT_END])
        rows.append(ids)
    width = max(len(row) for row in rows)
    return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_padded_duplex_batch_receives_explicit_bounds(batch_size):
    prompt_len = 4
    spans = [[(5, 15)]] * batch_size
    input_ids = _padded_ids(batch_size, prompt_len, spans)
    # Full-row coordinates: the span in ids starts at the <unit> token + 1.
    payloads = [_payload([[prompt_len + 1, prompt_len + 11]]) for _ in range(batch_size)]
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(batch_size, -1).contiguous()},
        {DUPLEX_REPLAY_KEY: payloads},
        model_config=None,
    )
    data = result["data"]
    assert len(data["audio_features"]) == batch_size
    assert len(data["audio_feature_lens"][0]) == 1
    assert data["audio_bounds"] == [[[prompt_len + 1, prompt_len + 11]]] * batch_size
    # The placeholder run the bounds cover is exactly the audio span in the ids.
    for sample in range(batch_size):
        start, end = data["audio_bounds"][sample][0]
        assert set(data["input_ids"][sample][start:end].tolist()) == {PLACEHOLDER}


def test_packed_duplex_batch_shifts_bounds_by_packed_offsets():
    prompt_len = 2
    spans = [[(3, 5)], [(3, 5)]]
    input_ids_rows = []
    for spans_one in spans:
        ids = [1] * prompt_len
        for start, end in spans_one:
            ids.extend([UNIT] + [PLACEHOLDER] * (end - start) + [UNIT_END])
        input_ids_rows.append(ids)
    packed = torch.tensor([sum(input_ids_rows, [])], dtype=torch.long)
    # rmpad: position ids reset per sample make _is_packed_batch true.
    position_ids = torch.tensor([list(range(len(input_ids_rows[0]))) + list(range(len(input_ids_rows[1])))])
    row_len = len(input_ids_rows[0])
    payloads = [
        _payload([[prompt_len + 1, prompt_len + 3]]),
        _payload([[prompt_len + 1, prompt_len + 3]]),
    ]
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": packed, "position_ids": position_ids},
        {DUPLEX_REPLAY_KEY: payloads},
        model_config=None,
    )
    data = result["data"]
    # After packing, both samples' media fold into one pseudo-row.
    assert len(data["audio_bounds"]) == 1
    first, second = data["audio_bounds"][0]
    assert first == [prompt_len + 1, prompt_len + 3]
    assert second == [row_len + prompt_len + 1, row_len + prompt_len + 3]
    # The folded features keep both samples' clips in order.
    assert len(data["audio_feature_lens"][0]) == 2


def test_missing_replay_mels_fail_closed():
    input_ids = _padded_ids(1, 2, [[(3, 5)]])
    with pytest.raises(ValueError, match="mel features or audio bounds"):
        MiniCPMThinkerAdapter.prepare_model_inputs(
            {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1])[None]},
            {DUPLEX_REPLAY_KEY: [{"audio_bounds": [[3, 5]]}]},
            model_config=None,
        )


def test_count_mismatch_fails_closed():
    input_ids = _padded_ids(2, 2, [[(3, 5)], [(3, 5)]])
    with pytest.raises(ValueError, match="replay payloads"):
        MiniCPMThinkerAdapter.prepare_model_inputs(
            {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(2, -1).contiguous()},
            {DUPLEX_REPLAY_KEY: [_payload([[3, 5]])]},
            model_config=None,
        )
