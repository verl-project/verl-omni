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
"""CPU tests for the dual-GRPO AR generation padding."""

import pytest
import torch

from verl_omni.agent_loop.composite_agent_loop import _pad_llm_generation_outputs


def test_pads_ragged_llm_responses_to_max_new_tokens():
    early_eos_ids = torch.tensor([[101, 2054, 8667, 102]])  # stopped at 4 tokens
    full_ids = torch.tensor([[101, 2054, 8667, 2055, 6844, 2900]])  # hit max_new_tokens
    early_eos_log_probs = torch.randn(1, 4, 11)

    padded_early, mask_early, padded_early_lp = _pad_llm_generation_outputs(
        early_eos_ids, early_eos_log_probs, max_new_tokens=6, pad_token_id=0
    )
    padded_full, mask_full, padded_full_lp = _pad_llm_generation_outputs(
        full_ids, None, max_new_tokens=6, pad_token_id=0
    )

    assert padded_early.shape == (1, 6)
    assert padded_full.shape == (1, 6)
    torch.testing.assert_close(padded_early[0, :4], early_eos_ids[0])
    torch.testing.assert_close(padded_early[0, 4:], torch.zeros(2, dtype=torch.long))
    assert mask_early.tolist() == [[1, 1, 1, 1, 0, 0]]
    assert mask_full.tolist() == [[1, 1, 1, 1, 1, 1]]
    assert padded_early_lp.shape == (1, 6, 11)
    torch.testing.assert_close(padded_early_lp[0, :4], early_eos_log_probs[0])
    torch.testing.assert_close(padded_early_lp[0, 4:], torch.zeros(2, 11))
    assert padded_full_lp is None

    # differently sized samples must now batch
    batched_ids = torch.cat([padded_early, padded_full], dim=0)
    assert batched_ids.shape == (2, 6)
    batched_mask = torch.cat([mask_early, mask_full], dim=0)
    assert batched_mask.shape == (2, 6)


def test_full_length_response_passes_through_unchanged():
    ids = torch.tensor([[5, 6, 7]])
    log_probs = torch.randn(1, 3, 4)
    padded_ids, mask, padded_log_probs = _pad_llm_generation_outputs(ids, log_probs, 3, pad_token_id=0)
    torch.testing.assert_close(padded_ids, ids)
    torch.testing.assert_close(padded_log_probs, log_probs)
    assert mask.tolist() == [[1, 1, 1]]


def test_rejects_response_longer_than_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        _pad_llm_generation_outputs(torch.tensor([[1, 2, 3]]), None, 2, pad_token_id=0)
