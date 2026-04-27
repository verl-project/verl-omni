# Copyright 2026 Amazon.com Inc and/or its affiliates
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
import torch
from tensordict import TensorDict

from verl_omni.workers.utils.padding import embeds_padding_2_no_padding


def test_embeds_padding_2_no_padding_varying_lengths():
    """Test that padding tokens are stripped correctly when sequences have different valid lengths."""
    batch_size = 3
    max_seq_len = 20
    dim = 16

    # Simulate different valid lengths: 20, 15, 10 (rest are padding zeros)
    valid_lens = [20, 15, 10]
    prompt_embeds = torch.randn(batch_size, max_seq_len, dim)
    prompt_embeds_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.int32)
    for i, vlen in enumerate(valid_lens):
        prompt_embeds_mask[i, :vlen] = 1

    data = TensorDict(
        {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        },
        batch_size=batch_size,
    )

    result = embeds_padding_2_no_padding(data)

    assert result["prompt_embeds"].is_nested

    # Each sample's nested embedding should have the correct stripped length
    embeds_nested = result["prompt_embeds"]
    for i, vlen in enumerate(valid_lens):
        sample_embed = embeds_nested[i]
        assert sample_embed.shape[0] == vlen, f"Sample {i}: expected {vlen} tokens, got {sample_embed.shape[0]}"
        # Values should match the original (unpadded portion)
        torch.testing.assert_close(sample_embed, prompt_embeds[i, :vlen, :])


def test_embeds_padding_2_no_padding_uniform_length():
    """When all sequences are fully valid (no padding), no tokens should be dropped."""
    batch_size = 4
    max_seq_len = 12
    dim = 8

    prompt_embeds = torch.randn(batch_size, max_seq_len, dim)
    # All positions valid
    prompt_embeds_mask = torch.ones(batch_size, max_seq_len, dtype=torch.int32)

    data = TensorDict(
        {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        },
        batch_size=batch_size,
    )

    result = embeds_padding_2_no_padding(data)

    assert result["prompt_embeds"].is_nested
    embeds_nested = result["prompt_embeds"]
    for i in range(batch_size):
        assert embeds_nested[i].shape[0] == max_seq_len
        torch.testing.assert_close(embeds_nested[i], prompt_embeds[i])


def test_embeds_padding_2_no_padding_nested_tensor_ragged_idx():
    """test nested tensors from embeds_padding_2_no_padding with _ragged_idx=1"""
    from verl.utils import tensordict_utils as tu

    batch_size = 4
    dim = 16
    valid_lens = [10, 7, 12, 5]
    max_seq_len = max(valid_lens)

    prompt_embeds = torch.randn(batch_size, max_seq_len, dim)
    prompt_embeds_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.int32)
    for i, vlen in enumerate(valid_lens):
        prompt_embeds_mask[i, :vlen] = 1

    data = TensorDict(
        {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask},
        batch_size=batch_size,
    )
    result = embeds_padding_2_no_padding(data)

    # 1. _ragged_idx must be 1 (ragged at seq_len, not at embedding dim).
    nested_embeds = result["prompt_embeds"]
    assert nested_embeds.is_nested
    ragged_idx = getattr(nested_embeds, "_ragged_idx", None)
    assert ragged_idx == 1, (
        f"Expected _ragged_idx=1 for (bs, [seq_len], dim) nested tensor, got {ragged_idx}. "
        "This breaks concat_nested_tensors and chunk_tensordict (fixed in verl commit 5d4bc46f)."
    )

    # 2. concat_nested_tensors across two identical batches must preserve shapes and values.
    nested_a = result["prompt_embeds"]
    nested_b = result["prompt_embeds"]
    concatenated = tu.concat_nested_tensors([nested_a, nested_b])
    assert concatenated.is_nested
    expected_lens = valid_lens + valid_lens
    for i, exp_len in enumerate(expected_lens):
        sample = concatenated[i]
        assert sample.shape == (exp_len, dim), (
            f"concat sample {i}: expected shape ({exp_len}, {dim}), got {sample.shape}"
        )
        orig_i = i % batch_size
        torch.testing.assert_close(sample, prompt_embeds[orig_i, : valid_lens[orig_i], :])

    # 3. chunk_tensordict must split nested tensors along the batch dim, not embed dim.
    chunks = tu.chunk_tensordict(result, chunks=2)
    assert len(chunks) == 2
    chunk_sizes = [batch_size // 2, batch_size - batch_size // 2]
    offset = 0
    for chunk, expected_size in zip(chunks, chunk_sizes, strict=True):
        chunk_embeds = chunk["prompt_embeds"]
        assert chunk_embeds.is_nested
        for j in range(expected_size):
            orig_i = offset + j
            sample = chunk_embeds[j]
            assert sample.shape == (valid_lens[orig_i], dim), (
                f"chunk sample (orig {orig_i}): expected ({valid_lens[orig_i]}, {dim}), got {sample.shape}"
            )
            torch.testing.assert_close(sample, prompt_embeds[orig_i, : valid_lens[orig_i], :])
        offset += expected_size


if __name__ == "__main__":
    test_embeds_padding_2_no_padding_varying_lengths()
    test_embeds_padding_2_no_padding_uniform_length()
    test_embeds_padding_2_no_padding_nested_tensor_ragged_idx()
    print("All padding tests passed!")
