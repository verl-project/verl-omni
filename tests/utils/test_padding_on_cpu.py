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

from verl.workers.utils.padding import embeds_padding_2_no_padding


def test_embeds_padding_2_no_padding_varying_lengths():
    """Test that padding tokens are stripped correctly when sequences have different valid lengths."""
    batch_size = 3
    max_seq_len = 20
    dim = 16
    num_steps = 8

    # Simulate different valid lengths: 20, 15, 10 (rest are padding zeros)
    valid_lens = [20, 15, 10]
    prompt_embeds = torch.randn(batch_size, max_seq_len, dim)
    prompt_embeds_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.int32)
    for i, vlen in enumerate(valid_lens):
        prompt_embeds_mask[i, :vlen] = 1
    response_mask = torch.ones(batch_size, num_steps)

    data = TensorDict(
        {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "response_mask": response_mask,
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


if __name__ == "__main__":
    test_embeds_padding_2_no_padding_varying_lengths()
    print("All padding tests passed!")
