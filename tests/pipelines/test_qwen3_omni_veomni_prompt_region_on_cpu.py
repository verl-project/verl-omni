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
"""Prompt boundaries in the packed VeOmni input layout."""

import importlib.util
from pathlib import Path

import torch
from tensordict import TensorDict

# Load the CPU helpers without importing optional rollout packages.
_path = Path(__file__).resolve().parents[2] / "verl_omni/pipelines/qwen3_omni/veomni.py"
_spec = importlib.util.spec_from_file_location("qwen3_omni_veomni_helpers", _path)
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
build_prompt_region_mask = adapter._build_prompt_region_mask


def _packed_batch(sequences, response_lengths):
    """Pack ``sequences`` the way verl's NO_PADDING path hands them to an engine."""

    return TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.tensor(seq) for seq in sequences], layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor(
                [torch.ones(n, dtype=torch.bool) for n in response_lengths], layout=torch.jagged
            ),
        },
        batch_size=len(sequences),
    )


def test_build_prompt_region_mask_splits_every_sequence():
    # Two packed sequences, 4+2 and 3+3 tokens: prompt first, then response.
    micro_batch = _packed_batch([[1] * 6, [2] * 6], response_lengths=[2, 3])

    mask = build_prompt_region_mask(micro_batch, packed_length=12)

    torch.testing.assert_close(
        mask,
        torch.tensor([[True] * 4 + [False] * 2 + [True] * 3 + [False] * 3]),
    )


def test_build_prompt_region_mask_pads_to_the_packed_width():
    # pad_to_length right-pads the packed sequence; the pad is never prompt.
    micro_batch = _packed_batch([[1, 1, 1, 1]], response_lengths=[1])

    mask = build_prompt_region_mask(micro_batch, packed_length=8)

    torch.testing.assert_close(mask, torch.tensor([[True, True, True] + [False] * 5]))


def test_build_prompt_region_mask_returns_none_without_response_mask():
    micro_batch = TensorDict(
        {"input_ids": torch.nested.as_nested_tensor([torch.tensor([1, 2, 3])], layout=torch.jagged)},
        batch_size=1,
    )

    assert build_prompt_region_mask(micro_batch, packed_length=3) is None
