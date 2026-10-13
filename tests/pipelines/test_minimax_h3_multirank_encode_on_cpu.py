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
"""Multi-rank CPU tests for MiniMax H3 encode_prompt collective synchronization.

Verifies that rank 0 (with prepared inputs) and rank > 0 (with prepared=None)
follow the exact same collective sequence for T2VA, FL2VA, and Ref2VA under both
success and fault propagation paths.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MiniMaxH3RolloutWeightSyncMixin as NFTWeightSyncMixin,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import (
    MiniMaxH3WeightSyncMixin as FlowGRPOWeightSyncMixin,
)


class _CollectiveCoordinator:
    """Simulates upstream multi-rank lockstep collectives between rank 0 and rank > 0."""

    def __init__(self):
        self.step = 0
        self.rank0_exception = None
        self.broadcast_data = None

    def execute_rank(self, rank: int, prepared, tokenizer, fail_rank0: bool = False):
        if rank == 0:
            if fail_rank0:
                self.rank0_exception = ValueError("Simulated rank 0 preparation failure")
            else:
                # Upstream rank 0 uses self.tokenizer to build presentation
                tokens = tokenizer(prepared.prompt)
                self.broadcast_data = tokens["input_ids"]

        # Collective 1: _broadcast_rank0_exception
        if self.rank0_exception is not None:
            if rank == 0:
                raise self.rank0_exception
            raise RuntimeError(f"[rank 0] {type(self.rank0_exception).__name__}: {self.rank0_exception}")

        # Collective 2: broadcast tensors across ranks
        assert self.broadcast_data is not None
        return torch.tensor(self.broadcast_data), torch.ones(len(self.broadcast_data))


@pytest.mark.parametrize("mixin_cls", [NFTWeightSyncMixin, FlowGRPOWeightSyncMixin])
@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_minimax_h3_encode_prompt_multirank_success(mixin_cls, task):
    """Rank 0 and Rank 1 both delegate to super().encode_prompt and follow identical collectives."""
    coordinator = _CollectiveCoordinator()

    class Parent:
        def encode_prompt(self, prepared):
            rank = getattr(self, "rank", 0)
            return coordinator.execute_rank(rank, prepared, self.tokenizer, fail_rank0=False)

    class TestPipeline(mixin_cls, Parent):
        def __init__(self, rank: int, prompt_ids: torch.Tensor):
            self.rank = rank
            self._h3_prompt_ids = prompt_ids
            self.tokenizer = lambda text: {"input_ids": [999]}

    prompt_ids = torch.tensor([101, 102, 103])
    pipe_rank0 = TestPipeline(rank=0, prompt_ids=prompt_ids)
    pipe_rank1 = TestPipeline(rank=1, prompt_ids=prompt_ids)

    orig_tok0 = pipe_rank0.tokenizer
    orig_tok1 = pipe_rank1.tokenizer

    # Upstream only prepares inputs on rank 0; rank > 0 receives prepared=None
    prepared_rank0 = SimpleNamespace(prompt="a golden retriever", media=SimpleNamespace(task=task))
    prepared_rank1 = None

    hidden0, tags0 = pipe_rank0.encode_prompt(prepared_rank0)
    hidden1, tags1 = pipe_rank1.encode_prompt(prepared_rank1)

    # Rank 0 used the token override, passing pretokenized prompt_ids through
    assert hidden0.tolist() == [101, 102, 103]
    # Rank 1 received the broadcast without deadlock
    assert hidden1.tolist() == [101, 102, 103]
    assert tags0.tolist() == tags1.tolist() == [1.0, 1.0, 1.0]

    # Tokenizer is cleanly restored on rank 0, and untouched on rank 1
    assert pipe_rank0.tokenizer is orig_tok0
    assert pipe_rank1.tokenizer is orig_tok1


@pytest.mark.parametrize("mixin_cls", [NFTWeightSyncMixin, FlowGRPOWeightSyncMixin])
@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_minimax_h3_encode_prompt_multirank_fault_propagation(mixin_cls, task):
    """Fault on rank 0 propagates to rank 1 via collective exception without deadlock, restoring tokenizer."""
    coordinator = _CollectiveCoordinator()

    class Parent:
        def encode_prompt(self, prepared):
            rank = getattr(self, "rank", 0)
            return coordinator.execute_rank(rank, prepared, self.tokenizer, fail_rank0=True)

    class TestPipeline(mixin_cls, Parent):
        def __init__(self, rank: int, prompt_ids: torch.Tensor):
            self.rank = rank
            self._h3_prompt_ids = prompt_ids
            self.tokenizer = lambda text: {"input_ids": [999]}

    prompt_ids = torch.tensor([201, 202])
    pipe_rank0 = TestPipeline(rank=0, prompt_ids=prompt_ids)
    pipe_rank1 = TestPipeline(rank=1, prompt_ids=prompt_ids)

    orig_tok0 = pipe_rank0.tokenizer
    orig_tok1 = pipe_rank1.tokenizer

    prepared_rank0 = SimpleNamespace(prompt="a soaring eagle", media=SimpleNamespace(task=task))
    prepared_rank1 = None

    # Rank 0 raises original exception; tokenizer must be restored via finally block
    with pytest.raises(ValueError, match="Simulated rank 0 preparation failure"):
        pipe_rank0.encode_prompt(prepared_rank0)

    assert pipe_rank0.tokenizer is orig_tok0

    # Rank 1 raises the propagated exception matching upstream behavior
    with pytest.raises(RuntimeError, match=r"\[rank 0\] ValueError: Simulated rank 0 preparation failure"):
        pipe_rank1.encode_prompt(prepared_rank1)

    assert pipe_rank1.tokenizer is orig_tok1


@pytest.mark.parametrize("mixin_cls", [NFTWeightSyncMixin, FlowGRPOWeightSyncMixin])
def test_minimax_h3_encode_prompt_bypasses_override_when_no_prompt_ids(mixin_cls):
    """When _h3_prompt_ids is None, both ranks bypass override and delegate directly to parent."""
    parent_mock = MagicMock(return_value=("parent_hidden", "parent_tags"))

    class Parent:
        def encode_prompt(self, prepared):
            return parent_mock(prepared)

    class TestPipeline(mixin_cls, Parent):
        def __init__(self):
            self._h3_prompt_ids = None
            self.tokenizer = "real-tok"

    pipe = TestPipeline()
    prepared = SimpleNamespace(prompt="test", media=SimpleNamespace(task="t2va"))
    result = pipe.encode_prompt(prepared)

    assert result == ("parent_hidden", "parent_tags")
    assert pipe.tokenizer == "real-tok"
    parent_mock.assert_called_once_with(prepared)
