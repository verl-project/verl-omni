# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the BSHD layout used by Megatron R3 router replay."""

import pytest
import torch

from verl_omni.workers.engine.megatron.router_replay import build_bshd_router_replay_targets

LAYERS, TOPK = 2, 2


def _routes(lengths):
    parts = [torch.arange(n * LAYERS * TOPK).reshape(n, LAYERS, TOPK) + 100 * i for i, n in enumerate(lengths)]
    return torch.nested.nested_tensor(parts, layout=torch.jagged).to(torch.int16)


def _inputs(lengths):
    return torch.nested.nested_tensor([torch.zeros(n, dtype=torch.long) for n in lengths], layout=torch.jagged)


def test_pads_routes_into_seq_major_model_order():
    lengths, padded_len = [3, 2], 4
    targets, mask = build_bshd_router_replay_targets(
        _routes(lengths), _inputs(lengths), torch.ones(2, 2), padded_len
    )

    assert targets.shape == (padded_len * 2, LAYERS, TOPK)
    view, parts = targets.view(padded_len, 2, LAYERS, TOPK), _routes(lengths).unbind()
    for sample_id, length in enumerate(lengths):
        for position in range(length):
            assert torch.equal(view[position, sample_id], parts[sample_id][position])
        assert torch.count_nonzero(view[length:, sample_id]) == 0
    # Every row is replayed except the final generated token, which has no route.
    assert mask.view(padded_len, 2).tolist() == [[True, True], [True, False], [False, False], [False, False]]


def test_missing_final_route_gets_a_placeholder_row():
    targets, mask = build_bshd_router_replay_targets(_routes([2]), _inputs([3]), torch.ones(1, 2), 3)

    view, (routes,) = targets.view(3, 1, LAYERS, TOPK), _routes([2]).unbind()
    assert torch.equal(view[0, 0], routes[0]) and torch.equal(view[1, 0], routes[1])
    assert torch.count_nonzero(view[2, 0]) == 0
    assert mask.tolist() == [True, True, False]


def test_nested_loss_mask_marks_response_samples_only():
    loss_mask = torch.nested.nested_tensor([torch.ones(2), torch.zeros(1)], layout=torch.jagged)
    _, mask = build_bshd_router_replay_targets(_routes([3, 3]), _inputs([3, 3]), loss_mask, 3)

    assert mask.view(3, 2).tolist() == [[True, False], [True, False], [False, False]]


def test_rejects_unreplayable_inputs():
    with pytest.raises(RuntimeError, match="routed_experts"):
        build_bshd_router_replay_targets(None, _inputs([2]), torch.ones(1, 1), 2)
    with pytest.raises(RuntimeError, match="jagged"):
        build_bshd_router_replay_targets(_routes([2]), torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 1), 2)
    with pytest.raises(RuntimeError, match="route rows"):
        build_bshd_router_replay_targets(_routes([1]), _inputs([4]), torch.ones(1, 1), 4)
    with pytest.raises(RuntimeError, match="above 2"):
        build_bshd_router_replay_targets(_routes([4]), _inputs([4]), torch.ones(1, 1), 2)
