# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""BSHD replay targets for Megatron R3 router replay in verl-omni."""

from __future__ import annotations

import torch


def build_bshd_router_replay_targets(
    routes: torch.Tensor | None,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    padded_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place rollout routes into the padded BSHD layout the omni forward sees.

    verl's replay helper packs routes to THD; the omni Megatron forward is BSHD,
    whose flattened token order is ``[seq, batch]``. Rollout records one route
    row per token, so a sample carries ``len`` rows or ``len - 1`` (the final
    generated token has no route to replay).

    Returns:
        Targets ``[padded_len * bs, layers, topk]`` in model token order and the
        bool mask ``[padded_len * bs]``; unmasked rows keep native routes.
    """
    if routes is None or not input_ids.is_nested:
        raise RuntimeError("R3 router replay requires jagged training inputs and rollout 'routed_experts'.")
    parts = list(routes.unbind())
    lengths = input_ids.offsets().diff().tolist()
    if len(parts) != len(lengths):
        raise RuntimeError(f"R3 router replay has {len(parts)} route sequences for {len(lengths)} samples.")
    if loss_mask.is_nested:
        has_response = [bool(part.any()) for part in loss_mask.unbind()]
    else:
        has_response = (loss_mask.sum(dim=-1) > 0).tolist()

    layers, topk = parts[0].shape[1:]
    batch_size = len(lengths)
    targets = parts[0].new_zeros((batch_size, padded_len, layers, topk))
    replay_mask = torch.zeros((batch_size, padded_len), dtype=torch.bool, device=targets.device)
    for sample_id, (part, length) in enumerate(zip(parts, lengths, strict=True)):
        if length > padded_len:
            raise RuntimeError(f"R3 router replay sample {sample_id} has {length} tokens, above {padded_len}.")
        if part.shape[0] == length - 1:
            part = torch.cat([part, part.new_zeros((1, layers, topk))], dim=0)
        elif part.shape[0] != length:
            raise RuntimeError(
                f"R3 router replay sample {sample_id} has {part.shape[0]} route rows for {length} tokens; "
                "expected equal lengths or one missing final route."
            )
        targets[sample_id, :length] = part
        if has_response[sample_id]:
            replay_mask[sample_id, : max(length - 1, 0)] = True

    return (
        targets.permute(1, 0, 2, 3).reshape(padded_len * batch_size, layers, topk).contiguous(),
        replay_mask.permute(1, 0).reshape(padded_len * batch_size).contiguous(),
    )
