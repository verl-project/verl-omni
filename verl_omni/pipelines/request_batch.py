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

"""Helpers for request-level diffusion batching in vLLM-Omni rollout adapters."""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "collate_prompt_mask",
    "collate_prompt_rows",
    "split_diffusion_output_by_request",
]


def _to_prompt_row(value: Any, *, device: torch.device, field_name: str) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor[0]
    raise ValueError(f"Request-batch {field_name} must be 1D or single-row 2D, got shape={tuple(tensor.shape)}.")


def _get_prompt_field(prompt: Any, aliases: tuple[str, ...]) -> Any:
    if isinstance(prompt, str) or not hasattr(prompt, "get"):
        return None
    for name in aliases:
        value = prompt.get(name)
        if value is None:
            additional = prompt.get("additional_information")
            if isinstance(additional, dict):
                value = additional.get(name)
        if value is not None:
            return value
    return None


def _rows_from_default(
    value: torch.Tensor | list[int] | None,
    *,
    device: torch.device,
    field_name: str,
) -> tuple[torch.Tensor | None, list[int] | None]:
    if value is None:
        return None, None
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{field_name} must be 1D or 2D, got shape={tuple(tensor.shape)}.")
    return tensor, [int(tensor.shape[1])] * int(tensor.shape[0])


def collate_prompt_rows(
    prompts: list[Any],
    aliases: tuple[str, ...],
    default_value: torch.Tensor | list[int] | None,
    *,
    device: torch.device,
    field_name: str,
    pad_value: int = 0,
) -> tuple[torch.Tensor | None, list[int] | None]:
    default_rows, default_lengths = _rows_from_default(default_value, device=device, field_name=field_name)
    if default_rows is not None:
        if len(prompts) > 1 and default_rows.shape[0] != len(prompts):
            raise ValueError(
                f"Batched {field_name} default must have one row per request; "
                f"got {default_rows.shape[0]} rows for {len(prompts)} requests."
            )
        return default_rows, default_lengths

    rows = [
        _to_prompt_row(
            _get_prompt_field(prompt, aliases),
            device=device,
            field_name=field_name,
        )
        for prompt in prompts
    ]
    if not any(row is not None for row in rows):
        return None, None
    if not all(row is not None for row in rows):
        raise ValueError(f"Cannot batch requests with a mix of provided and missing {field_name}.")

    typed_rows = [row for row in rows if row is not None]
    target_len = max(int(row.shape[0]) for row in typed_rows)
    result = torch.full(
        (len(typed_rows), target_len),
        pad_value,
        dtype=typed_rows[0].dtype,
        device=typed_rows[0].device,
    )
    lengths: list[int] = []
    for idx, row in enumerate(typed_rows):
        row_len = int(row.shape[0])
        result[idx, :row_len] = row
        lengths.append(row_len)
    return result, lengths


def collate_prompt_mask(
    prompts: list[Any],
    aliases: tuple[str, ...],
    default_value: torch.Tensor | list[int] | None,
    *,
    device: torch.device,
    field_name: str,
    token_lengths: list[int] | None,
    target_seq_len: int | None,
) -> torch.Tensor | None:
    mask, _ = collate_prompt_rows(
        prompts,
        aliases,
        default_value,
        device=device,
        field_name=field_name,
        pad_value=0,
    )
    if mask is not None:
        mask = mask != 0
        if target_seq_len is not None:
            if mask.shape[1] < target_seq_len:
                padded = torch.zeros((mask.shape[0], target_seq_len), dtype=torch.bool, device=mask.device)
                padded[:, : mask.shape[1]] = mask
                mask = padded
            elif mask.shape[1] > target_seq_len:
                mask = mask[:, :target_seq_len]
        return mask

    if token_lengths is None or target_seq_len is None:
        return None

    mask = torch.zeros((len(token_lengths), target_seq_len), dtype=torch.bool, device=device)
    for idx, row_len in enumerate(token_lengths):
        mask[idx, :row_len] = True
    return mask


def _slice_batch_value(value: Any, start: int, stop: int, expected_batch_size: int) -> Any:
    # Slice only when leading size matches the packed batch; leave shared T/L axes alone.
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[start:stop] if value.ndim > 0 and value.shape[0] == expected_batch_size else value
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, start, stop, expected_batch_size) for item in value)
    if isinstance(value, list):
        return value[start:stop] if len(value) == expected_batch_size else value
    return value


def split_diffusion_output_by_request(
    result: Any,
    req: Any,
    *,
    num_outputs_per_prompt: int,
) -> list[Any]:
    outputs: list[Any] = []
    custom_output = result.custom_output or {}
    expected_batch_size = req.num_reqs * num_outputs_per_prompt
    for idx in range(req.num_reqs):
        start = idx * num_outputs_per_prompt
        stop = (idx + 1) * num_outputs_per_prompt
        outputs.append(
            result.__class__(
                output=_slice_batch_value(result.output, start, stop, expected_batch_size),
                trajectory_timesteps=_slice_batch_value(result.trajectory_timesteps, start, stop, expected_batch_size),
                trajectory_latents=_slice_batch_value(result.trajectory_latents, start, stop, expected_batch_size),
                trajectory_log_probs=_slice_batch_value(result.trajectory_log_probs, start, stop, expected_batch_size),
                trajectory_decoded=_slice_batch_value(result.trajectory_decoded, start, stop, expected_batch_size),
                error=result.error,
                error_status_code=result.error_status_code,
                error_type=result.error_type,
                aborted=result.aborted,
                abort_message=result.abort_message,
                post_process_func=result.post_process_func,
                custom_output={
                    key: _slice_batch_value(value, start, stop, expected_batch_size)
                    for key, value in custom_output.items()
                },
                finished=result.finished,
                chunk_index=result.chunk_index,
                total_chunks=result.total_chunks,
                stage_durations=dict(result.stage_durations),
                peak_memory_mb=result.peak_memory_mb,
                to_cpu=result.to_cpu,
            )
        )
    return outputs
