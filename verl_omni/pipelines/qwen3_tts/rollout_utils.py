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
"""Helpers for validating Qwen3-TTS rollout outputs."""

import torch

QWEN3_TTS_REPLAY_KEY = "qwen3_tts_talker_replay"


def align_audio_codes(audio_codes: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Recover residual codebooks using codec-0 policy tokens as an exact invariant.

    The engine can prepend placeholder rows and need not emit residual codes
    for the last sampled token, because that token is never consumed as the
    context for another policy token. Thus ``token_ids[:-1]`` must match
    exactly; a final residual row is retained only when the engine provides it.
    """
    if audio_codes.ndim != 2 or audio_codes.shape[-1] != 16:
        raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
    if not token_ids:
        return audio_codes[:0].long()
    raw_codes = audio_codes.long()
    policy_ids = torch.as_tensor(token_ids, dtype=torch.long, device=raw_codes.device)
    required = len(token_ids) - 1
    candidates: list[tuple[int, int]] = []
    if required == 0:
        candidates.append((0, 0))
    else:
        for start in range(raw_codes.shape[0] - required + 1):
            if torch.equal(raw_codes[start : start + required, 0], policy_ids[:required]):
                has_final = int(
                    raw_codes.shape[0] - start >= len(token_ids)
                    and raw_codes[start + required, 0] == policy_ids[required]
                )
                candidates.append((required + has_final, start))
    if not candidates:
        raise RuntimeError(
            "Could not exactly align Qwen3-TTS residual codebooks with the sampled codec-0 policy: "
            f"response_length={len(token_ids)}, raw_codec_rows={raw_codes.shape[0]}."
        )

    copy_length = max(length for length, _ in candidates)
    best_starts = [start for length, start in candidates if length == copy_length]
    if len(best_starts) != 1:
        raise RuntimeError(
            "Ambiguous Qwen3-TTS codec alignment: multiple residual-codebook spans match the sampled policy."
        )
    start = best_starts[0]
    codes = raw_codes.new_zeros((len(token_ids), 16))
    if copy_length:
        codes[:copy_length] = raw_codes[start : start + copy_length]
    codes[:, 0] = policy_ids
    if required and not torch.equal(codes[:required, 0], policy_ids[:required]):
        raise RuntimeError("Qwen3-TTS codec alignment invariant failed after recovery.")
    return codes
