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
"""Distillation loss dispatch for the omni hidden-state (nitrobrew) path.

The verl logits processor calls ``loss_function(student_logits=..., data=...)``
for topk-style losses. ``use_topk=True`` in the omni hidden loss_settings is a
side-channel that makes verl run the logits processor with the full
``student_logits``; this module's wrapper intercepts that call and dispatches
to the chunked-vocab nitrobrew kernel instead of the top-k KL.

The final policy-loss aggregation is handled by re-registering ``nitrobrew`` /
``nitrobrew_reverse_kl`` into verl's loss registry (a plain ``no_padding`` +
``clamp_min(0)`` aggregation, mirroring verl PR 6194), so verl's
``distillation_loss`` / ``distillation_ppo_loss`` plumbing is reused unchanged.
"""

import logging
import os

import torch
from verl.trainer.distillation import (
    DistillationLossSettings,
    distillation_ppo_loss,
    register_distillation_loss,
)
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.distillation.nitrobrew_loss import (
    compute_nitrobrew_multi_kl,
    compute_nitrobrew_multi_reverse_kl,
    hidden_token_count,
)
from verl_omni.workers.config.omni.distillation import HIDDEN_STATE_LOSS_MODES

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Final-policy-loss aggregation (registered into verl's loss registry)
# ---------------------------------------------------------------------------


@register_distillation_loss(DistillationLossSettings(names=["nitrobrew", "nitrobrew_reverse_kl"], use_topk=True))  # type: ignore[arg-type]
def compute_nitrobrew_loss_aggregate(
    config,
    distillation_config,
    model_output: dict,
    data,
):
    """Aggregate per-token nitrobrew KL computed in the logits processor."""
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == response_mask_bool.shape

    # log_prob_min_clamp makes the computed KL no longer a true divergence; it
    # can go negative where the student locally outperforms the teacher on
    # clamped tokens. Floor at zero to prevent negative losses acting as reward.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, {}


# ---------------------------------------------------------------------------
# Logits-processor dispatch (intercepts the student_logits call)
# ---------------------------------------------------------------------------


def _distillation_loss_settings_use_hidden(distillation_config) -> bool:
    settings = distillation_config.distillation_loss.loss_settings
    return bool(getattr(settings, "use_hidden_states", False))


def _compute_nitrobrew_in_logits_processor(
    distillation_config,
    data,
    student_logits: torch.Tensor,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Run the chunked nitrobrew KL on the full student_logits tensor."""

    loss_mode = distillation_config.distillation_loss.loss_mode
    reverse = loss_mode == "nitrobrew_reverse_kl"

    teacher_hidden_states = data["teacher_hidden_states"]
    # NonTensorData flattened by tensordict on read-back.
    unembeds_map = data["teacher_unembeds"]
    key_to_id = data["teacher_key_to_id"]

    # key_to_id maps teacher key -> int id (MOPD-aware). Rearrange unembeds by id.
    unembeds = {int(key_to_id[k]): W for k, W in unembeds_map.items()}

    # Per-token route: single-teacher degenerate to all-zero id; multi-teacher
    # expands per-sequence ids over the jagged token offsets (see note below).
    key_ids = _per_token_teacher_key_ids(data, teacher_hidden_states, key_to_id)

    fn = compute_nitrobrew_multi_reverse_kl if reverse else compute_nitrobrew_multi_kl
    return fn(
        student_logits=student_logits,
        teacher_hidden_states=teacher_hidden_states,
        teacher_key_ids=key_ids,
        teacher_unembeds=unembeds,
        config=distillation_config,
        data_format=data_format,
    )


def _per_token_teacher_key_ids(data, teacher_hidden_states, key_to_id):
    """Build [T] int teacher id per rmpad token.

    Single teacher: every token routes to the sole unembedding -> all zeros.
    Multi teacher: ``data["teacher_key_ids"]`` (per-sequence int ids) is
    expanded over the jagged token offsets of the hidden tensor so each token
    is grouped under its originating teacher. If the per-sequence ids are not
    carried (older recipe), falls back to the single-teacher constant.
    """
    multi_teacher = len(key_to_id) > 1
    per_seq = data.get("teacher_key_ids", None) if multi_teacher else None
    if per_seq is None:
        if multi_teacher:
            logger.warning("multi-teacher hidden OPD without per-token key ids; routing all to id 0")
        T = hidden_token_count(teacher_hidden_states)
        return torch.zeros(T, dtype=torch.long, device=teacher_hidden_states.device)

    key_ids = per_seq.to(device=teacher_hidden_states.device, dtype=torch.long)
    if teacher_hidden_states.is_nested:
        seq_lens = torch.diff(teacher_hidden_states.offsets())
        return torch.repeat_interleave(key_ids, seq_lens)
    return key_ids


def omni_distillation_ppo_loss(
    config,
    distillation_config,
    model_output: dict | None = None,
    data=None,
    dp_group=None,
    student_logits: torch.Tensor | None = None,
    data_format: str = "thd",
):
    """Loss function used both as the logits processor and the final policy loss.

    Mirrors verl's ``distillation_ppo_loss`` but dispatches the logits-processor
    call to the chunked nitrobrew kernel when the hidden loss mode is active.
    The final-policy-loss path delegates to verl unchanged (its registry now
    knows the nitrobrew aggregate).
    """
    loss_mode = distillation_config.distillation_loss.loss_mode
    use_hidden = loss_mode in HIDDEN_STATE_LOSS_MODES

    if student_logits is not None and use_hidden:
        return _compute_nitrobrew_in_logits_processor(distillation_config, data, student_logits, data_format)

    return distillation_ppo_loss(
        config,
        distillation_config,
        model_output,
        data,
        dp_group,
        student_logits,
        data_format,
    )
