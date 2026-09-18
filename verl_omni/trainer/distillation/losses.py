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
"""Distillation loss dispatch for the omni hidden-state (nitrobrew) path."""

import logging
import os

import torch
from tensordict import NonTensorData, NonTensorStack
from verl.trainer.distillation import (
    distillation_ppo_loss,
    register_distillation_loss,
)
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.distillation.nitrobrew_loss import (
    compute_nitrobrew_multi_kl,
    compute_nitrobrew_multi_reverse_kl,
    hidden_token_count,
)
from verl_omni.workers.config.omni.distillation import HIDDEN_STATE_LOSS_MODES, OmniDistillationLossSettings

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Final-policy-loss aggregation (registered into verl's loss registry)
# ---------------------------------------------------------------------------


@register_distillation_loss(
    OmniDistillationLossSettings(names=["nitrobrew", "nitrobrew_reverse_kl"], use_hidden_states=True)
)  # type: ignore[arg-type]
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


def _per_seq_teacher_keys(data) -> list[str] | None:
    """Read the per-sequence teacher routing keys carried by the agent loop."""
    per_seq = data.get("teacher_key", None)
    if per_seq is None:
        return None
    if isinstance(per_seq, NonTensorStack):
        per_seq = list(per_seq)
    elif not isinstance(per_seq, list | tuple):
        per_seq = [per_seq]
    return [item.data if isinstance(item, NonTensorData) else item for item in per_seq]


def _per_token_teacher_key_ids(data, teacher_hidden_states, key_to_id):
    """Build [T] int teacher id per packed token.

    Multi teacher: resolve ``data["teacher_key"]`` through ``key_to_id`` and
    expand over the packed token layout; single teacher: all zeros.
    """
    device = teacher_hidden_states.device
    T = hidden_token_count(teacher_hidden_states)
    if len(key_to_id) <= 1:
        return torch.zeros(T, dtype=torch.long, device=device)

    keys = _per_seq_teacher_keys(data)
    if keys is None:
        logger.warning("multi-teacher hidden OPD without per-sequence teacher_key; routing all tokens to id 0")
        return torch.zeros(T, dtype=torch.long, device=device)

    unknown = sorted(set(keys) - set(key_to_id))
    if unknown:
        raise ValueError(f"teacher_key values {unknown} have no id mapping; known keys: {sorted(key_to_id)}.")
    per_seq_ids = torch.tensor([key_to_id[k] for k in keys], dtype=torch.long, device=device)

    if teacher_hidden_states.is_nested:
        # unbind rather than offsets(): CPU nested tensors use the slow-path
        # layout and do not expose offsets().
        seq_lens = torch.tensor([t.shape[0] for t in teacher_hidden_states.unbind()], device=device)
        return torch.repeat_interleave(per_seq_ids, seq_lens)
    # Flat [bsz, S, D] packing: one id per (sequence, position) pair.
    return per_seq_ids.repeat_interleave(teacher_hidden_states.shape[1])


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
    call to the chunked nitrobrew kernel for hidden loss modes.
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
