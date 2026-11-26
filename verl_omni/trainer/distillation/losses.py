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
    distillation_loss,
    distillation_ppo_loss,
    register_distillation_loss,
)
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.distillation.nitrobrew_loss import (
    compute_nitrobrew_multi_kl,
    compute_nitrobrew_multi_reverse_kl,
    hidden_token_count,
)
from verl_omni.workers.config.omni.distillation import (
    HIDDEN_STATE_LOSS_MODES,
    OmniDistillationLossSettings,
    is_self_distillation_enabled,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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


def _compute_nitrobrew_in_logits_processor(
    distillation_config,
    data,
    student_logits: torch.Tensor,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Chunked nitrobrew KL over the full student_logits tensor."""

    loss_mode = distillation_config.distillation_loss.loss_mode
    reverse = loss_mode == "nitrobrew_reverse_kl"

    teacher_hidden_states = data["teacher_hidden_states"]
    # NonTensorData flattened by tensordict on read-back.
    unembeds_map = data["teacher_unembeds"]
    key_to_id = data["teacher_key_to_id"]

    # Rearrange unembeds to be keyed by int teacher id (MOPD-aware).
    unembeds = {int(key_to_id[k]): W for k, W in unembeds_map.items()}

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
    per_seq = data.get("teacher_key", None)
    if per_seq is None:
        return None
    if isinstance(per_seq, NonTensorStack):
        per_seq = list(per_seq)
    elif not isinstance(per_seq, list | tuple):
        per_seq = [per_seq]
    return [item.data if isinstance(item, NonTensorData) else item for item in per_seq]


def _per_token_teacher_key_ids(data, teacher_hidden_states, key_to_id):
    """[T] teacher ids over the packed token layout; single teacher -> all zeros."""
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


def _mask_like_response_mask(raw_response_mask: torch.Tensor, padded_bool_mask: torch.Tensor) -> torch.Tensor:
    """Convert a padded boolean mask back to the layout and dtype of ``raw_response_mask``."""
    if raw_response_mask.is_nested:
        pieces = [
            padded_bool_mask[i, : piece.size(0)].to(dtype=raw_response_mask.dtype)
            for i, piece in enumerate(raw_response_mask.unbind())
        ]
        return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)
    return padded_bool_mask.to(dtype=raw_response_mask.dtype)


def _differentiable_zero(model_output: dict, fallback_device: torch.device) -> torch.Tensor:
    """Return a scalar zero anchored to the autograd graph when model outputs require grad."""
    for key in ("log_probs", "distillation_losses"):
        tensor = model_output.get(key)
        if isinstance(tensor, torch.Tensor):
            vals = tensor.values() if tensor.is_nested else tensor
            return vals.reshape(-1)[:0].sum(dtype=torch.float32)
    return torch.zeros((), dtype=torch.float32, device=fallback_device)


def _clone_data_with_response_mask(data, new_mask: torch.Tensor):
    """Return a shallow dict/TensorDict view of ``data`` with ``response_mask`` replaced."""
    if hasattr(data, "clone"):
        cloned = data.clone(recurse=False)
        cloned["response_mask"] = new_mask
        return cloned
    cloned = dict(data)
    cloned["response_mask"] = new_mask
    return cloned


def _compute_self_distillation_ppo_loss(
    config,
    distillation_config,
    model_output: dict,
    data,
    dp_group=None,
):
    """Combine per-sample self-distillation with optional GSPO/GRPO fallback on groups lacking context."""
    raw_response_mask = data["response_mask"]
    if raw_response_mask.is_nested:
        response_mask = raw_response_mask.bool().to_padded_tensor(False)
    else:
        response_mask = raw_response_mask.bool()

    sample_mask = data["self_distill_mask"]
    if sample_mask.is_nested:
        sample_mask = sample_mask.bool().to_padded_tensor(False)
    else:
        sample_mask = sample_mask.bool()
    sample_mask = sample_mask.to(device=response_mask.device)
    if sample_mask.ndim == 1:
        sample_mask_2d = sample_mask.unsqueeze(1).expand_as(response_mask)
    else:
        sample_mask_2d = sample_mask

    distill_mask = response_mask & sample_mask_2d
    loss_cfg = distillation_config.distillation_loss
    fallback_to_pg = bool(distillation_config.self_teacher.fallback_to_pg)
    if loss_cfg.use_task_rewards:
        pg_mask = response_mask
    elif fallback_to_pg:
        pg_mask = response_mask & ~sample_mask_2d
    else:
        pg_mask = torch.zeros_like(response_mask)

    zero_scalar = _differentiable_zero(model_output, response_mask.device)
    zero_metric = torch.zeros((), dtype=torch.float32, device=response_mask.device)

    metrics: dict = {}
    if distill_mask.any():
        distill_data = _clone_data_with_response_mask(data, _mask_like_response_mask(raw_response_mask, distill_mask))
        distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, distill_data)
        metrics.update(distill_metrics)
    else:
        distill_loss = zero_scalar

    if pg_mask.any():
        pg_data = _clone_data_with_response_mask(data, _mask_like_response_mask(raw_response_mask, pg_mask))
        policy_loss, policy_metrics = ppo_loss(config, model_output, pg_data, dp_group)
        metrics.update(policy_metrics)
    else:
        policy_loss = zero_scalar

    distill_coef = loss_cfg.distillation_loss_coef if loss_cfg.use_task_rewards else 1.0
    total_loss = policy_loss + distill_loss * distill_coef
    metrics["distillation/loss"] = Metric(value=distill_loss.detach(), aggregation=AggregationType.SUM)
    metrics["self_distillation/distilled_tokens"] = Metric(
        value=distill_mask.sum().to(dtype=torch.float32), aggregation=AggregationType.SUM
    )
    metrics["self_distillation/fallback_tokens"] = Metric(
        value=(response_mask & ~sample_mask_2d).sum().to(dtype=torch.float32) if fallback_to_pg else zero_metric,
        aggregation=AggregationType.SUM,
    )
    return total_loss, metrics


def omni_distillation_ppo_loss(
    config,
    distillation_config,
    model_output: dict | None = None,
    data=None,
    dp_group=None,
    student_logits: torch.Tensor | None = None,
    data_format: str = "thd",
    **kwargs,
):
    """Same entry as verl's ``distillation_ppo_loss``: logits processor + final
    policy loss, with hidden modes dispatched to the chunked nitrobrew kernel
    and self-distillation fallback masking handled when ``self_distill_mask``
    is present."""
    loss_mode = distillation_config.distillation_loss.loss_mode
    use_hidden = loss_mode in HIDDEN_STATE_LOSS_MODES

    if student_logits is not None and use_hidden:
        return _compute_nitrobrew_in_logits_processor(distillation_config, data, student_logits, data_format)

    if (
        student_logits is None
        and data is not None
        and is_self_distillation_enabled(distillation_config)
        and "self_distill_mask" in data
    ):
        return _compute_self_distillation_ppo_loss(
            config=config,
            distillation_config=distillation_config,
            model_output=model_output,
            data=data,
            dp_group=dp_group,
        )

    return distillation_ppo_loss(
        config,
        distillation_config,
        model_output,
        data,
        dp_group,
        student_logits,
        data_format,
        **kwargs,
    )
