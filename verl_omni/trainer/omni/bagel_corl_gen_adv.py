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
"""GEN FlowGRPO advantage for Bagel Co-RL (Joint-Training) (RFC flatten → GenAdv → composite update).

UND token GRPO stays on verl V1 ``_compute_advantage``. This module is the GEN
slice: same ``compute_advantage`` helper the diffusion trainers use, grouped by
``gen_group_uid`` (the S seeds of one ``generate_image`` call) when S >= 2, and by
task (``und_group_uid``: every image the N sibling episodes drew) when S == 1.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from verl_omni.trainer.diffusion.ray_diffusion_trainer import compute_advantage

logger = logging.getLogger(__name__)

__all__ = ["apply_gen_flowgrpo_advantage", "build_gen_flowgrpo_proto", "select_gen_advantages_for_step"]


def select_gen_advantages_for_step(micro_batch, step: int, *, require_bagel_corl_gen: bool = False):
    """GEN timestep advantages: ``bagel_corl_gen['advantages'][:, step]`` only.

    When ``require_bagel_corl_gen`` is True (Bagel Co-RL (Joint-Training) GEN engine path), refuse to
    fall back to UND / poisoned ``micro_batch['advantages']``.
    """
    from verl.utils import tensordict_utils as tu

    lane = tu.get_non_tensor_data(micro_batch, "bagel_corl_lane", default=None)
    if require_bagel_corl_gen and lane != "gen":
        raise ValueError("bagel_corl GEN step requires an explicitly tagged GEN microbatch")
    return micro_batch["advantages"][:, step]


def _logprob_1d(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return None
    return tensor


def _as_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _stack_padded(tensors: list[torch.Tensor], *, pad_value: float = 0.0) -> torch.Tensor:
    """Stack tensors that may differ on dim0 (timesteps); pad to max length."""
    if not tensors:
        raise ValueError("empty tensor list")
    max_t = max(int(t.shape[0]) for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[0] == max_t:
            padded.append(tensor)
            continue
        pad_shape = (max_t - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad = tensor.new_full(pad_shape, pad_value)
        padded.append(torch.cat([tensor, pad], dim=0))
    return torch.stack(padded, dim=0)


def _flowgrpo_group_index(usable: list[tuple[Mapping[str, Any], Any, float, str, Any, Any]]) -> list[str]:
    """FlowGRPO group id per usable GEN row."""
    call_sizes: dict[str, int] = {}
    for item in usable:
        call_sizes[item[3]] = call_sizes.get(item[3], 0) + 1
    index: list[str] = []
    for row, _, _, call_uid, _, _ in usable:
        task_uid = row.get("und_group_uid")
        if call_sizes[call_uid] >= 2 or not task_uid:
            index.append(call_uid)
        else:
            index.append(f"task:{task_uid}")
    return index


def build_gen_flowgrpo_proto(gen_batch: list[Mapping[str, Any]]) -> DataProto | None:
    """Stack flattened GEN rows into a diffusion ``DataProto`` for FlowGRPO + actor.

    Requires live traj stash: ``all_latents`` + ``timesteps`` + ``rollout_log_probs``
    (plus ``gen_group_uid`` / ``rm_score``). Refuse logprobs-only soft path that would
    skip GEN ``diffusion_loss``.

    Empty ``gen_batch`` (pattern 3 / ``K=0``) returns ``None`` so UND still trains.
    """
    usable: list[tuple[Mapping[str, Any], torch.Tensor, float, str, torch.Tensor, torch.Tensor]] = []
    missing_traj = 0
    # Reason counters for the "rows arrived but none are trainable" case, which is otherwise a
    # silent skip: with ``ENABLE_RM=0`` every row carries ``rm_score=None``, so a step's
    # ``gen/skipped_no_groups: 1`` / ``has_complete_gen_groups: 0`` are correct but opaque.
    # Naming the missing column here turns "GEN never trains" into one readable line.
    missing_score = missing_logprob = missing_uid = 0
    # Split the log-prob counter by *cause*, because "missing" is ambiguous and the two have different fixes:
    # ``absent`` -- the rebuild handed us ``None``: the value was dropped between generation and advantage (TQ round-
    # trip / row rebuild).
    missing_logprob_absent = missing_logprob_empty = 0
    for row in gen_batch:
        raw_logprob = row.get("rollout_log_probs")
        logprob = _logprob_1d(raw_logprob)
        score = row.get("rm_score")
        uid = row.get("gen_group_uid")
        latents = _as_tensor(row.get("all_latents"))
        timesteps = _as_tensor(
            row.get("all_timesteps") if row.get("all_timesteps") is not None else row.get("timesteps")
        )
        if latents is None or timesteps is None:
            if uid is not None or logprob is not None:
                missing_traj += 1
            continue
        if logprob is None or score is None or uid is None:
            if score is None:
                missing_score += 1
            if logprob is None:
                missing_logprob += 1
                if raw_logprob is None:
                    missing_logprob_absent += 1
                else:
                    missing_logprob_empty += 1
            if uid is None:
                missing_uid += 1
            continue
        usable.append((row, logprob, float(score), str(uid), latents, timesteps))
    if not usable:
        if missing_traj:
            raise RuntimeError(
                "bagel_corl: GEN batch lacks all_latents/timesteps for FlowGRPO "
                f"({missing_traj} row(s)); refuse soft-fallback to rollout_log_probs / skip GEN loss. "
                "Fix live GEN traj stash (calculate_log_probs + algo.noise_level>0)."
            )
        if gen_batch:
            logger.warning(
                "bagel_corl: %d GEN row(s) arrived but none are trainable "
                "(rm_score missing=%d, rollout_log_probs missing=%d [absent=%d empty=%d], "
                "gen_group_uid missing=%d). "
                "FlowGRPO needs all three and this step trains UND only. Read the counts, not "
                "an assumption: with the reward lane ON, 'rm_score missing=0' means the reward "
                "is fine and the row died on a different column. For the log-prob column, "
                "'absent' means the vector was dropped between generation and this point "
                "(TQ round-trip / row rebuild); 'empty' means a zero-length vector reached "
                "here, which the sampler cannot produce with the key present, so treat it as a "
                "plumbing bug rather than an engine setting.",
                len(gen_batch),
                missing_score,
                missing_logprob,
                missing_logprob_absent,
                missing_logprob_empty,
                missing_uid,
            )
        return None

    from verl_omni.trainer.omni.bagel_corl_batch import episode_numeric_id

    max_steps = max(item[1].numel() for item in usable)
    latent_list, timestep_list, prompt_token_ids, old_list, masks, parents = [], [], [], [], [], []
    has_reference = any(row.get("ref_prev_sample_mean") is not None for row, *_ in usable)
    reference_means = []
    for row, logprob, _, _, latents, timesteps in usable:
        timesteps = timesteps.reshape(-1).float()
        count = logprob.numel()
        if count != timesteps.numel() or latents.shape[0] != count + 1:
            raise ValueError("GEN transition lengths require T logprobs, T timesteps, and T+1 latents")
        if (
            not torch.isfinite(timesteps).all()
            or not torch.isfinite(logprob).all()
            or not torch.isfinite(latents).all()
        ):
            raise ValueError("GEN trajectory contains nonfinite transition inputs")
        if (timesteps <= 0).any():
            raise ValueError("GEN recorded stochastic transition timestep must be positive")
        episode_uid = row.get("episode_uid")
        if episode_uid is None:
            raise ValueError("GEN trajectory requires parent episode_uid for unique ownership")
        parents.append(episode_numeric_id(str(episode_uid)))
        pad = max_steps - count
        if has_reference:
            reference = _as_tensor(row.get("ref_prev_sample_mean"))
            if reference is None or reference.shape != (count, *latents.shape[1:]):
                raise ValueError("GEN reference means must be supplied for every row with T latent-shaped transitions")
            if not torch.isfinite(reference).all():
                raise ValueError("GEN reference means contain nonfinite transitions")
            reference_means.append(
                torch.cat([reference.float(), reference[-1:].float().expand(pad, *reference.shape[1:])])
            )
        # Repeat a finite transition timestep/latent instead of synthetic zero SDE inputs.
        latent_list.append(torch.cat([latents.float(), latents[-1:].float().expand(pad, *latents.shape[1:])]))
        timestep_list.append(torch.cat([timesteps, timesteps[:1].expand(pad)]))
        old_list.append(torch.cat([logprob, logprob.new_zeros(pad)]))
        masks.append(torch.cat([torch.ones(count), torch.zeros(pad)]))
        prompt_token_ids.append(row.get("prompt_token_ids"))
    scores = torch.tensor([[item[2]] for item in usable], dtype=torch.float32)
    uids = np.array(_flowgrpo_group_index(usable), dtype=object)
    batch_tensors = {
        "old_log_probs": torch.stack(old_list),
        "rm_scores": scores,
        "sample_level_scores": scores,
        "all_latents": torch.stack(latent_list),
        "all_timesteps": torch.stack(timestep_list),
        "gen_transition_valid": torch.stack(masks),
        "gen_valid": torch.ones(len(usable)),
        "parent_episode_id": torch.tensor(parents, dtype=torch.long),
        "gen_row_id": torch.arange(len(usable), dtype=torch.long),
    }
    if has_reference:
        batch_tensors["ref_prev_sample_mean"] = torch.stack(reference_means)

    batch = TensorDict(batch_tensors, batch_size=[len(usable)])
    # Build the object array explicitly 1-D rather than ``np.array(prompt_token_ids, dtype=object)``.
    prompt_token_ids_arr = np.empty(len(prompt_token_ids), dtype=object)
    for index, ids in enumerate(prompt_token_ids):
        prompt_token_ids_arr[index] = ids

    non_tensor: dict[str, Any] = {
        "uid": uids,
        "gen_group_uid": np.array([item[3] for item in usable], dtype=object),
        "prompt_token_ids": prompt_token_ids_arr,
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


def fold_gen_prompt_token_ids(proto: DataProto) -> None:
    """Move the GEN prompt token ids from ``non_tensor_batch`` onto the batch itself."""
    non_tensor = getattr(proto, "non_tensor_batch", None) or {}
    if "prompt_token_ids" in non_tensor:
        from verl.utils import tensordict_utils as tu

        tu.assign_non_tensor(proto.batch, prompt_token_ids=non_tensor["prompt_token_ids"])


def apply_gen_flowgrpo_advantage(
    gen_batch: list[Mapping[str, Any]] | None,
    *,
    adv_estimator: str = "flow_grpo",
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    algo_config: Any = None,
) -> tuple[DataProto | None, dict[str, float]]:
    """RFC GenAdv: FlowGRPO within each group of ``_flowgrpo_group_index``.

    Returns ``(None, skip metrics)`` when there is no GEN view (``K=0``). Rows that
    claim GEN without latents raise — refuse soft-skip of ``diffusion_loss``.
    """
    metrics: dict[str, float] = {
        "gen/skipped_no_groups": 1.0,
        "gen/num_rows": 0.0,
        "has_complete_gen_groups": 0.0,
        "gen/has_traj": 0.0,
    }
    if not gen_batch:
        return None, metrics

    proto = build_gen_flowgrpo_proto(list(gen_batch))
    metrics["gen/num_rows"] = float(len(gen_batch))
    if proto is None:
        return None, metrics

    num_timesteps = proto.batch["old_log_probs"].shape[1]
    proto.batch["sample_level_rewards"] = proto.batch["sample_level_scores"].expand(-1, num_timesteps)
    proto = compute_advantage(
        proto,
        adv_estimator=adv_estimator,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
        config=algo_config,
    )
    # Fold non-tensor traj metadata onto the batch so update_actor can materialize GEN.
    fold_gen_prompt_token_ids(proto)
    metrics["gen/skipped_no_groups"] = 0.0
    metrics["gen/num_usable_rows"] = float(len(proto))
    metrics["has_complete_gen_groups"] = 1.0
    metrics["gen/has_traj"] = 1.0
    return proto, metrics
