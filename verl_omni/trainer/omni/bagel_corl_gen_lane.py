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
"""GEN (FlowGRPO) lane of ``bagel_corl_sync``: old log-prob, advantage, and the flags the composite reads."""

from __future__ import annotations

import logging
from typing import Any

import torch
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.advantages import (
    apply_gen_flowgrpo_advantage,
    fold_gen_prompt_token_ids,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.diffusion import gen_norm_adv_by_std
from verl_omni.trainer.omni.bagel_corl_metrics import step_metric_keys

logger = logging.getLogger(__name__)

#: UND token estimators the GEN estimator must never collide with.
_UND_TOKEN_ESTIMATORS = frozenset({"grpo", "gspo", "gae", "rloo", "reinforce_plus_plus"})

#: ``extra`` keys the advantage phase hands to the actor update.
GEN_LANE_CARRIED_KEYS = ("has_complete_gen_groups", "num_gen_rows", "bagel_corl_gen")


def check_estimators_disjoint(gen_estimator: str, und_estimator: str) -> None:
    """GEN FlowGRPO and UND token GRPO are separate tensors; one estimator for both is a config bug."""
    if gen_estimator == und_estimator and und_estimator in _UND_TOKEN_ESTIMATORS:
        raise ValueError(
            f"Bagel Co-RL (Joint-Training) GEN adv_estimator={gen_estimator!r} collides with UND token estimator; "
            "set actor_rollout_ref.model.algorithm=flow_grpo (GEN) and leave algorithm.adv_estimator for UND"
        )


def gen_old_log_prob(proto, lane, divisor: int):
    """Recompute GEN old log-probs on the workers, padding to the dispatch divisor when K varies."""
    pad_size = 0
    if divisor > 1 and len(proto) % divisor != 0:
        unpadded_n = len(proto)
        proto, pad_size = pad_dataproto_to_divisor(proto, size_divisor=divisor)
        logger.info(
            "bagel_corl GEN old_log_prob: padded %d -> %d rows for dispatch divisor %d", unpadded_n, len(proto), divisor
        )
    old = lane._compute_old_log_prob(proto)
    return unpad_dataproto(old, pad_size) if pad_size else old


def _skipped_metrics() -> dict:
    return {"gen/skipped_no_groups": 1.0, "gen/num_rows": 0.0, "has_complete_gen_groups": 0.0}


def gen_advantage(proto, gen_rows: list, *, extra: dict, algo, gen_estimator: str, lane) -> tuple[Any, dict]:
    """FlowGRPO advantage for the GEN proto; ``lane`` is the worker-backed V1 lane or ``None`` (CPU tests)."""
    if proto is None:
        return None, _skipped_metrics()
    old_batch = extra.get("bagel_corl_gen_old")
    if old_batch is not None and "old_log_probs" in old_batch.keys():
        proto.batch["old_log_probs"] = old_batch["old_log_probs"]
        if "old_prev_sample_mean" in old_batch.keys():
            proto.batch["old_prev_sample_mean"] = old_batch["old_prev_sample_mean"]
    if lane is not None:
        gen_proto = lane._compute_advantage(proto)
        if "all_latents" not in gen_proto.batch.keys():
            raise RuntimeError("bagel_corl GEN V1 advantage returned no all_latents; refuse skip GEN loss.")
        fold_gen_prompt_token_ids(gen_proto)
        return gen_proto, {
            "gen/skipped_no_groups": 0.0,
            "gen/num_rows": float(len(gen_proto)),
            "gen/num_usable_rows": float(len(gen_proto)),
            "has_complete_gen_groups": 1.0,
            "gen/has_traj": 1.0,
        }
    gen_norm = gen_norm_adv_by_std(algo)
    return apply_gen_flowgrpo_advantage(
        gen_rows,
        adv_estimator=gen_estimator,
        norm_adv_by_std_in_grpo=gen_norm if gen_norm is not None else bool(algo.get("norm_adv_by_std_in_grpo", True)),
        global_std=bool(algo.get("global_std", True)),
        algo_config=algo,
    )


def gen_proto_metrics(gen_proto, metrics: dict) -> None:
    """Advantage spread, effective transitions and zero-std group fraction of the GEN proto."""
    adv = gen_proto.batch["advantages"]
    valid = gen_proto.batch["gen_transition_valid"].bool()
    metrics["gen/adv_std"] = float(adv[valid].float().std(unbiased=False).item())
    metrics["gen/effective_transitions"] = float(valid.sum().item())
    metrics["gen/valid_rows"] = float(gen_proto.batch["gen_valid"].sum().item())
    scores = gen_proto.batch["sample_level_scores"].reshape(-1)
    uids = gen_proto.non_tensor_batch["uid"]
    group_stds = [
        float(scores[torch.tensor(uids == uid, device=scores.device)].std(unbiased=False)) for uid in set(uids.tolist())
    ]
    metrics["gen/zero_std_group_fraction"] = sum(v == 0.0 for v in group_stds) / len(group_stds)


def stamp_gen_lane(extra: dict, metrics: dict, gen_proto, gen_metrics: dict) -> bool:
    """Record the GEN lane outcome on ``extra`` and ``metrics``; returns ``has_complete_gen_groups``."""
    metrics.update(gen_metrics)
    for key in step_metric_keys(extra):
        metrics.setdefault(key, extra[key])
    has_complete = bool(gen_metrics.get("has_complete_gen_groups"))
    extra["has_complete_gen_groups"] = has_complete
    extra["skip_gen"] = not has_complete
    extra["num_gen_rows"] = int(gen_metrics.get("gen/num_usable_rows") or gen_metrics.get("gen/num_rows") or 0)
    extra["bagel_corl_gen"] = None if gen_proto is None else gen_proto.batch
    metrics["gen/skipped_no_groups"] = float(gen_metrics.get("gen/skipped_no_groups", 0.0 if has_complete else 1.0))
    return has_complete


def carry_gen_lane(src: dict, dst: dict, has_complete: bool) -> None:
    """Copy the GEN lane flags from the pre-``super`` ``extra`` onto the returned batch's carrier."""
    for key in (*GEN_LANE_CARRIED_KEYS, *step_metric_keys(src)):
        if key in src:
            dst[key] = src[key]
    dst["skip_gen"] = not has_complete


def validate_batch_ownership(batch, extra: dict) -> None:
    """Check every GEN row's ``parent_episode_id`` against the UND rows, in tensor columns so DP slices it."""
    import transfer_queue as tq
    from tensordict import TensorDict

    from verl_omni.trainer.omni.bagel_corl_batch import validate_episode_ownership

    gen_view = extra.get("bagel_corl_gen")
    parents = gen_view["parent_episode_id"] if gen_view is not None else torch.empty(0, dtype=torch.long)
    if isinstance(batch, TensorDict):
        ids = batch["bagel_episode_id"]
        batch["bagel_episode_id"] = validate_episode_ownership(ids, parents).to(ids.device)
    elif hasattr(batch, "partition_id"):
        ids = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["bagel_episode_id"])[
            "bagel_episode_id"
        ]
        tags = list(getattr(batch, "tags", None) or [])
        padding = [bool(tag.get("is_padding", False)) for tag in tags] if len(tags) == len(batch.keys) else None
        checked = validate_episode_ownership(ids, parents, padding)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict({"bagel_episode_id": checked}, batch_size=[len(checked)]),
        )


def attach_gen_flags(batch, extra: dict, metrics: dict) -> bool:
    """Put the GEN flags where the composite reads them (``extra_info`` or TensorDict non-tensor data)."""
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    has_complete = bool(extra.get("has_complete_gen_groups"))
    extra["skip_gen"] = not has_complete
    extra["has_complete_gen_groups"] = has_complete
    if not has_complete:
        metrics["gen/skipped_no_groups"] = 1.0
    for key in step_metric_keys(extra):
        metrics[key] = extra[key]
    if isinstance(batch, TensorDict):
        tu.assign_non_tensor_data(batch, "skip_gen", extra["skip_gen"])
        tu.assign_non_tensor_data(batch, "has_complete_gen_groups", has_complete)
        tu.assign_non_tensor_data(batch, "num_gen_rows", int(extra.get("num_gen_rows") or 0))
        if extra.get("bagel_corl_gen") is not None:
            tu.assign_non_tensor_data(batch, "bagel_corl_gen", extra["bagel_corl_gen"])
    return has_complete
