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

from collections import defaultdict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig


def _apply_bypass_rc(
    log_prob: torch.Tensor,  # (B,) current policy log-prob
    old_log_prob: torch.Tensor,  # (B,) == rollout_log_prob in bypass
    rc_cfg,  # RolloutCorrectionConfig
    data: TensorDict,  # modified in-place
    metrics: dict,  # modified in-place
) -> None:
    """Compute per-step IS/RS for bypass mode and stash weights into ``data``."""
    log_prob_2d = log_prob.unsqueeze(-1)  # current policy log-prob (π_θ)
    rollout_lp_2d = old_log_prob.unsqueeze(-1)  # rollout policy log-prob (π_rollout)
    response_mask = torch.ones_like(log_prob_2d)

    # In bypass mode, RS checks current→rollout drift: pass current as old_log_prob, rollout as rollout_log_prob.
    # This matches the mathematical intent: RS mask is applied to exp(log_prob - rollout_log_prob).
    is_weights_proto, modified_mask, rc_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob_2d,  # current policy (π_θ)
        rollout_log_prob=rollout_lp_2d,  # rollout policy (π_rollout)
        response_mask=response_mask,
        rollout_is=rc_cfg.rollout_is,
        rollout_is_threshold=rc_cfg.rollout_is_threshold,
        rollout_is_batch_normalize=rc_cfg.rollout_is_batch_normalize,
        rollout_rs=rc_cfg.rollout_rs,
        rollout_rs_threshold=rc_cfg.rollout_rs_threshold,
    )

    # ppo_clip: PPO ratio handles IS, only RS mask is applied.
    assert rc_cfg.loss_type == "ppo_clip", f"Only loss_type='ppo_clip' is supported, got {rc_cfg.loss_type!r}"
    weights: torch.Tensor | None = None

    if rc_cfg.rollout_rs:
        rs_mask = modified_mask
        weights = rs_mask if weights is None else weights * rs_mask

    if weights is not None:
        existing = data.get("rollout_is_weights", None)
        data["rollout_is_weights"] = (
            weights.squeeze(-1).to(dtype=log_prob.dtype)
            if existing is None
            else existing * weights.squeeze(-1).to(dtype=log_prob.dtype)
        )

    for k, v in rc_metrics.items():
        metrics[k] = Metric(value=float(v), aggregation=AggregationType.MEAN)


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")
    loss_func = get_diffusion_loss_fn(loss_mode)

    # Rollout Correction bypass mode only applies to log-prob policy-gradient losses.
    if "log_probs" in loss_func.required_model_output_keys:
        log_prob = model_output["log_probs"]
        old_log_prob = data["old_log_probs"]
        rc_cfg = config.rollout_correction
        # Rollout Correction bypass mode: compute IS/RS weights per-step and
        # stash ``rollout_is_weights`` into ``data`` before loss dispatch.
        if rc_cfg.bypass_mode:
            _apply_bypass_rc(log_prob, old_log_prob, rc_cfg, data, metrics)

    loss_func.validate_inputs(loss_name=loss_mode, model_output=model_output, data=data)
    loss_result = loss_func(config=config, model_output=model_output, data=data)
    loss_value = loss_result.loss
    metrics_values = loss_result.metrics

    metrics_values = Metric.from_dict(metrics_values, aggregation=AggregationType.MEAN)

    metrics.update(metrics_values)
    if loss_result.add_loss_metric:
        metrics["actor/loss"] = Metric(value=loss_value, aggregation=AggregationType.MEAN)

    if config.use_kl_loss:
        loss_func = get_diffusion_loss_fn("kl")
        loss_func.validate_inputs(loss_name="kl", model_output=model_output, data=data)
        kl_result = loss_func(config=config, model_output=model_output, data=data)
        loss_value += kl_result.loss * config.kl_loss_coef
        metrics.update(Metric.from_dict(kl_result.metrics, aggregation=AggregationType.MEAN))
        metrics["kl_coef"] = config.kl_loss_coef
        if kl_result.add_loss_metric:
            metrics["actor/weighted_kl_loss"] = Metric(
                value=kl_result.loss * config.kl_loss_coef,
                aggregation=AggregationType.MEAN,
            )

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    loss_value = loss_value / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size > 1:
        loss_value = loss_value * sp_size

    return loss_value, metrics


def _as_pylist(v):
    """Coerce a per-row non-tensor field to a plain Python list.

    On the actor batch, uid/sj_score arrive as NonTensorStack handles whose np.asarray/iteration
    indexes a 0-dim tensordict and raises; .tolist() is the safe path.
    """
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.reshape(-1).detach().cpu().tolist()
    if isinstance(v, list | tuple):
        return list(v)
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(v)


def build_online_dpo_pair_indices(uids, scores) -> list[int]:
    """One adjacent (chosen, rejected) pair per prompt group.

    Groups sample indices by prompt uid, sorts each group by preference score descending, and emits
    [top, bottom] so the flat list alternates chosen/rejected. Groups with fewer than two members
    (e.g. an aborted sibling) are skipped; ties are kept and masked in the loss.
    """
    uid_vals = _as_pylist(uids)
    vals = _as_pylist(scores)
    if uid_vals is None or vals is None or len(uid_vals) != len(vals):
        raise ValueError("DPO pairing needs one uid per score.")
    groups: dict = defaultdict(list)
    for i, u in enumerate(uid_vals):
        groups[u].append(i)
    out: list[int] = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        s = sorted(idxs, key=lambda i: float(vals[i]), reverse=True)
        out.extend([s[0], s[-1]])
    return out


def tts_dpo_loss(config, model_output, data: TensorDict, dp_group=None):
    """Online DPO on the talker codec-0 sequence log-prob, judged pairwise within each uid group.

    For every prompt (uid) the batch holds rollout.n candidates; the judge's sj_score picks the best
    (chosen) and worst (rejected). Per sequence we sum the current-policy and frozen-reference
    log-probs over the response, form the implicit reward r = sum(log pi_theta - log pi_ref), and
    minimize -logsigmoid(beta*(r_chosen - r_rejected)) plus an optional lambda*NLL(chosen) anchor.
    beta and lambda come from DPOPolicyLossConfig (dpo_beta / dpo_nll_lambda).

    Pairing is internal: the config keeps a prompt's candidates in one in-order micro-batch per rank
    (use_dynamic_bsz false + one micro-batch + balance_batch false), so grouping by uid here recovers
    the pairs without a reorder pass. The per-rank weighted mean is the whole optimizer-step gradient;
    FSDP averages it across data-parallel ranks.
    """
    if getattr(config, "use_dynamic_bsz", False):
        raise ValueError("Online DPO requires actor.use_dynamic_bsz=false so a uid group stays in one micro-batch.")

    log_prob = no_padding_2_padding(model_output["log_probs"], data)  # (B, resp_len), current policy
    sel = data.select("response_mask", "ref_log_prob").to_padded_tensor()
    mask = sel["response_mask"].to(torch.float32)
    ref_log_prob = sel["ref_log_prob"].to(torch.float32)

    seq_logp = (log_prob.float() * mask).sum(-1)  # (B,) sum log pi_theta over response
    seq_ref = (ref_log_prob * mask).sum(-1)  # (B,) sum log pi_ref over response
    implicit_reward = seq_logp - seq_ref  # (B,) DPO implicit reward (pre-beta)

    uids = _as_pylist(tu.get_non_tensor_data(data, "uid", default=None))
    scores = _as_pylist(tu.get_non_tensor_data(data, "sj_score", default=None))
    if uids is None or scores is None:
        raise RuntimeError(
            "Online DPO needs `uid` and `sj_score` in the batch; the AudioJudgeRewardManager must emit sj_score."
        )
    sc = [float(x) for x in scores]
    idx = build_online_dpo_pair_indices(uids, sc)
    if not idx:
        raise RuntimeError(
            "Online DPO formed no preference pairs on this micro-batch; need rollout.n>=2 and each uid "
            "group co-located on one DP rank (trainer.balance_batch=false)."
        )
    device = implicit_reward.device
    order = torch.as_tensor(idx, dtype=torch.long, device=device)
    r = implicit_reward.index_select(0, order)
    sl = seq_logp.index_select(0, order)
    m = mask.index_select(0, order)
    # One flag per pair: 1 = real preference, 0 = judge tie (equal scores) -> zero gradient.
    valid = torch.tensor(
        [1.0 if sc[idx[k]] != sc[idx[k + 1]] else 0.0 for k in range(0, len(idx), 2)],
        dtype=torch.float32,
        device=device,
    )

    beta = float(config.policy_loss.get("dpo_beta", 0.1))
    chosen, rejected = r[0::2], r[1::2]
    inside = beta * (chosen - rejected)
    per_pair = -F.logsigmoid(inside)
    denom = valid.sum().clamp(min=1.0)
    loss = (per_pair * valid).sum() / denom

    # RPO/DPOP anchor: L = L_DPO + lambda*NLL(chosen). A per-token SFT loss on the chosen sequence so
    # its log-prob cannot collapse while the model merely widens the chosen-rejected margin.
    nll_lambda = float(config.policy_loss.get("dpo_nll_lambda", 0.0))
    chosen_tok = m[0::2].sum(-1).clamp(min=1.0)
    nll_chosen = -(sl[0::2] / chosen_tok)  # (n/2,) per-token NLL of chosen
    nll_term = (nll_chosen * valid).sum() / denom
    loss = loss + nll_lambda * nll_term

    with torch.no_grad():
        acc = ((inside > 0).float() * valid).sum() / denom
        margin = ((chosen - rejected) * valid).sum() / denom
        mean = AggregationType.MEAN
        metrics = {
            "actor/dpo_loss": Metric(value=loss, aggregation=mean),
            "actor/dpo_acc": Metric(value=acc, aggregation=mean),
            "actor/dpo_margin": Metric(value=margin, aggregation=mean),
            "actor/dpo_tie_frac": Metric(value=1.0 - valid.mean(), aggregation=mean),
            "actor/logp_chosen": Metric(value=sl[0::2].mean(), aggregation=mean),
            "actor/logp_rejected": Metric(value=sl[1::2].mean(), aggregation=mean),
            "actor/nll_chosen": Metric(value=(nll_chosen * valid).sum() / denom, aggregation=mean),
            "actor/dpo_beta": beta,
            "actor/dpo_nll_lambda": nll_lambda,
        }
    return loss, metrics
