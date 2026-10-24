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

import torch
from tensordict import TensorDict
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.trainer.omni.omni_algos import get_omni_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, OmniActorConfig


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


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None, use_kl_loss=None):
    """Compute loss for diffusion model"""
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")
    loss_func = get_diffusion_loss_fn(loss_mode)
    loss_func.validate_inputs(loss_name=loss_mode, model_output=model_output, data=data)

    # Rollout Correction bypass mode only applies to log-prob policy-gradient losses.
    if "log_probs" in loss_func.required_model_output_keys:
        log_prob = model_output["log_probs"]
        old_log_prob = data["old_log_probs"]
        # Read it optionally: Rollout Correction is an AR-era node of the *diffusion* actor config,
        # but the Bagel Co-RL (Joint-Training) composite hands the GEN branch the outer AR trainer's
        # ``OmniActorConfig`` instead (``_rewrite_bagel_corl_configs``), which declares no
        # ``rollout_correction`` at all. Absent means "not configured", i.e. the same thing
        # ``bypass_mode=False`` means.
        rc_cfg = getattr(config, "rollout_correction", None)
        # Rollout Correction bypass mode: compute IS/RS weights per-step and
        # stash ``rollout_is_weights`` into ``data`` before loss dispatch.
        if rc_cfg is not None and rc_cfg.bypass_mode:
            _apply_bypass_rc(log_prob, old_log_prob, rc_cfg, data, metrics)

    loss_result = loss_func(config=config, model_output=model_output, data=data)
    loss_value = loss_result.loss
    metrics_values = loss_result.metrics

    metrics_values = Metric.from_dict(metrics_values, aggregation=AggregationType.MEAN)

    metrics.update(metrics_values)
    if loss_result.add_loss_metric:
        metrics["actor/loss"] = Metric(value=loss_value, aggregation=AggregationType.MEAN)

    if config.use_kl_loss if use_kl_loss is None else use_kl_loss:
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

    # ``use_distill_loss`` / ``distill_loss_mode`` / ``distill_loss_coef`` live on the *diffusion*
    # actor config (``DiffusionActorConfig``); the Co-RL (Joint-Training) composite hands this
    # function the outer AR trainer's ``OmniActorConfig`` (``_rewrite_bagel_corl_configs``), which
    # declares none of them. Read them optionally, mirroring
    # ``diffusion_trainer_utils._validate_distill_config``'s rule that "``loss_mode`` is a distill
    # mode" already activates the term -- only an absent *and* unselected node reads as "off".
    activated_by_mode = loss_mode in ("distill_kl", "distill_fm_mse")
    if bool(getattr(config, "use_distill_loss", False)) or activated_by_mode:
        distill_mode = getattr(config, "distill_loss_mode", None) or (loss_mode if activated_by_mode else "distill_kl")
        distill_coef = float(getattr(config, "distill_loss_coef", 1.0))
        loss_func = get_diffusion_loss_fn(distill_mode)
        loss_func.validate_inputs(loss_name=distill_mode, model_output=model_output, data=data)
        distill_result = loss_func(config=config, model_output=model_output, data=data)
        loss_value += distill_result.loss * distill_coef
        metrics.update(Metric.from_dict(distill_result.metrics, aggregation=AggregationType.MEAN))
        metrics["distill_coef"] = distill_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    if not tu.get_non_tensor_data(data, "corl_loss_normalized", default=False):
        loss_value = loss_value / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size > 1:
        loss_value = loss_value * sp_size

    return loss_value, metrics


def _bagel_corl_view(data: TensorDict, key: str):
    """Unwrap ``bagel_corl_{und,gen}`` NonTensorData views; never invent a fallback."""
    return tu.get_non_tensor_data(data, key, default=None)


_LANE_SCALAR_TYPES = (bool, int, float, str, type(None))


def _inherit_micro_batch_scalars(outer: TensorDict, view) -> None:
    """Copy the engine's non-tensor *scalars* from the micro-batch onto a lane view.

    ``diffusion_loss`` reads ``gradient_accumulation_steps`` / ``sp_size`` as non-tensor data off
    whatever batch it is handed, but ``bagel_composite_loss`` hands it the ``bagel_corl_gen`` view,
    which the engine never stamps those keys onto. Only **scalars** are copied: a per-row list on
    the micro-batch is not necessarily row-aligned with the lane view, and the view's own per-row
    fields must win.
    """
    if outer is None or view is None or not isinstance(view, TensorDict):
        return
    try:
        keys = list(outer.keys())
    except Exception:
        return
    for key in keys:
        try:
            value = tu.get_non_tensor_data(outer, key, default=None)
        except Exception:
            continue
        if not isinstance(value, _LANE_SCALAR_TYPES):
            continue
        try:
            if tu.get_non_tensor_data(view, key, default=None) is not None:
                continue
            tu.assign_non_tensor(view, **{key: value})
        except Exception:
            continue


def _first_tensor_device(payload, depth: int = 2) -> torch.device | None:
    """Device of the first ``torch.Tensor`` reachable from ``payload`` within ``depth`` levels."""
    if isinstance(payload, torch.Tensor):
        return payload.device
    if depth <= 0 or payload is None:
        return None
    if isinstance(payload, dict) or hasattr(payload, "keys"):
        try:
            keys = list(payload.keys())
        except Exception:
            return None
        for key in keys:
            try:
                child = payload.get(key) if hasattr(payload, "get") else payload[key]
            except Exception:
                continue
            found = _first_tensor_device(child, depth - 1)
            if found is not None:
                return found
    return None


def _align_view_to_device(view, device: torch.device | None) -> None:
    """Move every tensor inside a non-tensor-stashed loss view onto ``device``, in place.

    ``bagel_corl_{und,gen}`` are parked as **non-tensor** data (``tu.assign_non_tensor``), and
    ``TensorDict.to`` does not recurse into non-tensor payloads, so the engine moves the outer
    micro-batch without touching these views while ``build_gen_flowgrpo_proto`` allocates on the
    CPU. Aligning here keeps the fix at the one place that knows the target device.
    """
    if view is None or device is None:
        return
    keys = None
    try:
        keys = list(view.keys())
    except Exception:
        return
    for key in keys:
        try:
            value = view.get(key) if hasattr(view, "get") else view[key]
        except Exception:
            continue
        if isinstance(value, torch.Tensor) and value.device != device:
            try:
                view[key] = value.to(device)
            except Exception:
                # Read-only or exotic container: leave it, the caller will report a clear error.
                pass


def _bagel_lane_weight(config, key: str, *, default: float | str = 1.0, cast=float):
    """Read a Bagel Co-RL (Joint-Training) lane knob from ``config.diffusion_loss`` (SoT: DiffusionLossConfig).

    Older configs without the field keep the default; never invent a fallback for
    garbage values — cast errors fail loud.
    """
    try:
        raw = config.diffusion_loss.get(key, default)
    except (AttributeError, TypeError):
        raw = default
    return cast(raw)


def _singleton_loss_tensor(value, index):
    """Keep a batch dimension without unsupported jagged batch slicing or detaching gradients."""
    if value.is_nested:
        return torch.nested.as_nested_tensor([value.unbind(0)[index]], layout=value.layout)
    return value[index : index + 1]


def bagel_und_episode_loss(config, model_output, data, dp_group=None):
    """Average each episode's token objective before global episode normalization."""
    from verl.utils.py_functional import append_to_dict
    from verl.workers.utils.losses import ppo_loss

    normalizer = float(tu.get_non_tensor_data(data, "und_episode_normalizer", default=1.0))
    if normalizer <= 0:
        raise ValueError("UND episode normalizer must be positive")
    terms, metrics = [], {}
    for i in range(len(data)):
        # The ingress UND view can reference its parent as non-tensor metadata.
        # Do not recursively deepcopy that cycle; fields below are replaced, not mutated.
        tensor_keys = [key for key, value in data.items() if isinstance(value, torch.Tensor)]
        row = data.exclude(*tensor_keys)[i : i + 1].clone(recurse=False)
        for key in tensor_keys:
            row[key] = _singleton_loss_tensor(data[key], i)
        output = {
            key: _singleton_loss_tensor(value, i)
            for key, value in model_output.items()
            if isinstance(value, torch.Tensor)
        }
        valid = bool(row["response_mask"].any())
        if not valid:
            # Publish the same finite metric keys on all ranks without a 0/0 token mean.
            row["response_mask"] = torch.ones_like(row["response_mask"])
            for key in ("old_log_probs", "advantages", "ref_log_prob", "rollout_is_weights"):
                if key in row.keys():
                    row[key] = torch.zeros_like(row[key])
            metric_output = {key: torch.zeros_like(value) for key, value in output.items()}
        else:
            metric_output = output
        term, row_metrics = ppo_loss(config, metric_output, row, dp_group=dp_group)
        if not valid:
            raw = output["log_probs"]
            raw = raw.values() if raw.is_nested else raw
            term = torch.nan_to_num(raw).sum() * 0.0
        terms.append(term)
        append_to_dict(metrics, row_metrics)
        dp_size = float(tu.get_non_tensor_data(data, "corl_dp_size", default=1))
        for key, value in row_metrics.items():
            leaf = {
                "actor/pg_loss": "pg_loss",
                "actor/ppo_kl": "update_kl",
                "kl_loss": "ref_kl",
                "actor/entropy_loss": "entropy",
            }.get(key)
            if leaf:
                raw_value = float(value.aggregate()) if isinstance(value, Metric) else float(value)
                metric = Metric(value=raw_value * int(valid) / (normalizer * dp_size), aggregation=AggregationType.SUM)
                append_to_dict(metrics, {f"actor/und/{leaf}": metric})
    return torch.stack(terms).sum() / normalizer, metrics


def bagel_composite_loss(config, model_output, data, dp_group=None):
    """Composite Co-RL (Joint-Training) loss: UND token PPO + GEN FlowGRPO on *separate* advantage views.

    UND uses token GRPO advantages on the main / ``bagel_corl_und`` tensors.
    GEN uses FlowGRPO advantages only from ``bagel_corl_gen`` — never the UND
    token ``advantages`` field (that would cross-wire MoT GEN to token GRPO).
    """
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.losses import ppo_loss

    metrics = {}
    has_complete = bool(tu.get_non_tensor_data(data, "has_complete_gen_groups", default=False))
    skip_gen = bool(tu.get_non_tensor_data(data, "skip_gen", default=not has_complete))
    num_gen_rows = tu.get_non_tensor_data(data, "num_gen_rows", default=0) or 0
    # RFC §4.4: each branch is a mean over its own loss units, and the relative
    # scale is governed solely by the lane weights. GEN's ``diffusion_loss``
    # divides by ``len(micro_batches) * num_timesteps`` internally
    # (``diffusers_impl.py:893``); ``ppo_loss`` does not, because the UND pass hands
    # ``agg_loss`` ``dp_size=1`` and ``batch_num_tokens=None`` so its term is the
    # mean over *that micro-batch* (``core_algos.py:1169-1173``). The UND term
    # therefore has to be divided by ``len(micro_batches)`` here, or the
    # accumulator sums G undivided UND terms and the lane weights silently drift
    # by G. Feed ``ppo_loss`` the DP-reduced global token count instead and the
    # division is wrong by another factor of ``dp_size`` -- see
    # ``run_und_token_forward_backward`` for the full derivation and the failure
    # that pins it.
    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)

    loss_value = None
    model_output = model_output if isinstance(model_output, dict) else {}
    # Both lane views are parked as *non-tensor* data, so the engine's ``micro_batch.to(device)``
    # never reaches them (see ``_align_view_to_device``). Align them with the forward that just ran.
    lane_device = _first_tensor_device(model_output) or _first_tensor_device(data)
    und_output = model_output.get("und")
    und_data = _bagel_corl_view(data, "bagel_corl_und")
    _align_view_to_device(und_data, lane_device)
    _inherit_micro_batch_scalars(data, und_data)
    # Legacy single-dict UND forward: only when an explicit UND view exists or
    # the payload is tagged und — never treat a GEN-only diffusion forward as UND.
    if und_output is None and und_data is not None and "log_probs" in model_output:
        und_output = model_output
    if und_output is None and und_data is None and model_output.get("modality") == "und":
        und_output = model_output
        und_data = data
    if und_data is None and und_output is not None and "bagel_corl_gen" not in data.keys():
        # Pure UND step (no GEN view attached): main batch holds token GRPO advantages.
        und_data = data

    if und_output is not None and und_data is not None and "log_probs" in und_output:
        if tu.get_non_tensor_data(und_data, "und_episode_normalizer", default=None) is not None:
            und_loss, und_metrics = bagel_und_episode_loss(config, und_output, und_data, dp_group=dp_group)
        else:
            und_loss, und_metrics = ppo_loss(config, und_output, und_data, dp_group=dp_group)
            if gradient_accumulation_steps:
                und_loss = und_loss / gradient_accumulation_steps
        loss_value = und_loss * _bagel_lane_weight(config, "loss_weight_und")
        metrics.update(und_metrics)
        for key, value in und_metrics.items():
            leaf = {
                "actor/pg_loss": "pg_loss",
                "actor/ppo_kl": "update_kl",
                "kl_loss": "ref_kl",
                "actor/entropy_loss": "entropy",
            }.get(key)
            if leaf:
                metrics.setdefault(f"actor/und/{leaf}", value)

    gen_data = _bagel_corl_view(data, "bagel_corl_gen")
    if tu.get_non_tensor_data(data, "bagel_corl_lane", default=None) == "gen":
        gen_data = data
    elif gen_data is not None and model_output.get("gen") is None and und_output is None:
        raise ValueError("GEN timestep loss requires row-aligned tagged data, not the full bagel_corl_gen view")
    _align_view_to_device(gen_data, lane_device)
    _inherit_micro_batch_scalars(data, gen_data)
    if skip_gen or not has_complete or int(num_gen_rows) <= 0 or gen_data is None:
        metrics["gen/skipped_no_groups"] = Metric(value=1.0, aggregation=AggregationType.MEAN)
        if gen_data is None and has_complete and not skip_gen and int(num_gen_rows) > 0:
            metrics["gen/missing_flowgrpo_view"] = Metric(value=1.0, aggregation=AggregationType.MEAN)
        if loss_value is None:
            loss_value = torch.zeros((), requires_grad=True)
        return loss_value, metrics

    gen_output = model_output.get("gen")
    if gen_output is None and "log_probs" in model_output and und_output is None:
        gen_output = model_output
    if gen_output is None or "log_probs" not in gen_output:
        metrics["gen/skipped_no_groups"] = Metric(value=1.0, aggregation=AggregationType.MEAN)
        if loss_value is None:
            loss_value = torch.zeros((), requires_grad=True)
        return loss_value, metrics

    # Fail closed: GEN must see FlowGRPO advantages from bagel_corl_gen only.
    if isinstance(gen_data, TensorDict):
        if "advantages" not in gen_data.keys():
            raise ValueError("bagel_corl_gen is missing FlowGRPO advantages; refusing to use UND token advantages")
    elif not hasattr(gen_data, "keys") or "advantages" not in gen_data:
        raise ValueError("bagel_corl_gen must carry FlowGRPO advantages for MoT GEN")

    gen_regularizer = _bagel_lane_weight(config, "gen_regularizer", default="latent_kl", cast=str)
    if gen_regularizer == "velocity_mse":
        raise NotImplementedError(
            "gen_regularizer='velocity_mse' (UniGRPO Eq. 8) lands in Phase 2; use 'latent_kl' for now"
        )

    if "ref_log_prob" in gen_data.keys() or "ref_prev_sample_mean" in gen_data.keys():
        raise NotImplementedError(
            "Co-RL GEN reference regularization requires valid-transition normalization and is unsupported; "
            "omit GEN reference tensors. UND reference KL remains supported."
        )

    gen_loss, gen_metrics = diffusion_loss(
        config,
        gen_output,
        gen_data,
        dp_group=dp_group,
        use_kl_loss=False,
    )
    metrics.update(gen_metrics)
    normalized = bool(tu.get_non_tensor_data(data, "corl_loss_normalized", default=False))
    dp_size = float(tu.get_non_tensor_data(data, "corl_dp_size", default=1))
    metric_weight = 1.0
    if normalized:
        normalizer = float(tu.get_non_tensor_data(data, "loss_normalizer", default=None))
        metric_weight = float(data["loss_unit_weight"].sum().item()) / (normalizer * dp_size)
    for key, value in gen_metrics.items():
        leaf = {
            "actor/loss": "pg_loss",
            "actor/pg_clipfrac": "clipfrac",
            "actor/ratio_std": "ratio_std",
            "actor/ratio_mean": "ratio_mean",
            "actor/ratio_square_mean": "ratio_square_mean",
            "actor/ppo_kl": "update_kl",
        }.get(key)
        if leaf:
            raw_value = float(value.aggregate()) if isinstance(value, Metric) else float(value)
            metrics[f"actor/gen/{leaf}"] = (
                Metric(value=raw_value * metric_weight, aggregation=AggregationType.SUM) if normalized else value
            )
    metrics["actor/gen/pg_loss"] = Metric(value=gen_loss.detach().item() / dp_size, aggregation=AggregationType.SUM)
    metrics["gen/skipped_no_groups"] = Metric(value=0.0, aggregation=AggregationType.MEAN)
    if loss_value is None:
        loss_value = gen_loss * _bagel_lane_weight(config, "loss_weight_gen")
    else:
        loss_value = loss_value + gen_loss * _bagel_lane_weight(config, "loss_weight_gen")
    return loss_value, metrics


def omni_loss(config: OmniActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for omni AR direct-preference training."""

    metrics = {}

    loss_mode = config.omni_loss.loss_mode
    loss_func = get_omni_loss_fn(loss_mode)
    loss_func.validate_inputs(model_output=model_output, data=data)
    loss_result = loss_func(config=config, model_output=model_output, data=data)
    loss_value = loss_result.loss
    metrics.update(Metric.from_dict(loss_result.metrics, aggregation=AggregationType.MEAN))
    metrics["actor/loss"] = Metric(value=loss_value, aggregation=AggregationType.MEAN)

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    if gradient_accumulation_steps:
        loss_value = loss_value / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size is not None and sp_size > 1:
        loss_value = loss_value * sp_size

    return loss_value, metrics
