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
"""Bagel UND+GEN Co-RL (Joint-Training) trainer: one post-gather composite update + weight publish."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import fields
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer

from verl_omni.agent_loop.bagel_corl_lib import (
    aggregate_episode_metrics,
    aggregate_reward_breakdown_metrics,
    und_record_field_keys,
)
from verl_omni.trainer.omni.bagel_corl_diff_v1 import DiffusionV1GenLane
from verl_omni.trainer.omni.bagel_corl_gen_adv import (
    apply_gen_flowgrpo_advantage,
    build_gen_flowgrpo_proto,
    fold_gen_prompt_token_ids,
)
from verl_omni.trainer.omni.bagel_corl_replay_buffer import install_auxiliary_row_filter
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.config.diffusion import DiffusionRolloutConfig, DiffusionSamplingConfig

logger = logging.getLogger(__name__)

# Bagel Co-RL (Joint-Training) agent knobs with no code default (RFC §5 "knob source of truth"):
# ``actor_rollout_ref.rollout.agent.*`` is the single SoT, and
# ``verl_omni/utils/config.py`` validates S >= 2 / max_generate_passes == 1 at
# launch. The trainer must not invent values, or a missing/bad recipe knob would
# be masked from that validation.
_BAGEL_AGENT_REQUIRED_KNOBS = ("gen_samples_per_call", "max_generate_passes", "max_und_turns")

# Fields accepted by DiffusionModelConfig (plus _target_). Omni YAML keys outside this
# set must be stripped before instantiate/omega_conf_to_dataclass.
_DIFFUSION_MODEL_KEYS = {
    "path",
    "architecture",
    "transformer_config",
    "algorithm",
    "local_path",
    "tokenizer_path",
    "local_tokenizer_path",
    "hf_config",
    "model_type",
    "composite_mode",
    "load_tokenizer",
    "tokenizer",
    "processor",
    "extra_tokenizers",
    "extra_tokenizer_map",
    "use_shm",
    "trust_remote_code",
    "custom_chat_template",
    "external_lib",
    "enable_gradient_checkpointing",
    "attn_backend",
    "lora_rank",
    "lora_alpha",
    "lora_init_weights",
    "lr_gen",
    "corl_freeze_und",
    "corl_freeze_gen",
    "corl_no_optimizer_step",
    "target_modules",
    "target_parameters",
    "exclude_modules",
    "lora",
    "lora_adapter_path",
    "policy_state_adapters",
    "lora_dtype",
    "mtp",
    "pipeline",
    "algo",
    "fsdp_layer_prefixes",
    "config_path",
    "transformer_subfolder",
}


def _bagel_role_histogram(tags) -> dict[str, int]:
    """Count a ``KVBatchMeta`` row-tag list by ``bagel_role`` (+ padding), for step diagnostics.

    A GEN stale in a sampled batch is invisible in the training metrics but fatal a few frames
    later (``assert len(output) == len(batch)``), so the batch's own composition is worth a line.
    """
    counts: dict[str, int] = {}
    for tag in tags or []:
        if not isinstance(tag, dict):
            continue
        role = str(tag.get("bagel_role") or "untagged")
        if tag.get("is_padding"):
            role += ":pad"
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_tq_kv_get_result(data, keys: list[str]) -> dict[str, dict]:
    """Normalize ``transfer_queue.kv_batch_get`` output into ``{key: {"fields": row}}``.

    ``kv_batch_get`` returns a columnar TensorDict (one entry per requested key, in
    request order). It is **not** a plain dict. Fail-loud on any shape we do not
    recognize so a silent ``{}`` can never drop the GEN lane.
    """
    if data is None:
        return {}

    # Columnar TensorDict (the canonical TQ shape): each field is a stacked column.
    td = data
    if hasattr(td, "items") and hasattr(td, "get") and not isinstance(td, dict | list):
        # Iterate ``td[field]`` (not ``td.items()``): TensorDict ``__getitem__`` already
        # unwraps non-tensor columns to plain lists, while ``.items()`` yields wrapped
        # NonTensorData entries whose ``[i]`` does not index the underlying data.
        if hasattr(td, "keys") and callable(getattr(td, "keys", None)):
            try:
                field_names = [str(k) for k in td.keys()]
                out: dict[str, dict] = {}
                for i, key in enumerate(keys):
                    row: dict[str, Any] = {}
                    for field_name in field_names:
                        row[field_name] = _index_column(td[field_name], i)
                    out[str(key)] = {"fields": row}
                if out:
                    return out
            except (TypeError, KeyError, IndexError):
                pass

    if isinstance(data, dict):
        # Keyed dict {key: row} — only if every key maps back to a requested key.
        if all(k in keys for k in data.keys()) and keys:
            out = {}
            for key in keys:
                row = data.get(key)
                if row is None:
                    continue
                if isinstance(row, dict) and "fields" in row:
                    # Already row-shaped ({"fields": {...}}): pass through. Wrapping
                    # again would nest "fields" twice and every downstream
                    # ``row["<column>"]`` lookup (gen_group_uid, rollout_log_probs, …)
                    # would KeyError, silently dropping the whole GEN lane.
                    out[str(key)] = row
                else:
                    out[str(key)] = {"fields": row if isinstance(row, dict) else {"value": row}}
            return out
        # Columnar dict {field: [values...]} optionally carrying a "keys" column.
        got_keys = data.get("keys")
        if got_keys is None:
            got_keys = keys
        got_keys = [str(k) for k in got_keys]
        out = {}
        for i, key in enumerate(got_keys):
            row = {}
            for field_name, col in data.items():
                if field_name == "keys":
                    continue
                try:
                    row[field_name] = col[i]
                except (TypeError, IndexError, KeyError):
                    row[field_name] = None
            out[key] = {"fields": row}
        return out

    if isinstance(data, list | tuple):
        # Positional list: zip with requested keys.
        out = {}
        for i, (key, row) in enumerate(zip(keys, data, strict=False)):
            if row is None:
                continue
            out[str(key)] = {"fields": row if isinstance(row, dict) else {"value": row}}
        return out

    raise ValueError(
        f"bagel_corl: unrecognized transfer_queue.kv_batch_get return type {type(data)!r}; "
        "refusing to silently drop the GEN lane"
    )


def _index_column(col, i: int):
    """Index a stacked TQ column (Tensor / NonTensorStack / list) at position ``i``.

    ``NonTensorStack.__getitem__`` yields a ``NonTensorData`` wrapper — unwrap it so
    field dicts carry plain values (a wrapper would silently break
    ``split_und_gen_metas`` iteration and ``build_gen_flowgrpo_proto``). Tensors are
    returned as-is (never touch ``Tensor.data``).
    """
    if col is None:
        return None
    try:
        value = col[i]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        from tensordict import NonTensorData
    except ImportError:  # pragma: no cover - tensordict is a hard dependency here
        return value
    if isinstance(value, NonTensorData):
        return value.data
    return value


def _metric_scalar(metrics: dict, key: str) -> float | None:
    """Read ``metrics[key]`` as a float, or ``None`` when it is absent/not scalar.

    The step ``metrics`` dict is heterogeneous by construction: the engine layer writes
    plain floats (``actor/grad_norm`` via ``engine_workers.py:212``), ``rollout_corr/*``
    arrives as ``float`` from ``compute_rollout_correction_and_add_to_batch``, and the
    composite loss writes ``verl.utils.metric.Metric`` objects (which carry ``.values``,
    not ``.value``). A tripwire that assumed one of the three would silently read
    ``None`` on the run it exists to catch, so unwrap all of them.
    """
    value = metrics.get(key)
    if value is None:
        return None
    aggregate = getattr(value, "aggregate", None)
    if callable(aggregate):
        try:
            value = aggregate()
        except (TypeError, ValueError):
            return None
    if hasattr(value, "numel") and hasattr(value, "item"):
        # Duck-typed rather than ``isinstance(value, torch.Tensor)``: this module does not
        # import torch (it is a Trainer, not a worker), and a scalar tensor would otherwise
        # raise out of ``float(...)`` below.
        try:
            if value.numel() != 1:
                return None
            value = value.item()
        except (RuntimeError, TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _actor_merges_lora(model) -> bool:
    """True when the recipe asks the *actor* to merge LoRA into the base weights.

    ``actor_rollout_ref.model.lora.merge=True`` (required on vllm-omni >= 0.24, see
    ``run_agentic_bagel_rpco_lora.sh``) makes the trainer publish **merged** full weights:
    ``EngineWorker._update_weights`` reports ``peft_config=None``
    (``self.peft_merge = model_config.lora.get("merge", False)`` in
    ``verl/workers/engine_workers.py``), so every rollout replica takes the standard
    full-weight update and **no adapter is ever added engine-side**.

    Tolerates both shapes this codebase hands around: the resolved ``DictConfig``
    (``.get``) and the converted dataclass (attribute access).
    """
    lora_cfg = model.get("lora") if hasattr(model, "get") else getattr(model, "lora", None)
    if lora_cfg is None:
        return False
    if hasattr(lora_cfg, "get"):
        return bool(lora_cfg.get("merge", False))
    return bool(getattr(lora_cfg, "merge", False))


def _as_ray_actor_class(cls: Any) -> Any:
    """Return ``cls`` as a Ray actor class, tolerating both ``verl`` shapes.

    ``GlobalRequestLoadBalancer`` is ``@ray.remote``-decorated in the older sibling
    ``verl``, but a *plain* class in the newer installed one, which wraps it at
    instantiation time -- see ``verl.experimental.teacher_loop.teacher_model``, where the
    same construction reads ``ray.remote(GlobalRequestLoadBalancer).remote(...)``. The
    launch recipe picks whichever ``verl`` passes its probe (a stale sibling is rejected
    outright, see ``run_agentic_bagel_rpco_lora.sh``), so both spellings have to work:
    ``.remote`` is present only on the decorated class, and re-decorating an
    already-decorated class is an error rather than a no-op. Only the *decorated* class
    advertises ``remote``, so that attribute is the version marker.
    """
    if hasattr(cls, "remote"):
        return cls
    import ray

    return ray.remote(cls)


#: ``extra`` → step-log allowlist for the dual-lane bookkeeping. Explicit because the bus
#: carries far more than should be logged, and every one of these was absent from the step
#: log at some point (see the note inside ``_compute_advantage``).
_BAGEL_CORL_STEP_METRIC_KEYS: tuple[str, ...] = (
    "und/no_image_credit",
    "gen/dropped_incomplete_groups",
    "episode/J",
    "episode/K",
    "episode/K_zero",
)

#: Prefix form of the same allowlist, for the keys whose names are data-dependent: the
#: per-dim reward breakdown is one key per dim present on the batch, so it cannot be
#: spelled as a literal tuple (``bagel_corl_lib.aggregate_reward_breakdown_metrics``).
_BAGEL_CORL_STEP_METRIC_PREFIXES: tuple[str, ...] = (
    "bagel_corl/reward_dim/",
    "bagel_corl/reward_valid/",
    "bagel_corl/reward_no_image/",
)


def _bagel_corl_step_metric_keys(extra: dict) -> list[str]:
    """Step-log keys ``extra`` is currently carrying, in a stable order.

    Only present keys are returned, so a caller never has to re-test for ``None`` before
    copying -- and an absent measurement stays absent rather than being published as one.
    """
    keys = [key for key in _BAGEL_CORL_STEP_METRIC_KEYS if extra.get(key) is not None]
    keys.extend(
        sorted(
            key
            for key, value in extra.items()
            if value is not None and key.startswith(_BAGEL_CORL_STEP_METRIC_PREFIXES)
        )
    )
    return keys


def _flatten_rpc_results(value: Any) -> list[Any]:
    """Flatten ``AsyncOmni.collective_rpc``'s per-stage, per-worker nesting into one list."""
    if isinstance(value, list | tuple):
        return [leaf for item in value for leaf in _flatten_rpc_results(item)]
    return [value]


@register_trainer("bagel_corl_sync")
class OmniBagelCoRLTrainerSync(OmniPPOTrainerSync):
    """Synchronous Bagel Co-RL (Joint-Training): serial J-episode gather, then one UND+GEN optimizer step.

    Weight sync runs once per step in ``on_step_end``: ``_publish_und_ar_weights`` pushes the
    merged policy to the UND AR (Thinker) replica, then the inherited naive publish reaches the
    GEN hybrids. No mid-episode sync.
    Replay uses the sync ``ReplayBuffer`` even though ``trainer_mode`` is not the string ``sync``.

    ``main_omni`` defaults to omni ``OmniModelConfig`` / ``RolloutConfig``. Bagel Co-RL (Joint-Training) needs
    the diffusion model + rollout surfaces (``algorithm``, ``pipeline``, ``algo``,
    ``rollout_adapter``) for FlowGRPO GEN / weight sync while keeping the omni actor for UND
    ``ppo_loss``. Rewrite those configs and inject ``diffusion_loss`` before tokenizer /
    worker init.

    Do **not** only add ``rollout_adapter`` onto omni ``RolloutConfig``: ``init_model``
    instantiates via Hydra ``_target_``, and verl's ``RolloutConfig`` rejects that kwarg.

    Parent is intentional PPO V1 (``OmniPPOTrainerSync`` / ``TaskRunnerV1`` / token GRPO),
    not diffusion ``PolicyGradientDiffusionTrainerV1Sync.fit()`` and not OCR's legacy
    ``PolicyGradientRayTrainer``. GEN lane binds
    ``PolicyGradientDiffusionTrainerV1._compute_old_log_prob`` /
    ``_compute_advantage`` onto the same ``actor_rollout_wg``. Rewrite +
    ``ActorRolloutRefWorker`` unlock GEN *loss* on ``PPODiffusersFSDPEngine``.
    Composite ``forward_backward_batch`` (``bagel_corl_composite``) runs UND
    ``BagelForCoRL.compute_und_log_prob`` then GEN timestep FlowGRPO in one
    ``train_batch`` / one ``optimizer.step``. Same training step: UND token GRPO via
    ``super()._compute_advantage`` (TQ ``advantages``); GEN FlowGRPO via the bound
    diffusion V1 ``_compute_advantage`` into ``extra_info["bagel_corl_gen"]`` only
    (grouped by ``gen_group_uid``), including traj ``all_latents`` when stashed.
    Estimators are decoupled: UND uses ``algorithm.adv_estimator`` (omni default
    ``grpo``); GEN uses ``model.algorithm`` / ``flow_grpo``. Incomplete or traj-less
    GEN views skip GEN only; UND still updates.
    Inherited Omni ``resume_generation_replicas`` is a poor fit for ``bagel_single_stage``
    (diffusion V1 sleeps replicas instead) — revisit after GEN serving is live.
    """

    def _rewrite_bagel_corl_configs(self) -> None:
        """Retarget model/rollout to diffusion configs; inject actor ``diffusion_loss``."""
        raw = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True) or {}
        filtered = {k: v for k, v in raw.items() if k in _DIFFUSION_MODEL_KEYS}
        filtered["_target_"] = "verl_omni.workers.config.diffusion.DiffusionModelConfig"
        filtered.setdefault("algorithm", "flow_grpo")
        filtered["model_type"] = "diffusion_model"
        filtered.setdefault("architecture", "OmniBagelForConditionalGeneration")
        filtered.setdefault("composite_mode", "bagel_corl")
        # UniGRPO per-expert LRs: the GEN (*_moe_gen) optimizer group runs at its
        # own LR; UND keeps the base actor.optim.lr. Recipe-visible, configurable.
        filtered.setdefault("lr_gen", 3e-5)
        filtered.setdefault("trust_remote_code", True)
        # Bagel Co-RL is defined by the *disjoint* UND/GEN LoRA split (RFC §4.0.2), so the
        # target list must be explicit. The omni model yaml default is the string
        # ``all-linear``; ``validate_disjoint_lora_targets`` would iterate it character by
        # character and abort model construction with the useless
        # "unknown names: ['-', 'a', 'e', 'i', 'l', 'n', 'r']". Fail loud here instead,
        # where the recipe author can act on it.
        lora_targets = filtered.get("target_modules")
        if not isinstance(lora_targets, list | tuple):
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.model.target_modules as an explicit "
                f"list of UND/GEN LoRA module names, got {lora_targets!r} (RFC §4.0.2 dual-lane LoRA)"
            )

        # Strip omni-only rollout keys (do_sample, over_sample_rate, …) before Hydra instantiate.
        # Keep agent / multi_turn / trace / response_length: AgentLoopWorker requires them.
        rollout_allowed = {f.name for f in fields(DiffusionRolloutConfig)}
        rollout_raw = OmegaConf.to_container(self.config.actor_rollout_ref.rollout, resolve=True) or {}
        rollout_filtered = {k: v for k, v in rollout_raw.items() if k in rollout_allowed}
        rollout_filtered["_target_"] = "verl_omni.workers.config.diffusion.DiffusionRolloutConfig"
        rollout_filtered.setdefault("rollout_adapter", "default")
        # Omni YAML can leave free_cache_engine=null; falsy disables AsyncOmni.sleep and the
        # first update_weights then OOMs with the actor FSDP summon and the live Omni server
        # both resident.
        rollout_filtered["free_cache_engine"] = True
        rollout_filtered.setdefault("enable_sleep_mode", True)
        agent = dict(rollout_filtered.get("agent") or {})
        agent["_target_"] = "verl_omni.workers.config.omni.BagelCorlAgentLoopConfig"
        # RFC §5: ``gen_samples_per_call`` / ``max_generate_passes`` /
        # ``max_und_turns`` have a single SoT of
        # ``actor_rollout_ref.rollout.agent.*`` and **no code fallback** — the recipe
        # (or yaml) must set them, and ``verl_omni/utils/config.py`` validates S >= 2
        # and ``max_generate_passes == 1`` at launch. Inventing values here would
        # silently mask a missing/bad recipe knob from that validation.
        missing = [k for k in _BAGEL_AGENT_REQUIRED_KNOBS if agent.get(k) is None]
        if missing:
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.rollout.agent."
                f"{', '.join(missing)} (RFC §5 knob SoT, no code default)"
            )
        agent.setdefault(
            "agent_loop_manager_class",
            "verl_omni.agent_loop.bagel_corl_tq.BagelCorlAgentLoopManagerTQ",
        )
        agent.setdefault("default_agent_loop", "bagel_multiturn_agent")
        agent.setdefault("und_ar_serving_ready", False)
        rollout_filtered["agent"] = agent
        # Live GEN traj stash requires FlowGRPO SDE + logprobs (refuse ODE soft-skip).
        rollout_filtered["calculate_log_probs"] = True
        algo = dict(rollout_filtered.get("algo") or {})
        algo.setdefault("noise_level", 0.7)
        algo.setdefault("sde_window_size", 2)
        algo.setdefault("sde_window_range", [0, 7])
        algo.setdefault("sde_type", "sde")
        rollout_filtered["algo"] = algo
        pipeline = dict(rollout_filtered.get("pipeline") or {})
        pipeline.setdefault("num_inference_steps", 10)
        # UniGRPO (arXiv:2603.23500): CFG doubles per-step evaluations and branches the
        # rollout graph — prohibitive for multi-turn episodes. The Joint-Training DEFAULT
        # stays CFG-free (1.0); a recipe wanting canonical BAGEL quality opts in with
        # ``+actor_rollout_ref.rollout.pipeline.cfg_text_scale=4.0``. An unconditional
        # assignment here would make the whole GEN lane CFG-free with no recipe override.
        #
        # ``setdefault`` (not an assignment) is what keeps RFC §5 intact: the value below
        # is mirrored onto ``model.pipeline`` verbatim, so whatever the recipe picks stays
        # equal on both sides and the parity assert holds.
        #
        # This node is NOT what validation reads. ``VisualRewardManager`` selects
        # ``rollout.val_kwargs.pipeline`` when validating, and that subtree carries only the
        # keys the recipe names (``num_inference_steps`` / ``algo.noise_level``), so its
        # ``cfg_text_scale`` comes from ``DiffusionPipelineConfig``'s own default -- which
        # is 4.0, NOT this value. Validation therefore does not inherit a CFG-free rollout,
        # and it does not inherit a canonical one either; it is pinned at 4.0 regardless.
        # Here those happen to agree, so val is guided at 50 steps (the canonical arm of
        # the A/B). The one real consequence is that a recipe which sets this to 1.0 still
        # gets guided validation, i.e. there is no guided-vs-free val comparison for free.
        # Note also ``cfg_img_scale`` differs on that path: ``DiffusionPipelineConfig``
        # defaults it to 1.0 while this node sets 1.5, so val denoises with text guidance
        # only (2 forwards) where rollout also applies image guidance (3).
        pipeline.setdefault("cfg_text_scale", 1.0)
        # Default image guidance alongside it, so ``model.pipeline`` never lacks the key
        # (the parity assert below reads it) and both sides land on one value. The rollout
        # request would otherwise fall back to 1.5 while training's typed default is 1.0 —
        # see the assert for why that divergence only bites once text CFG is enabled.
        # 1.0 mirrors ``BAGEL_FLOWGRPO_CFG_DEFAULTS["cfg_img_scale"]``; a recipe wanting
        # canonical Bagel image guidance opts in via the recipe's CFG_IMG_SCALE.
        pipeline.setdefault("cfg_img_scale", 1.0)
        # Neither the omni rollout config nor its yaml declares a ``pipeline`` node —
        # the recipe reaches it with ``+actor_rollout_ref.rollout.pipeline.*`` — so the
        # dict built here must carry ``_target_`` exactly like ``diffusion_rollout.yaml``
        # does. Without it Hydra hands a plain mapping to the typed
        # ``pipeline: DiffusionPipelineConfig`` field, and every ``pipeline.<attr>``
        # read (``diffusion_agent_loop``, ``VisualRewardManager``) breaks afterwards.
        pipeline["_target_"] = "verl_omni.workers.config.diffusion.DiffusionPipelineConfig"
        rollout_filtered["pipeline"] = pipeline

        # ``val_kwargs`` needs the same retarget for a different reason: unlike ``pipeline``,
        # the omni schema *does* declare this node -- but with the AR ``_target_``
        # ``verl.workers.config.SamplingConfig``, while ``DiffusionRolloutConfig`` types the
        # field as ``DiffusionSamplingConfig``. SamplingConfig has no ``pipeline``/``algo``
        # subtree, so the recipe's ``+…val_kwargs.pipeline.num_inference_steps=50`` (and the
        # ``algo.noise_level`` beside it) compose fine as new keys and then kill the first
        # ``init_model`` on every actor rank:
        #   InstantiationException: Error in call to target
        #   'verl.workers.config.rollout.SamplingConfig':
        #   TypeError("SamplingConfig.__init__() got an unexpected keyword argument 'pipeline'")
        #   full_key: actor_rollout_ref.rollout.val_kwargs
        # ``do_sample`` is AR-only and has to go with the target, so filter as well as retarget
        # -- exactly what the parent node above does. This is also what makes the validate
        # branch of ``composite_agent_loop`` work at all: it reads
        # ``config.val_kwargs.pipeline`` / ``.algo`` / ``.seed``, none of which exist on the
        # AR SamplingConfig.
        val_allowed = {f.name for f in fields(DiffusionSamplingConfig)}
        val_kwargs = {
            k: v for k, v in (rollout_filtered.get("val_kwargs") or {}).items() if k in val_allowed and k != "_target_"
        }
        val_kwargs["_target_"] = "verl_omni.workers.config.diffusion.DiffusionSamplingConfig"
        rollout_filtered["val_kwargs"] = val_kwargs

        # ``cfg_text_scale`` is a transition-kernel knob, not a style knob: the
        # GEN importance ratio is only unbiased when training and rollout use the
        # SAME value (RFC §4.8.5 / §5). The training side reads
        # ``model.pipeline.cfg_text_scale`` via
        # ``BagelDiffusion._get_cfg_params``, which falls back to
        # ``BAGEL_FLOWGRPO_CFG_DEFAULTS`` = 4.0 — so leaving it unset here silently
        # trained with CFG while rollout ran CFG-free.
        #
        # ``omni_model.yaml`` declares no ``pipeline`` node at all, so anything set
        # here has to be the whole node, not a patch onto an existing one. Mirror what
        # ``diffusion_model.yaml`` does for the stock diffusion path and make the model
        # node the rollout pipeline itself (rollout is the knob SoT, RFC §5): that keeps
        # ``num_inference_steps`` / ``height`` / ``width`` — used by the training adapter
        # for the sigma schedule and latent position ids — in lockstep with rollout, and
        # carries the ``_target_`` that turns this into a real ``DiffusionPipelineConfig``.
        # A bare ``{"cfg_text_scale": 1.0}`` dict here would instead replace the dataclass
        # default and then fail every ``model_config.pipeline.<attr>`` read.
        filtered["pipeline"] = dict(pipeline)
        # Same reasoning for the SDE kernel: the training adapter scores each recorded
        # transition with ``model.algo.noise_level``, and ``DiffusionModelConfig.algo``
        # otherwise falls back to ``DiffusionRolloutAlgoConfig``'s 1.0 while rollout
        # samples at 0.7. That scored every GEN transition under the wrong Gaussian, a
        # constant log-ratio of ~+0.25 that clipped every GEN update.
        filtered["algo"] = {**algo, "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutAlgoConfig"}

        with open_dict(self.config):
            self.config.actor_rollout_ref.model = OmegaConf.create(filtered)
            self.config.actor_rollout_ref.rollout = OmegaConf.create(rollout_filtered)
            actor = self.config.actor_rollout_ref.actor
            if actor.get("diffusion_loss") is None:
                actor.diffusion_loss = OmegaConf.create(
                    {
                        "_target_": "verl_omni.workers.config.diffusion.DiffusionLossConfig",
                        "loss_mode": filtered.get("algorithm", "flow_grpo"),
                    }
                )
            else:
                if actor.diffusion_loss.get("loss_mode") is None:
                    actor.diffusion_loss.loss_mode = filtered.get("algorithm", "flow_grpo")
                # ``_target_`` binds this node to ``DiffusionLossConfig`` and supplies every knob
                # the loss reads as an attribute (``adv_clip_max``, ``clip_ratio``, ``dpo_beta``).
                # A recipe that overrides *inside* the node creates a plain dict and leaves it
                # unbound. Stamp the schema unconditionally so an override cannot strip the
                # declared defaults.
                if actor.diffusion_loss.get("_target_") is None:
                    actor.diffusion_loss["_target_"] = "verl_omni.workers.config.diffusion.DiffusionLossConfig"
            # verl AgentLoopBase always reads data.continuous_token (struct). Omni data
            # schemas may omit it; missing key aborts every episode → empty TQ →
            # "no materializable trajectories".
            data = self.config.get("data")
            if data is None:
                self.config.data = data = OmegaConf.create({})
            if data.get("continuous_token") is None:
                data.continuous_token = OmegaConf.create({"enable": False, "model_family": "auto"})
            elif data.continuous_token.get("enable") is None:
                data.continuous_token.enable = False

        # RFC §5: ``cfg_text_scale`` is a transition-kernel knob whose single SoT is
        # ``model.pipeline`` AND ``rollout.pipeline``. A mismatch is NON-CONFORMANT —
        # the GEN importance ratio would be computed under a different kernel than
        # the rollout that produced the trajectory. Assert it at launch, not later.
        # (In practice the block above copies one dict into both nodes, so this can only
        # trip if some later code rewrites one side independently — which is exactly the
        # regression worth trapping.)
        model_cfg = float(self.config.actor_rollout_ref.model.pipeline.cfg_text_scale)
        rollout_cfg = float(self.config.actor_rollout_ref.rollout.pipeline.cfg_text_scale)
        if model_cfg != rollout_cfg:
            raise ValueError(
                "bagel_corl_sync CFG parity violated: "
                f"model.pipeline.cfg_text_scale={model_cfg} != "
                f"rollout.pipeline.cfg_text_scale={rollout_cfg}. Training and rollout "
                "must share one transition kernel (RFC §5); set both to the same value "
                "(1.0 for CFG-free, 4.0 for canonical BAGEL)."
            )
        # ``cfg_img_scale`` is a transition-kernel knob too, and it is NOT self-correcting:
        # the rollout request falls back to 1.5 when the key is absent
        # (``pipeline_bagel.py``) while training reads ``model.pipeline``, whose typed
        # default is 1.0. The mismatch is invisible at ``cfg_text_scale == 1.0`` because
        # ``generate_image`` gates its entire CFG block on ``cfg_text_scale > 1.0`` -- so
        # an unset ``cfg_img_scale`` used to be inert and turns into a real kernel
        # divergence (3 forwards/step vs 2) the moment text CFG is enabled. Pin it.
        model_img = float(self.config.actor_rollout_ref.model.pipeline.cfg_img_scale)
        rollout_img = float(self.config.actor_rollout_ref.rollout.pipeline.cfg_img_scale)
        if model_img != rollout_img:
            raise ValueError(
                "bagel_corl_sync CFG parity violated: "
                f"model.pipeline.cfg_img_scale={model_img} != "
                f"rollout.pipeline.cfg_img_scale={rollout_img}. Image guidance branches "
                "the denoise graph exactly like text guidance, so training and rollout "
                "must agree (RFC §5). Pin both via "
                "'+actor_rollout_ref.rollout.pipeline.cfg_img_scale=1.5'."
            )

    def _init_tokenizer(self):
        self._rewrite_bagel_corl_configs()
        model_config: DiffusionModelConfig = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.model, DiffusionModelConfig
        )
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _build_replay_buffer(self):
        # ``trainer_mode`` is temporarily "sync" so the base builds the sync
        # ``ReplayBuffer`` (and resolves its knob set) while the *configured* mode stays
        # ``bagel_corl_sync`` everywhere else; restore it in ``finally`` regardless.
        original_mode = self.trainer_mode
        self.trainer_mode = "sync"
        try:
            buffer = super()._build_replay_buffer()
        finally:
            self.trainer_mode = original_mode
        # The dual-lane writer tags every GEN seed row ``is_auxiliary`` so it stays out of a
        # sampled batch: the GEN lane reaches the trainer through the UND row's
        # ``child_gen_keys`` column, and the UND pass cannot score a seed row (no
        # ``prompts``/``responses``). Sibling ``verl`` 0.9.0.dev enforces that in
        # ``_materialize_batch``; the installed 0.10.0.dev has no such filter at all, so the
        # first run with a healthy UND lane (K >= 1, i.e. GEN rows exist for the first time)
        # died at step 0 with ``KeyError: 'prompts'`` inside ``_compute_reward_colocate`` --
        # one requested UND field dropped by ``BatchMeta.select_fields`` because a GEN row was
        # in the same batch. Apply the rule on this side rather than trusting the install; see
        # ``bagel_corl_replay_buffer`` for the measurements.
        return install_auxiliary_row_filter(buffer)

    def _init_resource_pool_mgr(self):
        """Use VeRL-Omni ``ActorRolloutRefWorker`` so Bagel FSDP + composite loss are available."""
        import ray

        super()._init_resource_pool_mgr()
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker as OmniActorRolloutRefWorker

        remote_cls = ray.remote(OmniActorRolloutRefWorker)
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = remote_cls

    def _expected_s(self) -> int:
        """Seeds per ``generate_image`` call (RFC §5 knob; no silent default).

        The recipe sets ``agent.gen_samples_per_call``; a fallback here would make
        the dual-lane gate accept a group size nobody configured, so fail loud.
        """
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        raw = agent.get("gen_samples_per_call")
        if raw is None:
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.rollout.agent.gen_samples_per_call; "
                "the dual-lane GEN seed gate cannot infer S"
            )
        return int(raw)

    def _gen_adv_estimator(self) -> str:
        """GEN FlowGRPO estimator — never reuse ``algorithm.adv_estimator`` (UND token GRPO)."""
        model = self.config.actor_rollout_ref.model
        algo = self.config.algorithm
        # Prefer diffusion model.algorithm / explicit gen override; omni default adv_estimator is token grpo.
        for candidate in (
            algo.get("gen_adv_estimator"),
            model.get("algorithm"),
        ):
            if candidate:
                return str(candidate)
        return "flow_grpo"

    @staticmethod
    def _extra_info(batch) -> dict:
        extra = getattr(batch, "extra_info", None)
        if extra is None:
            extra = {}
            if hasattr(batch, "extra_info"):
                batch.extra_info = extra
        return extra

    def _diffusion_v1_gen_lane(self) -> DiffusionV1GenLane:
        """GEN hooks from ``PolicyGradientDiffusionTrainerV1`` on this job's workers.

        The lane carries its own GEN estimator so ``_compute_advantage`` never reads
        the UND token ``algorithm.adv_estimator`` (RFC §4.4).
        """
        lane = getattr(self, "_diff_v1_gen_lane", None)
        wg = getattr(self, "actor_rollout_wg", None)
        gen_estimator = self._gen_adv_estimator()
        if (
            lane is None
            or getattr(lane, "actor_rollout_wg", None) is not wg
            or getattr(lane, "gen_adv_estimator", None) != gen_estimator
        ):
            self._diff_v1_gen_lane = DiffusionV1GenLane(self, gen_adv_estimator=gen_estimator)
        return self._diff_v1_gen_lane

    def _compute_old_log_prob(self, batch, metrics: dict):
        """UND: PPO V1 token old-logprob. GEN: diffusion V1 ``infer_actor_batch`` (required when K>0)."""
        from verl_omni.trainer.omni.bagel_corl_parity import FrozenPolicyCapture

        capture = FrozenPolicyCapture.from_environment(self)
        rollout_rows = None
        if capture is not None:
            rollout_rows, _ = capture.fetch_und(batch, _normalize_tq_kv_get_result)
            for row in rollout_rows:
                capture.provenance(row)

        def compute_und():
            base_recompute = super(OmniBagelCoRLTrainerSync, self)._compute_old_log_prob
            if capture is None:
                return base_recompute(batch, metrics)
            return capture.recompute_und(batch, metrics, base_recompute, _normalize_tq_kv_get_result, rollout_rows)

        extra = self._extra_info(batch)
        gen_rows = self._gen_batch_from_step(batch)
        logger.info(
            "bagel_corl old_log_prob batch_rows=%d roles=%s gen_rows=%d",
            len(batch),
            _bagel_role_histogram(getattr(batch, "tags", None)),
            len(gen_rows),
        )
        proto = build_gen_flowgrpo_proto(gen_rows)
        if proto is None:
            return compute_und()
        if "all_latents" not in proto.batch.keys():
            raise RuntimeError(
                "bagel_corl GEN old_log_prob requires all_latents; refuse rollout_log_probs-only recompute skip."
            )
        if getattr(self, "actor_rollout_wg", None) is None:
            # CPU unit tests: no worker group. Packed traj logprobs stay on proto for GEN advantage.
            metrics["gen/old_log_prob_recomputed"] = 0.0
            return compute_und()
        # The GEN proto's row count is data-dependent: one row per *trainable* GEN call, and K
        # varies per episode, so ``len(proto)`` is not necessarily a multiple of the actor DP
        # size. ``actor_rollout_wg.infer_actor_batch`` dispatches through
        # ``_split_args_kwargs_data_proto`` -> ``TensorDict.chunk(chunks=dp_size)``, whose guard
        # is ``assert len(td) % chunks == 0`` (``verl/utils/tensordict_utils.py``). Pad with the
        # same helper ``_balance_batch`` uses for the UND/main batch, then drop the repeated rows
        # so ``bagel_corl_gen_old`` stays aligned with ``gen_rows``.
        # ``pad_dataproto_to_divisor`` appends its copies at the tail and
        # ``unpad_dataproto`` slices ``[:-pad_size]``, so the two are symmetric.
        divisor = self._gen_dispatch_divisor()
        pad_size = 0
        if divisor > 1 and len(proto) % divisor != 0:
            unpadded_n = len(proto)
            proto, pad_size = pad_dataproto_to_divisor(proto, size_divisor=divisor)
            logger.info(
                "bagel_corl GEN old_log_prob: padded %d -> %d rows for dispatch divisor %d",
                unpadded_n,
                len(proto),
                divisor,
            )
        if capture is not None:
            import copy

            replay_proto = copy.deepcopy(proto)
            proto = copy.deepcopy(replay_proto)
        old = self._diffusion_v1_gen_lane()._compute_old_log_prob(proto)
        if capture is not None:
            # Both actor calls receive the exact same frozen input, before any update.
            proto = copy.deepcopy(replay_proto)
            repeat = self._diffusion_v1_gen_lane()._compute_old_log_prob(proto)
            capture_proto = unpad_dataproto(replay_proto, pad_size) if pad_size else replay_proto
            capture_old = unpad_dataproto(old, pad_size) if pad_size else old
            capture_repeat = unpad_dataproto(repeat, pad_size) if pad_size else repeat
            capture.write_gen(capture_proto, capture_old, capture_repeat, gen_rows, rollout_rows)
        if pad_size:
            old = unpad_dataproto(old, pad_size)
        extra["bagel_corl_gen_old"] = old.batch
        metrics["gen/old_log_prob_recomputed"] = 1.0
        return compute_und()

    def _gen_dispatch_divisor(self) -> int:
        """Row-count divisor the GEN lane's worker-group dispatch requires.

        Mirrors ``DiffusionV1TrainerBase._balance_batch``: ``dp_size`` comes from the
        actor's dispatch info, and the row count must also satisfy the actor mini-batch
        multiple that downstream chunking assumes.
        """
        import math

        wg = getattr(self, "actor_rollout_wg", None)
        dp_size = 1
        if wg is not None and hasattr(wg, "_query_dispatch_info"):
            info = wg._query_dispatch_info("actor")
            if isinstance(info, dict):
                dp_size = max(info.values()) + 1 if info else 1
            elif isinstance(info, list | tuple | set):
                dp_size = max(info) + 1 if info else 1
            else:
                # ONE_TO_ALL dispatch may return a scalar dp rank per worker.
                dp_size = int(info) + 1 if info is not None else 1
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * rollout_n
        return max(1, math.lcm(int(dp_size), int(mini)))

    def _gen_batch_from_step(self, batch) -> list:
        """GEN rows for FlowGRPO: dual-lane ``child_gen_keys`` or ``extra['gen_batch']`` only."""
        extra = self._extra_info(batch)
        if extra.get("gen_batch") is not None:
            return list(extra["gen_batch"])

        from verl_omni.agent_loop.bagel_corl_tq import split_und_gen_metas

        und_records = self._und_records_from_batch(batch)
        if und_records and any("child_gen_keys" in (r.get("fields") or r) for r in und_records):
            gen_by_key = self._fetch_gen_records_by_keys(batch, und_records)
            _, gen_batch = split_und_gen_metas(und_records, gen_by_key)
            # Batch aggregation (mean J/K over siblings, summed dropped groups,
            # fraction without image credit) — first-row-wins collapse hid every
            # episode but one (audit module A).
            aggregated = aggregate_episode_metrics(und_records)
            # Reward breakdown + the ``used_image_credit`` split ride the same allowlist:
            # training otherwise publishes only the scalar ``critic/score/mean``, whose
            # swing tracks ``episode/K_zero`` at -0.71 and hides the dim that is moving.
            breakdown = aggregate_reward_breakdown_metrics(und_records)
            if not breakdown:
                # An empty breakdown puts the step log back exactly where it was before the
                # breakdown existed -- one unattributable scalar -- and it fails silently,
                # because a missing key is indistinguishable from a step that simply did not
                # publish it. Say so on the step it happens, naming the fields the rows
                # actually carry, so the next reader is not left guessing which shape the
                # agent loop handed over.
                logger.warning(
                    "bagel_corl reward breakdown EMPTY: und_records=%d row_field_keys=%s",
                    len(und_records),
                    und_record_field_keys(und_records, limit=24),
                )
            aggregated.update(breakdown)
            for key, value in aggregated.items():
                if extra.get(key) is None:
                    extra[key] = value
            extra["gen_batch"] = gen_batch
            return gen_batch

        bagel = extra.get("bagel_corl") if isinstance(extra.get("bagel_corl"), dict) else {}
        meta = getattr(batch, "meta_info", None) or {}
        bagel_meta = meta.get("bagel_corl") if isinstance(meta, dict) else {}
        if bagel.get("gen_batch") or (isinstance(bagel_meta, dict) and bagel_meta.get("gen_batch")):
            raise RuntimeError(
                "bagel_corl: GEN rows found only on nested bagel_corl.gen_batch / meta_info; "
                "dual-lane extra['gen_batch'] or child_gen_keys is required (legacy nested path removed)."
            )
        # Loud on the driver side (Ray only forwards WARNING+ from the agent-loop workers, so the
        # worker's own dual-lane pack log never reaches this log). Without it a skipped GEN lane and
        # a genuinely empty one are indistinguishable in the step metrics.
        logger.info(
            "bagel_corl gen_batch empty (GEN lane will skip): und_records=%d has_child_gen_keys=%s "
            "has_tq_handle=%s ntb_keys=%s batch=%s",
            len(und_records),
            bool(und_records) and any("child_gen_keys" in (r.get("fields") or {}) for r in und_records),
            hasattr(batch, "keys") and hasattr(batch, "partition_id"),
            sorted((getattr(batch, "non_tensor_batch", None) or {}).keys()),
            type(batch).__name__,
        )
        return []

    def _und_records_from_batch(self, batch) -> list[dict]:
        extra = self._extra_info(batch)
        if extra.get("und_batch"):
            return [{"fields": row} for row in extra["und_batch"]]
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        if ntb.get("child_gen_keys") is not None:
            length = len(ntb["child_gen_keys"])
            records = []
            for i in range(length):
                fields = {}
                for key, col in ntb.items():
                    try:
                        fields[key] = col[i]
                    except (TypeError, IndexError, KeyError):
                        continue
                records.append({"fields": fields})
            return records
        # TensorDict carriers (the replay-buffer / balanced-batch shape) keep the dual-lane
        # bookkeeping in ``extra_fields`` instead of ``child_gen_keys``; the v1 advantage phase can
        # be handed a TensorDict rather than the rollout ``KVBatchMeta``, for which
        # ``_extra_fields_from_tq`` returns ``None`` (no ``partition_id``). Without this branch the
        # GEN lane is skipped for the whole step even with complete S-groups.
        if ntb.get("extra_fields") is not None:
            extras = list(ntb["extra_fields"])
            child_col = ntb.get("child_gen_keys")
            child_col = list(child_col) if child_col is not None else None
            tags = list(getattr(batch, "tags", None) or [])
            indices = (
                [i for i, tag in enumerate(tags) if not tag.get("is_padding", False)]
                if len(tags) == len(extras)
                else range(len(extras))
            )
            records = []
            for idx in indices:
                mapping = extras[idx] if isinstance(extras[idx], dict) else {}
                fields = dict(mapping)
                if "child_gen_keys" not in fields and isinstance(mapping.get("extra_fields"), dict):
                    fields.update(mapping["extra_fields"])
                if child_col is not None and idx < len(child_col):
                    fields = self._merge_child_gen_keys(fields, child_col[idx])
                records.append({"fields": fields})
            return records
        extras = self._extra_fields_from_tq(batch)
        if extras is None:
            return []
        keys = list(getattr(batch, "keys", []) or [])
        if keys and len(extras) != len(keys):
            raise RuntimeError(
                f"bagel_corl UND extra_fields fetch returned {len(extras)} row(s) for {len(keys)} key(s); "
                "row↔key alignment is what the GEN gather and the J/K aggregation rely on"
            )
        # ``upsample_batch_to_divisible_size`` builds its synthetic rows by deep-copying the first
        # row's fields *and* tag, so a padding row still carries its template's ``child_gen_keys``,
        # ``extra_fields`` and J/K. Counting it as an episode would fetch the template's GEN rows
        # twice and fold its J/K into the batch mean a second time, so drop it here. The tags live
        # on the KVBatchMeta in key order, which is exactly the fetch order.
        tags = list(getattr(batch, "tags", None) or [])
        indices = (
            [i for i, tag in enumerate(tags) if not tag.get("is_padding", False)]
            if len(tags) == len(extras)
            else range(len(extras))
        )
        records = []
        for idx in indices:
            extra_row = extras[idx]
            mapping = extra_row if isinstance(extra_row, dict) else {}
            fields = dict(mapping)
            if "child_gen_keys" not in fields and isinstance(mapping.get("extra_fields"), dict):
                fields.update(mapping["extra_fields"])
            records.append({"fields": fields})
        return records

    def _fetch_gen_records_by_keys(self, batch, und_records: list[dict]) -> dict[str, dict]:
        """Lookup GEN TQ rows by ``child_gen_keys``. Returns key → {fields: ...}.

        ``transfer_queue.kv_batch_get`` returns a **columnar TensorDict** (one entry
        per requested key, in request order), not a keyed dict. Treating it as a dict
        silently yields ``{}`` and drops the whole GEN lane. Normalize every known
        return shape here and fail-loud on an unrecognized one instead of silently
        skipping GEN.
        """
        extra = self._extra_info(batch)
        if isinstance(extra.get("gen_by_key"), dict):
            return {str(k): ({"fields": v} if "fields" not in v else v) for k, v in extra["gen_by_key"].items()}

        keys: list[str] = []
        for rec in und_records:
            fields = rec.get("fields") or rec
            for child in fields.get("child_gen_keys") or []:
                # Order-preserving dedupe: TQ's KVBatchMeta rejects duplicate keys, and a batch
                # may legitimately point at the same seed row from more than one gather path.
                if str(child) not in keys:
                    keys.append(str(child))
        if not keys:
            return {}
        if not hasattr(batch, "partition_id"):
            raise RuntimeError(
                f"bagel_corl: batch has child_gen_keys={len(keys)} but no partition_id; cannot fetch GEN TQ rows"
            )

        import transfer_queue as tq

        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id)
        out = _normalize_tq_kv_get_result(data, keys)
        missing = [k for k in keys if k not in out]
        if missing:
            raise RuntimeError(
                f"bagel_corl GEN TQ fetch incomplete: {len(out)}/{len(keys)} rows (first missing: {missing[:3]})"
            )
        return out

    @classmethod
    def _extra_fields_from_tq(cls, batch):
        if not hasattr(batch, "keys") or not hasattr(batch, "partition_id"):
            return None
        try:
            import transfer_queue as tq

            data = tq.kv_batch_get(
                keys=list(batch.keys), partition_id=batch.partition_id, select_fields=["extra_fields", "child_gen_keys"]
            )
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("bagel_corl UND extra_fields TQ fetch failed") from exc
        if isinstance(data, dict):
            if "extra_fields" in data:
                return data["extra_fields"]
            return cls._rows_in_batch_key_order(batch, data)
        # ``kv_batch_get`` hands back a **columnar TensorDict** (one entry per requested key, in
        # request order), not a dict and not a keyed mapping; iterating it yields *field names*.
        # Read the column instead, exactly like ``ReplayBuffer._dapo_filtered_keys`` does.
        if "extra_fields" in data.keys():
            rows = list(data["extra_fields"])
            # ``extra_fields`` and ``child_gen_keys`` come back as **sibling columns**; the
            # dual-lane put writes ``child_gen_keys`` both inside the blob and as its own column.
            # Returning only the first column drops the second for any row whose blob omitted it.
            # Re-attach it before handing rows back.
            if "child_gen_keys" in data.keys():
                child_col = list(data["child_gen_keys"])
                rows = [
                    OmniBagelCoRLTrainerSync._merge_child_gen_keys(row, child_col[i] if i < len(child_col) else None)
                    for i, row in enumerate(rows)
                ]
            return rows
        if "child_gen_keys" in data.keys():
            return list(data["child_gen_keys"])
        return data

    @staticmethod
    def _rows_in_batch_key_order(batch, data: dict):
        """Normalize a per-key mapping (``{tq_key: row}``) into batch-key order."""
        keys = [str(k) for k in (getattr(batch, "keys", None) or [])]
        if keys and all(k in data for k in keys):
            return [data[k] for k in keys]
        return list(data.values())

    @staticmethod
    def _merge_child_gen_keys(row, child_keys):
        """Give a fetched row the ``child_gen_keys`` column when its own blob omitted it."""
        if not isinstance(row, dict) or not child_keys:
            return row
        if row.get("child_gen_keys"):
            return row
        merged = dict(row)
        merged["child_gen_keys"] = list(child_keys)
        return merged

    def _compute_advantage(self, batch, metrics: dict):
        """Same step, separate tensors: GEN FlowGRPO → ``bagel_corl_gen``; UND token GRPO → TQ via ``super``.

        Do **not** write FlowGRPO into TQ ``advantages`` (that field is UND token GRPO only).
        Do **not** pass ``algorithm.adv_estimator`` (omni default ``grpo``) into GEN.
        """
        extra = self._extra_info(batch)
        # NOTE: the dual-lane metrics (``episode/J``, ``episode/K``, ``gen/dropped_incomplete_groups``,
        # ``und/no_image_credit``) do not exist on ``extra`` yet -- ``_gen_batch_from_step`` below is
        # what runs ``aggregate_episode_metrics`` and populates them. The copy therefore has to happen
        # *after* the gather, or every one of them stays absent from the step log.

        algo = self.config.algorithm
        gen_estimator = self._gen_adv_estimator()
        und_estimator = str(algo.get("adv_estimator", "grpo"))
        if gen_estimator == und_estimator and und_estimator in {"grpo", "gspo", "gae", "rloo", "reinforce_plus_plus"}:
            raise ValueError(
                f"Bagel Co-RL (Joint-Training) GEN adv_estimator={gen_estimator!r} collides with UND token estimator; "
                "set actor_rollout_ref.model.algorithm=flow_grpo (GEN) and leave algorithm.adv_estimator for UND"
            )
        proto = build_gen_flowgrpo_proto(self._gen_batch_from_step(batch))
        gen_proto = None
        gen_metrics: dict = {
            "gen/skipped_no_groups": 1.0,
            "gen/num_rows": 0.0,
            "has_complete_gen_groups": 0.0,
        }
        if proto is not None:
            old_batch = extra.get("bagel_corl_gen_old")
            if old_batch is not None and "old_log_probs" in old_batch.keys():
                proto.batch["old_log_probs"] = old_batch["old_log_probs"]
                if "old_prev_sample_mean" in old_batch.keys():
                    proto.batch["old_prev_sample_mean"] = old_batch["old_prev_sample_mean"]
            if getattr(self, "actor_rollout_wg", None) is not None:
                gen_proto = self._diffusion_v1_gen_lane()._compute_advantage(proto)
                if "all_latents" not in gen_proto.batch.keys():
                    raise RuntimeError("bagel_corl GEN V1 advantage returned no all_latents; refuse skip GEN loss.")
                # The diffusion V1 entry point returns the advantage proto but does not carry the
                # GEN prompt ids onto the batch the way ``apply_gen_flowgrpo_advantage`` does, and
                # the diffusion training adapter reads them from the micro-batch
                # (``diffusers_training_adapter.py:152``). Without this the GEN loss dies with
                # ``KeyError: 'prompt_token_ids'`` at the last step, after the whole rollout,
                # reward and advantage computation has already been paid for.
                fold_gen_prompt_token_ids(gen_proto)
                gen_metrics = {
                    "gen/skipped_no_groups": 0.0,
                    "gen/num_rows": float(len(gen_proto)),
                    "gen/num_usable_rows": float(len(gen_proto)),
                    "has_complete_gen_groups": 1.0,
                    "gen/has_traj": 1.0,
                }
            else:
                # CPU tests without a worker group: same FlowGRPO helper, still requires traj.
                gen_proto, gen_metrics = apply_gen_flowgrpo_advantage(
                    self._gen_batch_from_step(batch),
                    adv_estimator=gen_estimator,
                    norm_adv_by_std_in_grpo=bool(algo.get("norm_adv_by_std_in_grpo", True)),
                    global_std=bool(algo.get("global_std", True)),
                    algo_config=algo,
                )
        if gen_proto is not None:
            adv = gen_proto.batch["advantages"]
            valid = gen_proto.batch["gen_transition_valid"].bool()
            metrics["gen/adv_std"] = float(adv[valid].float().std(unbiased=False).item())
            metrics["gen/effective_transitions"] = float(valid.sum().item())
            metrics["gen/valid_rows"] = float(gen_proto.batch["gen_valid"].sum().item())
            scores = gen_proto.batch["sample_level_scores"].reshape(-1)
            uids = gen_proto.non_tensor_batch["uid"]
            group_stds = [
                float(scores[torch.tensor(uids == uid, device=scores.device)].std(unbiased=False))
                for uid in set(uids.tolist())
            ]
            metrics["gen/zero_std_group_fraction"] = sum(v == 0.0 for v in group_stds) / len(group_stds)
        metrics.update(gen_metrics)
        # Now that the gather has run, the dual-lane bookkeeping is on ``extra`` (see the note at the
        # top of this method); publish it so a skipped GEN lane is attributable from the step log.
        for metric_key in _bagel_corl_step_metric_keys(extra):
            metrics.setdefault(metric_key, extra[metric_key])
        has_complete = bool(gen_metrics.get("has_complete_gen_groups"))
        extra["has_complete_gen_groups"] = has_complete
        extra["skip_gen"] = not has_complete
        extra["num_gen_rows"] = int(gen_metrics.get("gen/num_usable_rows") or gen_metrics.get("gen/num_rows") or 0)
        # FlowGRPO advantages live only here — never merge into TQ UND advantages.
        extra["bagel_corl_gen"] = None if gen_proto is None else gen_proto.batch
        skipped = gen_metrics.get("gen/skipped_no_groups", 1.0 if not has_complete else 0.0)
        metrics["gen/skipped_no_groups"] = float(skipped)
        logger.info(
            "bagel_corl_sync advantage und=%s gen=%s skip_gen=%s J=%s K=%s K0=%s",
            und_estimator,
            gen_estimator,
            extra["skip_gen"],
            metrics.get("episode/J"),
            metrics.get("episode/K"),
            metrics.get("episode/K_zero"),
        )
        # UND token GRPO writes advantages/returns onto episode TQ keys.
        #
        # ``super()`` hands back a *different* ``KVBatchMeta``: the v1 base ends with
        # ``batch = tq.kv_batch_put(...)``, and ``kv_batch_put`` builds its result from a meta it
        # creates internally (``extra_info=batch_meta.extra_info``), so the returned meta carries
        # the ``KVBatchMeta`` dataclass default ``{}`` -- none of the flags written to ``extra``
        # above survive the hand-off, and ``step()`` then rebinds ``batch`` to that new object
        # (``verl/trainer/ppo/v1/trainer_base.py:444``). Re-publish the lane's flags onto the meta
        # ``super()`` actually returns, which is the object ``_update_actor`` will be handed.
        result = super()._compute_advantage(batch, metrics)
        carried = self._extra_info(result)
        for key in (
            "has_complete_gen_groups",
            "num_gen_rows",
            "bagel_corl_gen",
            *_bagel_corl_step_metric_keys(extra),
        ):
            if key in extra:
                carried[key] = extra[key]
        carried["skip_gen"] = not has_complete
        return result

    def _update_actor(self, batch, metrics: dict):
        """Single composite ``update_actor`` after the N-sibling gather. Skip GEN when no complete S-groups."""
        from verl.utils import tensordict_utils as tu

        extra = self._extra_info(batch)
        # Ownership lives in tensor columns so DP and optimizer minibatching slice it.
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

        has_complete = bool(extra.get("has_complete_gen_groups"))
        extra["skip_gen"] = not has_complete
        extra["has_complete_gen_groups"] = has_complete
        if not has_complete:
            metrics["gen/skipped_no_groups"] = 1.0
        for key in _bagel_corl_step_metric_keys(extra):
            metrics[key] = extra[key]
        # The flags have to ride on whatever carrier this trainer's bus uses, and ``_extra_info``
        # above already put them where the actor will look: the v1 bus hands us a ``KVBatchMeta``,
        # whose ``extra_info`` dict the TQ dispatch layer copies onto the actor's TensorDict as
        # non-tensor data (``verl/utils/transferqueue_utils.py:160-177``) -- which is exactly where
        # the composite reads them back (``bagel_corl_composite.py:129,131`` and
        # ``diffusers_impl.py:954,956``). ``tu.assign_non_tensor_data`` is only valid for a real
        # TensorDict; calling it on the meta asserts (``tensordict_utils.py:44``), and
        # ``hasattr(batch, "keys")`` is *not* a TensorDict test -- ``KVBatchMeta.keys`` is its list
        # of TQ keys.
        from tensordict import TensorDict

        if isinstance(batch, TensorDict):
            tu.assign_non_tensor_data(batch, "skip_gen", extra["skip_gen"])
            tu.assign_non_tensor_data(batch, "has_complete_gen_groups", has_complete)
            tu.assign_non_tensor_data(batch, "num_gen_rows", int(extra.get("num_gen_rows") or 0))
            if extra.get("bagel_corl_gen") is not None:
                tu.assign_non_tensor_data(batch, "bagel_corl_gen", extra["bagel_corl_gen"])
        logger.info(
            "bagel_corl_sync update_actor skip_gen=%s policy_version=%s J=%s K=%s K0=%s",
            extra.get("skip_gen"),
            getattr(self, "global_steps", None),
            metrics.get("episode/J"),
            metrics.get("episode/K"),
            metrics.get("episode/K_zero"),
        )
        updated = super()._update_actor(batch, metrics)
        # ``metrics`` now carries ``actor/*`` (renamed by ``trainer_base._update_actor``)
        # alongside this step's ``rollout_corr/*``. See ``_divergence_tripwire``.
        for key, value in list(metrics.items()):
            if key.startswith(("actor/und/", "actor/gen/")):
                metrics[key.removeprefix("actor/")] = value
        if "rollout_corr/kl" in metrics:
            metrics["und/rollout_actor_logp_gap"] = metrics["rollout_corr/kl"]
        if "actor/entropy" in metrics:
            metrics["und/entropy"] = metrics["actor/entropy"]
        if "gen/ratio_square_mean" in metrics and "gen/ratio_mean" in metrics:
            metrics["gen/ratio_std"] = math.sqrt(
                max(0.0, metrics["gen/ratio_square_mean"] - metrics["gen/ratio_mean"] ** 2)
            )
        self._divergence_tripwire(metrics)
        return updated

    #: Runaway-policy tripwire thresholds (see ``_divergence_tripwire``).
    #:
    #: Both sit an order of magnitude above anything a healthy run reaches and below the
    #: readings that made the ``_084724`` collapse unrecoverable. Measured across the
    #: 20261012 device-0-3 runs: every run that stopped before step 47 held
    #: ``rollout_corr/kl`` < 0.7 and ``actor/grad_norm`` < 2.3, while
    #: ``bagel_corl_rm1_0123_20261012_084724`` reached kl 9.3 / grad_norm 3.4 by step 51,
    #: 92 / 156 by step 90, and 175 / 310 by step 200 without anything in the logs saying
    #: so. ``rollout_corr/kl`` is the diagnostic the collapse shows in first.
    DIVERGENCE_KL_LIMIT = 20.0
    DIVERGENCE_GRAD_NORM_LIMIT = 100.0

    def _divergence_tripwire(self, metrics: dict) -> None:
        """Warn -- or abort -- the step the actor leaves its trust region.

        ``bagel_corl_rm1_0123_20261012_084724`` (devices 0-3) spent its last ~150 steps
        producing an unusable policy and said nothing: ``actor/grad_norm`` 0.32 -> 310 and
        ``rollout_corr/kl`` 0.03 -> 175 while ``actor/entropy`` fell to 0.02 and
        ``rollout_corr/training_ppl`` overflowed to ``inf`` (the training forward assigning
        ~``e^-65`` to tokens the rollout engine sampled at p~0.92). Neither number is
        trended anywhere, so the only way to notice was to read the run afterwards.

        This recipe has no KL anchor (``use_kl_loss=false``) and no entropy bonus
        (``entropy_coeff=0``), so a step that has already left the region cannot come back
        on its own -- the point of the tripwire is to make that visible on the step it
        happens rather than after the budget is spent. Default is a loud WARNING plus
        ``bagel_corl/divergence_warning`` in the metrics; set
        ``BAGEL_CORL_DIVERGENCE_ABORT=1`` to fail the run instead (a checkpoint this far out
        of the region is not a usable starting point for a restart).

        Called after ``super()._update_actor``, so both keys are already on ``metrics``:
        ``actor/grad_norm`` is the *pre-clip* norm the engine reports, and
        ``rollout_corr/kl`` is the rollout-vs-training log-prob gap computed once per step
        from the same batch. Both are per-step diagnostics, so the thresholds are exact
        rather than a heuristic on a trend.
        """
        kl = _metric_scalar(metrics, "rollout_corr/kl")
        grad_norm = _metric_scalar(metrics, "actor/grad_norm")

        def _runaway(value: float | None, limit: float) -> bool:
            if value is None:
                return False
            # ``nan`` comparisons are always False, but a non-finite grad norm or KL is the
            # loudest possible divergence signal -- do not let it slip through the ``>``.
            return math.isnan(value) or value > limit

        triggers = []
        if _runaway(kl, self.DIVERGENCE_KL_LIMIT):
            triggers.append(f"rollout_corr/kl={kl:.4g} > {self.DIVERGENCE_KL_LIMIT:g}")
        if _runaway(grad_norm, self.DIVERGENCE_GRAD_NORM_LIMIT):
            triggers.append(f"actor/grad_norm={grad_norm:.4g} > {self.DIVERGENCE_GRAD_NORM_LIMIT:g}")

        metrics["bagel_corl/divergence_warning"] = 1.0 if triggers else 0.0
        if not triggers:
            self._divergence_warned = False
            return
        if getattr(self, "_divergence_warned", False):
            # One warning per excursion: the metric above keeps the series complete, and a
            # collapsing run would otherwise repeat this on every remaining step.
            return
        self._divergence_warned = True
        step = getattr(self, "global_steps", None)
        entropy = _metric_scalar(metrics, "actor/entropy")
        message = (
            f"bagel_corl_sync step={step}: actor/rollout mismatch or gradient health threshold exceeded "
            f"({'; '.join(triggers)}"
            + (f"; actor/entropy={entropy:.4g}" if entropy is not None else "")
            + "). Check serving synchronization, probability conventions, and update scale; "
            "check NORM_ADV_BY_STD / loss_weight_* / optim.lr, and inspect "
            "checkpoint health before resuming."
        )
        if os.environ.get("BAGEL_CORL_DIVERGENCE_ABORT", "0") == "1":
            raise RuntimeError(message)
        logger.warning("%s Set BAGEL_CORL_DIVERGENCE_ABORT=1 to fail instead of warn.", message)

    def _und_deploy_config_path(self) -> str | None:
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        path = agent.get("und_deploy_config")
        if path:
            return str(path)
        ek = self.config.actor_rollout_ref.rollout.get("engine_kwargs") or {}
        vo = (ek.get("vllm_omni") or {}) if hasattr(ek, "get") else {}
        path = vo.get("und_deploy_config") if hasattr(vo, "get") else None
        return str(path) if path else None

    @staticmethod
    def _und_ar_stage_ids(deploy_config: str) -> tuple[int, ...]:
        """Stage ids declared by the UND AR deploy config (``bagel_think``: 0 + 1).

        The UND AR replica declares its width through
        ``tensor_model_parallel_size`` (see ``_build_und_ar_entrypoint_config``),
        and a plain top-level engine arg is copied into *every* stage by the
        deploy-config path (``build_stage_runtime_overrides`` applies non-
        orchestrator keys to each stage; for the DiT stage it is then folded into
        ``parallel_config``). Each stage therefore has to be pinned back to one
        rank, or a stage holding a single ``devices`` entry is asked for two.
        Read the ids from the deploy config instead of hard-coding them so a stage
        added or collapsed upstream cannot silently lose the pin.
        """
        if not os.path.isfile(deploy_config):
            raise ValueError(
                f"bagel_corl_sync cannot read agent.und_deploy_config={deploy_config!r} to derive the "
                "UND AR stage ids. Without them the per-stage tensor_parallel_size pin cannot be "
                "emitted, and every stage would inherit the replica-width value."
            )
        stages = OmegaConf.load(deploy_config).get("stages") or []
        stage_ids = tuple(int(stage["stage_id"]) for stage in stages if "stage_id" in stage)
        if not stage_ids:
            raise ValueError(f"bagel_corl_sync: {deploy_config!r} declares no stages to pin")
        return stage_ids

    @staticmethod
    def _und_stage_device_indices(deploy_config: str) -> dict[int, int]:
        """Highest logical device index each stage of the UND AR deploy config asks for.

        Stage ``devices`` are logical indices into the replica's own
        ``CUDA_VISIBLE_DEVICES`` (``run_bagel_und_ar_serve.sh`` relies on the same
        convention), so the pool width has to cover the widest one or the stage's
        engine core dies during init with an unrelated-looking error.
        """
        return {
            stage_id: max(indices)
            for stage_id, indices in OmniBagelCoRLTrainerSync._und_stage_devices(deploy_config).items()
        }

    @staticmethod
    def _und_stage_devices(deploy_config: str) -> dict[int, list[int]]:
        """Logical device indices each stage of the UND AR deploy config declares, in order."""
        stages = OmegaConf.load(deploy_config).get("stages") or []
        devices: dict[int, list[int]] = {}
        for stage in stages:
            if "stage_id" not in stage:
                continue
            runtime = stage.get("runtime") or {}
            raw = runtime.get("devices", stage.get("devices"))
            if raw is None:
                continue
            indices = [int(token) for token in str(raw).replace(",", " ").split() if token.strip()]
            if indices:
                devices[int(stage["stage_id"])] = indices
        return devices

    @staticmethod
    def _und_stage_max_model_len(deploy_config: str) -> int | None:
        """Widest ``max_model_len`` the UND AR deploy config declares across its stages.

        The entrypoint's ``max_model_len`` is a plain (non-orchestrator) engine arg, so
        the deploy-config path copies it into *every* stage and it therefore **wins**
        over each stage's own value. The builder is thus the single source of truth for
        the AR replica's context, and its value has to cover what ``_und_decode``
        actually feeds the engine: the decode prompt of UND turn ``j`` is
        ``prompt_ids + response_ids``, i.e. the *whole episode so far*.

        Sizing it as ``max_prompt + max_resp`` leaves no headroom for the turn being
        generated once the response budget is spent, which the AR strategy rejects with
        ``ValueError: Prompt length (...) meets or exceeds the model's maximum context
        length (...)``. Read the value back instead of hard-coding it, so the yaml's
        documented budget
        (16384: the ~8625-token-per-image MM-encoder floor, sized so
        ``max_num_batched_tokens == max_num_seqs * max_model_len``) is what the engine
        actually gets and the two cannot drift.
        """
        if not os.path.isfile(deploy_config):
            return None
        stages = OmegaConf.load(deploy_config).get("stages") or []
        declared = [int(stage["max_model_len"]) for stage in stages if stage.get("max_model_len") is not None]
        return max(declared) if declared else None

    # Bagel publishes weights only: its top-level ``config.json`` is
    # ``model_type: bagel`` with no ``auto_map`` and no modeling code, so
    # ``transformers.AutoConfig`` cannot resolve it. The MoT sub-configs ship
    # alongside in the snapshot; the UND AR (Thinker) replica is an LLM and needs
    # the language sub-config.
    UND_HF_SUBCONFIG = "llm_config.json"

    # ``load_format`` for the UND AR replica's own config, and it is **not** a style
    # choice -- ``dummy`` here means the Thinker decodes from randomly initialised
    # weights for the whole run.
    #
    # The AR replica is a *separate* Ray actor over a slice of the actor pool
    # (``init_colocated`` -> ``RolloutMode.COLOCATED``), so the inherited publish never
    # hands it weights: the V1 trainer forces ``checkpoint_engine.backend="naive"`` and
    # ``CheckpointEngineManager.update_weights`` early-returns for that backend after
    # publishing through the *actor* to its own in-process GEN server
    # (``verl/checkpoint_engine/base.py``). ``_publish_und_ar_weights`` pushes the policy
    # to the Thinker stage only; the DiT stage keeps the checkpoint it loads at launch.
    #
    # Which is why the value has to be pinned *here*. ``verl`` 0.9.0.dev (the sibling
    # checkout the 2026-10-13/14 runs resolved through ``PYTHONPATH``) promoted exactly
    # this case in ``vllm_async_server.__init__``:
    #
    #     if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
    #         logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
    #         self.config.load_format = "auto"
    #
    # The installed ``verl`` 0.10.0.dev -- what the launcher now falls back to, because the
    # sibling tree is too old for this tree -- dropped that promotion and passes
    # ``load_format`` straight through. An unset key therefore resolves to
    # ``RolloutConfig.load_format``'s default, ``"dummy"``, and the Thinker stage is
    # launched with ``--load_format dummy``.
    #
    # Measured on 2026-10-15 (``bagel_corl_prfix_t07_20261015_170336_a1``, devices 4-7):
    # ``actor/entropy`` 11.17 ~= ln(vocab) (a uniform distribution, i.e. no learned
    # preference at all), ``episode/K=0.0``, ``episode/K_zero=1.0``,
    # ``und/no_image_credit=1.0``, ``gen/skipped_no_groups=1.0``, ``has_complete_gen_groups=0.0``,
    # ``rollout_corr/rollout_ppl`` 46.5 against ``training_ppl`` 247736, and all
    # 36 val samples x 8 turns looping tokens like ``\tbuf_widget`` / ``💨third`` with no
    # ``<tool_call>`` anywhere -- so the GEN lane gets no group to score and the whole
    # Co-RL step optimises a zero advantage. A direct probe of the live replica
    # (``http://<host>:35045/v1/chat/completions``, greedy, "Say hello in one short
    # sentence.") answered " navigator prer navigator prer navigator prer ..."; the
    # Thinker stage's engine log carries no checkpoint-loading line at all, while every
    # engine that does load (the reward server, both DiT stages) logs its shard progress.
    #
    # ``auto`` is what the standalone AR gate (``run_bagel_und_ar_serve.sh`` -> ``vllm
    # serve``, whose own default is ``auto``) has always used, so this pin restores the
    # proven configuration. The GEN hybrid must keep ``dummy``: its weights are pushed by
    # the actor, and loading the checkpoint there would only waste a copy.
    UND_AR_LOAD_FORMAT = "auto"

    #: Stage of the ``bagel_think`` deploy that holds the policy LLM (the Thinker). Only this
    #: stage receives per-step weights; stage 1 is the DiT, which UND never trains.
    UND_AR_POLICY_STAGE_ID = 0

    #: Check both a merged LoRA target and the untied output head after publication.
    UND_AR_SYNC_CHECK_PARAM = "language_model.model.layers.0.self_attn.o_proj.weight"
    UND_AR_SYNC_CHECK_PARAMS = (UND_AR_SYNC_CHECK_PARAM, "language_model.lm_head.weight")

    def _resolve_und_hf_config_path(self, model, agent) -> str | None:
        """HF config path for the UND AR replica (RFC §4.7).

        ``OmniModelConfig.__post_init__`` calls ``AutoConfig.from_pretrained`` on
        ``hf_config_path``, falling back to ``path``. For a Bagel checkpoint that
        fallback raises ``ValueError: ... model type ``bagel`` but Transformers
        does not recognize this architecture``. Prefer the explicit
        ``agent.und_hf_config_path`` knob; otherwise derive the LLM sub-config
        from the snapshot so no recipe has to hand-set it. Non-Bagel checkpoints
        (e.g. Qwen3-Omni) resolve from ``path`` directly and get ``None``.
        """
        explicit = agent.get("und_hf_config_path")
        if explicit:
            return str(explicit)
        path = model.get("path")
        if not path:
            return None
        try:
            from verl_omni.utils.fs import resolve_model_local_dir

            local_dir = resolve_model_local_dir(str(path))
        except Exception as exc:  # noqa: BLE001 - optional derivation only
            logger.warning("bagel_corl_sync cannot locate model snapshot %s: %s", path, exc)
            return None
        candidate = os.path.join(local_dir, self.UND_HF_SUBCONFIG)
        return candidate if os.path.isfile(candidate) else None

    def _build_und_ar_entrypoint_config(self):
        """Clone entry config for a standalone UND AR ``LLMServerManager``.

        GEN keeps the rewritten diffusion hybrid stack. UND needs Omni model +
        ``output_mode=ar`` + bagel_think deploy — never ``bagel_single_stage``.
        """
        from omegaconf import OmegaConf, open_dict

        und_deploy = self._und_deploy_config_path()
        if not und_deploy:
            raise ValueError("bagel_corl_sync dual-role requires agent.und_deploy_config")

        model = self.config.actor_rollout_ref.model
        rollout = self.config.actor_rollout_ref.rollout
        agent = rollout.get("agent") or {}
        und_hf_config_path = self._resolve_und_hf_config_path(model, agent)
        und_n_gpus = int(agent.get("und_n_gpus") or 1)
        und_util = float(agent.get("und_gpu_memory_utilization") or 0.40)
        max_prompt = int(self.config.data.get("max_prompt_length") or rollout.get("prompt_length") or 1024)
        max_resp = int(self.config.data.get("max_response_length") or rollout.get("response_length") or max_prompt)
        # The AR replica's context has to serve any turn the UND loop is allowed to start.
        # ``run_serial_episode`` stops at ``max_prompt + max_resp`` (the episode's whole
        # context: ``_und_decode`` passes ``prompt_ids + response_ids`` as the decode prompt),
        # so an engine context below that cannot fit even one generated token and the AR
        # strategy refuses with ``ValueError: Prompt length (...) meets or exceeds the model's
        # maximum context length (...)``.
        # Prefer ``agent.und_max_model_len``, else the deploy config's own declared
        # budget (``bagel_corl_deploy_ar.yaml``: 16384), else the bare episode size.
        und_max_model_len = int(
            agent.get("und_max_model_len") or self._und_stage_max_model_len(und_deploy) or (max_prompt + max_resp)
        )
        if und_max_model_len < max_prompt + max_resp:
            raise ValueError(
                f"bagel_corl_sync: UND AR max_model_len={und_max_model_len} is smaller than the episode "
                f"context the loop has to serve (max_prompt_length={max_prompt} + "
                f"max_response_length={max_resp} = {max_prompt + max_resp}). The loop starts a turn while "
                "len(prompt_ids)+len(response_ids) is still below that budget, and the AR strategy rejects "
                "a prompt that leaves no room for generation ('Prompt length (...) meets or exceeds the "
                "model's maximum context length'). Lower data.max_prompt_length/data.max_response_length, "
                "or raise agent.und_max_model_len / the deploy config's max_model_len."
            )

        und_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        # ``actor_rollout_ref.model`` is the *actor's* config, and the recipe runs it with
        # ``lora.merge=True`` (mandatory on vllm-omni >= 0.24) -- i.e. the UND replica below
        # receives merged full weights and never an adapter. Its engine is launched by
        # ``verl/workers/rollout/vllm_rollout/vllm_async_server.py``, which enables LoRA
        # whenever the *effective* rank is > 0: it reads ``model.lora.rank``, falls back to
        # the top-level ``model.lora_rank`` ("FIXME: fallback to lora_rank for now") and only
        # zeroes that fallback when ``model.lora.merge`` is set. This builder copies
        # ``lora_rank``/``lora_alpha`` but not ``lora.merge``, so the AR engine would be
        # launched with ``enable_lora=True, max_loras=1`` with nothing to apply -- the Punica
        # kernels launch against an unpopulated ``lora_a_stacked`` (the actor syncs merged
        # weights, so no ``add_lora`` ever runs) and the first UND decode hangs forever.
        #
        # Pinning the AR replica's declared rank to 0 restores the intended "full weights, no
        # engine-side adapter" state -- the same state the GEN replica reaches through
        # ``lora_as_adapter=False`` in ``vllm_omni_async_server.py`` -- and keeps the AR engine
        # on the plain ``load_weights`` path that the merged sync uses.
        und_lora_rank = 0 if _actor_merges_lora(model) else int(model.get("lora_rank") or 0)
        with open_dict(und_cfg):
            und_cfg.actor_rollout_ref.model = OmegaConf.create(
                {
                    "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
                    "path": model.get("path"),
                    "tokenizer_path": model.get("tokenizer_path") or model.get("path"),
                    "model_type": "omni_model",
                    "architecture": model.get("architecture") or "OmniBagelForConditionalGeneration",
                    "trust_remote_code": bool(model.get("trust_remote_code", True)),
                    "composite_mode": "bagel_corl",
                    "lora_rank": und_lora_rank,
                    "lora_alpha": int(model.get("lora_alpha") or 0),
                    # OmniModelConfig.__post_init__ runs AutoConfig.from_pretrained on
                    # hf_config_path (falling back to ``path``). Bagel publishes weights
                    # only: its config.json is model_type "bagel" with no auto_map and no
                    # modeling code, so transformers cannot resolve it. Use the explicit
                    # agent.und_hf_config_path when set, else the snapshot's LLM sub-config.
                    "hf_config_path": und_hf_config_path,
                    # No OmniModelBase adapter is registered for
                    # ("OmniBagelForConditionalGeneration", "thinker") — the registry only
                    # knows Qwen3-Omni — and OmniModelConfig.__post_init__ only consults it
                    # when load_tokenizer=True. The UND AR replica is served over HTTP by
                    # vllm-omni (tokenisation happens server-side) and never goes through a
                    # training engine, so skip client-side tokenizer/processor loading.
                    "load_tokenizer": False,
                }
            )
            # ``to_container`` raises on ``None``, so the fallback has to be applied first --
            # otherwise a config without ``checkpoint_engine`` (the documented "naive"
            # default the ``or`` implies) dies here with "Input cfg is not an OmegaConf
            # config object (NoneType)".
            _ckpt = rollout.get("checkpoint_engine")
            ckpt_engine = OmegaConf.to_container(_ckpt, resolve=True) if _ckpt is not None else None
            ckpt_engine = ckpt_engine or {"backend": "naive"}
            und_cfg.actor_rollout_ref.rollout = OmegaConf.create(
                {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm_omni",
                    # The AR replica's engine is launched by this config and is never
                    # handed weights by the trainer, so it has to load the checkpoint
                    # itself -- leaving the key out resolves to ``RolloutConfig``'s
                    # ``"dummy"`` and silently costs the run its UND lane. See
                    # ``UND_AR_LOAD_FORMAT`` for the measurement and the ``verl`` history.
                    "load_format": self.UND_AR_LOAD_FORMAT,
                    # ``tensor_model_parallel_size`` here is the replica's *width
                    # declaration*, not a stage-level parallelism knob. It is what makes
                    # the colocated server see more than one card:
                    # ``RolloutReplica.__init__`` derives ``world_size = TP * DP * PP``,
                    # slices ``gpus_per_replica_node = min(n_gpus_per_node, world_size)``
                    # workers out of the pool ``_ensure_dual_role_rollout`` hands to
                    # ``init_colocated`` (one worker per ``und_n_gpus``-wide slice), and
                    # ``vLLMReplica.launch_servers`` builds the server's
                    # ``CUDA_VISIBLE_DEVICES`` from exactly ``gpus_per_replica_node``
                    # workers (``assert len(self.workers) == world_size``). A width of 1
                    # leaves the server ``CUDA_VISIBLE_DEVICES=0`` only, and stage devices
                    # are *logical* indices into that string, so both ``bagel_think``
                    # stages silently land on GPU 0 and the checkpoint copies pile onto one
                    # card; the first ``update_weights`` then dies in ``cumem create_and_map``
                    # when the GEN rank re-maps its level-1 sleep ("Wake-up failed on Rank 0").
                    #
                    # Declaring the width as TP (rather than data_parallel_size) keeps the
                    # launch free of extra engine args: the DP path makes
                    # ``launch_servers`` inject ``data_parallel_size_local`` alongside DP,
                    # and that key is *not* in vLLM-Omni's ``OrchestratorArgs`` (so it is
                    # not filtered as an orchestrator field) while DP *is* a pipeline-wide
                    # field that reaches every stage -- a stage pinned back to DP=1 would
                    # then abort with "data_parallel_size_local (2) must be <=
                    # data_parallel_size (1)". TP alone carries no such companion arg.
                    # ``bagel_corl_deploy_ar.yaml`` then spreads the stages over the two
                    # cards; ``stage_overrides`` below pins both stages back to TP=1 (a
                    # plain top-level engine arg reaches every stage through the deploy
                    # config path, and a TP=2 stage with one ``devices`` entry cannot
                    # start). Mirrors the Qwen3-Omni AR recipe, which also declares its
                    # replica width through ``tensor_model_parallel_size`` and leaves the
                    # per-stage engine args to the stage config.
                    "tensor_model_parallel_size": und_n_gpus,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n_gpus_per_node": und_n_gpus,
                    "nnodes": 1,
                    "prompt_length": max_prompt,
                    "response_length": max_resp,
                    # From the deploy config, never ``max_prompt + max_resp`` -- see
                    # ``_und_stage_max_model_len`` and the invariant check above.
                    "max_model_len": und_max_model_len,
                    # Bagel registers as encoder-decoder in vLLM-Omni, which disables
                    # chunked MM input and pins the MM encoder budget to
                    # max_num_batched_tokens. One image is ~8625 tokens, so the
                    # RolloutConfig default (8192) aborts AR engine init with
                    # "max_tokens_per_mm_item (8625) is larger than max_num_batched_tokens".
                    # Mirror bagel_corl_deploy_ar.yaml stage 0.
                    "max_num_batched_tokens": int(agent.get("und_max_num_batched_tokens") or 16384),
                    "gpu_memory_utilization": und_util,
                    "enforce_eager": True,
                    "free_cache_engine": True,
                    "enable_sleep_mode": True,
                    # ``bagel_corl_deploy_ar.yaml`` pins *both* stages to
                    # ``enable_prefix_caching: false``, but a plain top-level engine arg is
                    # copied into every stage by the deploy-config path and beats the
                    # per-stage value, so the upstream ``RolloutConfig`` default (True)
                    # reaches the DiT stage and vllm-omni's ``_DiffusionConfigProjection``
                    # rejects it:
                    #   Value error, enable_prefix_caching=True requires
                    #   diffusion_kv_mode='paged_scheduler'; set diffusion_kv_mode=
                    #   'paged_scheduler' or disable enable_prefix_caching
                    # Measured 2026-10-14 (bagel_corl_prfix_t07_20261014_061143_a1), after the
                    # GEN lane already carried the same pin from the recipe (a16e0b2).
                    "enable_prefix_caching": False,
                    "disable_log_stats": True,
                    "checkpoint_engine": ckpt_engine,
                    "disaggregation": {"enabled": False},
                    # Nested rollout sub-configs are only instantiated into their dataclass
                    # when the node carries ``_target_`` (``omega_conf_to_dataclass`` uses
                    # ``_convert_="partial"``). Without it the AR server dies on
                    # ``self.config.prometheus.enable`` in vllm_async_server.launch_server.
                    # Keep it disabled: the AR replica must not bind a second exporter.
                    "prometheus": {"_target_": "verl.workers.config.PrometheusConfig", "enable": False},
                    "engine_kwargs": {
                        "vllm_omni": {
                            "output_mode": "ar",
                            "deploy_config": und_deploy,
                            # ``tensor_model_parallel_size`` above only declares how wide
                            # the colocated replica is; ``--stage-overrides`` is the
                            # per-stage mechanism run_bagel_und_ar_serve.sh already uses
                            # for its two-card Thinker/DiT split. A plain top-level engine arg
                            # reaches every stage through the
                            # deploy-config path, so both ``bagel_think`` stages would
                            # otherwise be asked for two ranks while
                            # ``bagel_corl_deploy_ar.yaml`` gives each of them a single
                            # ``devices`` entry (the DiT stage even folds it into its
                            # diffusion ``parallel_config``). Stage-scoped keys are applied
                            # after the plain ones, so these win, and the deploy yaml's
                            # per-stage ``devices`` stay authoritative.
                            "stage_overrides": {
                                str(stage_id): {"tensor_parallel_size": 1}
                                for stage_id in self._und_ar_stage_ids(und_deploy)
                            },
                            # The policy push carries Thinker names only; the DiT stage
                            # would reject them.
                            "weight_sync_stage_ids": [self.UND_AR_POLICY_STAGE_ID],
                        }
                    },
                    "agent": OmegaConf.to_container(agent, resolve=True) or {},
                }
            )
        return und_cfg

    @staticmethod
    def _validate_und_ar_pool(
        und_rollout, und_n_gpus: int, deploy_config: str, actor_world_size: int | None = None
    ) -> None:
        """Fail loud when the AR replica's pool and its deploy config cannot agree.

        Three invariants, all otherwise surfacing as unrelated deep-stack failures:

        * ``world_size = TP * DP * PP`` (declared in ``_build_und_ar_entrypoint_config``)
          must equal the ``und_n_gpus``-wide actor-pool slice handed to ``init_colocated``,
          or ``vLLMReplica.launch_servers`` trips ``assert len(self.workers) == world_size``.
        * the pool must be at least as wide as the widest logical stage device. Stage
          ``devices`` are indices into the replica's own ``CUDA_VISIBLE_DEVICES``, so a
          1-wide pool cannot resolve ``devices: "1"`` (the layout the AR deploy yaml
          ships); a run configured that way died as
          ``StageEngineCoreProc_stage0_replica0 ... ValueError: No available memory for the
          cache blocks`` -> ``Orchestrator initialization failed: ... Failed core proc(s): {}``.
        * the pool must **divide** the actor pool, because that is how it is carved out:
          ``_ensure_dual_role_rollout`` takes the first slice with
          ``split_resource_pool(actor_pool, split_size=und_n_gpus)``, which cannot return a
          ragged split. On a 4-card pool the valid widths are therefore {1, 2, 4}; a width
          that does not divide the pool dies as
          ``AssertionError: split_size must be a divisor of world_size``.

        ``actor_world_size`` is the actor PG width when the caller can resolve it; ``None``
        (e.g. from a unit test that only exercises the config consistency) skips the
        divisibility check rather than guessing.
        """
        declared_world_size = (
            int(und_rollout.get("tensor_model_parallel_size") or 1)
            * int(und_rollout.get("data_parallel_size") or 1)
            * int(und_rollout.get("pipeline_model_parallel_size") or 1)
        )
        if declared_world_size != und_n_gpus:
            raise ValueError(
                f"bagel_corl_sync: the UND AR replica declares world_size={declared_world_size} "
                f"(TP*DP*PP) but agent.und_n_gpus={und_n_gpus} workers are handed to it. "
                "The replica's width declaration and the actor-pool slice must match."
            )
        if actor_world_size is not None and actor_world_size % und_n_gpus:
            raise ValueError(
                f"bagel_corl_sync: agent.und_n_gpus={und_n_gpus} does not divide the actor pool "
                f"(n_gpus_per_node={actor_world_size}). The UND AR slice is taken with "
                "split_resource_pool, which requires an exact split, so this would otherwise die "
                "as 'split_size must be a divisor of world_size'. Pick a divisor of the pool "
                "(on a 4-card pool: UND_N_GPUS in {1, 2, 4})."
            )
        stage_devices = OmniBagelCoRLTrainerSync._und_stage_device_indices(deploy_config)
        widest_stage_device = 1 + max(stage_devices.values(), default=0)
        if und_n_gpus < widest_stage_device:
            raise ValueError(
                f"bagel_corl_sync: agent.und_n_gpus={und_n_gpus} cannot cover the stage devices in "
                f"{deploy_config!r} (needs >= {widest_stage_device}). Stage devices are logical "
                "indices into the replica's CUDA_VISIBLE_DEVICES; either raise UND_N_GPUS or put "
                "every stage back on logical device 0 (1-GPU smoke test)."
            )

    def _und_actor_pool_size(self) -> int | None:
        """Width of the actor pool the UND AR replica is carved out of, or ``None``.

        Used only to feed ``_validate_und_ar_pool``'s divisibility check. Returns ``None``
        rather than raising when the pool cannot be resolved -- this is a pre-flight check,
        and ``_ensure_dual_role_rollout`` reports a missing pool itself.
        """
        try:
            actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
            actor_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        except Exception:  # noqa: BLE001 - a missing pool is reported by _ensure_dual_role_rollout
            return None
        # ``split_resource_pool`` reads exactly this attribute, so validating against it
        # checks the same quantity the assert that failed would have.
        world_size = getattr(actor_pool, "world_size", None)
        return int(world_size) if world_size else None

    def _ensure_dual_role_rollout(self) -> None:
        """Start UND AR colocated on the actor GPU pool; wrap GEN+UND behind one client.

        Standalone UND fails when ``trainer.n_gpus_per_node`` already claims every
        visible card (``Total available GPUs 0``). Reward/teacher use the same
        ``init_colocated`` pattern on the actor placement group.
        """
        from verl.single_controller.ray.base import split_resource_pool
        from verl.trainer.ppo.utils import Role
        from verl.utils.ray_utils import auto_await
        from verl.workers.rollout.llm_server import (
            DEFAULT_ROUTING_CACHE_SIZE,
            GlobalRequestLoadBalancer,
            LLMServerClient,
        )
        from verl.workers.rollout.replica import get_rollout_replica_class

        from verl_omni.workers.rollout.bagel_dual_role_llm_server import BagelDualRoleLLMServerClient

        if getattr(self, "_bagel_dual_role_ready", False):
            return
        if not bool((self.config.actor_rollout_ref.rollout.get("agent") or {}).get("und_ar_serving_ready")):
            raise RuntimeError(
                "bagel_corl_sync dual-role init requires agent.und_ar_serving_ready=True "
                "(prove spike_und_hermes.py first)"
            )

        und_cfg = self._build_und_ar_entrypoint_config()
        und_rollout = und_cfg.actor_rollout_ref.rollout
        und_model = und_cfg.actor_rollout_ref.model
        # Fail loud, not silently useless. This engine is the one replica nothing in the
        # training loop can feed weights to (naive backend publishes through the actor to
        # its own GEN server only), so a ``dummy`` load format here is not a default to
        # inherit but a run-long decode from random weights: every UND turn becomes a
        # token loop, ``episode/K`` stays 0, and the Co-RL step trains on a zero advantage.
        # Checked before the first side effect of the dual-role init (no replica is created,
        # no resource pool is split, no manager is touched), so a misconfigured launch dies
        # immediately instead of after the GEN+UND stack has been brought up. See
        # ``UND_AR_LOAD_FORMAT``.
        und_load_format = str(und_rollout.get("load_format") or "").strip().lower()
        if und_load_format in ("", "dummy"):
            raise RuntimeError(
                "bagel_corl_sync: the UND AR replica would be launched with "
                f"load_format={und_load_format or '<unset>'}, but it is a COLOCATED replica that "
                "the naive checkpoint backend never publishes to -- it would decode from random "
                "weights for the whole run. Pin actor_rollout_ref.rollout.load_format to "
                f"{self.UND_AR_LOAD_FORMAT!r} in _build_und_ar_entrypoint_config."
            )
        gen_manager = self.llm_server_manager
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        und_n_gpus = int(agent.get("und_n_gpus") or 1)
        if und_n_gpus < 1:
            raise ValueError(f"agent.und_n_gpus must be >= 1, got {und_n_gpus}")

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        # One UND AR replica: take the first ``und_n_gpus``-wide slice of the actor PG. Both
        # the declared width and the stage devices are validated against it first (see
        # ``_validate_und_ar_pool``), so a stale ``UND_N_GPUS`` reports itself here instead
        # of dying inside vLLM's engine core. ``und_deploy`` is non-``None`` because
        # ``_build_und_ar_entrypoint_config`` above already failed loud without it.
        und_deploy = self._und_deploy_config_path()
        self._validate_und_ar_pool(und_rollout, und_n_gpus, und_deploy, self._und_actor_pool_size())
        split_pools = split_resource_pool(actor_pool, split_size=und_n_gpus)
        if not split_pools:
            raise RuntimeError("bagel_corl_sync: actor resource pool is empty; cannot colocate UND AR")
        und_pool = split_pools[0]

        hybrid_n = len(gen_manager.rollout_replicas)
        replica_cls = get_rollout_replica_class(str(und_rollout.get("name") or "vllm_omni"))
        und_replica = replica_cls(
            replica_rank=hybrid_n,
            config=und_rollout,
            model_config=und_model,
            gpus_per_node=int(und_rollout.get("n_gpus_per_node") or und_n_gpus),
            name_suffix="bagel_und_ar",
        )
        logger.info(
            "bagel_corl_sync colocating UND AR on actor pool start_rank=%s und_n_gpus=%s und_deploy=%s",
            hybrid_n,
            und_n_gpus,
            und_deploy,
        )
        # ``vllm_async_server`` turns a nonzero ``model_config.lora_rank`` into
        # ``enable_lora=True`` for this engine. With the actor merging its LoRA the AR
        # replica must stay at 0 (see ``_build_und_ar_entrypoint_config``): a nonzero rank
        # here means the AR engine was launched with Punica LoRA layers and *no* adapter,
        # which hung the first decode of step 0 inside ``punica_gpu.add_lora_linear``.
        logger.info(
            "bagel_corl_sync UND AR engine lora_rank=%s actor_lora_merge=%s "
            "(0 => plain full-weight sync, no engine-side adapter)",
            int(und_model.get("lora_rank", 0) or 0),
            _actor_merges_lora(self.config.actor_rollout_ref.model),
        )

        @auto_await
        async def _init_und():
            await und_replica.init_colocated(und_pool)

        _init_und()

        und_replicas = [und_replica]
        self.und_rollout_replicas = und_replicas
        # Registration makes ``checkpoint_manager.sleep_replicas()`` sleep the UND AR replica
        # with the GEN hybrids. It does NOT publish weights to it: the V1 trainer forces
        # ``checkpoint_engine.backend="naive"`` (verl/trainer/ppo/v1/trainer_base.py), and
        # ``CheckpointEngineManager.update_weights`` early-returns for that backend without
        # touching ``self.replicas``. ``_publish_und_ar_weights`` does the UND publish.
        self.checkpoint_manager.add_replicas(und_replicas)
        logger.info(
            "bagel_corl_sync UND AR replica registered for sleep/wake "
            "replica=%s und_n_gpus=%s (weights published by _publish_und_ar_weights)",
            getattr(und_replica, "replica_rank", None),
            und_n_gpus,
        )

        # ``GlobalRequestLoadBalancer`` is decorated in one ``verl`` and plain in the other;
        # ``_as_ray_actor_class`` absorbs that difference (see its docstring). Getting this
        # wrong kills ``on_init_end`` with
        # "type object 'GlobalRequestLoadBalancer' has no attribute 'remote'".
        und_lb = _as_ray_actor_class(GlobalRequestLoadBalancer).remote(
            servers={und_replica.server_address: und_replica.server_handle},
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=bool(getattr(und_rollout, "full_determinism", False)),
        )
        und_client = LLMServerClient(config=und_cfg, load_balancer_handle=und_lb)
        self._bagel_dual_client = BagelDualRoleLLMServerClient(
            config=self.config,
            und_client=und_client,
            gen_client=gen_manager.get_client(),
        )
        self._bagel_dual_role_ready = True

    def get_llm_client(self):
        """Return the dual-role client (UND AR + GEN diffusion)."""
        self._ensure_dual_role_rollout()
        return self._bagel_dual_client

    def _gen_rollout_replicas(self) -> list:
        """The GEN (hybrid) replicas: everything registered except the UND AR pool.

        The parent registers the GEN replicas and we ``add_replicas`` the UND AR pool on top
        (RFC §4.11), so the set difference is exactly the pool the parent's own wake touches.
        Identity comparison, not equality: replica objects are not value types.
        """
        registered = getattr(getattr(self, "checkpoint_manager", None), "replicas", None) or []
        und_ids = {id(r) for r in (getattr(self, "und_rollout_replicas", None) or [])}
        return [r for r in registered if id(r) not in und_ids]

    def _wake_und_rollout_replicas(self) -> None:
        """Wake the colocated UND AR replica(s) after ``on_sample_end`` slept them.

        ``checkpoint_manager.sleep_replicas()`` sleeps *every* registered replica,
        and ``_ensure_dual_role_rollout`` registers the UND AR replica there. Nothing
        in the naive sync path wakes it again: ``CheckpointEngineManager.update_weights``
        short-circuits for ``backend='naive'`` without touching ``self.replicas``, and the
        actor-side ``EngineWorker.update_weights`` only resumes its *own* colocated server
        (``self.rollout``, the GEN engine). The AR engine therefore stays asleep with
        **both** tags still set and the next rollout's first UND decode dies in
        ``AsyncOmni.generate``:

            RuntimeError: Generation rejected: Engine is partially or fully asleep.
            Currently sleeping tags: ['weights', 'kv_cache'].

        Both tags are requested on purpose. ``vLLMOmniHttpServer.wake_up`` defaults to
        ``_get_wake_up_tags() == ["weights"]``, while ``AsyncOmni`` keeps its own
        ``_sleeping_tags`` and rejects generation while *any* tag is still sleeping;
        that is why the actor-side naive sync resumes weights and kv_cache in two
        separate calls. Asking for both in one wake clears the tag set outright, and is
        a no-op (``wake_up`` returns early on an already-warm engine) when the replica is
        awake, e.g. at ``on_init_end`` where ``init_colocated`` created it after
        ``_setup``'s sleep.
        """
        replicas = getattr(self, "und_rollout_replicas", None) or []
        self._wake_replicas(replicas, label="UND AR")

    def _wake_rollout_replicas(self) -> None:
        """Wake *both* colocated pools after a sleep, GEN included.

        ``checkpoint_manager.sleep_replicas()`` sleeps every registered replica, so the GEN
        pool is asleep too. The parent's naive publish only resumes the actor's *own*
        colocated GEN server, and that call passes no tags -- ``vLLMOmniHttpServer.wake_up``
        then falls back to ``_get_wake_up_tags() == ["weights"]``, while ``AsyncOmni`` keeps
        rejecting generation until *every* tag is cleared. The GEN pool therefore never has
        its ``kv_cache`` tag cleared by anything, so its rollout dies mid-run with
        ``RuntimeError: Generation rejected: Engine is partially or fully asleep.`` followed by
        ``StageDiffusionProc executor reported permanent failure``. This is the same gap
        ``_wake_und_rollout_replicas`` closes for the AR pool, one pool over.
        """
        und = getattr(self, "und_rollout_replicas", None) or []
        gen = self._gen_rollout_replicas()
        # Per-step wake topology. The omni server exposes no "which tags are still sleeping"
        # query, so the trail comes from the trainer side: if a rollout later dies with
        # ``Engine is partially or fully asleep``, this line shows the pools and the tags each
        # was asked for. Emitted before any early return so a run with no UND AR replica still
        # reports the GEN side.
        logger.info(
            "bagel_corl_sync step=%s wake topology: und_ar=%d gen=%d (both woken explicitly "
            "with ['weights', 'kv_cache']; the parent publish alone requests no tags and so "
            "resolves to ['weights'] through _get_wake_up_tags)",
            getattr(self, "global_steps", None),
            len(und),
            len(gen),
        )
        self._wake_replicas(und, label="UND AR")
        self._wake_replicas(gen, label="GEN")

    @staticmethod
    def _wake_replicas(replicas: list, *, label: str) -> None:
        """Wake every server of ``replicas`` with both tags, one at a time, then report the count.

        Both tags are requested on purpose (see ``_wake_und_rollout_replicas``); this is a
        no-op for an already-warm engine because ``wake_up`` returns early in that case.

        **Sequential, not gathered.** The wake is a ``cuMemCreate`` re-map -- it needs a large
        *physically contiguous* region (see ``verl_omni.utils.rollout_wake``) -- so gathering
        every replica at once makes the pools contend for it and the wake dies with
        ``CUDA Error: out of memory at .../cumem_allocator.cpp``. Awaiting each server in turn
        removes the self-contention for free; the servers are independent and the wake is
        idempotent, so nothing depends on the concurrency.
        """
        if not replicas:
            return

        from verl.utils.ray_utils import auto_await

        @auto_await
        async def _wake_all() -> None:
            for replica in replicas:
                for server in replica.servers:
                    await server.wake_up.remote(tags=["weights", "kv_cache"])

        _wake_all()
        logger.info(
            "bagel_corl_sync woke %s %s replica(s) (weights+kv_cache, sequentially) after sleep",
            len(replicas),
            label,
        )

    def _publish_und_ar_weights(self) -> None:
        """Push the merged actor policy into the UND AR Thinker and verify it landed.

        The inherited naive publish only reaches the actor's own GEN servers, so without this
        the Thinker would decode from the launch checkpoint for the whole run while the UND
        lane trains on its samples.

        Must run while the GEN replicas are asleep. Each UND server is woken with both tags
        first: vLLM only offloads on ``sleep`` from a fully awake engine, so a weights-only
        wake would leave the Thinker resident through the GEN publish. After the push, two
        Thinker parameters are fingerprinted on the engine and compared with what the actor
        sent, and the server goes back to sleep; ``_wake_rollout_replicas`` restores it
        (level-1 sleep keeps the pushed weights in host memory).
        """
        import time

        import ray

        from verl_omni.workers.rollout.vllm_rollout.weight_fingerprint import fingerprints_match

        replicas = getattr(self, "und_rollout_replicas", None) or []
        if not replicas:
            return
        receiver_ranks = self._und_stage_devices(self._und_deploy_config_path()).get(self.UND_AR_POLICY_STAGE_ID)
        if not receiver_ranks:
            raise RuntimeError(
                f"bagel_corl_sync: UND AR deploy config declares no devices for stage "
                f"{self.UND_AR_POLICY_STAGE_ID}; cannot route the policy push"
            )
        check_names = self.UND_AR_SYNC_CHECK_PARAMS
        check_name = check_names[0]
        started = time.perf_counter()
        num_tensors = 0
        fingerprint = None
        for replica in replicas:
            if len(replica.servers) != 1:
                raise RuntimeError(
                    f"bagel_corl_sync: UND AR replica {replica.replica_rank} has {len(replica.servers)} "
                    "servers; the policy push assumes a single-node replica"
                )
            server = replica.servers[0]
            ray.get(server.wake_up.remote(tags=["weights", "kv_cache"]))
            try:
                results = ray.get(
                    self.actor_rollout_wg.update_bagel_und_ar_weights(
                        server_handle=server,
                        und_replica_rank=replica.replica_rank,
                        receiver_ranks=receiver_ranks,
                        checksum_name=check_name,
                        checksum_names=list(check_names),
                        global_steps=self.global_steps,
                    )
                )
                stats = next((r for r in _flatten_rpc_results(results) if r is not None), None)
                checksums = {} if stats is None else stats.get("checksums", {})
                for name in check_names:
                    if checksums.get(name) is None:
                        raise RuntimeError(
                            f"bagel_corl_sync: actor export carried no {name!r}; UND AR name routing "
                            "no longer matches the Thinker"
                        )
                    engine_values = [
                        v
                        for v in _flatten_rpc_results(
                            ray.get(server.collective_rpc.remote("policy_param_checksum", kwargs={"name": name}))
                        )
                        if v is not None
                    ]
                    if not engine_values or not all(fingerprints_match(checksums[name], v) for v in engine_values):
                        raise RuntimeError(
                            f"bagel_corl_sync step={self.global_steps}: UND AR replica {replica.replica_rank} "
                            f"did not take the pushed weights; {name} fingerprint sent={checksums[name]!r} "
                            f"engine={engine_values!r}"
                        )
            finally:
                ray.get(server.sleep.remote())
            num_tensors += int(stats["num_tensors"])
            fingerprint = stats["checksum"]

        previous = getattr(self, "_und_ar_last_fingerprint", None)
        self._und_ar_last_fingerprint = fingerprint
        sync_metrics = {
            "bagel_corl/und_sync/num_tensors": float(num_tensors),
            "bagel_corl/und_sync/seconds": time.perf_counter() - started,
            "bagel_corl/und_sync/fingerprint_delta": 0.0 if previous is None else abs(fingerprint - previous),
        }
        pending = getattr(self, "_pending_sync_metrics", None) or {}
        pending.update(sync_metrics)
        self._pending_sync_metrics = pending
        logger.info(
            "bagel_corl_sync step=%s published policy to %d UND AR replica(s): %d tensors, %.1fs, "
            "%s fingerprint verified (delta since last publish %s)",
            self.global_steps,
            len(replicas),
            num_tensors,
            sync_metrics["bagel_corl/und_sync/seconds"],
            check_name,
            sync_metrics["bagel_corl/und_sync/fingerprint_delta"],
        )

    def _bagel_rm_enabled(self) -> bool:
        reward_cfg = getattr(self.config, "reward", None)
        if reward_cfg is None:
            return False
        try:
            return bool(reward_cfg.get("reward_model", {}).get("enable", False))
        except (AttributeError, TypeError):
            return False

    def _ensure_reward_loop_manager(self):
        """Create the colocated ``OmniRewardLoopManager`` (RFC §4.2).

        Its worker handles serve both consumers: the inherited episode-reward
        ``_compute_score`` and the mid-loop GEN scoring adapter
        (``bagel_corl_rm``). Creation mirrors the diffusion V1 trainer.
        """
        if getattr(self, "reward_loop_manager", None) is not None:
            return self.reward_loop_manager
        from verl.trainer.ppo.utils import Role

        from verl_omni.reward_loop import OmniRewardLoopManager

        resource_pool = None
        if getattr(self, "use_rm", False) and getattr(self, "resource_pool_manager", None) is not None:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        self.reward_loop_manager = OmniRewardLoopManager(config=self.config, rm_resource_pool=resource_pool)
        return self.reward_loop_manager

    def get_reward_handles(self):
        """Handles the pinned ``TaskRunnerV1`` forwards into the agent-loop manager.

        The manager passes them to every ``BagelCorlAgentLoopWorkerTQ``, which
        binds the GEN-side handle for in-loop GEN scoring.

        Normalised to ``None`` when there are none. ``RewardLoopManager.
        reward_loop_worker_handles`` returns ``self.reward_loop_workers`` -- an *empty
        list*, not ``None`` -- whenever ``reward_model.enable=False``, which the Co-RL
        recipe sets alongside ``reward.num_workers=0`` (the in-loop Bagel reward is the
        only scorer). The agent loop then tests ``is not None`` and calls
        ``random.choice(...)`` on it, so any episode that finishes with
        ``reward_score is None`` kills the worker with

            IndexError: Cannot choose from an empty sequence

        which masks the real failure. Returning ``None`` makes the
        agent loop skip that branch and lets the genuine error surface.
        """
        manager = getattr(self, "reward_loop_manager", None)
        if manager is not None:
            workers = getattr(manager, "reward_loop_workers", None)
            if workers:
                return workers
        getter = getattr(super(), "get_reward_handles", None)
        handles = getter() if callable(getter) else None
        return list(handles) if handles else None

    # Row-scoped dataset fields ``VisualRewardManager`` dispatches on: ``data_source`` picks
    # the scorer (``unicot_reflection`` / ``unicot_breakdown``), ``reward_model`` carries its
    # ``ground_truth``.
    REWARD_ROW_FIELDS = ("data_source", "reward_model", "extra_info")

    def _compute_reward_colocate(self, batch, metrics: dict | None = None):
        """Score a batch with the colocated RM, handing it the row fields the scorer needs.

        The installed ``verl`` (0.10.0.dev) reads only ``prompts``/``responses``/``raw_prompt``
        here, so every episode reached ``VisualRewardManager`` with ``data_source=''`` and no
        ``ground_truth`` and scored a structural zero: ``critic/rewards/mean``,
        ``critic/advantages/*`` and ``actor/pg_loss`` were exactly 0 for all 85 steps of
        ``bagel_corl_prfix_t07_20261015_190942_a1`` (devices 4-7), with the RM pool warning
        "episode row carries no reward_model.ground_truth (data_source='')". The older sibling
        ``verl`` carried this selection locally, which is why the 2026-10-13 runs learned; the
        recipe now rejects that checkout as too old, so the rule lives here instead.

        With ``reward.reward_model.enable=1`` and no RM resource pool this is the only scorer
        (``reward_loop_worker_handles`` is None), so the agent loop cannot cover for it.
        """
        import numpy as np
        import torch
        import transfer_queue as tq
        from tensordict import TensorDict
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu

        assert self.reward_loop_manager is not None, "RewardLoopManager is None"

        fields = ["prompts", "responses", "raw_prompt", *self.REWARD_ROW_FIELDS]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        prompts = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        prompt_mask = self._lengths_to_mask(prompt_lengths, prompts.size(1))
        response_mask = self._lengths_to_mask(response_lengths, responses.size(1))
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        raw_prompts = list(data["raw_prompt"])
        raw_prompt_arr = np.empty(len(raw_prompts), dtype=object)
        raw_prompt_arr[:] = raw_prompts
        non_tensor_batch: dict[str, Any] = {"raw_prompt": raw_prompt_arr}
        for key in self.REWARD_ROW_FIELDS:
            values = data.pop(key, None)
            if values is None:
                continue
            # TQ returns ``NonTensorData`` wrappers here; a wrapped ``reward_model`` has no
            # ``ground_truth`` key and would score the same silent zero.
            items = [tu.unwrap_non_tensor_data(value) for value in list(values)]
            arr = np.empty(len(items), dtype=object)
            arr[:] = items
            non_tensor_batch[key] = arr

        rm_input = DataProto(
            batch=TensorDict(
                {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
                batch_size=len(batch),
            ),
            non_tensor_batch=non_tensor_batch,
        )
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=tu.get_tensordict(write_back))
        return batch

    def on_init_end(self):
        # Build UND AR before the first weight publish so both pools see step-0 weights.
        self._ensure_dual_role_rollout()
        if self._bagel_rm_enabled():
            self._ensure_reward_loop_manager()
        # A resumed checkpoint is already loaded into the actor (``_setup``), so this is the
        # resumed policy, not the launch checkpoint the Thinker loaded.
        self._publish_und_ar_weights()
        super().on_init_end()
        # ``_setup`` slept the GEN replica(s) to load the checkpoint and the parent hook
        # woke them through the actor; the AR replica is created after that sleep, so this
        # is a no-op today and a guard if the creation order ever moves. Both pools are
        # woken, since the parent's tagless wake leaves each one's ``kv_cache`` set.
        self._wake_rollout_replicas()

    def on_step_end(self):
        # ``on_sample_end`` slept every replica, so the UND push runs first, before the
        # parent's naive publish wakes the GEN servers onto the same GPUs.
        with marked_timer("update_und_weights", self.timing_raw, color="red"):
            self._publish_und_ar_weights()
        super().on_step_end()
        # The parent hook's naive publish only resumed the actor's own GEN server, and it
        # requests no tags, so neither pool is fully awake here: the AR engine is still
        # asleep and every rollout since would fail on its first UND decode, while the GEN
        # engine is awake on weights but still holds ``kv_cache``. See
        # ``_wake_und_rollout_replicas`` / ``_wake_rollout_replicas``.
        self._wake_rollout_replicas()

    def on_validate_end(self):
        """Re-wake the AR replica after ``_validate``'s colocated-reward sleep.

        ``_validate`` sleeps **all** replicas when ``reward_loop_manager.
        reward_loop_worker_handles is None`` (a colocated RM). Today's recipe runs with
        ``reward.reward_model.enable=False`` + ``num_workers=0``, whose handle list is empty
        -- not ``None`` -- so that branch is skipped; ``ENABLE_RM=1`` flips it on. Its
        follow-up ``update_weights()`` is the naive short-circuit that only wakes the actor's
        GEN server, so without this the AR engine would be asleep for the *next* step's
        rollout: the same failure ``_wake_und_rollout_replicas`` fixes, one step later. Also
        covers ``val_before_train=True``, where ``on_step_end`` has not run yet.

        Harmless when nothing slept: ``AsyncOmni.wake_up`` logs "already warm" and returns
        without touching the engine.
        """
        super().on_validate_end()
        self._wake_rollout_replicas()

    def _validate(self):
        """V1 validation, plus the RFC §8.2 evidence record.

        Was NON-CONFORMANT: ``test_freq`` unset meant the required validation
        evidence had no execution path. The cadence is restored by the recipe
        (``trainer.test_freq``); this hook records what the validated episodes
        actually exercise — in-episode J/K, pattern coverage, and whether UND got
        image credit — because §8.2 gates PR1 on seeing pattern-1/2/3 episodes.

        Wraps the parent's per-batch publish so both rollout pools are fully awake again
        before the *next* validation batch rolls out. The parent's loop
        (``verl/trainer/ppo/v1/trainer_base.py:1034-1037``) does, once per validation batch::

            self.agent_loop_manager.generate_sequences(batch)   # needs the engines awake
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()        # hand the GPU to the colocated RM
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()        # naive: ['weights'], GEN server only

        That branch is entered only when the RM is *colocated*, i.e. only with
        ``ENABLE_RM=1``. Its publish resumes just the actor's own GEN server and passes no
        tags, so every batch after the first starts its rollout against a sleeping engine, while
        ``on_validate_end`` only wakes once the whole loop is over -- too late for the batches in
        between. A later batch's rollout dies with ``RuntimeError: Generation rejected: Engine is
        partially or fully asleep.`` and the diffusion stage reports permanent failure. Nothing
        about the step count is special: it is the first validation that spans more than one batch.

        The wrap is scoped to this call rather than installed permanently, so the training path
        keeps using its single wake in ``on_step_end``.
        """
        manager = getattr(self, "checkpoint_manager", None)
        publish = getattr(manager, "update_weights", None)
        if publish is None:
            val_metrics = super()._validate()
        else:

            def _publish_then_wake(*args, **kwargs):
                result = publish(*args, **kwargs)
                self._wake_rollout_replicas()
                return result

            try:
                manager.update_weights = _publish_then_wake
                val_metrics = super()._validate()
            finally:
                manager.update_weights = publish
        evidence = {
            key: float(value)
            for key, value in (val_metrics or {}).items()
            if isinstance(value, int | float)
            and (
                "episode/J" in str(key)
                or "episode/K" in str(key)
                or "pattern" in str(key).lower()
                or "no_image_credit" in str(key)
                or "skipped_no_groups" in str(key)
            )
        }
        val_metrics = dict(val_metrics or {})
        val_metrics["val/rfc82_evidence_keys"] = float(len(evidence))
        if "episode/K" in evidence and evidence.get("episode/J") == 0 and evidence.get("episode/K") == 0:
            logger.warning(
                "bagel_corl_sync §8.2: validation produced no dual-lane episode evidence "
                "(J=0, K=0); the validation batch did not run the bagel_multiturn_agent loop."
            )
        logger.info("bagel_corl_sync §8.2 validation evidence: %s", evidence or "<none>")
        return val_metrics
