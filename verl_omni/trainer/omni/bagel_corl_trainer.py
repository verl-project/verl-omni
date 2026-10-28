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

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.advantages import (
    apply_gen_flowgrpo_advantage,
    build_gen_flowgrpo_proto,
    fold_gen_prompt_token_ids,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.diffusion import DiffusionV1GenLane, gen_norm_adv_by_std
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode import (
    aggregate_episode_metrics,
    aggregate_reward_breakdown_metrics,
    und_record_field_keys,
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
    """Index a stacked TQ column (Tensor / NonTensorStack / list) at position ``i``."""
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
    """Read ``metrics[key]`` as a float, or ``None`` when it is absent/not scalar."""
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
    """True when the recipe asks the *actor* to merge LoRA into the base weights."""
    lora_cfg = model.get("lora") if hasattr(model, "get") else getattr(model, "lora", None)
    if lora_cfg is None:
        return False
    if hasattr(lora_cfg, "get"):
        return bool(lora_cfg.get("merge", False))
    return bool(getattr(lora_cfg, "merge", False))


def _as_ray_actor_class(cls: Any) -> Any:
    """Return ``cls`` as a Ray actor class, tolerating both ``verl`` shapes."""
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
#: spelled as a literal tuple (``agentic_und_gen.episode.aggregate_reward_breakdown_metrics``).
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
    """Synchronous Bagel Co-RL (Joint-Training): serial J-episode gather, then one UND+GEN optimizer step."""

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
        agent["_target_"] = "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.config.BagelCorlAgentLoopConfig"
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
            "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.transfer_queue.BagelCorlAgentLoopManagerTQ",
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
        # UniGRPO (arXiv:2603.23500): CFG doubles per-step evaluations and branches the rollout graph — prohibitive
        # for multi-turn episodes.
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

        # ``val_kwargs`` needs the same retarget for a different reason: unlike ``pipeline``, the omni schema *does*
        # declare this node -- but with the AR ``_target_`` ``verl.workers.config.SamplingConfig``, while
        # ``DiffusionRolloutConfig`` types the field as ``DiffusionSamplingConfig``.
        val_allowed = {f.name for f in fields(DiffusionSamplingConfig)}
        val_kwargs = {
            k: v for k, v in (rollout_filtered.get("val_kwargs") or {}).items() if k in val_allowed and k != "_target_"
        }
        val_kwargs["_target_"] = "verl_omni.workers.config.diffusion.DiffusionSamplingConfig"
        rollout_filtered["val_kwargs"] = val_kwargs

        # ``cfg_text_scale`` is a transition-kernel knob, not a style knob: the GEN importance ratio is only unbiased
        # when training and rollout use the SAME value (RFC §4.8.5 / §5).
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

        # RFC §5: ``cfg_text_scale`` is a transition-kernel knob whose single SoT is ``model.pipeline`` AND
        # ``rollout.pipeline``.
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
        # ``cfg_img_scale`` is a transition-kernel knob too, and it is NOT self-correcting: the rollout request falls
        # back to 1.5 when the key is absent (``pipeline_bagel.py``) while training reads ``model.pipeline``, whose
        # typed default is 1.0.
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
        # The dual-lane writer tags every GEN seed row ``is_auxiliary`` so it stays out of a sampled batch: the GEN
        # lane reaches the trainer through the UND row's ``child_gen_keys`` column, and the UND pass cannot score a
        # seed row (no ``prompts``/``responses``).
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

        def compute_und():
            return super(OmniBagelCoRLTrainerSync, self)._compute_old_log_prob(batch, metrics)

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
        # The GEN proto's row count is data-dependent: one row per *trainable* GEN call, and K varies per episode, so
        # ``len(proto)`` is not necessarily a multiple of the actor DP size.
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
        old = self._diffusion_v1_gen_lane()._compute_old_log_prob(proto)
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

        from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.transfer_queue import split_und_gen_metas

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
        """Lookup GEN TQ rows by ``child_gen_keys``. Returns key → {fields: ...}."""
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
                    norm_adv_by_std_in_grpo=(
                        gen_norm
                        if (gen_norm := gen_norm_adv_by_std(algo)) is not None
                        else bool(algo.get("norm_adv_by_std_in_grpo", True))
                    ),
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

        from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.batch import validate_episode_ownership

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
        # The flags have to ride on whatever carrier this trainer's bus uses, and ``_extra_info`` above already put
        # them where the actor will look: the v1 bus hands us a ``KVBatchMeta``, whose ``extra_info`` dict the TQ
        # dispatch layer copies onto the actor's TensorDict as non-tensor data
        # (``verl/utils/transferqueue_utils.py:160-177``) -- which is exactly where the composite reads them back
        # (``bagel_corl_composite.py:129,131`` and ``diffusers_impl.py:954,956``).
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

    #: Trend view of the per-step scalars. ``critic/score/mean`` averages 16-32 episodes and
    #: is white noise step to step (lag-1 autocorrelation 0.06, std 0.067 on ``v0s9x7t2``)
    #: against a trend of a few hundredths per 100 steps. A bias-corrected EMA with ~20-step
    #: memory is the curve to read for "is training reward rising".
    METRIC_EMA_BETA = 0.95
    METRIC_EMA_KEYS = (
        "critic/score/mean",
        "bagel_corl/reward_dim/image/mean",
        "episode/J",
        "episode/K_zero",
    )

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
        state = self.__dict__.setdefault("_metric_ema", {})
        for key in self.METRIC_EMA_KEYS:
            value = _metric_scalar(metrics, key)
            if value is None or not math.isfinite(value):
                continue
            ema, n = state.get(key, (0.0, 0))
            ema = self.METRIC_EMA_BETA * ema + (1.0 - self.METRIC_EMA_BETA) * value
            n += 1
            state[key] = (ema, n)
            metrics[f"bagel_corl/ema/{key}"] = ema / (1.0 - self.METRIC_EMA_BETA**n)

    # : Runaway-policy tripwire thresholds (see ``_divergence_tripwire``). : : Both sit an order of magnitude above
    # anything a healthy run reaches and below the : readings that made the ``_084724`` collapse unrecoverable.
    DIVERGENCE_KL_LIMIT = 20.0
    DIVERGENCE_GRAD_NORM_LIMIT = 100.0

    def _divergence_tripwire(self, metrics: dict) -> None:
        """Warn or optionally abort on provisional rollout/actor mismatch diagnostics."""
        kl = _metric_scalar(metrics, "rollout_corr/kl")
        grad_norm = _metric_scalar(metrics, "actor/grad_norm")

        def _runaway(value: float | None, limit: float) -> bool:
            if value is None:
                return False
            # ``nan`` comparisons are always False, but a non-finite grad norm or KL is the
            # loudest possible divergence signal -- do not let it slip through the ``>``.
            return not math.isfinite(value) or value > limit

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
        """Stage ids declared by the UND AR deploy config (``bagel_think``: 0 + 1)."""
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
        """Highest logical device index each stage of the UND AR deploy config asks for."""
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
        """Widest ``max_model_len`` the UND AR deploy config declares across its stages."""
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

    # ``load_format`` for the UND AR replica's own config, and it is **not** a style choice -- ``dummy`` here means
    # the Thinker decodes from randomly initialised weights for the whole run.
    UND_AR_LOAD_FORMAT = "auto"

    #: Stage of the ``bagel_think`` deploy that holds the policy LLM (the Thinker). Only this
    #: stage receives per-step weights; stage 1 is the DiT, which UND never trains.
    UND_AR_POLICY_STAGE_ID = 0

    #: Check both a merged LoRA target and the untied output head after publication.
    UND_AR_SYNC_CHECK_PARAM = "language_model.model.layers.0.self_attn.o_proj.weight"
    UND_AR_SYNC_CHECK_PARAMS = (UND_AR_SYNC_CHECK_PARAM, "language_model.lm_head.weight")

    def _resolve_und_hf_config_path(self, model, agent) -> str | None:
        """HF config path for the UND AR replica (RFC §4.7)."""
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
        # ``actor_rollout_ref.model`` is the *actor's* config, and the recipe runs it with ``lora.merge=True``
        # (mandatory on vllm-omni >= 0.24) -- i.e. the UND replica below receives merged full weights and never an
        # adapter.
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
                    "logprobs_mode": rollout.get("logprobs_mode", "processed_logprobs"),
                    # ``tensor_model_parallel_size`` here is the replica's *width declaration*, not a stage-level
                    # parallelism knob.
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
                            # ``tensor_model_parallel_size`` above only declares how wide the colocated replica is;
                            # ``--stage-overrides`` is the per-stage mechanism run_bagel_und_ar_serve.sh already uses
                            # for its two-card Thinker/DiT split.
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
        """Fail loud when the AR replica's pool and its deploy config cannot agree."""
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
                "(verify Hermes generate_image calls on the UND AR replica first)"
            )

        und_cfg = self._build_und_ar_entrypoint_config()
        und_rollout = und_cfg.actor_rollout_ref.rollout
        und_model = und_cfg.actor_rollout_ref.model
        # Fail loud, not silently useless.
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
        """Wake the colocated UND AR replica(s) after ``on_sample_end`` slept them."""
        replicas = getattr(self, "und_rollout_replicas", None) or []
        self._wake_replicas(replicas, label="UND AR")

    def _wake_rollout_replicas(self) -> None:
        """Wake *both* colocated pools after a sleep, GEN included."""
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
        """Wake every server of ``replicas`` with both tags, one at a time, then report the count."""
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
        """Push the merged actor policy into the UND AR Thinker and verify it landed."""
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
        (``agentic_und_gen.reward``). Creation mirrors the diffusion V1 trainer.
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
        """Handles the pinned ``TaskRunnerV1`` forwards into the agent-loop manager."""
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
        """Score a batch with the colocated RM, handing it the row fields the scorer needs."""
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
        # A partial dual-lane publication cannot attest a loaded policy version.
        self._bagel_corl_published_weight_version = None
        self._publish_und_ar_weights()
        super().on_init_end()
        # ``_setup`` slept the GEN replica(s) to load the checkpoint and the parent hook
        # woke them through the actor; the AR replica is created after that sleep, so this
        # is a no-op today and a guard if the creation order ever moves. Both pools are
        # woken, since the parent's tagless wake leaves each one's ``kv_cache`` set.
        self._wake_rollout_replicas()
        self._bagel_corl_published_weight_version = self.global_steps

    def on_step_end(self):
        # ``on_sample_end`` slept every replica, so the UND push runs first, before the
        # parent's naive publish wakes the GEN servers onto the same GPUs.
        # A partial dual-lane publication cannot attest a loaded policy version.
        self._bagel_corl_published_weight_version = None
        with marked_timer("update_und_weights", self.timing_raw, color="red"):
            self._publish_und_ar_weights()
        super().on_step_end()
        # The parent hook's naive publish only resumed the actor's own GEN server, and it
        # requests no tags, so neither pool is fully awake here: the AR engine is still
        # asleep and every rollout since would fail on its first UND decode, while the GEN
        # engine is awake on weights but still holds ``kv_cache``. See
        # ``_wake_und_rollout_replicas`` / ``_wake_rollout_replicas``.
        self._wake_rollout_replicas()
        self._bagel_corl_published_weight_version = self.global_steps

    def on_validate_end(self):
        """Re-wake the AR replica after ``_validate``'s colocated-reward sleep."""
        super().on_validate_end()
        self._wake_rollout_replicas()

    def _validate(self):
        """V1 validation, plus the RFC §8.2 evidence record."""
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
