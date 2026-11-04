# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail-fast validation shared by VeRL-Omni trainer entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf


def _select(config: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from an OmegaConf node or a plain config object.

    OmegaConf (hydra) nodes go through ``OmegaConf.select`` so struct flags and
    interpolations behave exactly as in the composed trainer config;
    instantiated dataclasses fall back to attribute access. A ``None`` result
    collapses to ``default`` either way.
    """
    if OmegaConf.is_config(config):
        value = OmegaConf.select(config, path, default=default)
        return default if value is None else value
    value = config
    for part in path.split("."):
        if value is None:
            return default
        if hasattr(value, "get"):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
    return default if value is None else value


def _validate_delta_sharded(config: Any) -> None:
    """Fail-fast gates for the ``omni_delta_sharded`` checkpoint engine backend (RFC #38).

    The delta path syncs full weights to standalone rollout replicas and diffs each
    rank's shard against the last export, so it is only defined for full-weight
    separate_async runs; anything else raises here instead of degrading silently.
    """
    backend = _select(config, "actor_rollout_ref.rollout.checkpoint_engine.backend")
    if backend == "delta_sharded":
        raise ValueError(
            "checkpoint_engine.backend='delta_sharded' is verl's sglang-only delta backend; verl's "
            "CheckpointEngineWorker rejects it for the vllm_omni rollout. Use backend='omni_delta_sharded': "
            "verl-omni registers the same DeltaShardedCheckpointEngine under that name and consumes its "
            "flushes in the vllm-omni worker extension."
        )
    if backend != "omni_delta_sharded":
        return

    mode = _select(config, "trainer.v1.trainer_mode")
    valid_modes = ("separate_async", "omni_separate_async")
    if mode not in valid_modes:
        raise ValueError(
            f"actor_rollout_ref.rollout.checkpoint_engine.backend='omni_delta_sharded' requires "
            f"trainer.v1.trainer_mode in {list(valid_modes)} (got {mode!r})."
        )
    if mode == "separate_async" and not _select(config, "trainer.use_v1", False):
        raise ValueError(
            "checkpoint_engine.backend='omni_delta_sharded' requires the v1 diffusion trainer "
            "(trainer.use_v1=true); the legacy trainer does not drive the delta sync flow."
        )

    model_key = "actor_rollout_ref.model"
    # A resolver ValueError means LoRA was requested in some spelling; its flat-key
    # hint is a dead end where the flat key is refused too.
    delta_lora_error = (
        "checkpoint_engine.backend='omni_delta_sharded' requires full-weight training; LoRA is "
        "refused in either merge mode because the shard export's names and values differ "
        "from the adapter (merge=false) and merged (merge=true) full exports."
    )
    try:
        settings = resolve_lora_config(_select(config, model_key))
    except ValueError as e:
        raise ValueError(delta_lora_error) from e
    if settings.enabled:
        raise ValueError(
            f"{delta_lora_error} Resolved LoRA settings for {model_key}: "
            f"rank={settings.rank}, adapter_path={settings.adapter_path!r}, merge={settings.merge}."
        )

    for qat_path in ("actor_rollout_ref.actor.fsdp_config.qat.enable", "actor_rollout_ref.actor.megatron.qat.enable"):
        if _select(config, qat_path, False):
            raise ValueError(
                f"checkpoint_engine.backend='omni_delta_sharded' does not support QAT ({qat_path}=true): "
                "the quantized, fused full export does not share coordinates with the shard export."
            )


def reject_engine_side_expert_parallel(
    omni_kwargs: Any,
    *,
    prefix: str = "actor_rollout_ref.rollout",
) -> None:
    """Refuse engine-side EP flags, which bypass the parallel-size accounting.

    The flat spelling is overwritten by verl's ``launch_server`` on the AR path
    and forwarded unvalidated by the diffusion ingress; the nested one is copied
    into the engine arguments verbatim; and the ``stage_overrides`` resolver
    applies its entries over the generated deploy config's stage values. All
    three can turn EP on or off regardless of the validated size, so all three
    are refused. Both layers check because the rollout replica is validated
    independently of the driver.

    Args:
        omni_kwargs: The ``engine_kwargs.vllm_omni`` mapping.
        prefix: Config path used in error messages.

    Raises:
        ValueError: If any spelling requests expert parallelism.
    """
    if not hasattr(omni_kwargs, "get"):
        return
    if omni_kwargs.get("enable_expert_parallel", False):
        raise ValueError(
            f"{prefix}.engine_kwargs.vllm_omni.enable_expert_parallel is not a supported spelling: "
            "verl's launch_server overwrites the engine arg from "
            f"{prefix}.expert_parallel_size, and the vLLM-Omni CLI rejects --enable-expert-parallel "
            f"under --omni. Set {prefix}.expert_parallel_size instead — the rollout writes the "
            "per-stage flag into the generated deploy config."
        )
    parallel_config = omni_kwargs.get("parallel_config") or {}
    if hasattr(parallel_config, "get") and parallel_config.get("enable_expert_parallel", False):
        raise ValueError(
            f"{prefix}.engine_kwargs.vllm_omni.parallel_config.enable_expert_parallel bypasses the "
            f"parallel-size accounting: the rollout replica sizes its ranks from "
            f"{prefix}.expert_parallel_size, and the engine-side flag is then set from that size. "
            f"Set {prefix}.expert_parallel_size instead and drop the engine_kwargs entry."
        )
    for key, value in (omni_kwargs.get("stage_overrides") or {}).items():
        if hasattr(value, "get") and value.get("enable_expert_parallel", False):
            raise ValueError(
                f"{prefix}.engine_kwargs.vllm_omni.stage_overrides[{key!r}].enable_expert_parallel "
                "bypasses the parallel-size accounting: the vLLM-Omni resolver applies stage "
                "overrides over the generated deploy config, so the flag would be honored without "
                f"accounting. Set {prefix}.expert_parallel_size instead and drop the "
                "stage_overrides entry."
            )


def coerce_expert_parallel_size(raw_ep_size: Any, *, prefix: str = "actor_rollout_ref.rollout") -> int:
    """Return the EP degree, treating an unset value as "off".

    Args:
        raw_ep_size: The configured size, before coercion.
        prefix: Config path used in error messages.

    Returns:
        int: The degree; 1 means EP is off.

    Raises:
        ValueError: If the value is not a whole positive integer.
    """
    if raw_ep_size is None:
        return 1
    if isinstance(raw_ep_size, bool):  # bool is an int subclass
        raise ValueError(f"{prefix}.expert_parallel_size must be a positive integer, got {raw_ep_size!r}.")
    if isinstance(raw_ep_size, float) and not raw_ep_size.is_integer():
        raise ValueError(f"{prefix}.expert_parallel_size must be a whole number, got {raw_ep_size!r}.")
    try:
        ep_size = int(raw_ep_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}.expert_parallel_size must be an integer, got {raw_ep_size!r}.") from exc
    if ep_size < 1:
        raise ValueError(f"{prefix}.expert_parallel_size must be a positive integer, got {raw_ep_size!r}.")
    return ep_size


def validate_expert_parallel_degree(
    raw_ep_size: Any,
    tp_size: int,
    dp_size: int,
    *,
    prefix: str = "actor_rollout_ref.rollout",
) -> int:
    """Validate an EP size against the parallel layout and return the degree.

    Single source of truth for the size contract shared by the driver gate and
    the rollout replica. vLLM derives the degree as TP x DP, and the vllm-omni
    deploy config cannot express DP, so only EP == TP with DP == 1 is honored.

    Args:
        raw_ep_size: The configured size, before coercion.
        tp_size: Tensor-parallel size of the rollout replica.
        dp_size: Data-parallel size of the rollout replica.
        prefix: Config path used in error messages.

    Returns:
        int: The validated degree (1 means EP is off).

    Raises:
        ValueError: If the size is not a positive integer, disagrees with TP x DP,
            or pairs a DP > 1 with EP.
    """
    ep_size = coerce_expert_parallel_size(raw_ep_size, prefix=prefix)
    if ep_size == 1:
        return ep_size
    if ep_size != tp_size * dp_size:
        raise ValueError(
            f"{prefix}.expert_parallel_size={ep_size} must equal tensor_model_parallel_size * "
            f"data_parallel_size (={tp_size} * {dp_size} = {tp_size * dp_size}): expert parallelism "
            "is a boolean over the existing TP x DP ranks, not an independent world dimension. "
            "Lower expert_parallel_size or raise the TP/DP sizes."
        )
    if dp_size != 1:
        raise ValueError(
            f"{prefix}.expert_parallel_size={ep_size} with data_parallel_size={dp_size} is not wired: "
            "the vllm-omni rollout realizes DP as separate TP-sized rollout replicas and the generated "
            "deploy config has no data_parallel_size field, so a stage can only shard experts over its "
            f"own TP ranks ({tp_size}). Set data_parallel_size=1 (expert_parallel_size then equals TP), "
            "or leave expert_parallel_size=1."
        )
    return ep_size


def _validate_expert_parallel(config: Any) -> None:
    """Honor-or-reject gate for rollout expert parallelism.

    Adds the driver-level rules that need the composed config: the diffusion
    rollout has no consumer for the knob, and both ``engine_kwargs.vllm_omni``
    spellings of the engine flag bypass the size accounting. The size contract is
    shared with the rollout replica via :func:`validate_expert_parallel_degree`;
    whether the model can run EP needs the architecture, so that half lives in
    the replica.
    """
    rollout = "actor_rollout_ref.rollout"
    raw_ep_size = _select(config, f"{rollout}.expert_parallel_size", 1)

    # Before the EP==1 early return: both spellings bypass the size accounting.
    reject_engine_side_expert_parallel(
        _select(config, f"{rollout}.engine_kwargs.vllm_omni", None) or {},
        prefix=rollout,
    )

    tp_size = int(_select(config, f"{rollout}.tensor_model_parallel_size", 1))
    dp_size = int(_select(config, f"{rollout}.data_parallel_size", 1))
    ep_size = validate_expert_parallel_degree(raw_ep_size, tp_size, dp_size, prefix=rollout)
    if ep_size == 1:
        return

    # Diffusion-only field; verl's RolloutConfig has no such key.
    if _select(config, f"{rollout}.rollout_attn_backend", None) is not None:
        raise ValueError(
            f"{rollout}.expert_parallel_size={ep_size} is not supported by the diffusion rollout: "
            "no diffusion stage consumes the knob and vLLM-Omni gates diffusion EP per pipeline, so a "
            "pipeline must opt in through its own stage config. Leave expert_parallel_size=1."
        )


def validate_config(config: Any) -> None:
    """Validate configuration values that otherwise trigger silent fallbacks."""
    if _select(config, "actor_rollout_ref.actor.enable_timestep_staging", False):
        sp_size = _select(config, "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 1)
        if sp_size != 1:
            raise ValueError("Timestep staging requires ulysses_sequence_parallel_size=1.")

    _validate_expert_parallel(config)

    _validate_delta_sharded(config)

    if _select(config, "actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation", False):
        strategy = _select(config, "actor_rollout_ref.actor.strategy")
        if strategy not in ("fsdp", "fsdp2"):
            raise ValueError(
                "actor.use_no_sync_for_gradient_accumulation=true requires actor.strategy "
                f"fsdp or fsdp2, got {strategy!r}."
            )

    resume_mode = _select(config, "trainer.resume_mode")
    valid_resume_modes = ("disable", "auto", "resume_path")
    if resume_mode not in valid_resume_modes:
        raise ValueError(f"Unknown trainer.resume_mode={resume_mode!r}. Available options: {list(valid_resume_modes)}.")
    if resume_mode == "resume_path" and not _select(config, "trainer.resume_from_path"):
        raise ValueError("trainer.resume_from_path must be set when trainer.resume_mode='resume_path'.")

    total_steps = _select(config, "trainer.total_training_steps")
    if total_steps is not None:
        try:
            total_steps = int(total_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.") from exc
        if total_steps <= 0:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.")

    _validate_dynamic_resource_scheduling(config)


def _validate_dynamic_resource_scheduling(config: Any) -> None:
    """Refuse the fully_async_policy scheduler on v1/v0 entrypoints.

    ``async_training.use_dynamic_resource_scheduling`` drives verl's
    ``DynamicResourceController`` (verl#6556) on the MessageQueue
    ``fully_async_policy`` stack. ``main_omni`` / ``main_diffusion`` /
    ``main_diffusion_v1`` never construct that controller, so the flag would
    be a silent no-op. The v1 analog is ``hybrid_rollout.enable_switch``.
    """
    if not _select(config, "async_training.use_dynamic_resource_scheduling", False):
        return
    raise ValueError(
        "async_training.use_dynamic_resource_scheduling=true is the fully_async_policy "
        "DynamicResourceController (verl#6556) and is not wired on main_omni / "
        "main_diffusion / main_diffusion_v1. For v1 separate_async, set "
        "trainer.v1.separate_async.hybrid_rollout.enable_switch=true instead."
    )


# ---------------------------------------------------------------------------
# LoRA config surface
# ---------------------------------------------------------------------------

# The only nested model.lora keys verl-omni reads. Everything else nested is
# either Megatron grammar (below) or a mistake.
NESTED_LORA_READ_KEYS = frozenset({"rank", "merge"})

# Nested model.lora keys that verl's pinned default config tree injects into
# every composed config (origin: verl/trainer/config/model/hf_model.yaml).
# They are Megatron grammar no FSDP engine here reads: none of them is read
# (rank/merge live in NESTED_LORA_READ_KEYS; adapter_path raises via the
# resolver when set without its flat twin), and they are tolerated-but-unread
# until a verl pin bump prunes them at the source.
MEGATRON_PINNED_LORA_KEYS = frozenset(
    {
        "type",
        "alpha",
        "dropout",
        "target_modules",
        "exclude_modules",
        "dropout_position",
        "lora_A_init_method",
        "lora_B_init_method",
        "a2a_experimental",
        "dtype",
        "adapter_path",
        "freeze_vision_model",
        "freeze_vision_projection",
        "freeze_language_model",
    }
)

KNOWN_POLICY_STATES = ("default", "old", "reference")


@dataclass(frozen=True)
class LoRASettings:
    """Normalized LoRA policy resolved from ``actor_rollout_ref.model``.

    The single authority for LoRA configuration: trainer entry points read
    this instead of re-deriving the flat vs nested spellings. Engine wrap
    gates still read the raw keys until engine gating unifies.
    """

    rank: int  # flat lora_rank (a nested-only lora.rank raises in the resolver)
    alpha: int  # flat lora_alpha (the nested alpha key is Megatron-only, unread)
    adapter_path: str | None  # flat lora_adapter_path (a nested-only adapter_path raises)
    merge: bool  # nested lora.merge: fuse adapters into base weights before sync
    adapters: tuple[str, ...]  # normalized policy_state_adapters, "default" forced first
    enabled: bool  # rank > 0 or adapter_path is not None


def resolve_lora_config(model_config: Any) -> LoRASettings:
    """Resolve the flat and nested LoRA keys into one validated settings object.

    Accepts an OmegaConf node or the instantiated model config dataclass.
    Fail-closed: conflicting spellings and unread nested keys raise ``ValueError``.

    - rank: flat ``lora_rank`` only. A nested ``lora.rank`` > 0 without the flat
      key raises ("set the flat key"): every engine gate still reads the flat
      spelling, so adopting the nested value would let the trainer believe LoRA
      is enabled while the engine never wraps PEFT. Nested-only adoption returns
      when engine gating unifies through this resolver (#685).
    - adapter_path: flat ``lora_adapter_path`` only, same raise-on-nested-only
      rule as rank.
    - alpha: flat ``lora_alpha`` only. The nested ``lora.alpha`` key is Megatron
      grammar and is never read: every composed omni config still carries
      ``lora.alpha: 32`` next to ``lora_alpha: 16`` (both injected from verl's
      default tree), so a disagreeing nested alpha is ignored until one of
      those two defaults changes.
    """
    nested = _select(model_config, "lora", None) or {}
    if not hasattr(nested, "keys"):
        raise ValueError(f"actor_rollout_ref.model.lora must be a mapping, got {type(nested).__name__}.")
    allowed_keys = NESTED_LORA_READ_KEYS | MEGATRON_PINNED_LORA_KEYS
    for key in nested.keys():
        if key not in allowed_keys:
            raise ValueError(
                f"actor_rollout_ref.model.lora.{key!r} is not read by any verl-omni engine. "
                "Remove the override; if the key comes from a newer verl default config, "
                "this build needs a verl pin bump."
            )

    # A CLI override can arrive as the string "8"; compare ints, not the raw values.
    rank = int(_select(model_config, "lora_rank", 0) or 0)
    nested_rank = int(nested.get("rank", 0) or 0)
    if rank > 0 and nested_rank > 0 and rank != nested_rank:
        raise ValueError(
            f"actor_rollout_ref.model.lora.rank={nested_rank} conflicts with "
            f"actor_rollout_ref.model.lora_rank={rank}; set only one spelling."
        )
    if nested_rank > 0 and rank <= 0:
        raise ValueError(
            f"actor_rollout_ref.model.lora.rank={nested_rank} is Megatron grammar the "
            "engines do not gate on; set actor_rollout_ref.model.lora_rank instead "
            "(nested-only adoption returns with engine gating unification, #685)."
        )

    adapter_path = _select(model_config, "lora_adapter_path", None) or None
    nested_adapter_path = nested.get("adapter_path")
    if adapter_path and nested_adapter_path and adapter_path != nested_adapter_path:
        raise ValueError(
            f"actor_rollout_ref.model.lora.adapter_path={nested_adapter_path!r} conflicts with "
            f"actor_rollout_ref.model.lora_adapter_path={adapter_path!r}; set only one spelling."
        )
    if nested_adapter_path and not adapter_path:
        raise ValueError(
            f"actor_rollout_ref.model.lora.adapter_path={nested_adapter_path!r} is Megatron "
            "grammar the engines do not gate on; set actor_rollout_ref.model.lora_adapter_path "
            "instead (nested-only adoption returns with engine gating unification, #685)."
        )

    return LoRASettings(
        rank=int(rank),
        alpha=int(_select(model_config, "lora_alpha", 0) or 0),
        adapter_path=adapter_path,
        merge=bool(nested.get("merge", False)),
        adapters=_normalize_policy_state_adapters(_select(model_config, "policy_state_adapters", ("default",))),
        enabled=rank > 0 or adapter_path is not None,
    )


def _normalize_policy_state_adapters(value: Any) -> tuple[str, ...]:
    """Dedupe policy states, validate names, and force "default" first.

    "default" is the trained policy and must be the primary adapter; a config
    listing e.g. ["old", "default"] must not silently promote "old" to primary.
    """
    adapters: list[str] = []
    for adapter in value or ():
        if adapter not in KNOWN_POLICY_STATES:
            raise ValueError(f"Unknown policy state adapter {adapter!r}; expected one of {list(KNOWN_POLICY_STATES)}.")
        if adapter not in adapters:
            adapters.append(adapter)
    return ("default", *(adapter for adapter in adapters if adapter != "default"))
