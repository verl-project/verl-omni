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
import argparse
import asyncio
import dataclasses
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import ray
import torch
import vllm_omni.entrypoints.cli.serve
from verl.utils.profiler import build_rollout_dist_profiler
from verl.utils.tracking import RLInsightLogger
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout import ServerAdapter
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs, OrchestratorArgs, orchestrator_field_names
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.lora.request import LoRARequest

from verl_omni.utils.net_utils import get_non_ephemeral_free_port
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig, OmniModelConfig
from verl_omni.workers.rollout.base import get_rollout_sequence_parallel_size, get_rollout_world_size
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import (
    colocate_zmq_handle,
    export_colocate_zmq_namespace,
    namespace_colocate_zmq_handle,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Sentinel: ``None`` is a valid cached value (LoRA not loaded).
# TODO: Move LoRA request cache resolution/invalidation upstream to AsyncOmni.
_LORA_REQUEST_CACHE_MISS = object()

# Lazily-computed upstream argument defaults, used to forward only explicitly
# set engine arguments (see ``_drop_defaulted_engine_args``; tracked in vllm-omni#8503).
_ENGINE_ARGS_DEFAULTS: dict | None = None

# Lazily-computed ``RolloutConfig`` field defaults. These are the defaults of the
# *source* schema the engine args are read from -- a different schema from
# ``OmniEngineArgs``, with different defaults for the keys the two share.
_SOURCE_CONFIG_DEFAULTS: dict | None = None

# ``deploy_config path -> (mtime, keys pinned by every stage block)``.
_DEPLOY_PINNED_KEYS: dict[str, tuple[float, frozenset[str]]] = {}


def _source_config_defaults() -> dict:
    """Field defaults of the rollout config the engine args are copied from.

    ``OmniEngineArgs`` is not the only schema in play: ``RolloutConfig`` owns the
    same keys with *different* defaults, and ``_is_defaulted_value`` only ever
    compared against the former. Two of those disagreements are load-bearing on
    this box:

    * ``max_num_seqs`` defaults to ``1024`` here but to ``None`` there, so a
      rollout that never asked for a batch size forwards ``1024`` as if it were
      an explicit override;
    * ``enable_prefix_caching`` defaults to ``True`` here and to ``None`` there,
      which is what fed ``enable_prefix_caching=True`` into the DiT stage of
      ``bagel_think`` and tripped ``_DiffusionConfigProjection``.

    Both then reach *every* stage (the deploy-config path copies plain engine
    args into each stage, where they beat that stage's own pin) -- so the
    engine's ``BagelPipeline`` stage aborts engine init with "does not support
    request-level batching. Use max_num_seqs=1", which is how the 2026-10-14
    ``bagel_corl_prfix_t07`` attempts all died. See
    ``_drop_defaulted_engine_args`` for when a source default may be dropped.
    """
    global _SOURCE_CONFIG_DEFAULTS
    if _SOURCE_CONFIG_DEFAULTS is None:
        defaults: dict = {}
        for cls in reversed(RolloutConfig.__mro__):
            if not dataclasses.is_dataclass(cls):
                continue
            for field in dataclasses.fields(cls):
                if field.default is not dataclasses.MISSING:
                    defaults[field.name] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    defaults[field.name] = field.default_factory()
        _SOURCE_CONFIG_DEFAULTS = defaults
    return _SOURCE_CONFIG_DEFAULTS


def _deploy_stage_pinned_keys(deploy_config: Any) -> frozenset[str]:
    """Keys that *every* ``stages[*]`` block of *deploy_config* pins explicitly.

    Only keys pinned by all stages are returned: those are the ones whose value
    the deploy config fully determines, so a plain global arg equal to a source
    default cannot be carrying intent. A key pinned by *some* stages is left out
    on purpose -- there the remaining stages would fall back to the engine's own
    default, which is not always the safe direction (``gpu_memory_utilization``
    is ``0.5`` in ``RolloutConfig`` and ``0.92`` in ``OmniEngineArgs``).

    An unreadable or absent config returns the empty set, so callers degrade to
    the pre-existing behaviour.
    """
    if not deploy_config:
        return frozenset()
    path = Path(deploy_config)
    candidates = [path] if path.is_absolute() else [path, Path.cwd() / path]
    for candidate in candidates:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        cached = _DEPLOY_PINNED_KEYS.get(str(candidate))
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            import yaml

            with open(candidate, encoding="utf-8") as handle:
                document = yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001 - malformed configs are reported by vllm-omni
            return frozenset()
        stages = document.get("stages") or []
        blocks = [stage for stage in stages if isinstance(stage, dict)]
        if not blocks:
            return frozenset()
        pinned = set(blocks[0])
        for block in blocks[1:]:
            pinned &= set(block)
        # Identity/plumbing keys are not engine arguments and never travel anyway.
        pinned -= {"stage_id", "devices", "input_connectors", "default_sampling_params"}
        result = frozenset(pinned)
        _DEPLOY_PINNED_KEYS[str(candidate)] = (mtime, result)
        return result
    return frozenset()


def _validate_collective_rpc_reply(method: str, result: Any) -> None:
    """Reject worker ``error``, ``todo``, and ``supported=False`` replies; allow ``None``."""
    if isinstance(result, list | tuple):
        if not result:
            raise RuntimeError(f"{method} failed: empty worker results")
        for item in result:
            _validate_collective_rpc_reply(method, item)
    elif isinstance(result, dict):
        if result.get("error") or result.get("todo") or result.get("supported") is False:
            raise RuntimeError(f"{method} failed: {result}")


def _is_defaulted_value(key: str, value: Any, default: Any) -> bool:
    """Return True if *value* matches the upstream default for *key*.

    ``None`` counts as defaulted for every field: the vLLM-Omni CLI parser
    emits ``None`` for the flags it declares ``bool | None`` (``enable_lora``,
    ``mm_device_do_normalize``), so it is the parser's *unset* sentinel and can
    never be an explicit value. The resolved dataclass defaults for those two
    are ``False`` and ``True`` -- *not* ``None`` -- so equality alone forwards
    them, and vllm-omni's diffusion stage ingress validates key *presence*:
    stage 1 of the mixed ``bagel_think`` pipeline aborts engine init with
    "Unknown diffusion config field(s) for stage 1: 'enable_lora',
    'mm_device_do_normalize'". Dropping them restores the engine's own default
    for the stage that consumes them.
    """
    if value is None:
        return True
    if value == default:
        if isinstance(value, bool) != isinstance(default, bool):
            return False
        return True
    if default is None and value in ("", {}, []):
        return True
    if isinstance(default, str) and isinstance(value, dict | list):
        try:
            return json.loads(default) == value
        except (ValueError, TypeError):
            return False
    if isinstance(value, str) and isinstance(default, dict | list):
        try:
            return json.loads(value) == default
        except (ValueError, TypeError):
            return False
    return False


def _restore_raw_compilation_config(engine_args: dict, args: Any) -> None:
    """Restore the raw user ``compilation_config`` dict to pass strict stage validation."""
    raw_compilation_config = getattr(args, "compilation_config", None)
    if raw_compilation_config is None:
        engine_args.pop("compilation_config", None)
    else:
        if isinstance(raw_compilation_config, str):
            raw_compilation_config = json.loads(raw_compilation_config)
        engine_args["compilation_config"] = raw_compilation_config


def _drop_defaulted_engine_args(engine_args: dict, deploy_config: Any = None) -> dict:
    """Keep only explicitly-set engine arguments, dropping upstream defaults.

    ``deploy_config`` is the vLLM-Omni deploy yaml this server is about to launch.
    Its per-stage pins are authoritative for the keys it pins on *every* stage, so
    a value that merely reproduces a ``RolloutConfig`` default is dropped rather
    than forwarded on top of them (see ``_source_config_defaults`` and
    ``_deploy_stage_pinned_keys``). Without a deploy config the filtering is
    unchanged.
    """
    global _ENGINE_ARGS_DEFAULTS
    if _ENGINE_ARGS_DEFAULTS is None:
        model = engine_args.get("model", "")
        defaults = asdict(OmniEngineArgs(model=model))
        defaults.update(asdict(OrchestratorArgs()))
        _ENGINE_ARGS_DEFAULTS = defaults
    defaults = _ENGINE_ARGS_DEFAULTS
    source_defaults = _source_config_defaults()
    pinned_by_every_stage = _deploy_stage_pinned_keys(deploy_config)

    def _is_default(key: str, value: Any) -> bool:
        if _is_defaulted_value(key, value, defaults[key]):
            return True
        # A source-schema default is the caller saying "I never set this". It must
        # not ride in as an explicit override and outrank the deploy config's own
        # per-stage pin -- a plain global arg beats a per-stage value.
        return key in pinned_by_every_stage and _is_defaulted_value(key, value, source_defaults.get(key))

    return {
        key: value
        for key, value in engine_args.items()
        if key == "model" or key not in defaults or not _is_default(key, value)
    }


# Control RPCs the diffusion worker cannot serve in the pinned vllm_omni.
#
# ``clear_prompt_embed_cache`` / ``get_prompt_embed_cache_stats`` live on
# ``DiffusionModelRunner``, but ``collective_rpc`` resolves
# ``getattr(self.worker, method)`` against the *worker*
# (``DiffusionWorkerWithvLLMOmniColocateWorkerExtension``) and nothing bridges worker →
# ``model_runner``, so the engine answers ``{"supported": False, "error": "… has no
# attribute 'clear_prompt_embed_cache'"}`. The engine logs a full traceback per attempt
# **before** returning that dict, so this is not catchable client-side — the only way to
# stop the log spam is to stop issuing the call once it is known to be unserviceable.
#
# Latched per method so a transient stage failure (engine asleep, replica detached) does
# not permanently disable a method that works; only a *missing-method* error latches.


def _rpc_reports_missing_method(results: Any, method: str) -> bool:
    """Whether an engine control RPC failed because the target lacks ``method``.

    Matches only the missing-attribute failure. Other failures (a detached replica, a
    sleeping stage) return the same ``{"supported": False}`` shape from the stage pool
    and must *not* latch the method off.

    Args:
        results: Per-stage results returned by ``AsyncOmni.collective_rpc``.
        method: RPC method name that was called.

    Returns:
        True when some stage reported the method as a missing attribute.
    """
    if not isinstance(results, list | tuple):
        results = [results]
    for item in results:
        if not isinstance(item, dict) or item.get("supported") is not False:
            continue
        error = str(item.get("error") or "")
        # e.g. "RPC 'clear_prompt_embed_cache' failed on worker rank(s): rank 0:
        #       AttributeError: 'DiffusionWorker…Extension' object has no attribute
        #       'clear_prompt_embed_cache'"
        if "has no attribute" in error and method in error:
            return True
    return False


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_config(self, config):
        """Select one mode strategy before initializing its rollout config."""
        engine_kwargs = getattr(config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        # TODO (mike): drop this once the legacy omni training script is removed.
        # It should be automatically inferred from the model config.
        strategy_cls = ARStrategy if omni_kwargs.get("output_mode", "diffusion") == "ar" else DiffusionStrategy
        self._generate_strategy = strategy_cls(self)
        self._rollout_flags: dict[int, dict] = {}
        # Number of ``generate`` calls accepted but not yet returned, across *both* lanes.
        # Used by ``wait_for_requests_to_drain`` before any sleep; see that method for why the
        # engine's own ``request_states`` cannot answer this.
        self._inflight_generations = 0
        rollout_config = self._generate_strategy.init_config(config)
        if getattr(rollout_config, "seed", None) is None:
            rollout_config.seed = 42
        return rollout_config

    def _init_model_config(self, model_config):
        return self._generate_strategy.init_model_config(model_config)

    def _validate_configs(self) -> None:
        self._generate_strategy.validate_configs()

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Run strategy post-init and preserve the replica device list."""
        export_colocate_zmq_namespace()
        if get_rollout_sequence_parallel_size(self.config) > 1:
            self.replica_world_size = get_rollout_world_size(self.config)
            profiler = self.profiler_controller
            self.profiler_controller = build_rollout_dist_profiler(
                self.replica_rank,
                self.replica_world_size,
                config=self.config.profiler if profiler.config is not None else None,
                tool_config=profiler.tool_config,
            )
        if getattr(self.config, "full_determinism", False):
            from verl.workers.engine.utils import enable_full_determinism

            rollout_seed = getattr(self.config, "seed", 42)
            enable_full_determinism(seed=rollout_seed)
            os.environ["VERL_SEED"] = str(rollout_seed)
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        # Set before vllm-omni narrows per-stage visible devices; stage workers
        # remap their ZMQ ranks through this replica-level list.
        os.environ["VERL_ZMQ_BASE_VISIBLE_DEVICES"] = cuda_visible_devices
        self._generate_strategy.post_init(cuda_visible_devices)
        self._lora_request_cache: LoRARequest | None | object = _LORA_REQUEST_CACHE_MISS
        self._lora_resolve_lock = asyncio.Lock()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _apply_quantization(self) -> tuple[str | None, dict]:
        return self._generate_strategy.apply_quantization()

    def _get_override_generation_config(self) -> dict:
        return self._generate_strategy.override_generation_config()

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        device_type = ""
        try:
            from vllm.platforms import current_platform

            device_type = current_platform.device_type
        except Exception:
            pass

        return self._generate_strategy.worker_extension_cls(device_type)

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        self._generate_strategy.preprocess_engine_kwargs(engine_kwargs)

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        # TODO (mike): drop this patch once vllm-omni strips the serialized default
        # fault_tolerance_config at its kwargs boundary, or vLLM defaults it to None —
        # any dict is read as an explicit --fault-tolerance-config and auto-enabled,
        # failing without an external load balancer.
        if not engine_args.get("enable_fault_tolerance"):
            engine_args.pop("fault_tolerance_config", None)

        # ``from_cli_args`` only retains OmniEngineArgs fields. Restore the
        # OrchestratorArgs fields forwarded by verl before creating AsyncOmni.
        for key in orchestrator_field_names() - engine_args.keys():
            value = getattr(args, key, None)
            if value is not None:
                engine_args[key] = value

        # Restore raw user mapping so compilation runtime fields do not fail stage config validation.
        _restore_raw_compilation_config(engine_args, args)

        # Forward only explicitly-set engine args; stage config rejects unowned defaults.
        # The deploy config's per-stage pins are authoritative for the keys it pins
        # everywhere, so a value that only reproduces a ``RolloutConfig`` default is
        # dropped instead of overriding them (``max_num_seqs=1024`` vs the AR deploy
        # config's ``max_num_seqs: 1`` is what killed every BagelPipeline init).
        engine_args = _drop_defaulted_engine_args(engine_args, deploy_config=getattr(args, "deploy_config", None))
        # This rollout setting is explicit even when its value matches the
        # OmniEngineArgs default, so restore it after default filtering.
        engine_args["log_stats"] = not self.config.disable_log_stats

        deploy_config = getattr(args, "deploy_config", None)
        if deploy_config:
            engine_args["deploy_config"] = deploy_config

        self._generate_strategy.prepare_engine_args(engine_args, args)

        if getattr(self.config, "step_execution", False):
            engine_args["step_execution"] = True

        if engine_args.get("seed") is None:
            engine_args.pop("seed", None)

        # The port stays unbound until vllm-omni's rank-0 DiffusionWorker
        # listens on it; see get_non_ephemeral_free_port for why it must not
        # come from the ephemeral range.
        # TODO (mike): drop once vllm-omni passes a FileStore-backed
        # distributed_init_method to its workers instead of env:// MASTER_PORT.
        diffusion_master_port = get_non_ephemeral_free_port("127.0.0.1")

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(diffusion_master_port)
        logger.info("Using MASTER_PORT=%s for vLLM-Omni workers", os.environ["MASTER_PORT"])

        # rollout_attn_backend only exists on the diffusion rollout config, not AR text rollouts.
        attn_backend = getattr(self.config, "rollout_attn_backend", None)
        if attn_backend is not None:
            engine_args.pop("diffusion_attention_backend", None)
            engine_args["diffusion_attention_config"] = self.config.to_vllm_omni_attention_config()
            logger.info(
                "Setting diffusion_attention_config.default.backend=%s from rollout config",
                attn_backend,
            )

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        # Deploy config YAML is consumed by AsyncOmni above; clean up the temp dir.
        if getattr(self, "_temp_deploy_ctx", None) is not None:
            self._temp_deploy_ctx.cleanup()
            self._temp_deploy_ctx = None

        self.engine = engine_client
        if isinstance(self._generate_strategy, ARStrategy):
            # attach engine-level monkey patches
            await self.collective_rpc(method="monkey_patch_model")
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    async def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        *,
        require_ack: bool = False,
    ):
        """Dispatch a shared RPC to the stages selected by the active strategy."""
        stage_ids = self._generate_strategy.collective_rpc_stage_ids(method)
        if require_ack:
            return await self._collective_rpc_with_ack(method, timeout, args, kwargs, stage_ids)
        return await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            stage_ids=stage_ids,
        )

    async def _collective_rpc_with_ack(
        self, method: Any, timeout: float | None, args: tuple, kwargs: dict[str, Any] | None, stage_ids: Any
    ) -> list[Any]:
        """Validate one stage at a time because the pinned reply omits stage IDs."""
        num_stages = getattr(self.engine, "num_stages", None)
        if type(num_stages) is not int or num_stages <= 0:
            raise RuntimeError(f"{method} failed: unknown stage topology")
        if stage_ids is None:
            stage_ids = list(range(num_stages))
        if not isinstance(stage_ids, list | tuple) or not stage_ids:
            raise RuntimeError(f"{method} failed: no selected stages")
        if any(type(stage_id) is not int or not 0 <= stage_id < num_stages for stage_id in stage_ids):
            raise RuntimeError(f"{method} failed: unknown selected stage")

        # The pinned flattened reply has no stage IDs. Check each stage separately.
        replies: list[Any] = []
        for stage_id in dict.fromkeys(stage_ids):
            results = await self.engine.collective_rpc(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                stage_ids=[stage_id],
            )
            if not isinstance(results, list | tuple) or not results:
                raise RuntimeError(f"{method} failed: missing replica results for stage {stage_id}")
            _validate_collective_rpc_reply(method, results)
            replies.extend(results)
        return replies

    # -----------------------------------------------------------------------
    # wake_up hook: full wake must include kv_cache
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        # AsyncOmni.generate() rejects leftover sleeping tags. Weights-only left
        # kv_cache asleep and every generate() failed.
        return ["kv_cache", "weights"]

    def _resolve_sleep_level(self) -> int:
        """Level 1 is the correct phase-separation level for vllm-omni diffusion.

        Unlike upstream vLLM (whose LLM level-1 keeps weights resident and only
        drops KV cache), vllm-omni's diffusion-worker level-1 sleep offloads the
        whole "weights" pool — transformer + text encoder + VAE — to pinned host
        memory and unmaps the GPU pages, and ``wake_up(tags=["weights"])``
        restores them via DMA. Level 2 additionally discards the CPU copy but
        ``AsyncOmni.wake_up`` deliberately raises NotImplementedError after a
        level-2 sleep, and the trainer would have to re-upload the full
        pipeline each cycle. Keep 1 until both change.
        """
        return 1

    async def _wake_engine_vram_safe(self, tags: list[str]) -> None:
        """``engine.wake_up(tags)`` with a bounded retry for a transient VRAM re-map failure.

        The trainer reaches ``wake_up`` **directly** (``BagelCoRLTrainer._wake_rollout_replicas``,
        once per step for both lanes), so there is no caller-side retry to fall back on the way
        ``update_weights`` has one in :func:`verl_omni.workers.engine_workers._resume_rollout_vram_safe`.
        The failure is the transient sleep-mode allocator OOM (``CUDA Error: out of memory at
        .../cumem_allocator.cpp``), raised out of ``_validate_acks`` (it rides the ack, not the
        await), so the retry wraps both. Each retry first hands this process's cached blocks back
        to the driver: ``aggressive_empty_cache`` reclaims the ``reserved - allocated`` gap the
        ``cuMemCreate`` re-map is asking for. A failed wake leaves the engine asleep, so re-issuing
        is idempotent.
        """
        from verl.utils.memory_utils import aggressive_empty_cache

        from verl_omni.utils.rollout_wake import wake_with_oom_retry

        async def _wake_once():
            acks = await self.engine.wake_up(tags=tags)
            self._validate_acks("wake_up", acks)
            return acks

        await wake_with_oom_retry(
            _wake_once,
            flush=lambda: aggressive_empty_cache(force_sync=True),
            label=f"rollout wake_up(tags={tags})",
        )

    async def wake_up(self, tags: list[str] | None = None):
        if self.node_rank != 0:
            return
        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")
            return
        resolved_tags = tags if tags is not None else self._get_wake_up_tags()
        with RLInsightLogger.trace_state(
            f"vllm_wake_up[{','.join(resolved_tags)}]", state_lane_id=f"replica_{self.replica_rank}"
        ):
            await self._wake_engine_vram_safe(resolved_tags)
            await self.engine.resume_generation()
            await self._invalidate_policy_caches()

    async def set_global_steps(self, global_steps: int):
        if global_steps != self.global_steps:
            await self._invalidate_policy_caches()
        await super().set_global_steps(global_steps)

    async def prompt_embed_cache_stats(self) -> list[Any] | None:
        """Conditioning-cache counters from the GEN engine (RFC §4.4.4, R2).

        ``DiffusionModelRunner.get_prompt_embed_cache_stats`` owns the counters —
        each runner process holds its own cache, so there is no engine-level
        aggregate to read instead. The control RPC fans out per stage so a
        multi-stage deploy reports every cache rather than one silently-partial
        number. Returns ``None`` on non-head ranks.
        """
        if self.node_rank != 0:
            return None
        if self._prompt_embed_cache_rpc_unsupported("get_prompt_embed_cache_stats"):
            return []
        # ``self.engine`` is the ``AsyncOmni`` *client* (vllm_omni
        # ``entrypoints/async_omni.py``), whose control-plane entry point is
        # ``collective_rpc`` -- itself already a coroutine that delegates to the inner
        # ``AsyncOmniEngine.collective_rpc_async``. Calling ``_async`` here would reach
        # for the engine-internal method on the client object, which has no such
        # attribute.
        results = list(await self.engine.collective_rpc(method="get_prompt_embed_cache_stats"))
        if _rpc_reports_missing_method(results, "get_prompt_embed_cache_stats"):
            self._mark_prompt_embed_cache_rpc_unsupported("get_prompt_embed_cache_stats")
            return []
        return results

    async def _reset_frontend_mm_cache(self) -> None:
        """Clear the frontend multimodal cache; EngineCore.sleep wipes only the engine-side copy."""
        # Diffusion-only engines build no InputProcessor, so renderer is None.
        # TODO (mike): drop after vllm-omni fixes AsyncOmni.reset_mm_cache.
        renderer = self.engine.renderer
        if renderer is not None:
            await renderer.clear_mm_cache_async()

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")
            return
        with RLInsightLogger.trace_state("vllm_sleep", state_lane_id=f"replica_{self.replica_rank}"):
            # Never unmap engine memory under a live decode: ``AsyncOmni.sleep`` aborts in-flight
            # requests, and doing that to a request the worker is mid-way through killed a worker
            # outright (see ``wait_for_requests_to_drain``).
            await self.wait_for_requests_to_drain()
            acks = await self.engine.sleep(level=self._resolve_sleep_level())
            self._validate_acks("sleep", acks)
            await self._reset_frontend_mm_cache()
            await self._invalidate_policy_caches()

    async def release_kv_cache(self):
        """Free cache around a weight sync without discarding Omni weights.

        Sleeps both tags then wakes weights only so NCCL can write into the
        existing buffers. Do not resume generation here: kv_cache is still
        asleep and AsyncOmni.generate() rejects that state. resume_kv_cache()
        restores the cache and re-opens admission after the sync.
        """
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        with RLInsightLogger.trace_state("vllm_release_kv_cache", state_lane_id=f"replica_{self.replica_rank}"):
            # Same hazard as ``sleep``: this path frees the kv_cache device memory, so it must not
            # run while a request is still reading it.
            await self.wait_for_requests_to_drain()
            acks = await self.engine.sleep(level=self._resolve_sleep_level())
            self._validate_acks("sleep", acks)
            await self._reset_frontend_mm_cache()
            await self._invalidate_policy_caches()
            await self._wake_engine_vram_safe(["weights"])
            await self.engine.resume_generation()
            await self._invalidate_policy_caches()

    async def resume_kv_cache(self):
        """Restore kv_cache after a weight sync and re-open generate admission."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        with RLInsightLogger.trace_state("vllm_resume_kv_cache", state_lane_id=f"replica_{self.replica_rank}"):
            await self._wake_engine_vram_safe(["kv_cache"])
            await self.engine.resume_generation()
            await self._invalidate_policy_caches()

    async def resume_generation(self):
        if self.node_rank == 0:
            await self.engine.resume_generation()

    def _validate_acks(self, method: str, acks: Any) -> None:
        """Fail closed on any non-success sleep/wake handshake.

        Acks reach us in two shapes: vLLM's typed ``SleepAck`` objects, which carry
        ``status``/``error_msg`` attributes, and plain ``dict``s once the orchestrator
        has serialized them. Both therefore have to be judged on their ``status``,
        because a healthy sleep's platform-audit ack is itself a ``dict`` -- treating
        *any* ``dict`` as a failure is wrong.
        """
        for ack in acks or []:
            if isinstance(ack, dict):
                status = ack.get("status")
                error = ack.get("error_msg") or ack.get("error")
            else:
                status = getattr(ack, "status", None)
                error = getattr(ack, "error_msg", None)
            if status == "SUCCESS" and not error:
                continue
            raise RuntimeError(f"{method} failed on a stage: {error or ack!r}")

    # -----------------------------------------------------------------------
    # Generation delegates mode-specific behavior to the selected strategy.
    # -----------------------------------------------------------------------

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: torch.BoolTensor | None = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        # Read-modify-write rather than ``+=`` so a server built without ``_init_config`` (the CPU
        # tests use ``object.__new__``) still counts correctly instead of raising AttributeError.
        self._inflight_generations = getattr(self, "_inflight_generations", 0) + 1
        try:
            return await self._generate_strategy.generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                negative_prompt_ids=negative_prompt_ids,
                prompt_mask=prompt_mask,
                extra_prompt_ids=extra_prompt_ids,
                negative_extra_prompt_ids=negative_extra_prompt_ids,
                priority=priority,
            )
        finally:
            self._inflight_generations = max(0, self._inflight_generations - 1)

    # -----------------------------------------------------------------------
    # Shared LoRA state
    # -----------------------------------------------------------------------

    def _invalidate_lora_request_cache(self) -> None:
        """Drop cached LoRA state after weight sync or engine sleep/wake."""
        self._lora_request_cache = _LORA_REQUEST_CACHE_MISS

    async def _clear_prompt_embed_cache(self) -> None:
        """Flush the diffusion conditioning cache (RFC §4.4.0(b), R2).

        ``PromptEmbedCache`` holds **text-encoder outputs**, i.e. a function of
        ``(conditioning tokens, weights)``. §4.11 publishes new merged weights to
        both replicas every composite step, so an entry computed under
        ``policy_version = v-1`` conditions on a *different policy* than the one
        being trained at ``v``. Serving it would mix policies mid-episode and
        silently bias the importance ratio — the entries are exact within a weight
        version, which is why the rule is "tag and flush on publish" rather than
        "disable".

        Only the diffusion strategy owns such a cache; on an AR replica there is
        nothing to flush and no such runner method, so this is a no-op there.
        Failures degrade to a warning rather than breaking the weight-sync path:
        the cache is a performance device, and the worst case of a missed flush is
        a stale *conditioning*, which the §4.4.4 hit-rate metrics surface.
        """
        if self.node_rank != 0 or not isinstance(getattr(self, "_generate_strategy", None), DiffusionStrategy):
            return
        # Nothing to flush while the cache is off, which is the shipped default (R2 is
        # opt-in): ``install_prompt_embed_cache`` never ran, so the entry set is empty by
        # construction and the RPC has no work to do. It is not a harmless no-op in the
        # pinned vllm_omni, though -- the *worker* cannot serve it
        # (``DiffusionWorkerWithvLLMOmniColocateWorkerExtension`` has no
        # ``clear_prompt_embed_cache``; the method lives on ``DiffusionModelRunner`` and
        # nothing bridges worker -> model_runner), so ``collective_rpc`` raises inside the
        # stage's ``multiproc_executor`` and the diffusion stage counts that as a fatal
        # init error:
        #
        #   AttributeError: 'DiffusionWorkerWithvLLMOmniColocateWorkerExtension' object
        #   has no attribute 'clear_prompt_embed_cache'
        #     -> RuntimeError: Orchestrator initialization failed
        #
        # ``trainer.init()`` reaches here on every run via ``set_global_steps``, so an
        # unconditional flush tore down the very engine it was meant to protect (measured
        # 2026-10-15, ``bagel_corl_prfix_t07_20261015_102119_a1``). Skipping when the
        # cache is disabled costs nothing and keeps the flush honest when it is not.
        if not getattr(self.config, "enable_prompt_embed_cache", False):
            return
        if self._prompt_embed_cache_rpc_unsupported("clear_prompt_embed_cache"):
            return
        try:
            # ``collective_rpc`` (not ``collective_rpc_async``): see
            # ``prompt_embed_cache_stats`` -- the latter exists only on the inner
            # ``AsyncOmniEngine``, not on the ``AsyncOmni`` client held here. This call
            # is inside a try/except that degrades to a warning, so the wrong name used
            # to fail silently on every weight sync: the cache was never flushed, and
            # the only trace was a warning about stale entries.
            results = await self.engine.collective_rpc(method="clear_prompt_embed_cache")
        except Exception as exc:  # noqa: BLE001 — must not break weight sync
            logger.warning("prompt-embed cache flush failed; entries may be stale: %s", exc)
            return
        # The engine logs the traceback itself before returning this shape, so a missing
        # method cannot be caught above; latch it off so we issue it only once per server
        # rather than on every weight sync.
        if _rpc_reports_missing_method(results, "clear_prompt_embed_cache"):
            self._mark_prompt_embed_cache_rpc_unsupported("clear_prompt_embed_cache")

    def _prompt_embed_cache_rpc_unsupported(self, method: str) -> bool:
        """Whether ``method`` is already known to be unserviceable by this engine.

        Args:
            method: Control RPC name.

        Returns:
            True when a previous call reported the method as a missing attribute.
        """
        return method in getattr(self, "_pec_rpc_unsupported", frozenset())

    def _mark_prompt_embed_cache_rpc_unsupported(self, method: str) -> None:
        """Latch ``method`` off and explain it once.

        Args:
            method: Control RPC name that the engine reported as missing.
        """
        known = set(getattr(self, "_pec_rpc_unsupported", frozenset()))
        known.add(method)
        self._pec_rpc_unsupported = frozenset(known)
        logger.warning(
            "prompt-embed cache RPC %r is not serviceable by this engine: the diffusion "
            "worker does not expose the runner's method, so the RFC §4.4.4 counters stay "
            "empty and the conditioning cache is not flushed. Skipping further calls "
            "(this warning appears once per method).",
            method,
        )

    async def _invalidate_policy_caches(self) -> None:
        """Invalidate everything keyed on the served weights.

        The LoRA request cache is version-keyed on the *adapter*; the
        conditioning cache is version-keyed on the *text encoder*. Both are
        functions of the published weights, so both move together — which is why
        this wraps the two rather than leaving each call site to remember the pair
        (RFC §4.4.0: "the tag makes the invariant checkable rather than assumed").
        """
        self._invalidate_lora_request_cache()
        await self._clear_prompt_embed_cache()

    async def _resolve_lora_request(self) -> Optional[LoRARequest]:
        """Return the actor LoRA request when a LoRA adapter is loaded.

        ``list_loras`` is a diffusion busy-loop RPC serialized behind
        ``execute_fn``. Calling it on every ``generate`` blocks concurrent
        ``add_request`` calls until the current forward finishes, collapsing
        request batching to B≈1. Resolve once per weight version and cache.
        Invalidate via :meth:`_invalidate_lora_request_cache` on wake/sleep/step.
        """
        if not self.lora_as_adapter:
            return None

        if self._lora_request_cache is not _LORA_REQUEST_CACHE_MISS:
            return self._lora_request_cache  # type: ignore[return-value]

        async with self._lora_resolve_lock:
            if self._lora_request_cache is not _LORA_REQUEST_CACHE_MISS:
                return self._lora_request_cache  # type: ignore[return-value]
            try:
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            except TypeError:
                # Some engine backends return a non-iterable; treat as loaded.
                lora_loaded = True
            self._lora_request_cache = (
                LoRARequest(lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)
                if lora_loaded
                else None
            )
            return self._lora_request_cache  # type: ignore[return-value]

    async def wait_for_requests_to_drain(self, timeout_s: float | None = None) -> int:
        """Block until no ``generate`` call is in flight; return how many remain.

        ``AsyncOmni.sleep`` defaults to ``mode="abort"`` / ``reset_running_requests=True``, so a
        sleep issued while a decode is still running tears that request down *and* unmaps the
        memory the worker is reading it from.

        The server admission counter covers registration and cleanup windows around
        engine requests; the native registry's lifetime varies across serving versions.
        Count calls rather than IDs because the GEN S-group deliberately reuses one
        ``routing_key`` across siblings.

        On timeout it logs and returns the remainder instead of raising: the caller is a sleep
        and the engine's own abort semantics are the fallback.
        """
        if timeout_s is None:
            timeout_s = float(os.getenv("VERL_OMNI_DRAIN_TIMEOUT_S", "900"))
        deadline = time.monotonic() + timeout_s
        announced = False
        while True:
            remaining = getattr(self, "_inflight_generations", 0)
            if remaining <= 0:
                return 0
            if not announced:
                logger.info(
                    "[vLLMOmniHttpServer] waiting for %d in-flight generate call(s) to drain before sleep",
                    remaining,
                )
                announced = True
            if time.monotonic() >= deadline:
                logger.warning(
                    "[vLLMOmniHttpServer] %d generate call(s) still in flight after %.0fs; sleeping "
                    "anyway (the engine will abort them)",
                    remaining,
                    timeout_s,
                )
                return remaining
            await asyncio.sleep(0.5)

    # -----------------------------------------------------------------------
    # Abort: AsyncOmni has no `output_processor` (it routes through an
    # Orchestrator process and tracks state in `AsyncOmni.request_states`),
    # so the parent's AsyncLLM-specific implementation must be overridden.
    # -----------------------------------------------------------------------

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all in-flight requests on the AsyncOmni engine."""
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_all_requests(reset_prefix_cache)

        # ``engine.abort`` takes EXTERNAL ids; ``request_states`` is keyed by internal.
        in_flight: list[tuple[str, str, Any]] = []
        seen: set[str] = set()
        for state in engine.request_states.values():
            if not state.external_request_id or state.external_request_id in seen:
                continue
            seen.add(state.external_request_id)
            in_flight.append((state.request_id, state.external_request_id, state))

        request_ids = [external_id for _, external_id, _ in in_flight]

        aborted = False
        try:
            await asyncio.wait_for(
                engine.abort(request_ids), timeout=float(os.getenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "120"))
            )
            aborted = True
            # Pause even with nothing to abort: holds admission until resume_generation.
            await engine.pause_generation(
                mode="abort", wait_for_inflight_requests=False, clear_cache=reset_prefix_cache
            )
        except Exception:
            # If the engine abort RPC fails or times out, upstream has not
            # delivered terminals to waiting generator coroutines. Enqueue
            # synthetic abort outputs so active generators unblock and exit.
            if not aborted:
                for internal_id, _, state in in_flight:
                    self._enqueue_abort_output(internal_id, state)
            raise

        logger.info("Aborted %d request(s): %s", len(request_ids), request_ids)
        return {"aborted_count": len(request_ids), "request_ids": request_ids}

    def _enqueue_abort_output(self, internal_id: str, req_state: Any) -> None:
        """Synthesize a terminal abort OutputMessage and put it into a per-request queue.

        Ensures active generators in ``_process_orchestrator_results`` unblock from
        ``queue.get()`` and clean up if the orchestrator abort RPC raises or times out.
        """
        from vllm.outputs import CompletionOutput
        from vllm_omni.engine.messages import OutputMessage
        from vllm_omni.outputs import OmniRequestOutput

        completion = CompletionOutput(
            index=0,
            text="",
            token_ids=[],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="abort",
            stop_reason=None,
        )
        omni_output = OmniRequestOutput(
            request_id=internal_id,
            outputs=[completion],
            finished=True,
        )
        # Use the final stage so _process_single_result's stage_meta.final_output
        # check passes and the output is yielded (not silently dropped).
        final_stage_id = max(0, getattr(self.engine, "num_stages", 1) - 1)
        msg = OutputMessage(
            request_id=internal_id,
            stage_id=final_stage_id,
            engine_outputs=omni_output,
            finished=True,
        )
        queue = getattr(req_state, "queue", None)
        if queue is not None:
            queue.put_nowait(msg)

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a single in-flight request on the AsyncOmni engine."""
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_request(request_id, reset_prefix_cache)

        in_flight_state = None
        for state in engine.request_states.values():
            if state.external_request_id == request_id:
                in_flight_state = state
                break

        try:
            await asyncio.wait_for(
                engine.abort(request_id), timeout=float(os.getenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "120"))
            )
        except Exception:
            if in_flight_state is not None:
                self._enqueue_abort_output(in_flight_state.request_id, in_flight_state)
            raise

        logger.info("Aborted request: %s", request_id)
        return {"aborted": True, "request_id": request_id}


class vLLMOmniServerAdapter(ServerAdapter):
    """Reuse verl's transport with SP-aware replica and IPC rank mapping."""

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        """Require checked replies for IPC weight updates through the inherited sender."""
        if method != "update_weights_from_ipc":
            return await super()._execute_method(method, non_block=non_block, timeout=timeout, args=args, kwargs=kwargs)
        if not self._ensure_server_handle():
            return None
        future = self.server_handle.collective_rpc.remote(
            method, timeout=timeout, args=args, kwargs=kwargs, require_ack=True
        )
        return future if non_block else await future

    def __init__(self, config, model_config, device_mesh, replica_rank: int = -1):
        super().__init__(config, model_config, device_mesh, replica_rank=replica_rank)
        if get_rollout_sequence_parallel_size(self.config) == 1:
            self.zmq_handle = namespace_colocate_zmq_handle(self.zmq_handle)
            return

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        world_size = get_rollout_world_size(self.config)
        self.replica_rank = rank // world_size if replica_rank == -1 else replica_rank
        self.rollout_rank = rank % world_size
        self.node_rank = self.rollout_rank // local_world_size
        self._has_server = self.rollout_rank == 0
        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = colocate_zmq_handle(job_id, self.replica_rank, local_rank)


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig | RolloutConfig,
        model_config: DiffusionModelConfig | OmniModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        if get_rollout_sequence_parallel_size(self.config) > 1:
            self.world_size = get_rollout_world_size(self.config)
            self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
            if self.world_size % self.gpus_per_replica_node:
                raise ValueError(f"Replica size {self.world_size} must be divisible by GPUs per node {gpus_per_node}.")
            self.nnodes = self.world_size // self.gpus_per_replica_node
        self.server_class = ray.remote(vLLMOmniHttpServer)

    # The rollout worker actor class is verl's default
    # (ray.remote(CheckpointEngineWorker)): the omni_delta_sharded backend
    # passes verl's sglang-only "delta_sharded" gate by name and resolves
    # through CheckpointEngineRegistry like any other backend.

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
