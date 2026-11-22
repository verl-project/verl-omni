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
import json
import logging
import os
import uuid
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import ray
import torch
import vllm_omni.entrypoints.cli.serve
from omegaconf import DictConfig, OmegaConf
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
from vllm_omni.engine.messages import CollectiveRPCRequestMessage, CollectiveRPCResultMessage
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.lora.request import LoRARequest
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

from verl_omni.model_merger.utils import fingerprint, inventory, tree_files
from verl_omni.utils.net_utils import get_non_ephemeral_free_port
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig, OmniModelConfig
from verl_omni.workers.rollout.base import get_rollout_sequence_parallel_size, get_rollout_world_size
from verl_omni.workers.rollout.diffusion_llm_server import DEFAULT_REQUEST_TIMEOUT_S
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import _RolloutSubmission

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _abort_ack_timeout_s() -> float:
    """Timeout for rollout abort and scheduler RPC acknowledgements."""
    return float(os.getenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "120"))


class _RolloutAsyncOmni(AsyncOmni):
    """Bound lifecycle RPCs without duplicating upstream control logic."""

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None, stage_ids=None):
        """Use bounded, nonblocking transport for lifecycle control."""
        if method not in {
            "pause_scheduler",
            "resume_scheduler",
            "weight_sharing_drain",
            "synchronize_device",
            "export_runtime_lora",
            "restore_runtime_lora",
            "remove_lora",
            "export_step_requests",
            "release_step_requests",
            "restore_step_requests",
        }:
            return await super().collective_rpc(method, timeout, args, kwargs, stage_ids)
        if timeout is None:
            timeout = _abort_ack_timeout_s()
        rpc_id = uuid.uuid4().hex
        message = CollectiveRPCRequestMessage(
            rpc_id=rpc_id, method=method, timeout=timeout, args=tuple(args), kwargs=kwargs or {}, stage_ids=stage_ids
        )
        # Nonblocking submission also bounds failure when the request queue is full.
        rpc = asyncio.create_task(
            asyncio.to_thread(
                self.engine.rpc_client.execute,
                ("collective", rpc_id),
                message,
                timeout=timeout,
                timeout_message=f"{method} timed out after {timeout} seconds",
                block_on_submit=False,
            )
        )
        try:
            result = await asyncio.shield(rpc)
        except asyncio.CancelledError:
            # Settle the bounded transport waiter before propagating cancellation.
            await asyncio.gather(rpc, return_exceptions=True)
            raise
        if not isinstance(result, CollectiveRPCResultMessage):
            raise RuntimeError(f"unexpected collective RPC result type: {type(result).__name__}")
        return list(result.results)


# Sentinel: ``None`` is a valid cached value (LoRA not loaded).
# TODO: Move LoRA request cache resolution/invalidation upstream to AsyncOmni.
_LORA_REQUEST_CACHE_MISS = object()


@dataclass(frozen=True)
class _DiffusionStepHandoff:
    adapter_id: int
    policy_ref: str
    request_ids: frozenset[str]
    snapshots: list[dict[str, bytes]] | None


class _DiffusionCheckpointModel:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.loaded_files = self._file_state()
        if not self.loaded_files:
            raise ValueError("Diffusion checkpoint base model has no local files")
        self.digest = fingerprint(inventory(self.root, self.loaded_files))
        self.verify()

    def _file_state(self) -> dict[Path, tuple[int, int, int, int, int]]:
        files = {}
        for path in tree_files(self.root):
            st = path.stat()
            files[path] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        return files

    def verify(self) -> str:
        if self._file_state() != self.loaded_files:
            raise ValueError("Diffusion base model files changed during rollout startup")
        return self.digest


# Lazily-computed upstream argument defaults, used to forward only explicitly
# set engine arguments (see ``_drop_defaulted_engine_args``).
_ENGINE_ARGS_DEFAULTS: dict | None = None


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
    """Return True if *value* matches the upstream default for *key*."""
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


def _drop_defaulted_engine_args(engine_args: dict) -> dict:
    """Keep only explicitly-set engine arguments, dropping upstream defaults."""
    global _ENGINE_ARGS_DEFAULTS
    if _ENGINE_ARGS_DEFAULTS is None:
        model = engine_args.get("model", "")
        defaults = asdict(OmniEngineArgs(model=model))
        defaults.update(asdict(OrchestratorArgs()))
        _ENGINE_ARGS_DEFAULTS = defaults
    defaults = _ENGINE_ARGS_DEFAULTS
    return {
        key: value
        for key, value in engine_args.items()
        if key == "model" or key not in defaults or not _is_defaulted_value(key, value, defaults[key])
    }


def _normalize_rollout_logging_level() -> None:
    """Canonicalize VLLM_LOGGING_LEVEL before diffusion workers spawn.

    verl's rollout env pins ``WARN`` (verl/trainer/constants_ppo.py); vLLM's
    LoggingConfig at this pin accepts only canonical level names, and
    vllm-omni's diffusion workers feed the environment value straight into
    it. Only the alias is rewritten — an explicit user level passes through.

    TODO (mike): drop once verl sets ``WARNING`` directly in its rollout env
    (its constants_ppo.py pins the ``WARN`` alias).
    """
    if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "WARN":
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


# TODO (mike): drop these wrappers once vllm-omni's serve CLI accepts plain
# argparse subparsers (it only recognizes TrackingArgumentParser, while
# verl's launcher builds the rollout CLI with FlexibleArgumentParser).
class _TrackingSubparsers:
    """Create tracking parsers from a plain argparse subparsers action."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        self._subparsers = subparsers

    def add_parser(self, name: str, *args: Any, **kwargs: Any) -> TrackingArgumentParser:
        # ``add_parser`` builds kwargs for its fixed parser class, so construct
        # and register the tracking parser here instead.
        kwargs.pop("parser_class", None)
        kwargs.setdefault("prog", f"{self._subparsers._prog_prefix} {name}")
        parser = TrackingArgumentParser(**kwargs)
        self._subparsers._name_parser_map[name] = parser
        return parser

    def __getattr__(self, name: str) -> Any:
        return getattr(self._subparsers, name)


class _TrackingCliCommand:
    """Build one vllm-omni CLI command's subparser as a tracking parser.

    verl's launcher constructs the rollout CLI from a plain
    FlexibleArgumentParser, while the pinned vllm-omni's serve subparser
    only accepts TrackingArgumentParser (its explicit-args machinery).
    """

    def __init__(self, command):
        self._command = command

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> TrackingArgumentParser:
        return self._command.subparser_init(_TrackingSubparsers(subparsers))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._command, name)


class _TrackingCliModule:
    """Present vllm-omni's CLI modules with tracking-parser subcommands."""

    def __init__(self, module):
        self._module = module

    def cmd_init(self) -> list:
        return [_TrackingCliCommand(command) for command in self._module.cmd_init()]


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    _committed_lora_adapter_id: int | None = None
    _diffusion_step_handoff: _DiffusionStepHandoff | None = None
    _diffusion_checkpoint_model_fingerprint: str | None = None

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
        _normalize_rollout_logging_level()
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
        self._step_handoff_lock = asyncio.Lock()
        # Mirrors the workers' set_pending stash: the bare follow-up
        # update_weights_from_ipc routes as an adapter-delta sync.
        self._pending_lora_delta = False
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _apply_quantization(self) -> dict:
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
        return [_TrackingCliModule(vllm_omni.entrypoints.cli.serve)]

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

        # ``from_cli_args`` only retains OmniEngineArgs fields. Restore the
        # OrchestratorArgs fields forwarded by verl before creating AsyncOmni.
        for key in orchestrator_field_names() - engine_args.keys():
            value = getattr(args, key, None)
            if value is not None:
                engine_args[key] = value

        # The stage-config boundary rejects asdict-expanded compilation configs,
        # so restore the raw user mapping before dropping unowned defaults.
        _restore_raw_compilation_config(engine_args, args)
        engine_args = _drop_defaulted_engine_args(engine_args)
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

        checkpoint_model = None
        if (
            isinstance(self._generate_strategy, DiffusionStrategy)
            and self.rollout_mode == RolloutMode.HYBRID
            and self.model_config.algorithm == "flow_grpo"
            and engine_args.get("step_execution")
            and engine_args.get("max_cpu_loras") == 2
            and "dummy" not in self.config.load_format
        ):
            try:
                checkpoint_model = await asyncio.to_thread(_DiffusionCheckpointModel, Path(engine_args["model"]))
            except (OSError, ValueError):
                logger.info("Diffusion checkpoint recovery requires a verifiable local pipeline snapshot")
        engine_client = _RolloutAsyncOmni(**engine_args)
        if checkpoint_model is not None:
            try:
                self._diffusion_checkpoint_model_fingerprint = checkpoint_model.verify()
            except (OSError, ValueError):
                logger.warning("Could not verify the loaded diffusion checkpoint base model", exc_info=True)
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

    async def close_diffusion_engine(self) -> None:
        """Shut down the engine before Ray tears down the rollout actor."""
        self._submission_paused = True
        self._rejecting = True
        self._resume_event.set()
        self.engine.shutdown()

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
            acks = await self.engine.wake_up(tags=resolved_tags)
            self._validate_acks("wake_up", acks)
            # A sync may wake both pools before installing the adapter.
            # Explicit pauses remain held until the checkpoint owner resumes.
            if not await self.engine.is_sleeping() and not self._submission_paused:
                await self.engine.resume_generation()
            self._invalidate_lora_request_cache()

    async def set_global_steps(self, global_steps: int, *, lora_adapter_id: int | None = None):
        async with self._lora_resolve_lock:
            if lora_adapter_id is not None and not self._submission_paused:
                raise RuntimeError("Versioned LoRA commit requires paused rollout admission")
            if global_steps != self.global_steps or lora_adapter_id != self._committed_lora_adapter_id:
                self._invalidate_lora_request_cache()
            await super().set_global_steps(global_steps)
            self._committed_lora_adapter_id = lora_adapter_id

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")
            return
        with RLInsightLogger.trace_state("vllm_sleep", state_lane_id=f"replica_{self.replica_rank}"):
            acks = await self.engine.sleep(level=self._resolve_sleep_level())
            self._validate_acks("sleep", acks)
            self._invalidate_lora_request_cache()

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
            acks = await self.engine.sleep(level=self._resolve_sleep_level())
            self._validate_acks("sleep", acks)
            self._invalidate_lora_request_cache()
            acks = await self.engine.wake_up(tags=["weights"])
            self._validate_acks("wake_up", acks)
            self._invalidate_lora_request_cache()

    async def resume_kv_cache(self):
        """Restore kv_cache after a weight sync and re-open generate admission."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        with RLInsightLogger.trace_state("vllm_resume_kv_cache", state_lane_id=f"replica_{self.replica_rank}"):
            acks = await self.engine.wake_up(tags=["kv_cache"])
            self._validate_acks("wake_up", acks)
            await self.resume_generation()
            self._invalidate_lora_request_cache()

    def _validate_acks(self, method: str, acks: Any) -> None:
        """Fail closed on any non-success sleep/wake handshake."""
        for ack in acks or []:
            if isinstance(ack, dict):
                raise RuntimeError(f"{method} failed on a stage: {ack.get('error', ack)}")
            elif ack.status != "SUCCESS":
                raise RuntimeError(f"{method} failed on a stage: {getattr(ack, 'error_msg', None) or ack!r}")

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
        session_id: Optional[str] = None,
    ) -> DiffusionOutput | TokenOutput:
        # session_id (cross-request server affinity in the vLLM path) has no
        # AsyncOmni counterpart; routing stickiness is owned by the load balancer.
        if self._submission_paused and self._rejecting:
            return TokenOutput(token_ids=[], stop_reason="aborted", extra_fields={"global_steps": self.global_steps})
        timeout_s = float(os.getenv("VERL_OMNI_ADMISSION_TIMEOUT_S", str(DEFAULT_REQUEST_TIMEOUT_S)))
        async with asyncio.timeout(timeout_s):
            rejected = await self._park_until_admitted(request_id)
        if rejected is not None:
            return rejected
        submitted = _RolloutSubmission(self)

        def release_admission(_):
            self._admitting -= 1

        submitted.add_done_callback(release_admission)
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
                submitted=submitted,
            )
        finally:
            if not submitted.done():
                submitted.cancel()

    # -----------------------------------------------------------------------
    # Shared LoRA state
    # -----------------------------------------------------------------------

    def _invalidate_lora_request_cache(self) -> None:
        """Drop cached LoRA state after weight sync or engine sleep/wake."""
        self._lora_request_cache = _LORA_REQUEST_CACHE_MISS

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
            committed_id = self._committed_lora_adapter_id
            lora_id = VLLM_LORA_INT_ID if committed_id is None else committed_id
            try:
                lora_loaded = lora_id in await self.engine.list_loras()
            except TypeError:
                if committed_id is not None:
                    raise
                # Some engine backends return a non-iterable; treat as loaded.
                lora_loaded = True
            if not lora_loaded and committed_id is not None:
                raise RuntimeError(f"Missing committed LoRA adapter {committed_id}")
            self._lora_request_cache = (
                LoRARequest(
                    lora_name=VLLM_LORA_NAME if committed_id is None else str(lora_id),
                    lora_int_id=lora_id,
                    lora_path=VLLM_LORA_PATH,
                )
                if lora_loaded
                else None
            )
            return self._lora_request_cache  # type: ignore[return-value]

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass

    async def _step_handoff_rpc(self, method, timeout, *, args=(), kwargs=None):
        return await self._collective_rpc_with_ack(method, timeout, args, kwargs, [0])

    async def _drain_step_requests(self, request_ids: list[str], timeout: float) -> None:
        replies = await self._step_handoff_rpc("weight_sharing_drain", timeout, kwargs={"request_ids": request_ids})
        if replies != [True]:
            raise RuntimeError("Could not drain requests from the retiring rollout adapter")

    async def _retire_step_handoff(self, timeout: float) -> None:
        handoff = self._diffusion_step_handoff
        if handoff is None:
            return
        request_ids = sorted(handoff.request_ids)
        if request_ids:
            states = self.engine.request_states
            external_ids = [states[rid].external_request_id for rid in request_ids if rid in states]
            if external_ids:
                await asyncio.wait_for(self.engine.abort(external_ids), timeout=timeout)
            await self._drain_step_requests(request_ids, timeout)
        if await self._step_handoff_rpc("remove_lora", timeout, args=(handoff.adapter_id,)) != [True]:
            raise RuntimeError(f"Could not retire rollout adapter {handoff.adapter_id}")
        self._diffusion_step_handoff = None

    async def suspend_diffusion_steps(self, timeout: float, max_bytes: int) -> int:
        """Park resumable rollout work and reserve the other adapter slot."""
        async with self._step_handoff_lock:
            policy_ref = f"rollout-step-{self.global_steps}"
            handoff = self._diffusion_step_handoff
            if handoff is None or handoff.policy_ref != policy_ref:
                handoff = await self._suspend_diffusion_steps(policy_ref, timeout, max_bytes)
                self._diffusion_step_handoff = handoff
            return VLLM_LORA_INT_ID + int(handoff.adapter_id == VLLM_LORA_INT_ID)

    async def _suspend_diffusion_steps(self, policy_ref: str, timeout: float, max_bytes: int) -> _DiffusionStepHandoff:
        if self.engine.num_stages != 1:
            raise ValueError("Diffusion step handoff requires one rollout stage")
        self._submission_paused = True
        self._rejecting = False
        self._resume_event.clear()
        async with asyncio.timeout(timeout):
            while self._admitting:
                await asyncio.sleep(0.01)
        await self.engine.pause_generation(mode="step_boundary", timeout=timeout)

        previous = self._diffusion_step_handoff
        retired_ids = previous.request_ids if previous is not None else frozenset()
        if previous is not None:
            await self._retire_step_handoff(timeout)

        adapter = self._committed_lora_adapter_id or VLLM_LORA_INT_ID
        request_ids = frozenset(self.engine.request_states) - retired_ids
        if await self._step_handoff_rpc("pin_lora", timeout, args=(adapter,)) != [True]:
            raise RuntimeError(f"Could not pin rollout adapter {adapter}")
        snapshots = None
        if request_ids:
            options = {"policy_ref": policy_ref, "max_bytes": max_bytes}
            try:
                ((bundle,),) = await self._step_handoff_rpc("export_step_requests", timeout, kwargs=options)
                if bundle[0]:
                    await self._step_handoff_rpc("release_step_requests", timeout, args=(list(bundle[0]),))
                    snapshots = bundle
            except (RuntimeError, ValueError, TimeoutError):
                logger.warning("Diffusion step handoff falling back to whole-sample retry", exc_info=True)
                await self.abort_all_requests(reset_prefix_cache=False)
                await self._drain_step_requests(sorted(request_ids), timeout)
                request_ids = frozenset()
        logger.info("Diffusion handoff policy=%s snapshots=%d", policy_ref, len(snapshots[0]) if snapshots else 0)
        return _DiffusionStepHandoff(adapter, policy_ref, request_ids, snapshots)

    async def export_diffusion_checkpoint(self, session_keys: set[str], timeout: float, max_bytes: int) -> dict | None:
        """Collect the parked requests and the exact policy that produced them."""
        handoff = self._diffusion_step_handoff
        if handoff is None or handoff.snapshots is None:
            return None
        states = self.engine.request_states
        requests = {
            states[rid].external_request_id: rid
            for rid in handoff.snapshots[0]
            if rid in states and states[rid].external_request_id in session_keys
        }
        if not requests:
            return None
        model_fingerprint = self._diffusion_checkpoint_model_fingerprint
        if model_fingerprint is None:
            return None
        try:
            (adapters,) = await self._step_handoff_rpc(
                "export_runtime_lora", timeout, args=(handoff.adapter_id,), kwargs={"max_bytes": max_bytes}
            )
            if len(adapters) != len(handoff.snapshots) or not all(isinstance(part, bytes) for part in adapters):
                raise ValueError("Rollout checkpoint adapter topology does not match the parked requests")
        except (RuntimeError, ValueError, TimeoutError):
            logger.warning("Rollout checkpoint will use whole-sample retry", exc_info=True)
            return None
        return {
            "policy_ref": handoff.policy_ref,
            "model_fingerprint": model_fingerprint,
            "requests": requests,
            "snapshots": [{rid: rank[rid] for rid in requests.values()} for rank in handoff.snapshots],
            "adapters": adapters,
        }

    async def recover_diffusion_checkpoint(
        self, handoff, adapter_id: int, sampling_params: dict, timeout: float, max_bytes: int
    ) -> dict[str, DiffusionOutput]:
        """Finish the saved sessions while ordinary rollout admission is closed."""
        if not self._submission_paused or self._committed_lora_adapter_id in (None, adapter_id):
            raise RuntimeError("Checkpoint recovery requires closed admission and a separate committed adapter")
        engine = self.engine
        if not hasattr(engine, "restore_diffusion_step_streaming"):
            return {}
        model_fingerprint = self._diffusion_checkpoint_model_fingerprint
        if model_fingerprint is None or model_fingerprint != handoff["model_fingerprint"]:
            logger.warning("Ignoring diffusion checkpoint with a different or unverifiable base model")
            return {}
        await engine.pause_generation(mode="step_boundary", timeout=timeout)
        try:
            (replies,) = await self._step_handoff_rpc(
                "restore_runtime_lora", timeout, args=(handoff["adapters"][0],), kwargs={"max_bytes": max_bytes}
            )
            if replies != [True]:
                raise RuntimeError(f"Checkpoint LoRA restore was rejected: {replies}")
            async with asyncio.timeout(timeout), AsyncExitStack() as contexts:
                streams = {}
                for session, inner in handoff["requests"].items():
                    try:
                        streams[session] = await contexts.enter_async_context(
                            engine.restore_diffusion_step_streaming(
                                inner,
                                session,
                                [{inner: handoff["snapshots"][0][inner]}],
                                policy_ref=handoff["policy_ref"],
                                max_bytes=max_bytes,
                                timeout=timeout,
                            )
                        )
                    except (RuntimeError, ValueError, TimeoutError):
                        logger.warning("Checkpoint session %s will retry the whole sample", session, exc_info=True)
                if not streams:
                    return {}
                replies = await self._step_handoff_rpc("resume_scheduler", timeout)
                _validate_collective_rpc_reply("resume_scheduler", replies)
                outputs = await asyncio.gather(
                    *(self._generate_strategy._collect_last_output(stream) for stream in streams.values()),
                    return_exceptions=True,
                )
                version = int(handoff["policy_ref"].removeprefix("rollout-step-"))
                recovered = {}
                for session, output in zip(streams, outputs, strict=True):
                    if isinstance(output, RuntimeError | ValueError | TimeoutError) or output is None:
                        continue
                    if isinstance(output, BaseException):
                        raise output
                    sample = self._generate_strategy.process_output(output, None, sampling_params, global_steps=version)
                    if sample.stop_reason == "completed":
                        sample.extra_fields["rollout_denoising_resumed"] = True
                        recovered[session] = sample
                return recovered
        except (RuntimeError, ValueError, TimeoutError):
            logger.warning("Diffusion checkpoint will use whole-sample retry", exc_info=True)
            return {}
        finally:
            # A failed or timed-out request must settle before dropping its weights.
            await engine.pause_generation(mode="step_boundary", timeout=timeout)
            replies = await self._step_handoff_rpc("weight_sharing_drain", timeout)
            if replies != [True]:
                raise RuntimeError("Could not drain checkpoint recovery")
            replies = await self._step_handoff_rpc("remove_lora", timeout, args=(adapter_id,))
            if replies not in ([True], [False]):
                raise RuntimeError(f"Could not retire checkpoint LoRA {adapter_id}")

    async def restore_diffusion_steps(self, timeout: float, max_bytes: int) -> None:
        """Restore released denoise state before reopening admission."""
        handoff = self._diffusion_step_handoff
        assert handoff is not None
        if self._committed_lora_adapter_id in (None, handoff.adapter_id):
            raise RuntimeError("The next rollout adapter has not been committed")
        if handoff.snapshots is not None:
            try:
                options = {"policy_ref": handoff.policy_ref, "max_bytes": max_bytes}
                await self._step_handoff_rpc(
                    "restore_step_requests", timeout, args=(handoff.snapshots,), kwargs=options
                )
            except (RuntimeError, ValueError, TimeoutError):
                logger.warning("Diffusion step restore falling back to whole-sample retry", exc_info=True)
                await self.abort_all_requests(reset_prefix_cache=False)
                await self._retire_step_handoff(timeout)
                return
            logger.info(
                "Restored diffusion handoff policy=%s snapshots=%d", handoff.policy_ref, len(handoff.snapshots[0])
            )
        if handoff.request_ids & self.engine.request_states.keys():
            self._diffusion_step_handoff = replace(handoff, snapshots=None)
        else:
            await self._retire_step_handoff(timeout)

    # -----------------------------------------------------------------------
    # Abort: AsyncOmni has no `output_processor` (it routes through an
    # Orchestrator process and tracks state in `AsyncOmni.request_states`),
    # so the parent's AsyncLLM-specific implementation must be overridden.
    # -----------------------------------------------------------------------

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True, reject_request: bool = False, abort_only: bool = False
    ) -> dict[str, Any]:
        """Abort all in-flight requests on the AsyncOmni engine.

        ``abort_only`` cancels current engine requests and keeps admission open;
        the other flags shape the pause that otherwise follows. The inherited
        submission gate parks or rejects new calls until explicit resume.
        """
        if self.node_rank != 0:
            return {"aborted_count": 0, "request_ids": []}
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_all_requests(
                reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
            )
        if not abort_only:
            self._submission_paused = True
            self._rejecting = reject_request
            self._resume_event.clear()

        if not self.config.free_cache_engine and not getattr(self.config, "step_execution", False) and not abort_only:
            try:
                await engine.pause_generation(mode="keep", clear_cache=reset_prefix_cache)
            except BaseException:
                for internal_id, state in engine.request_states.items():
                    self._enqueue_abort_output(internal_id, state)
                raise

        # ``engine.abort`` takes EXTERNAL ids; ``request_states`` is keyed by internal.
        in_flight: list[tuple[str, str, Any]] = []
        seen: set[str] = set()
        initial_ids = set(engine.request_states)
        for state in engine.request_states.values():
            if not state.external_request_id or state.external_request_id in seen:
                continue
            seen.add(state.external_request_id)
            in_flight.append((state.request_id, state.external_request_id, state))

        request_ids = [external_id for _, external_id, _ in in_flight]

        if abort_only:
            try:
                await asyncio.wait_for(engine.abort(request_ids), timeout=_abort_ack_timeout_s())
            except BaseException:
                # Same cleanup as the pause path: synthetic terminals unblock waiting generators.
                for internal_id, _, state in in_flight:
                    self._enqueue_abort_output(internal_id, state)
                raise
            logger.info("Aborted %d request(s) (admission left open): %s", len(request_ids), request_ids)
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        aborted = False
        try:
            await asyncio.wait_for(engine.abort(request_ids), timeout=_abort_ack_timeout_s())
            aborted = True
            # Pause even with nothing to abort: holds admission until resume_generation.
            await engine.pause_generation(
                mode="abort", wait_for_inflight_requests=False, clear_cache=reset_prefix_cache
            )
        except BaseException:
            # If the engine abort RPC fails or times out, upstream has not
            # delivered terminals to waiting generator coroutines. Enqueue
            # synthetic abort outputs so active generators unblock and exit.
            if not aborted:
                for internal_id, _, state in in_flight:
                    self._enqueue_abort_output(internal_id, state)
            raise

        stragglers = {
            state.external_request_id: state
            for state in engine.request_states.values()
            if state.external_request_id and state.request_id not in initial_ids
        }
        if stragglers:
            try:
                await asyncio.wait_for(engine.abort(list(stragglers)), timeout=_abort_ack_timeout_s())
            except BaseException:
                for state in stragglers.values():
                    self._enqueue_abort_output(state.request_id, state)
                raise
            request_ids.extend(stragglers)

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
            await asyncio.wait_for(engine.abort(request_id), timeout=_abort_ack_timeout_s())
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
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"


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
        if is_teacher_model and isinstance(model_config, dict | DictConfig):
            engine_kwargs = (config.engine_kwargs or {}).get("vllm_omni") or {}
            if engine_kwargs.get("output_mode") == "ar":
                # Checkpoint workers construct the model config before the HTTP server does.
                model_config = OmegaConf.merge(
                    {"trust_remote_code": engine_kwargs.get("trust_remote_code", False)},
                    model_config,
                    {"_target_": "verl_omni.workers.config.OmniModelConfig"},
                )
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
