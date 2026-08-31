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
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Optional

import ray
import torch
import vllm_omni.entrypoints.cli.serve
from verl.utils.net_utils import get_free_port
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs, orchestrator_field_names
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.lora.request import LoRARequest

from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig, OmniModelConfig
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Sentinel: ``None`` is a valid cached value (LoRA not loaded).
_LORA_REQUEST_CACHE_MISS = object()

# Bound for the acked abort (``AsyncOmni.abort()``'s correlated RPC has no default
# timeout) and for the pre-sleep request drain. Replaces the old drain-phase
# timeout ``VERL_OMNI_ABORT_DRAIN_TIMEOUT_S``.
ABORT_ACK_TIMEOUT_S = float(os.getenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "120"))
_DRAIN_POLL_INTERVAL_S = 0.1


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
        # Startup probe: a platform that cannot even be imported must fail the
        # server launch, not silently fall back to the "" device type (#388 B4).
        from vllm.platforms import current_platform

        return self._generate_strategy.worker_extension_cls(current_platform.device_type)

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

        # ``from_cli_args`` only retains OmniEngineArgs fields. Restore the
        # OrchestratorArgs fields forwarded by verl before creating AsyncOmni.
        for key in orchestrator_field_names() - engine_args.keys():
            value = getattr(args, key, None)
            if value is not None:
                engine_args[key] = value

        deploy_config = getattr(args, "deploy_config", None)
        if deploy_config:
            engine_args["deploy_config"] = deploy_config

        self._generate_strategy.prepare_engine_args(engine_args, args)

        if getattr(self.config, "step_execution", False):
            engine_args["step_execution"] = True

        if engine_args.get("seed") is None:
            engine_args.pop("seed", None)

        diffusion_master_port, diffusion_master_sock = get_free_port("127.0.0.1", with_alive_sock=True)
        diffusion_master_sock.close()

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
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    # -----------------------------------------------------------------------
    # Sleep / wake lifecycle: delegated to AsyncOmni and ACK-validated
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    def _resolve_sleep_level(self) -> int:
        """
        # TODO (andy): use sleep_level=2 when vllm-omni implements wake_up
        after level-2 sleep AND the trainer syncs the full pipeline.
        """
        return 1

    def _validate_acks(self, method: str, acks: Any) -> None:
        """Fail closed on any non-success sleep/wake handshake (#433).

        ``AsyncOmni.sleep()/wake_up()`` re-raise AR EngineCore RPC failures, but
        diffusion handlers return ``OmniACK(status="ERROR")`` / ``{"supported":
        False, "error": ...}`` entries that the engine iterates without checking
        — for diffusion/mixed engines this validation is the only barrier.
        An empty ACK list is success ("already warm").
        """
        for ack in acks or []:
            if isinstance(ack, dict):
                if "error" in ack:
                    raise RuntimeError(f"{method} failed on a stage: {ack['error']}")
            elif getattr(ack, "status", None) != "SUCCESS":
                raise RuntimeError(f"{method} failed on a stage: {getattr(ack, 'error_msg', None) or ack!r}")

    async def _delegated_sleep(self) -> None:
        """Level-1 sleep through ``AsyncOmni.sleep()`` in every replica mode.

        Level 1 keeps non-actor pipeline weights (text encoder, VAE) offloaded
        and restorable — level 2 discards them and its wake is not implemented
        upstream. The engine sets an admission hold that only
        ``resume_generation()`` clears; the trainer-side bridge owns that, so
        this method must not resume by itself.
        """
        acks = await self.engine.sleep(level=self._resolve_sleep_level())
        self._validate_acks("sleep", acks)
        self._invalidate_lora_request_cache()
        # Frontend no-op on the omni engine (parity with verl's shape); the real
        # encoder-cache safety is EngineCore-side sleep clearing.
        await self.engine.reset_encoder_cache()

    async def _delegated_wake(self, tags: list[str] | None = None) -> None:
        """Wake through ``AsyncOmni.wake_up()`` (keyword ``tags=`` only).

        The first positional parameter of ``AsyncOmni.wake_up`` is ``stage_ids``
        — passing tags positionally would silently bind to the wrong argument.
        Does NOT call ``resume_generation``: that would re-open admission
        mid-weight-sync on the non-naive path (#492 §5.2).
        """
        resolved_tags = tags if tags is not None else self._get_wake_up_tags()
        acks = await self.engine.wake_up(tags=resolved_tags)
        self._validate_acks("wake_up", acks)
        self._invalidate_lora_request_cache()
        # Frontend no-op on the omni engine, kept for upstream-shape parity.
        await self.engine.reset_prefix_cache(reset_connector=True)

    async def wake_up(self, tags: list[str] | None = None):
        if self.node_rank != 0:
            return
        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")
            return
        await self._delegated_wake(tags)

    async def set_global_steps(self, global_steps: int):
        if global_steps != self.global_steps:
            self._invalidate_lora_request_cache()
        await super().set_global_steps(global_steps)

    async def sleep(self):
        """Level-1 delegated sleep in HYBRID and COLOCATED modes alike.

        The parent dispatches COLOCATED to ``engine.sleep(level=1)`` whose
        wake never clears the frontend admission hold, while HYBRID used raw
        ``collective_rpc`` — routing both modes through ``_delegated_sleep``
        makes the lifecycle uniform and ACK-validated.
        """
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")
            return
        await self._delegated_sleep()

    async def release_kv_cache(self):
        """Free cache around a weight sync without discarding Omni weights."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        await self._delegated_sleep()
        await self._delegated_wake(tags=["weights"])

    async def resume_kv_cache(self):
        """Restore after a weight sync. Both halves of the split are validated."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        await self._delegated_wake(tags=["kv_cache"])

    async def resume_generation(self):
        if self.node_rank == 0:
            await self.engine.resume_generation()

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

    async def wait_for_requests_to_drain(self):
        """Bounded drain: no in-flight request states may remain.

        ``AsyncOmni`` has no drain API. The acked abort in ``abort_all_requests``
        is the real quiesce; this poll is the fail-closed backstop — EngineCore
        auto-resumes leftover AR requests on a full wake, so callers (e.g.
        ``sleep_replicas``) must not proceed past a lingering request.
        """
        deadline = time.monotonic() + ABORT_ACK_TIMEOUT_S
        while self.engine.request_states:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"rollout engine still has {len(self.engine.request_states)} in-flight "
                    f"request(s) after {ABORT_ACK_TIMEOUT_S:.0f}s; refusing to proceed (#433)"
                )
            await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)

    # -----------------------------------------------------------------------
    # Abort: AsyncOmni has no `output_processor` (it routes through an
    # Orchestrator process and tracks state in `AsyncOmni.request_states`),
    # so the parent's AsyncLLM-specific implementation must be overridden.
    # -----------------------------------------------------------------------

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort every in-flight request, then pause — abort-then-pause.

        ``AsyncOmni.pause_generation`` never delivers tokens, so the batched
        ``abort()`` must run while generate is still live: it enqueues the real
        cumulative partial-token terminals (``finish_reason="abort"``) into each
        per-request queue. Pausing first would let EngineCore finish the requests
        ``FINISHED_ABORTED`` and the follow-up abort would synthesize *empty*
        ``token_ids`` instead. The pause afterwards is the idle boundary plus the
        admission hold that stays until ``resume_generation``.

        Known straggler window: a ``generate()`` that already passed the pause
        check between the abort and the pause is finished EngineCore-side with
        empty partials — the client falls back to whole-sample retry (the same
        degradation diffusion always gets), never a hang.
        """
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_all_requests(reset_prefix_cache)

        # Snapshot in-flight states (dedup by external id) BEFORE the abort:
        # the failure path needs the queue references, and ``engine.abort``
        # takes EXTERNAL ids while ``request_states`` is keyed by internal ones.
        in_flight: list[tuple[str, str, Any]] = []
        seen: set[str] = set()
        for state in engine.request_states.values():
            external_id = getattr(state, "external_request_id", None)
            if external_id is None or external_id in seen:
                continue
            seen.add(external_id)
            in_flight.append((state.request_id, external_id, state))

        request_ids = [external_id for _, external_id, _ in in_flight]

        aborted = False
        try:
            if request_ids:
                # One batched acked abort — never per-request loops (each call
                # is a correlated RPC) and never an empty list (a silent no-op).
                await asyncio.wait_for(engine.abort(request_ids), timeout=ABORT_ACK_TIMEOUT_S)
            aborted = True
            # Always pause, even with nothing to abort: the admission hold is
            # what keeps the engine quiesced until resume_generation. EngineCore
            # ``pause_scheduler(clear_cache=...)`` performs the real cache reset.
            await engine.pause_generation(
                mode="abort", wait_for_inflight_requests=False, clear_cache=reset_prefix_cache
            )
        except Exception:
            # Fail closed (#433/#388) — but never strand an in-flight generate()
            # on ``queue.get``: if the abort itself failed, nothing engine-side
            # enqueued terminals, so synthesize them before re-raising.
            if not aborted:
                for internal_id, _, state in in_flight:
                    self._enqueue_emergency_abort_output(internal_id, state)
            raise

        logger.info("Aborted %d request(s): %s", len(request_ids), request_ids)
        return {"aborted_count": len(request_ids), "request_ids": request_ids}

    def _enqueue_emergency_abort_output(self, internal_id: str, req_state: Any) -> None:
        """FAILURE PATH ONLY: synthesize a terminal so generate() cannot hang.

        On the success path the engine's acked abort already enqueues real
        cumulative partial-token terminals (plus its own synthetic fallback for
        requests without output). This helper exists solely for the case where
        ``engine.abort()`` itself raised — nothing was enqueued, and in-flight
        ``generate()`` coroutines would otherwise block forever on
        ``req_state.queue.get``.
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
        req_state.queue.put_nowait(msg)

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a single in-flight request — same abort-then-pause contract."""
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_request(request_id, reset_prefix_cache)

        in_flight_state = None
        for state in engine.request_states.values():
            if getattr(state, "external_request_id", None) == request_id:
                in_flight_state = state
                break

        aborted = False
        try:
            if in_flight_state is not None:
                await asyncio.wait_for(engine.abort(request_id), timeout=ABORT_ACK_TIMEOUT_S)
            aborted = True
            await engine.pause_generation(
                mode="abort", wait_for_inflight_requests=False, clear_cache=reset_prefix_cache
            )
        except Exception:
            if not aborted and in_flight_state is not None:
                self._enqueue_emergency_abort_output(in_flight_state.request_id, in_flight_state)
            raise

        logger.info("Aborted request: %s", request_id)
        return {"aborted": True, "request_id": request_id}


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig | RolloutConfig,
        model_config: DiffusionModelConfig | OmniModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
