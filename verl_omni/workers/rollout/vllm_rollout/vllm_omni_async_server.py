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

        # In vLLM 0.27, asdict converts the default FaultToleranceConfig dataclass into a dict.
        # OmniEngineArgs.__post_init__ auto-enables enable_fault_tolerance when fault_tolerance_config
        # is a dict, which causes create_engine_config to fail without an external load balancer.
        # Strip fault_tolerance_config when fault tolerance was not explicitly enabled.
        if not engine_args.get("enable_fault_tolerance"):
            engine_args.pop("fault_tolerance_config", None)

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
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    def _resolve_sleep_level(self) -> int:
        """
        # TODO (andy): use sleep_level=2 when vllm-omni implements wake_up
        after level-2 sleep AND the trainer syncs the full pipeline.
        """
        return 1

    async def wake_up(self, tags: list[str] | None = None):
        """Override parent to use collective_rpc instead of engine.wake_up().

        The parent (verl ``1927ad33``+) calls ``self.engine.wake_up(tags=...)``
        which triggers CUDA initialisation in this HTTP server process when
        running under vLLM-Omni (AsyncOmni engine).
        Use ``collective_rpc`` instead.

        # TODO (long): drop this override once vllm-omni wake_up
        without triggering GPU initialisation.
        """
        if self.node_rank != 0:
            return
        await self.engine.collective_rpc(
            "wake_up", kwargs={"tags": tags if tags is not None else self._get_wake_up_tags()}
        )
        self._invalidate_lora_request_cache()

    async def set_global_steps(self, global_steps: int):
        if global_steps != self.global_steps:
            self._invalidate_lora_request_cache()
        await super().set_global_steps(global_steps)

    async def _sleep_hybrid(self):
        """Preserve non-actor pipeline weights during hybrid training sleep.

        vLLM-Omni diffusion pipelines include components such as the text
        encoder and VAE that are loaded by the rollout server, but are not part
        of the trainable actor and therefore are not included in full-model
        weight syncs. Use level-1 sleep so those weights are offloaded and can
        be restored on wake-up instead of discarded by level-2 sleep.
        """
        self._invalidate_lora_request_cache()
        await self.engine.collective_rpc("sleep", kwargs={"level": self._resolve_sleep_level()})
        await self.engine.reset_encoder_cache()

    async def release_kv_cache(self):
        """Free cache around a weight sync without discarding Omni weights."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        self._invalidate_lora_request_cache()
        await self.engine.collective_rpc("sleep", kwargs={"level": self._resolve_sleep_level()})
        await self.engine.collective_rpc("wake_up", kwargs={"tags": ["weights"]})

    async def resume_kv_cache(self):
        """Restore after a weight sync. Route through collective_rpc like wake_up."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        await self.engine.collective_rpc("wake_up", kwargs={"tags": ["kv_cache"]})

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
        # TODO (mike): implement this once DP is supported.
        pass

    # -----------------------------------------------------------------------
    # Abort: AsyncOmni has no `output_processor` (it routes through an
    # Orchestrator process and tracks state in `AsyncOmni.request_states`),
    # so the parent's AsyncLLM-specific implementation must be overridden.
    # -----------------------------------------------------------------------

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all in-flight requests on the AsyncOmni engine.

        During ``on_step_end`` no new prompts are fed (the feed happens in
        ``step``/``_add_batch_to_generate``), so the in-flight set monotonically
        drains and the drain terminates quickly in practice.
        """
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_all_requests(reset_prefix_cache)

        try:
            # ---- Phase 1: drain in-flight requests naturally ----------------
            # Letting requests finish avoids the Orchestrator race that produces
            # "Dropping output for unknown req" and avoids whole-sample retries.
            drain_timeout_s = float(os.getenv("VERL_OMNI_ABORT_DRAIN_TIMEOUT_S", "120"))
            drain_poll_interval_s = 0.1
            drained = False
            drain_start = time.monotonic()
            last_count = -1
            while True:
                in_flight = len(engine.request_states)
                if in_flight == 0:
                    drained = True
                    break
                if time.monotonic() - drain_start >= drain_timeout_s:
                    logger.warning(
                        "abort_all_requests: drain timed out after %.1fs with %d request(s) still in-flight; "
                        "falling back to hard abort (these may produce 'Dropping output' warnings)",
                        drain_timeout_s,
                        in_flight,
                    )
                    break
                if in_flight != last_count:
                    logger.info(
                        "abort_all_requests: draining %d in-flight request(s) (%.1fs elapsed)",
                        in_flight,
                        time.monotonic() - drain_start,
                    )
                    last_count = in_flight
                await asyncio.sleep(drain_poll_interval_s)

            if drained:
                if reset_prefix_cache:
                    await self.clear_kv_cache()
                logger.info(
                    "abort_all_requests: drained all in-flight requests in %.2fs; no abort needed",
                    time.monotonic() - drain_start,
                )
                return {"aborted_count": 0, "request_ids": [], "drained": True}

            # ---- Phase 2: hard-abort the remainder (drain timed out) ---------
            # Snapshot in-flight states (internal_id -> ClientRequestState) BEFORE
            # engine.abort() pops them from AsyncOmni.request_states. We need the
            # per-request asyncio.Queue references to unblock the generate() coroutines.
            in_flight_states: list[tuple[str, Any]] = []
            seen: set[str] = set()
            for state in engine.request_states.values():
                ext = getattr(state, "external_request_id", None)
                if ext is None or ext in seen:
                    continue
                seen.add(ext)
                in_flight_states.append((state.request_id, state))

            request_ids = [s.external_request_id for _, s in in_flight_states]

            if request_ids:
                await engine.abort(request_ids)

            # Synthesize terminal abort OutputMessages and put them directly into
            # each per-request queue so _process_orchestrator_results can drain
            # and return. Without this, generate() hangs forever because the
            # Orchestrator already dropped the real abort output.
            for internal_id, state in in_flight_states:
                self._enqueue_abort_output(internal_id, state)

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}
        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    def _enqueue_abort_output(self, internal_id: str, req_state: Any) -> None:
        """Synthesize a terminal abort OutputMessage and put it into a per-request queue.

        ``_process_orchestrator_results`` reads from ``req_state.queue`` and
        expects ``OutputMessage`` (or ``ErrorMessage``) objects. We build a
        minimal ``OmniRequestOutput`` with ``finish_reason="abort"`` so that
        ``_process_single_result`` yields it and the active generation strategy maps it
        to ``stop_reason="aborted"``.
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
        """Abort a single in-flight request on the AsyncOmni engine."""
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_request(request_id, reset_prefix_cache)

        try:
            # Snapshot the in-flight state before engine.abort() pops it, so we
            # can unblock the generate() coroutine with a synthetic abort output.
            in_flight_state = None
            for state in engine.request_states.values():
                if getattr(state, "external_request_id", None) == request_id:
                    in_flight_state = state
                    break

            await engine.abort(request_id)

            if in_flight_state is not None:
                self._enqueue_abort_output(in_flight_state.request_id, in_flight_state)

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")
            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}
        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}


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
