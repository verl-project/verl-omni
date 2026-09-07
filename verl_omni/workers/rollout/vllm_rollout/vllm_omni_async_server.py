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
        if self.node_rank != 0:
            return
        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")
            return
        resolved_tags = tags if tags is not None else self._get_wake_up_tags()
        acks = await self.engine.wake_up(tags=resolved_tags)
        self._validate_acks("wake_up", acks)
        await self.engine.resume_generation()
        self._invalidate_lora_request_cache()

    async def set_global_steps(self, global_steps: int):
        if global_steps != self.global_steps:
            self._invalidate_lora_request_cache()
        await super().set_global_steps(global_steps)

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
        acks = await self.engine.sleep(level=self._resolve_sleep_level())
        self._validate_acks("sleep", acks)
        await self._reset_frontend_mm_cache()
        self._invalidate_lora_request_cache()

    async def release_kv_cache(self):
        """Free cache around a weight sync without discarding Omni weights."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        acks = await self.engine.sleep(level=self._resolve_sleep_level())
        self._validate_acks("sleep", acks)
        await self._reset_frontend_mm_cache()
        self._invalidate_lora_request_cache()
        acks = await self.engine.wake_up(tags=["weights"])
        self._validate_acks("wake_up", acks)
        await self.engine.resume_generation()
        self._invalidate_lora_request_cache()

    async def resume_kv_cache(self):
        """Restore after a weight sync."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        acks = await self.engine.wake_up(tags=["kv_cache"])
        self._validate_acks("wake_up", acks)
        await self.engine.resume_generation()
        self._invalidate_lora_request_cache()

    async def resume_generation(self):
        if self.node_rank == 0:
            await self.engine.resume_generation()

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
        """Abort all in-flight requests on the AsyncOmni engine."""
        engine = self.engine
        if getattr(engine, "output_processor", None) is not None:
            return await super().abort_all_requests(reset_prefix_cache)

        # ``engine.abort`` takes EXTERNAL ids; ``request_states`` is keyed by internal.
        in_flight: list[tuple[str, str, Any]] = []
        seen: set[str] = set()
        for state in engine.request_states.values():
            if state.external_request_id in seen:
                continue
            seen.add(state.external_request_id)
            in_flight.append((state.request_id, state.external_request_id, state))

        request_ids = [external_id for _, external_id, _ in in_flight]

        aborted = False
        try:
            # TODO (mike): multi-stage AR abort is broken upstream — the engine's
            # abort fallback terminal is stage_id=0 and the consume loop breaks on
            # finished non-final messages, so generate() exits empty. Single-stage /
            # thinker-only is correct here; needs a vllm-omni fix + pin bump.
            await asyncio.wait_for(
                engine.abort(request_ids), timeout=float(os.getenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "120"))
            )
            aborted = True
            # Pause even with nothing to abort: holds admission until resume_generation.
            await engine.pause_generation(
                mode="abort", wait_for_inflight_requests=False, clear_cache=reset_prefix_cache
            )
        except Exception:
            # Nothing engine-side enqueued terminals — synthesize them so
            # generate() cannot hang on queue.get.
            if not aborted:
                for internal_id, _, state in in_flight:
                    self._enqueue_abort_output(internal_id, state)
            raise

        if reset_prefix_cache:
            # pause_generation(clear_cache=True) wiped the engine-side mm cache;
            # drop the frontend copy too, or hash-only follow-ups finish empty.
            # TODO (mike): drop after vllm-omni fixes AsyncOmni.reset_mm_cache.
            await self._reset_frontend_mm_cache()

        logger.info("Aborted %d request(s): %s", len(request_ids), request_ids)
        return {"aborted_count": len(request_ids), "request_ids": request_ids}

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
