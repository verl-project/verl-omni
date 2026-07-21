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

import numpy as np
import ray
import torch
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.net_utils import get_free_port
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.cli.serve import run_headless as run_omni_headless
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.tracking_parser import TrackingNamespace

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.placement_guard import (
    estimate_outer_rollout_replicas,
    validate_vllm_omni_rollout_placement,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        """AR mode uses HFModelConfig; diffusion uses DiffusionModelConfig.

        Mode is selected by ``engine_kwargs.vllm_omni.output_mode`` ("ar" vs the
        default "diffusion").
        """
        engine_kwargs = getattr(self.config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        self._ar_mode = omni_kwargs.get("output_mode", "diffusion") == "ar"

        if self._ar_mode:
            return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """AR mode: derive max_model_len. Diffusion: no max_position_embeddings."""
        if self._ar_mode:
            if self.config.max_model_len is None:
                self.config.max_model_len = self.config.prompt_length + self.config.response_length
            logprobs_mode = (
                self.config.get("logprobs_mode", "processed_logprobs")
                if isinstance(self.config, dict)
                else getattr(self.config, "logprobs_mode", "processed_logprobs")
            )
            calculate_log_probs = (
                self.config.get("calculate_log_probs", False)
                if isinstance(self.config, dict)
                else getattr(self.config, "calculate_log_probs", False)
            )
            if calculate_log_probs and logprobs_mode not in {
                "raw_logprobs",
                "processed_logprobs",
            }:
                raise ValueError(
                    "vLLM-Omni AR rollout requires an explicit raw_logprobs or "
                    f"processed_logprobs mode, got {logprobs_mode!r}"
                )

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Diffusion needs a PIL→tensor converter; AR does not."""
        if not self._ar_mode:
            self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """AR mode uses the parent's LLM sampling config; diffusion has none."""
        if self._ar_mode:
            return vLLMHttpServer._get_override_generation_config(self)
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        return "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Strip the mode selector; in AR mode also drop diffusion-only kwargs and
        normalize underscore keys vLLM-Omni expects with dashes."""
        engine_kwargs.pop("output_mode", None)
        if self._ar_mode:
            engine_kwargs.pop("custom_pipeline", None)
            stage_init_timeout = engine_kwargs.get("stage_init_timeout") or engine_kwargs.get("stage-init-timeout")
            init_timeout = engine_kwargs.get("init_timeout") or engine_kwargs.get("init-timeout")
            if stage_init_timeout is not None:
                stage_init_timeout = int(stage_init_timeout)
                os.environ.setdefault("VLLM_OMNI_STARTUP_HANDSHAKE_TIMEOUT", str(stage_init_timeout))
                if init_timeout is None:
                    engine_kwargs["init_timeout"] = max(stage_init_timeout, 600)

            for underscore_key in (
                "stage_configs_path",
                "deploy_config",
                "stage_overrides",
                "async_chunk",
                "stage_init_timeout",
                "init_timeout",
            ):
                if underscore_key in engine_kwargs:
                    engine_kwargs[underscore_key.replace("_", "-")] = engine_kwargs.pop(underscore_key)

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _ensure_tracking_namespace(args: argparse.Namespace) -> argparse.Namespace:
        if hasattr(args, "get_explicit_kwargs_dict"):
            return args
        return TrackingNamespace(unfiltered_ns=args, explicit_keys=frozenset(vars(args)))

    @staticmethod
    def _visible_device_count() -> int | None:
        raw_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
        if not raw_devices:
            return None
        return len([device for device in raw_devices.split(",") if device.strip()])

    @staticmethod
    def _config_int(config: Any, name: str, default: int = 1) -> int:
        try:
            value = getattr(config, name)
        except (AttributeError, KeyError):
            value = config.get(name, default) if isinstance(config, dict) else default
        return int(value or default)

    def _run_ar_placement_preflight(self, args: argparse.Namespace):
        outer_replicas = estimate_outer_rollout_replicas(
            nnodes=int(getattr(self, "nnodes", 1) or 1),
            gpus_per_node=int(getattr(self, "gpus_per_node", 1) or 1),
            tensor_model_parallel_size=self._config_int(self.config, "tensor_model_parallel_size"),
            data_parallel_size=self._config_int(self.config, "data_parallel_size"),
            pipeline_model_parallel_size=self._config_int(self.config, "pipeline_model_parallel_size"),
        )
        preflight = validate_vllm_omni_rollout_placement(
            stage_configs_path=getattr(args, "stage_configs_path", None),
            outer_replicas=outer_replicas,
            visible_device_count=self._visible_device_count(),
            allow_physical_stage_devices=self._env_flag("VERL_OMNI_ALLOW_PHYSICAL_STAGE_DEVICES"),
        )
        logger.info("vLLM-Omni rollout placement preflight: %s", preflight)
        return preflight

    def _configure_omni_distributed_args(self, args: argparse.Namespace, *, headless: bool) -> None:
        """Map verl's multi-node replica contract onto vLLM-Omni launch args."""
        if self.nnodes <= 1 or not self._ar_mode:
            return

        omni_master_port = int(os.environ.get("VERL_OMNI_MASTER_ZMQ_PORT", self._master_port))
        dist_master_port = os.environ.get("VLLM_OMNI_DIST_MASTER_PORT")
        if dist_master_port:
            args.master_addr = self._master_address
            args.master_port = int(dist_master_port)

        args.stage_id = 0
        args.omni_master_address = self._master_address
        args.omni_master_port = omni_master_port
        args.omni_dp_size_local = 1
        args.worker_backend = "multi_process"
        args.headless = headless
        if getattr(args, "omni_lb_policy", None) is None:
            args.omni_lb_policy = "random"
        if getattr(args, "omni_heartbeat_timeout", None) is None:
            args.omni_heartbeat_timeout = 30.0

    def _release_omni_master_port_reservation(self) -> None:
        sock = getattr(self, "_master_sock", None)
        if sock is not None:
            sock.close()
            self._master_sock = None

    async def run_server(self, args: argparse.Namespace):
        args = self._ensure_tracking_namespace(args)
        self._configure_omni_distributed_args(args, headless=False)
        if self.nnodes > 1 and self._ar_mode:
            self._release_omni_master_port_reservation()
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        if self._ar_mode:
            self._run_ar_placement_preflight(args)
            os.environ.setdefault("VLLM_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE", "1")
            for timeout_key in ("stage_init_timeout", "init_timeout"):
                timeout_value = getattr(args, timeout_key, None)
                if timeout_value is not None:
                    engine_args[timeout_key] = int(timeout_value)
            engine_args["logprobs_mode"] = getattr(self.config, "logprobs_mode", "processed_logprobs")
            # AR mode: no diffusion pipeline. Drop None entries from
            # compilation_config that OmniEngineArgs may leave behind.
            if isinstance(engine_args.get("compilation_config"), dict):
                engine_args["compilation_config"] = {
                    k: v for k, v in engine_args["compilation_config"].items() if v is not None
                }
        else:
            # inject multi-stage yaml config
            deploy_config = getattr(args, "deploy_config", None)
            if deploy_config:
                engine_args["deploy_config"] = deploy_config

            import_external_libs(self.config.external_lib)

            self.config.resolve_algorithm(self.model_config)

            pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
                architecture=self.model_config.architecture,
                algorithm=self.model_config.algorithm,
            )
            # TODO (mike): read custom_pipeline from engine_args
            if pipeline_path is not None:
                engine_args["enable_dummy_pipeline"] = True
                engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

        if getattr(self.config, "step_execution", False):
            engine_args["step_execution"] = True

        diffusion_master_port, diffusion_master_sock = get_free_port("127.0.0.1", with_alive_sock=True)
        diffusion_master_sock.close()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(diffusion_master_port)
        logger.info("Using MASTER_PORT=%s for vLLM-Omni workers", os.environ["MASTER_PORT"])

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run a remote vLLM-Omni stage replica in a background thread."""
        args = self._ensure_tracking_namespace(args)
        self._configure_omni_distributed_args(args, headless=True)
        if self._ar_mode:
            self._run_ar_placement_preflight(args)
            os.environ.setdefault("VLLM_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE", "1")
        args.api_server_count = 0

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_omni_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc is not None:
                    logger.exception("vLLM-Omni headless server failed: %s", exc)
                else:
                    logger.error("vLLM-Omni headless server exited unexpectedly")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

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

    async def _sleep_hybrid(self):
        """Preserve non-actor pipeline weights during hybrid training sleep.

        vLLM-Omni diffusion pipelines include components such as the text
        encoder and VAE that are loaded by the rollout server, but are not part
        of the trainable actor and therefore are not included in full-model
        weight syncs. Use level-1 sleep so those weights are offloaded and can
        be restored on wake-up instead of discarded by level-2 sleep.
        """
        # TODO (andy): use `sleep_level=2` in the future when the
        #  trainer side incorporates the whole components of the model.
        await self.engine.collective_rpc("sleep", kwargs={"level": 1})
        await self.engine.reset_encoder_cache()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort every in-flight AsyncOmni request owned by the head replica."""
        if self.node_rank != 0:
            return {"aborted_count": 0, "request_ids": []}

        request_ids = list(getattr(self.engine, "request_states", {}))
        if request_ids:
            await self.engine._abort_internal_requests(request_ids)
        if reset_prefix_cache:
            await self.engine.reset_prefix_cache()
        return {"aborted_count": len(request_ids), "request_ids": request_ids}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort one AsyncOmni request by internal or external request id."""
        if self.node_rank != 0:
            return {"aborted": False, "request_id": request_id}

        request_states = getattr(self.engine, "request_states", {})
        if request_id in request_states:
            await self.engine._abort_internal_requests([request_id])
        elif any(getattr(state, "external_request_id", None) == request_id for state in request_states.values()):
            await self.engine.abort(request_id)
        else:
            return {"aborted": False, "request_id": request_id, "error": "request not found"}
        if reset_prefix_cache:
            await self.engine.reset_prefix_cache()
        return {"aborted": True, "request_id": request_id}

    async def resume_generation(self):
        """AsyncOmni aborts requests directly and does not pause the engine."""
        return

    # -----------------------------------------------------------------------
    # generate: shared pipeline; mode-specific steps branch on self._ar_mode
    # (_preprocess_input / _run_generation / _process_output).
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
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        prompt_ids = normalize_token_ids(prompt_ids)
        self._validate_generate_multimodal_args(
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        multi_modal_data = self._build_multi_modal_data(image_data, video_data)
        lora_request = await self._resolve_lora_request()
        prompt, params = self._preprocess_input(
            prompt_ids, sampling_params, multi_modal_data, lora_request, negative_prompt_ids, prompt_mask
        )
        final_res = await self._run_generation(prompt, params, request_id, lora_request, priority)
        return self._process_output(final_res, params, sampling_params)

    # -----------------------------------------------------------------------
    # Shared helpers for the AR and diffusion generate paths
    # -----------------------------------------------------------------------

    def _validate_generate_multimodal_args(
        self,
        *,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
    ) -> None:
        provided = {
            "image_data": image_data,
            "video_data": video_data,
            "audio_data": audio_data,
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        if self._ar_mode:
            unsupported = [name for name, value in provided.items() if value]
        else:
            unsupported = [name for name in ("audio_data", "mm_processor_kwargs") if provided[name]]
        if unsupported:
            mode = "AR text" if self._ar_mode else "diffusion"
            raise NotImplementedError(f"vLLM-Omni {mode} rollout does not support: {', '.join(unsupported)}")

    @staticmethod
    def _build_multi_modal_data(image_data: Optional[list[Any]], video_data: Optional[list[Any]]) -> dict[str, Any]:
        """Assemble the vLLM multi_modal_data dict from optional image/video inputs."""
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        return multi_modal_data

    async def _resolve_lora_request(self) -> Optional[LoRARequest]:
        """Build the actor LoRA request if a LoRA adapter is currently loaded.

        Wraps ``list_loras`` in ``try/except TypeError`` (a strict superset of the
        plain membership check): some engine backends return a non-iterable, in
        which case we assume the adapter is loaded. The diffusion path is unchanged
        in the normal (iterable) case.
        """
        if not self.lora_as_adapter:
            return None
        try:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
        except TypeError:
            lora_loaded = True
        if not lora_loaded:
            return None
        return LoRARequest(lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)

    @staticmethod
    def _map_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
        """Map a vLLM finish_reason to verl's stop_reason vocabulary."""
        if finish_reason == "abort":
            return "aborted"
        if finish_reason in ("stop", "length"):
            return "completed"
        return finish_reason

    # -----------------------------------------------------------------------
    # Mode-specific pipeline steps
    # -----------------------------------------------------------------------

    def _preprocess_input(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        lora_request: Optional[LoRARequest],
        negative_prompt_ids: Optional[list[int]],
        prompt_mask: torch.BoolTensor | None = None,
    ):
        """Build the engine prompt + sampling params for the active mode.

        Returns ``(prompt, params)`` consumed by ``_run_generation``.
        """
        if self._ar_mode:
            max_possible_tokens = self.config.max_model_len - len(prompt_ids)
            if max_possible_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids)}) meets or exceeds the model's maximum context length "
                    f"({self.config.max_model_len}), leaving no space for generation."
                )

            if "max_tokens" in sampling_params:
                max_tokens = sampling_params.pop("max_tokens")
            elif "max_new_tokens" in sampling_params:
                max_tokens = sampling_params.pop("max_new_tokens")
            else:
                max_tokens = min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_ids),
                )
            max_tokens = max(0, min(max_tokens, max_possible_tokens))

            # Normalize ``logprobs``: bare ``True`` -> 0 (sampled-token logprob),
            # preserve explicit int counts (incl. 0), fall back to None otherwise.
            logprobs = sampling_params.pop("logprobs", None)
            if logprobs is True:
                sampling_params["logprobs"] = 0
            elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
                sampling_params["logprobs"] = logprobs
            else:
                sampling_params["logprobs"] = None
            sampling_params.setdefault("repetition_penalty", getattr(self.config, "repetition_penalty", 1.0))
            params = SamplingParams(max_tokens=max_tokens, **sampling_params)

            prompt = {"prompt_token_ids": prompt_ids}
            if multi_modal_data:
                prompt["multi_modal_data"] = multi_modal_data
            return prompt, params

        # diffusion
        default_params_list = self.engine.default_sampling_params_list

        custom_prompt: OmniCustomPrompt = {"prompt_token_ids": prompt_ids}
        if prompt_mask is not None:
            custom_prompt["prompt_mask"] = prompt_mask
        if len(default_params_list) > 1:
            # Multi-stage pipelines tag the diffusion stage so the orchestrator can route inputs correctly.
            custom_prompt["modalities"] = ["image"]
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["multi_modal_data"] = multi_modal_data
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)
        # Multi-stage models use defaults for non-diffusion stages.
        params = default_params_list[:-1] + [diffusion_sampling_params]
        return custom_prompt, params

    async def _run_generation(self, prompt, params, request_id: str, lora_request, priority: int):
        """Drive the engine and return the final OmniRequestOutput."""
        if self._ar_mode:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
        else:
            generator = self.engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=params,
            )
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        return final_res

    def _process_output(self, final_res, params, sampling_params: dict[str, Any]):
        """Convert the engine output into the active mode's verl output dataclass."""
        if self._ar_mode:
            if final_res is None:
                raise RuntimeError("AR mode: vLLM-Omni engine yielded no output for the prompt.")

            req_output = final_res.request_output
            if req_output is None:
                raise RuntimeError("AR mode expects request_output with token IDs, but got None.")

            extra_fields = {"global_steps": self.global_steps}
            token_ids = req_output.outputs[0].token_ids
            log_probs = None
            if params.logprobs is not None:
                output_logprobs = req_output.outputs[0].logprobs
                if output_logprobs is None:
                    raise RuntimeError("AR mode requested logprobs, but vLLM-Omni returned none")
                if len(output_logprobs) != len(token_ids):
                    raise RuntimeError(
                        "AR mode logprob rows do not match generated tokens: "
                        f"logprobs={len(output_logprobs)} tokens={len(token_ids)}"
                    )
                log_probs = []
                for index, token_logprobs in enumerate(output_logprobs):
                    token_id = token_ids[index]
                    if token_id not in token_logprobs:
                        raise RuntimeError(
                            "AR mode sampled-token logprob is missing from vLLM-Omni output: "
                            f"index={index} token_id={token_id}"
                        )
                    log_prob = float(token_logprobs[token_id].logprob)
                    if not np.isfinite(log_prob):
                        raise RuntimeError(
                            "AR mode sampled-token logprob is non-finite: "
                            f"index={index} token_id={token_id} value={log_prob}"
                        )
                    log_probs.append(log_prob)

            finish_reason = req_output.outputs[0].finish_reason
            stop_reason = self._map_stop_reason(finish_reason)

            num_preempted = None
            if hasattr(req_output.outputs[0], "num_preempted"):
                num_preempted = req_output.outputs[0].num_preempted

            return TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason=stop_reason,
                num_preempted=num_preempted,
                extra_fields=extra_fields,
            )

        # diffusion
        assert final_res is not None
        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, torch.Tensor):
            diffusion_output = diffusion_output.float()
        elif isinstance(diffusion_output, np.ndarray):
            diffusion_output = torch.from_numpy(diffusion_output).float()
        else:
            diffusion_output = self._to_tensor(diffusion_output).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        def _maybe_unbatch(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value[0] if value.dim() > 0 else value
            if isinstance(value, np.ndarray):
                return value[0] if value.ndim > 0 else value
            if isinstance(value, list | tuple):
                return value[0] if value else None
            return value

        extra_fields = {k: _maybe_unbatch(v) for k, v in mm_output.items() if k != "all_log_probs"}
        extra_fields["global_steps"] = self.global_steps

        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        stop_reason = self._map_stop_reason(finish_reason)

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig | RolloutConfig,
        model_config: DiffusionModelConfig | HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
