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
import tempfile
import time
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import ray
import torch
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
import yaml
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_visible_devices_keyword
from verl.utils.import_utils import import_external_libs
from verl.utils.net_utils import get_free_port
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase, VllmOmniPipelineBase
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig, OmniModelConfig
from verl_omni.workers.rollout.replica import DiffusionOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Sentinel: ``None`` is a valid cached value (LoRA not loaded).
_LORA_REQUEST_CACHE_MISS = object()


def _drop_none_mapping_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_mapping_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_mapping_values(item) for item in value]
    return value


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
        """AR mode uses OmniModelConfig; diffusion uses DiffusionModelConfig.

        Mode is selected by ``engine_kwargs.vllm_omni.output_mode`` ("ar" vs the
        default "diffusion").
        """
        engine_kwargs = getattr(self.config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        # TODO (mike): drop this once the legacy omni training script is removed.
        # It should be automatically inferred from the model config
        self._ar_mode = omni_kwargs.get("output_mode", "diffusion") == "ar"
        self._rollout_flags: dict[int, dict] = {}

        if self._ar_mode:
            return omega_conf_to_dataclass(model_config, dataclass_type=OmniModelConfig)
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """AR mode: derive max_model_len. Diffusion: no max_position_embeddings."""
        if self._ar_mode:
            if self.config.max_model_len is None:
                self.config.max_model_len = self.config.prompt_length + self.config.response_length

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Diffusion needs a PIL→tensor converter; AR does not."""
        if not self._ar_mode:
            self._to_tensor = T.PILToTensor()
        self._lora_request_cache: LoRARequest | None | object = _LORA_REQUEST_CACHE_MISS
        self._lora_resolve_lock = asyncio.Lock()
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
        device_type = ""
        try:
            from vllm.platforms import current_platform

            device_type = current_platform.device_type
        except Exception:
            pass

        # vLLMOmniColocateWorkerExtension supports LoRA + weight updates for GPU.
        # vLLMOmniNPUColocateWorkerExtension additionally mixes in NPUColocateWorkerMixin
        # for NPU memory pool, sleep, and wake_up.
        # ar_mode uses vllm-ascend which already handles NPU natively, so the base extension suffices.
        if device_type != "npu" or self._ar_mode:
            return "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"
        else:
            return "verl_omni.workers.rollout.vllm_rollout.npu_utils.vLLMOmniNPUColocateWorkerExtension"

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
            # TODO (mike): drop this later
            # It should be automatically inferred from the model config
            pipeline_name = engine_kwargs.pop("pipeline_name", None)
            pipeline_mode = engine_kwargs.pop("pipeline_mode", "thinker_only")

            adapter_cls = OmniRolloutPipelineBase.get_class(pipeline_name)
            if adapter_cls is not None:
                # Generate deploy config using the adapter's stage topology.
                self._write_deploy_config(engine_kwargs, pipeline_name, adapter_cls, pipeline_mode)
                # Store per-stage rollout flags for downstream use.
                self._rollout_flags = adapter_cls.rollout_flags(pipeline_mode=pipeline_mode)
                # Merge pipeline-specific HF config overrides.
                adapter_overrides = adapter_cls.get_engine_hf_overrides(pipeline_mode=pipeline_mode)
                if adapter_overrides:
                    hf_overrides = engine_kwargs.get("hf_overrides", {})
                    if isinstance(hf_overrides, str):
                        hf_overrides = json.loads(hf_overrides)
                    hf_overrides.update(adapter_overrides)
                    engine_kwargs["hf_overrides"] = hf_overrides

            stage_init_timeout = engine_kwargs.get("stage_init_timeout") or engine_kwargs.get("stage-init-timeout")
            init_timeout = engine_kwargs.get("init_timeout") or engine_kwargs.get("init-timeout")
            if stage_init_timeout is not None and init_timeout is None:
                engine_kwargs["init_timeout"] = max(int(stage_init_timeout), 600)

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

    def _write_deploy_config(self, engine_kwargs: dict, pipeline_name: str, adapter_cls, pipeline_mode: str) -> None:
        """Write a deploy config YAML from the adapter's stage topology."""
        adapter_cls.ensure_pipeline_registered(pipeline_mode)
        stages = adapter_cls.build_stage_configs(pipeline_mode=pipeline_mode)
        pipeline_id = adapter_cls.get_pipeline_id(pipeline_mode)

        device_control_env = get_visible_devices_keyword()
        visible_devices = os.environ.get(device_control_env, "")
        tp_size = self.config.tensor_model_parallel_size

        deploy_dict: dict[str, object] = {"pipeline": pipeline_id}

        if visible_devices:
            # Stage configs use logical ids relative to the Ray actor's visible-device set.
            device_count = len([device for device in visible_devices.split(",") if device.strip()])
            devices = ",".join(str(device_id) for device_id in range(device_count))
            stage_ids = [s.stage_id for s in stages]
            deploy_dict["stages"] = [
                {
                    "stage_id": sid,
                    "devices": devices,
                    "tensor_parallel_size": tp_size,
                    "engine_extras": adapter_cls.get_stage_engine_extras(sid, pipeline_mode=pipeline_mode),
                }
                for sid in stage_ids
            ]
        else:
            raise RuntimeError(
                f"Environment variable `{device_control_env}` is not set, cannot generate deploy config."
            )

        yaml_str = yaml.dump(deploy_dict).strip()
        logger.info("Generated deploy config:\n%s", yaml_str)
        self._temp_deploy_ctx = tempfile.TemporaryDirectory(prefix="verl_omni_deploy_")
        deploy_path = os.path.join(self._temp_deploy_ctx.name, f"{pipeline_name}.yaml")
        try:
            with open(deploy_path, "w") as f:
                f.write(yaml_str)
        except OSError as e:
            self._temp_deploy_ctx.cleanup()
            self._temp_deploy_ctx = None
            raise RuntimeError(f"Failed to write deploy config to {deploy_path}: {e}") from e
        engine_kwargs["deploy_config"] = deploy_path

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        deploy_config = getattr(args, "deploy_config", None)
        if deploy_config:
            engine_args["deploy_config"] = deploy_config

        if self._ar_mode:
            for timeout_key in ("stage_init_timeout", "init_timeout"):
                timeout_value = getattr(args, timeout_key, None)
                if timeout_value is not None:
                    engine_args[timeout_key] = int(timeout_value)
            engine_args["logprobs_mode"] = getattr(self.config, "logprobs_mode", "processed_logprobs")
            # AR mode: no diffusion pipeline. Drop None entries from
            # compilation_config that OmniEngineArgs may leave behind.
            if isinstance(engine_args.get("compilation_config"), dict):
                engine_args["compilation_config"] = _drop_none_mapping_values(engine_args["compilation_config"])
        else:
            import_external_libs(self.config.external_lib)

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

        # rollout_attn_backend only exists on the diffusion rollout config, not AR text rollouts.
        attn_backend = getattr(self.config, "rollout_attn_backend", None)
        if attn_backend is not None:
            engine_args["diffusion_attention_backend"] = attn_backend
            logger.info("Setting diffusion_attention_backend=%s from rollout config", attn_backend)

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
        # TODO (andy): use `sleep_level=2` in the future when the
        #  trainer side incorporates the whole components of the model.
        self._invalidate_lora_request_cache()
        await self.engine.collective_rpc("sleep", kwargs={"level": 1})
        await self.engine.reset_encoder_cache()

    async def resume_generation(self):
        if self.node_rank == 0:
            await self.engine.resume_generation()

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
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        prompt_ids = normalize_token_ids(prompt_ids)
        multi_modal_data = self._build_multi_modal_data(image_data, video_data, audio_data)
        lora_request = await self._resolve_lora_request()
        prompt, params = self._preprocess_input(
            prompt_ids,
            sampling_params,
            multi_modal_data,
            lora_request,
            negative_prompt_ids,
            prompt_mask,
            mm_processor_kwargs,
            extra_prompt_ids,
            negative_extra_prompt_ids,
        )
        final_res = await self._run_generation(prompt, params, request_id, lora_request, priority)
        return self._process_output(final_res, params, sampling_params)

    # -----------------------------------------------------------------------
    # Shared helpers for the AR and diffusion generate paths
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_multi_modal_data(
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        audio_data: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Assemble the vLLM multi_modal_data dict from optional image/video/audio inputs."""
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data
        return multi_modal_data

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
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
    ):
        """Build the engine prompt + sampling params for the active mode.

        Returns ``(prompt, params)`` consumed by ``_run_generation``.
        """
        if self._ar_mode:
            if multi_modal_data:
                # Deduplicate already-expanded multimodal pad tokens to prevent
                # double-expansion inside vLLM-Omni.
                processor = getattr(self.model_config, "processor", None)
                if processor is not None and hasattr(processor, "dedup_pad_tokens"):
                    prompt_ids = processor.dedup_pad_tokens(prompt_ids)
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
            if mm_processor_kwargs:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs
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
        # Per-text-encoder token ids for multi-encoder models (e.g. SD3.5),
        # produced by the agent loop so pipelines never decode/re-encode text.
        if extra_prompt_ids is not None:
            custom_prompt["extra_prompt_ids"] = extra_prompt_ids
        if negative_extra_prompt_ids is not None:
            custom_prompt["negative_extra_prompt_ids"] = negative_extra_prompt_ids
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
                sampling_params_list=params,
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
                log_probs = [
                    logprobs[token_ids[i]].logprob for i, logprobs in enumerate(req_output.outputs[0].logprobs)
                ]

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
        # Handle aborted requests: the engine may yield a terminal output with
        # finish_reason="abort" and no images (e.g. when abort_all_requests
        # synthesizes an abort OutputMessage to unblock the generate() coroutine).
        # Return a DiffusionOutput with stop_reason="aborted" so the retry client
        # can retry the whole sample.
        if final_res is None or not final_res.images:
            finish_reason = "abort"
            if final_res is not None and final_res.request_output is not None:
                finish_reason = getattr(final_res.request_output, "finish_reason", None) or "abort"
            stop_reason = self._map_stop_reason(finish_reason)
            logger.debug(
                "diffusion rollout produced no image (finish_reason=%s); returning %s", finish_reason, stop_reason
            )
            return DiffusionOutput(
                diffusion_output=torch.empty(0),
                log_probs=None,
                stop_reason=stop_reason,
                num_preempted=None,
                extra_fields={"global_steps": self.global_steps},
            )

        assert final_res is not None
        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, torch.Tensor):
            diffusion_output = diffusion_output.float()
        elif isinstance(diffusion_output, np.ndarray):
            diffusion_output = torch.from_numpy(diffusion_output).float()
        else:
            diffusion_output = self._to_tensor(diffusion_output).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        custom_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = custom_output.get("all_log_probs")
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

        extra_fields = {k: _maybe_unbatch(v) for k, v in custom_output.items() if k != "all_log_probs"}
        multimodal_output = final_res.multimodal_output or {}
        if isinstance(multimodal_output, dict):
            for key, value in multimodal_output.items():
                extra_fields.setdefault(key, _maybe_unbatch(value))
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
        minimal ``OmniRequestOutput`` carrying a ``RequestOutput`` with
        ``finish_reason="abort"`` so that ``_process_single_result`` yields it
        and ``_process_output`` maps it to ``stop_reason="aborted"``.
        """
        from vllm.outputs import CompletionOutput, RequestOutput
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
        request_output = RequestOutput(
            request_id=internal_id,
            prompt=None,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[completion],
            finished=True,
        )
        omni_output = OmniRequestOutput(
            request_id=internal_id,
            finished=True,
            request_output=request_output,
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

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
