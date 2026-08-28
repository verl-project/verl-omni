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
import logging
from argparse import Namespace
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as T
from verl.utils.import_utils import import_external_libs
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import OmniStrategyBase

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_GPU_WORKER_EXTENSION = "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"
_NPU_WORKER_EXTENSION = "verl_omni.workers.rollout.vllm_rollout.npu_utils.vLLMOmniNPUColocateWorkerExtension"


def _diffusion_output_type(sampling_params: dict[str, Any]) -> str:
    output_type = sampling_params.get("output_type")
    if output_type is None:
        output_type = (sampling_params.get("extra_args") or {}).get("output_type")
    return output_type or "image"


def _pixel_output_to_uint8(output: torch.Tensor) -> torch.Tensor:
    """Quantize a rollout pixel tensor from float ``[0, 1]`` to uint8 once."""
    if output.dtype == torch.uint8:
        return output
    output = output.detach().to(dtype=torch.float32, copy=True)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Pixel rollout output must contain only finite values")
    output = output.clamp_(0, 1)
    return output.mul_(255).round_().to(dtype=torch.uint8)


def _rollout_metadata_groups(multimodal_output: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(multimodal_output, Mapping):
        return ()
    metadata = multimodal_output.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    groups = []
    for name in ("prompt_embeddings", "rl"):
        group = metadata.get(name)
        if isinstance(group, Mapping):
            groups.append(group)
    return tuple(groups)


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


class DiffusionStrategy(OmniStrategyBase):
    """Concrete diffusion strategy.

    Converts configs to ``DiffusionRolloutConfig``/``DiffusionModelConfig``,
    resolves the diffusion pipeline and prepares its engine args, builds an
    ``OmniCustomPrompt``, and produces ``DiffusionOutput`` from
    :meth:`process_output`.
    """

    rollout_config_cls = DiffusionRolloutConfig
    model_config_cls = DiffusionModelConfig

    def post_init(self, cuda_visible_devices: str) -> None:
        self.server._to_tensor = T.PILToTensor()

    def worker_extension_cls(self, device_type: str) -> str:
        if device_type == "npu":
            return _NPU_WORKER_EXTENSION
        return _GPU_WORKER_EXTENSION

    def prepare_engine_args(self, engine_args: dict[str, Any], args: Namespace) -> None:
        import_external_libs(self.server.config.external_lib)

        pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
            architecture=self.server.model_config.architecture,
            algorithm=self.server.model_config.algorithm,
        )
        # TODO (mike): read custom_pipeline from engine_args.
        if pipeline_path is not None:
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

            pipeline_cls = VllmOmniPipelineBase.get_class(
                architecture=self.server.model_config.architecture,
                algorithm=self.server.model_config.algorithm,
            )
            step_execution = getattr(self.server.config, "step_execution", False)
            if (
                pipeline_cls is not None
                and not getattr(pipeline_cls, "supports_request_batch", False)
                and not step_execution
                and int(engine_args.get("max_num_seqs") or 1) > 1
            ):
                logger.info(
                    "Pipeline %s does not support request-level batching; clamping max_num_seqs to 1.",
                    pipeline_cls.__name__,
                )
                engine_args["max_num_seqs"] = 1

        engine_args["enable_prompt_embed_cache"] = self.server.config.enable_prompt_embed_cache
        engine_args["prompt_embed_cache_size"] = self.server.config.prompt_embed_cache_size

    def preprocess_input(
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
    ) -> tuple[OmniCustomPrompt, list[Any]]:
        default_params_list = self.server.engine.default_sampling_params_list

        custom_prompt: OmniCustomPrompt = {"prompt_token_ids": prompt_ids}
        if prompt_mask is not None:
            custom_prompt["prompt_mask"] = prompt_mask
        if len(default_params_list) > 1:
            custom_prompt["modalities"] = ["image"]
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if extra_prompt_ids is not None:
            custom_prompt["extra_prompt_ids"] = extra_prompt_ids
        if negative_extra_prompt_ids is not None:
            custom_prompt["negative_extra_prompt_ids"] = negative_extra_prompt_ids
        if multi_modal_data:
            custom_prompt["multi_modal_data"] = multi_modal_data
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for key, value in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, key):
                sampling_kwargs[key] = value
            else:
                extra_args[key] = value
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)
        params = default_params_list[:-1] + [diffusion_sampling_params]
        return custom_prompt, params

    async def run_generation(
        self,
        prompt: Any,
        params: Any,
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
    ) -> Any:
        return await self._collect_last_output(
            self.server.engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=params,
            )
        )

    def process_output(self, final_res: Any, params: Any, sampling_params: dict[str, Any]) -> DiffusionOutput:
        output_type = _diffusion_output_type(sampling_params)
        if final_res is None or not final_res.images:
            finish_reason = "abort"
            if final_res is not None:
                req_out = getattr(final_res, "request_output", None) or final_res
                if hasattr(req_out, "outputs") and req_out.outputs:
                    finish_reason = getattr(req_out.outputs[0], "finish_reason", None) or "abort"
                else:
                    finish_reason = getattr(req_out, "finish_reason", None) or "abort"
            stop_reason = self._map_stop_reason(finish_reason)
            logger.debug(
                "diffusion rollout produced no image (finish_reason=%s); returning %s", finish_reason, stop_reason
            )
            return DiffusionOutput(
                diffusion_output=torch.empty(
                    0,
                    dtype=torch.float32 if output_type == "latent" else torch.uint8,
                ),
                log_probs=None,
                stop_reason=stop_reason,
                num_preempted=None,
                extra_fields={"global_steps": self.server.global_steps},
            )

        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, dict):
            for key in ("video", "image", "output", "audio"):
                if key in diffusion_output and diffusion_output[key] is not None:
                    diffusion_output = diffusion_output[key]
                    break
        rollout_audio: Any = None
        if isinstance(diffusion_output, tuple | list):
            rollout_audio = diffusion_output[1] if len(diffusion_output) > 1 else None
            diffusion_output = diffusion_output[0]
        if output_type == "latent":
            diffusion_output = torch.as_tensor(diffusion_output).float()
        else:
            if isinstance(diffusion_output, np.ndarray):
                diffusion_output = torch.from_numpy(diffusion_output)
            elif not isinstance(diffusion_output, torch.Tensor):
                diffusion_output = self.server._to_tensor(diffusion_output)
            diffusion_output = _pixel_output_to_uint8(diffusion_output)

        if sampling_params.get("logprobs", False):
            log_probs = _maybe_unbatch(final_res.trajectory_log_probs)
        else:
            log_probs = None

        extra_fields: dict[str, Any] = {"global_steps": self.server.global_steps}
        if final_res.trajectory_latents is not None:
            extra_fields["all_latents"] = _maybe_unbatch(final_res.trajectory_latents)
        if final_res.trajectory_timesteps is not None:
            extra_fields["all_timesteps"] = _maybe_unbatch(final_res.trajectory_timesteps)
        for metadata_group in _rollout_metadata_groups(final_res.multimodal_output):
            for key, value in metadata_group.items():
                if key in extra_fields:
                    raise ValueError(f"Duplicate rollout metadata field: {key}")
                extra_fields[key] = _maybe_unbatch(value)
        if rollout_audio is not None:
            extra_fields["audio"] = _maybe_unbatch(rollout_audio)
            extra_fields.setdefault("audio_sample_rate", 32000)

        req_output = getattr(final_res, "request_output", None) or final_res
        if hasattr(req_output, "outputs") and req_output.outputs:
            finish_reason = req_output.outputs[0].finish_reason or "stop"
        elif hasattr(req_output, "finish_reason"):
            finish_reason = req_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        stop_reason = self._map_stop_reason(finish_reason)
        num_preempted = self._extract_num_preempted(req_output)

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )
