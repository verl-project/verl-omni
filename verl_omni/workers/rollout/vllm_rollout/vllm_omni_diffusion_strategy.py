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
from verl.utils.import_utils import import_external_libs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_artifacts import (
    ARTIFACT_PREFIX,
    ARTIFACT_SPECS,
    PRIMARY_ARTIFACT,
    artifacts_from_fields,
    requested_artifact_names,
    select_artifact,
    validate_artifacts,
)
from verl_omni.pipelines.rollout_media import DiffusionIOSpec
from verl_omni.pipelines.rollout_request import OmniRolloutRequest, _alias_values_match
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
    if output_type is None:
        return "image"
    if not isinstance(output_type, str) or output_type not in ("image", "pt", "np", "pil", "both", "latent"):
        raise ValueError(f"Unsupported diffusion output_type: {output_type!r}")
    return output_type


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


def _unbatch_training_field(value: Any, *, context: str, name: str) -> Any:
    """Training groups declare B-leading tensors/lists; the RPC returns exactly one sample."""
    if isinstance(value, torch.Tensor | np.ndarray):
        if value.ndim == 0:
            return value
        if value.shape[0] != 1:
            raise ValueError(f"{context}, field={name!r}: expected batch size 1, got shape={tuple(value.shape)}")
        return value[0]
    if isinstance(value, list | tuple):
        if len(value) != 1:
            raise ValueError(f"{context}, field={name!r}: expected one batched item, got {len(value)}")
        return value[0]
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
        request: OmniRolloutRequest,
        sampling_params: dict[str, Any],
        lora_request: Optional[LoRARequest],
    ) -> tuple[dict[str, Any], list[Any]]:
        default_params_list = self.server.engine.default_sampling_params_list
        custom_prompt = dict(request.to_diffusion_prompt())
        if self.server.engine.engine.get_stage_metadata(0).stage_type != "diffusion":
            # Match AsyncOmniEngine's stage-0 preprocessing gate, independently of stage count.
            custom_prompt["prompt_token_ids"] = custom_prompt.pop("prompt_ids")
            custom_prompt["modalities"] = ["image"]

        sampling_kwargs: dict[str, Any] = {}
        explicit_extra = sampling_params.get("extra_args") or {}
        if not isinstance(explicit_extra, Mapping):
            raise TypeError("Diffusion sampling extra_args must be a mapping")
        extra_args: dict[str, Any] = dict(explicit_extra)
        _diffusion_output_type(sampling_params)
        for key, value in sampling_params.items():
            if key == "extra_args":
                continue
            if key in extra_args:
                if not _alias_values_match(value, extra_args[key]):
                    raise ValueError(f"Conflicting diffusion sampling field {key!r} and extra_args.{key}")
                del extra_args[key]
            if hasattr(OmniDiffusionSamplingParams, key):
                sampling_kwargs[key] = value
            else:
                extra_args[key] = value
        requested = requested_artifact_names(extra_args.get("requested_outputs"), context="diffusion request")
        if requested:
            io_spec = self._diffusion_io_spec()
            if io_spec is not None and (unknown := set(requested) - io_spec.artifacts.keys()):
                raise ValueError(f"Diffusion request asks for undeclared artifacts: {sorted(unknown)}")
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
        if priority != 0:
            raise ValueError("DiffusionStrategy does not support nonzero request priority")
        return await self._collect_last_output(
            self.server.engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=params,
            )
        )

    def _diffusion_io_spec(self) -> Optional[DiffusionIOSpec]:
        """Resolve the adapter's available named artifacts, independent of engine wire layout."""
        model_config = getattr(self.server, "model_config", None)
        if model_config is None:
            return None
        pipeline_cls = VllmOmniPipelineBase.get_class(
            architecture=model_config.architecture,
            algorithm=model_config.algorithm,
        )
        return getattr(pipeline_cls, "diffusion_io_spec", None)

    def process_output(self, final_res: Any, params: Any, sampling_params: dict[str, Any]) -> DiffusionOutput:
        output_type = _diffusion_output_type(sampling_params)
        req_output = getattr(final_res, "request_output", None) or final_res
        request_id = getattr(req_output, "request_id", getattr(final_res, "request_id", "unknown"))
        model_config = getattr(self.server, "model_config", None)
        context = (
            f"pipeline={getattr(model_config, 'architecture', 'unknown')}/"
            f"{getattr(model_config, 'algorithm', 'unknown')}, request_id={request_id}"
        )
        multimodal = getattr(final_res, "multimodal_output", None)
        metadata = multimodal.get("metadata", {}) if isinstance(multimodal, Mapping) else {}
        artifact_header = metadata.get("media_artifacts") if isinstance(metadata, Mapping) else None
        artifact_payload = None
        if artifact_header is not None:
            if (
                not isinstance(artifact_header, Mapping)
                or not {"primary", "specs", "preview", "audio"} <= artifact_header.keys()
            ):
                raise ValueError(
                    f"request_id={getattr(final_res, 'request_id', 'unknown')}: invalid media_artifacts header"
                )
            artifact_payload = {}
            sources = []
            specs = artifact_header["specs"]
            if not isinstance(specs, Mapping) or artifact_header["primary"] not in specs:
                raise ValueError(f"{context}: invalid named artifact declarations/primary")
            for key, value in multimodal.items():
                if key in {"metadata", "audio_sample_rate", "fps", "trajectory"}:
                    continue
                if key in ("image", "video", "audio") and isinstance(value, Mapping | list):
                    if isinstance(value, list):
                        if len(value) != 1:
                            raise ValueError(f"{context}: expected one named artifact payload")
                        value = value[0]
                    if not isinstance(value, Mapping):
                        raise ValueError(f"{context}: invalid named {key} payload")
                    sources.append(value)
                elif key in specs:
                    sources.append({key: value})
                else:
                    raise ValueError(f"{context}: undeclared multimodal output {key!r}")
            images = final_res.images or []
            if images:
                if len(images) != 1:
                    raise ValueError(f"{context}: expected one named artifact payload")
                image_payload = images[0]
                if not isinstance(image_payload, Mapping):
                    primary = artifact_header["primary"]
                    expected_kind = specs[primary]["modality"]
                    if getattr(final_res, "final_output_type", None) != expected_kind:
                        raise ValueError(
                            f"{context}: images fallback requires explicit final_output_type={expected_kind}"
                        )
                    image_payload = {primary: image_payload}
                sources.append(image_payload)
            for source in sources:
                for name, tensor in source.items():
                    if not isinstance(name, str):
                        raise ValueError(f"{context}: artifact names must be strings, got {name!r}")
                    if name in artifact_payload and (
                        getattr(artifact_payload[name], "dtype", None) != getattr(tensor, "dtype", None)
                        or not _alias_values_match(artifact_payload[name], tensor)
                    ):
                        raise ValueError(f"{context}: conflicting artifact={name!r} in multimodal_output and images")
                    artifact_payload[name] = tensor
            if not artifact_payload:
                raise ValueError(f"{context}: expected one named artifact payload")
        if artifact_header is None and isinstance(multimodal, Mapping) and multimodal.keys() - {"metadata"}:
            raise ValueError(f"{context}: named media_artifacts declaration required")
        if artifact_header is None and (final_res is None or not final_res.images):
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

        if artifact_header is None:
            raise ValueError(
                f"{context}: named media_artifacts declaration required; legacy tensor/tuple output is unsupported"
            )
        diffusion_output = artifact_payload
        io_spec = self._diffusion_io_spec()
        artifacts = {}
        primary_artifact = None
        audio_sample_rate: Optional[int] = None
        rollout_audio: Any = None
        if artifact_header is not None:
            primary_artifact = artifact_header["primary"]
            artifacts = artifacts_from_fields(
                {
                    ARTIFACT_SPECS: artifact_header["specs"],
                    PRIMARY_ARTIFACT: primary_artifact,
                    **{ARTIFACT_PREFIX + name: tensor for name, tensor in diffusion_output.items()},
                },
                context=context,
            )
            validate_artifacts(
                artifacts.items(),
                {name: artifact.spec for name, artifact in artifacts.items()},
                primary=primary_artifact,
                context=context,
                requested=sampling_params.get(
                    "requested_outputs", (sampling_params.get("extra_args") or {}).get("requested_outputs")
                ),
            )
            preview_name = artifact_header.get("preview")
            if preview_name is not None and (
                preview_name not in artifacts
                or artifacts[preview_name].spec.representation != "decoded"
                or artifacts[preview_name].spec.modality not in ("image", "video")
            ):
                raise ValueError(f"{context}: preview artifact={preview_name!r} must name decoded visual media")
            primary = artifacts[primary_artifact]
            expected_representation = "latent" if output_type == "latent" else "decoded"
            if primary.spec.representation != expected_representation:
                raise ValueError(
                    f"{context}, artifact={primary_artifact!r}: expected {expected_representation}, got {primary.spec}"
                )
            if io_spec is not None:
                for name, artifact in artifacts.items():
                    expected = io_spec.artifacts.get(name)
                    if expected is None:
                        raise ValueError(f"{context}: undeclared artifact={name!r}")
                    actual = artifact.spec
                    if (actual.modality, actual.representation, actual.layout) != (
                        expected.modality,
                        expected.representation,
                        expected.layout,
                    ) or (expected.sample_rate is not None and actual.sample_rate != expected.sample_rate):
                        raise ValueError(f"{context}, artifact={name!r}: expected {expected}, got {actual}")
            diffusion_output = primary.data
            if artifact_header.get("audio") is not None:
                audio_artifact = select_artifact(
                    artifacts, name=artifact_header["audio"], modality="audio", representation="decoded"
                )
                rollout_audio = audio_artifact.data
                audio_sample_rate = audio_artifact.spec.sample_rate

        if sampling_params.get("logprobs", False):
            log_probs = _unbatch_training_field(
                final_res.trajectory_log_probs, context=context, name="trajectory_log_probs"
            )
        else:
            log_probs = None

        extra_fields: dict[str, Any] = {"global_steps": self.server.global_steps}
        if final_res.trajectory_latents is not None:
            extra_fields["all_latents"] = _unbatch_training_field(
                final_res.trajectory_latents, context=context, name="trajectory_latents"
            )
        if final_res.trajectory_timesteps is not None:
            extra_fields["all_timesteps"] = _unbatch_training_field(
                final_res.trajectory_timesteps, context=context, name="trajectory_timesteps"
            )
        for metadata_group in _rollout_metadata_groups(final_res.multimodal_output):
            for key, value in metadata_group.items():
                if key in extra_fields:
                    raise ValueError(f"Duplicate rollout metadata field: {key}")
                extra_fields[key] = _unbatch_training_field(value, context=context, name=key)
        if artifacts:
            kind = artifacts[primary_artifact].spec.modality
            if extra_fields.get("media_kind", kind) != kind:
                raise ValueError(f"{context}: media_kind conflicts with primary artifact")
            extra_fields["media_kind"] = kind
            if rollout_audio is not None:
                if "audio" in extra_fields and not torch.equal(torch.as_tensor(extra_fields["audio"]), rollout_audio):
                    raise ValueError(f"{context}: audio metadata conflicts with named audio artifact")
                if extra_fields.get("audio_sample_rate", audio_sample_rate) != audio_sample_rate:
                    raise ValueError(f"{context}: audio_sample_rate conflicts with named audio artifact")
        if rollout_audio is not None:
            extra_fields["audio"] = rollout_audio
            extra_fields["audio_sample_rate"] = audio_sample_rate

        if hasattr(req_output, "outputs") and req_output.outputs:
            finish_reason = req_output.outputs[0].finish_reason or "stop"
        elif hasattr(req_output, "finish_reason"):
            finish_reason = req_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        stop_reason = self._map_stop_reason(finish_reason)
        num_preempted = self._extract_num_preempted(req_output)

        return DiffusionOutput(
            artifacts=artifacts,
            primary_artifact=primary_artifact,
            preview_artifact=artifact_header["preview"],
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )
