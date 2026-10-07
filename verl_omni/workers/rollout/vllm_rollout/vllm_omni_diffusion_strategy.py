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
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch
from verl.utils.import_utils import import_external_libs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_artifacts import (
    ARTIFACT_PREFIX,
    ARTIFACT_SPECS,
    PRIMARY_ARTIFACT,
    MediaArtifact,
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


def _media_artifact_header(multimodal: Any, *, context: str) -> Mapping[str, Any] | None:
    """Validate the named header before interpreting any engine payload."""
    metadata = multimodal.get("metadata", {}) if isinstance(multimodal, Mapping) else {}
    header = metadata.get("media_artifacts") if isinstance(metadata, Mapping) else None
    if header is None:
        if isinstance(multimodal, Mapping) and multimodal.keys() - {"metadata"}:
            raise ValueError(f"{context}: named media_artifacts declaration required")
        return None
    if not isinstance(header, Mapping) or not {"primary", "specs", "preview", "audio"} <= header.keys():
        raise ValueError(f"{context}: invalid media_artifacts header")
    if not isinstance(header["specs"], Mapping) or header["primary"] not in header["specs"]:
        raise ValueError(f"{context}: invalid named artifact declarations/primary")
    return header


def _merge_artifact_sources(sources: list[Mapping[str, Any]], *, context: str) -> dict[str, torch.Tensor]:
    """Merge complementary engine sources, rejecting non-tensors and conflicting aliases."""
    payload = {}
    for source in sources:
        for name, tensor in source.items():
            if not isinstance(name, str):
                raise ValueError(f"{context}: artifact names must be strings, got {name!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{context}, artifact={name!r}: expected tensor, got {type(tensor).__name__}")
            if name in payload and (
                payload[name].dtype != tensor.dtype or not _alias_values_match(payload[name], tensor)
            ):
                raise ValueError(f"{context}: conflicting artifact={name!r} in multimodal_output and images")
            payload[name] = tensor
    if not payload:
        raise ValueError(f"{context}: expected one named artifact payload")
    return payload


def _collect_artifact_payload(
    final_res: OmniRequestOutput, header: Mapping[str, Any], *, context: str
) -> dict[str, torch.Tensor]:
    """Read the pinned formatter's named mappings and explicitly labelled images fallback."""
    sources = []
    specs = header["specs"]
    for key, value in final_res.multimodal_output.items():
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
    if final_res.images:
        if len(final_res.images) != 1:
            raise ValueError(f"{context}: expected one named artifact payload")
        image_payload = final_res.images[0]
        if not isinstance(image_payload, Mapping):
            primary = header["primary"]
            expected_kind = specs[primary]["modality"]
            if final_res.final_output_type != expected_kind:
                raise ValueError(f"{context}: images fallback requires explicit final_output_type={expected_kind}")
            image_payload = {primary: image_payload}
        sources.append(image_payload)
    return _merge_artifact_sources(sources, context=context)


def _finish_reason(final_res: OmniRequestOutput | None, *, default: str) -> str:
    """Read completion state when supplied; pure diffusion outputs may omit it."""
    if final_res is None:
        return default
    if final_res.outputs:
        return final_res.outputs[0].finish_reason or default
    if hasattr(final_res, "finish_reason"):
        return final_res.finish_reason or default
    return default


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
        config = self.server.config
        parallel_config = engine_args.get("parallel_config")
        if parallel_config is not None:
            parallel_config = dict(parallel_config) if isinstance(parallel_config, Mapping) else asdict(parallel_config)

        # Resource-bearing degrees must agree with the config used by verl's allocator.
        topology = {
            "tensor_parallel_size": config.tensor_model_parallel_size,
            "ulysses_degree": config.ulysses_degree,
            "ring_degree": config.ring_degree,
            "sequence_parallel_size": config.ulysses_degree * config.ring_degree,
            "data_parallel_size": config.data_parallel_size,
            "pipeline_parallel_size": config.pipeline_model_parallel_size,
        }
        explicit_kwargs = {
            key.replace("-", "_"): value for key, value in (config.engine_kwargs.get("vllm_omni", {}) or {}).items()
        }
        for key, value in topology.items():
            cli_value = getattr(args, key, None)
            nested_value = (parallel_config or {}).get(key)
            aliases = {"ulysses_degree": "usp", "ring_degree": "ring"}
            explicit = key in explicit_kwargs or aliases.get(key) in explicit_kwargs
            if (cli_value is not None and cli_value != value and (explicit or cli_value != 1)) or (
                nested_value is not None and nested_value != value
            ):
                raise ValueError(
                    f"Conflicting {key}; configure topology through actor_rollout_ref.rollout, not engine_kwargs."
                )
            engine_args[key] = value
            if parallel_config is not None:
                parallel_config[key] = value

        for key, default in (("vae_patch_parallel_size", 1), ("vae_parallel_mode", "tile"), ("vae_use_tiling", False)):
            value = getattr(config, key)
            cli_value = getattr(args, key, None)
            if cli_value is not None and (cli_value != default or key in explicit_kwargs):
                if value not in (default, cli_value):
                    raise ValueError(f"Conflicting {key} in rollout config and engine_kwargs.")
                value = cli_value
            if key != "vae_use_tiling" and parallel_config is not None:
                nested_value = parallel_config.get(key)
                if nested_value is not None and (nested_value != getattr(config, key) or nested_value != value):
                    raise ValueError(f"Conflicting {key} in rollout config and parallel_config.")
                parallel_config[key] = value
            engine_args[key] = value

        # TODO(vllm-omni#7564): Drop this pin-compat shim; tracked in verl-omni#445.
        text_encoder_tp = config.text_encoder_tp_size
        cli_text_encoder_tp = getattr(args, "text_encoder_tp_size", None)
        if cli_text_encoder_tp is not None:
            if text_encoder_tp not in (1, cli_text_encoder_tp):
                raise ValueError("Conflicting text_encoder_tp_size in rollout config and engine_kwargs.")
            text_encoder_tp = cli_text_encoder_tp

        dit_world_size = config.tensor_model_parallel_size * config.ulysses_degree * config.ring_degree
        if parallel_config is not None:
            nested_text_encoder_tp = parallel_config.get("text_encoder_tp_size")
            if nested_text_encoder_tp is not None:
                if nested_text_encoder_tp != text_encoder_tp and (
                    cli_text_encoder_tp is not None or text_encoder_tp != 1
                ):
                    raise ValueError("Conflicting text_encoder_tp_size in rollout/engine_kwargs and parallel_config.")
                text_encoder_tp = nested_text_encoder_tp

        if text_encoder_tp < 1 or text_encoder_tp not in (1, dit_world_size):
            raise ValueError(f"text_encoder_tp_size must be 1 or equal to DiT group size ({dit_world_size}).")
        engine_args["text_encoder_tp_size"] = text_encoder_tp
        if config.ulysses_degree * config.ring_degree > 1:
            for key in ("cfg_parallel_size", "allgather_degree"):
                value = (parallel_config or {}).get(key, getattr(args, key, None))
                if value not in (None, 1):
                    raise ValueError(f"Rollout sequence parallelism requires {key}=1.")
            num_gpus = engine_args.get("num_gpus")
            if num_gpus not in (None, dit_world_size):
                raise ValueError(f"num_gpus must match the allocated DiT group size ({dit_world_size}).")
            engine_args["num_gpus"] = dit_world_size
        if parallel_config is not None:
            parallel_config["text_encoder_tp_size"] = text_encoder_tp
            engine_args["parallel_config"] = parallel_config

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
        model_config = self.server.model_config
        if model_config is None:
            return None
        pipeline_cls = VllmOmniPipelineBase.get_class(
            architecture=model_config.architecture,
            algorithm=model_config.algorithm,
        )
        if pipeline_cls is None or not hasattr(pipeline_cls, "diffusion_io_spec"):
            return None
        return pipeline_cls.diffusion_io_spec

    def _parse_media_artifacts(
        self,
        final_res: OmniRequestOutput,
        header: Mapping[str, Any],
        sampling_params: dict[str, Any],
        *,
        context: str,
    ) -> dict[str, MediaArtifact]:
        """Validate merged named media against requested outputs and adapter declarations."""
        payload = _collect_artifact_payload(final_res, header, context=context)
        primary_name = header["primary"]
        artifacts = artifacts_from_fields(
            {
                ARTIFACT_SPECS: header["specs"],
                PRIMARY_ARTIFACT: primary_name,
                **{ARTIFACT_PREFIX + name: tensor for name, tensor in payload.items()},
            },
            context=context,
        )
        validate_artifacts(
            artifacts.items(),
            {name: artifact.spec for name, artifact in artifacts.items()},
            primary=primary_name,
            context=context,
            requested=sampling_params.get(
                "requested_outputs", (sampling_params.get("extra_args") or {}).get("requested_outputs")
            ),
        )
        preview_name = header["preview"]
        if preview_name is not None and (
            preview_name not in artifacts
            or artifacts[preview_name].spec.representation != "decoded"
            or artifacts[preview_name].spec.modality not in ("image", "video")
        ):
            raise ValueError(f"{context}: preview artifact={preview_name!r} must name decoded visual media")
        primary = artifacts[primary_name]
        expected_representation = "latent" if _diffusion_output_type(sampling_params) == "latent" else "decoded"
        if primary.spec.representation != expected_representation:
            raise ValueError(
                f"{context}, artifact={primary_name!r}: expected {expected_representation}, got {primary.spec}"
            )
        io_spec = self._diffusion_io_spec()
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
        return artifacts

    def _training_output_fields(self, final_res: OmniRequestOutput, *, context: str) -> dict[str, Any]:
        """Remove only declared training batch axes and preserve algorithm-owned keys."""
        fields: dict[str, Any] = {"global_steps": self.server.global_steps}
        for name, source, value in (
            ("all_latents", "trajectory_latents", final_res.trajectory_latents),
            ("all_timesteps", "trajectory_timesteps", final_res.trajectory_timesteps),
        ):
            if value is not None:
                fields[name] = _unbatch_training_field(value, context=context, name=source)
        for metadata_group in _rollout_metadata_groups(final_res.multimodal_output):
            for key, value in metadata_group.items():
                if key in fields:
                    raise ValueError(f"Duplicate rollout metadata field: {key}")
                fields[key] = _unbatch_training_field(value, context=context, name=key)
        return fields

    def process_output(
        self, final_res: OmniRequestOutput | None, params: Any, sampling_params: dict[str, Any]
    ) -> DiffusionOutput:
        output_type = _diffusion_output_type(sampling_params)
        model_config = self.server.model_config
        pipeline = f"{model_config.architecture}/{model_config.algorithm}" if model_config is not None else "unknown"
        request_output = final_res
        if final_res is not None and hasattr(final_res, "request_output"):
            request_output = final_res.request_output or final_res
        request_id = request_output.request_id if request_output is not None else "unknown"
        context = f"pipeline={pipeline}, request_id={request_id}"
        header = _media_artifact_header(final_res.multimodal_output if final_res is not None else None, context=context)
        if header is None:
            if final_res is not None and final_res.images:
                raise ValueError(
                    f"{context}: named media_artifacts declaration required; legacy tensor/tuple output is unsupported"
                )
            finish_reason = _finish_reason(request_output, default="abort")
            stop_reason = self._map_stop_reason(finish_reason)
            logger.debug(
                "diffusion rollout produced no image (finish_reason=%s); returning %s", finish_reason, stop_reason
            )
            return DiffusionOutput(
                diffusion_output=torch.empty(0, dtype=torch.float32 if output_type == "latent" else torch.uint8),
                log_probs=None,
                stop_reason=stop_reason,
                num_preempted=None,
                extra_fields={"global_steps": self.server.global_steps},
            )

        artifacts = self._parse_media_artifacts(final_res, header, sampling_params, context=context)
        primary_name = header["primary"]
        primary = artifacts[primary_name]
        fields = self._training_output_fields(final_res, context=context)
        if fields.get("media_kind", primary.spec.modality) != primary.spec.modality:
            raise ValueError(f"{context}: media_kind conflicts with primary artifact")
        fields["media_kind"] = primary.spec.modality
        if header["audio"] is not None:
            audio = select_artifact(artifacts, name=header["audio"], modality="audio", representation="decoded")
            if "audio" in fields and not torch.equal(torch.as_tensor(fields["audio"]), audio.data):
                raise ValueError(f"{context}: audio metadata conflicts with named audio artifact")
            if fields.get("audio_sample_rate", audio.spec.sample_rate) != audio.spec.sample_rate:
                raise ValueError(f"{context}: audio_sample_rate conflicts with named audio artifact")
            fields["audio"] = audio.data
            fields["audio_sample_rate"] = audio.spec.sample_rate
        log_probs = (
            _unbatch_training_field(final_res.trajectory_log_probs, context=context, name="trajectory_log_probs")
            if sampling_params.get("logprobs", False)
            else None
        )
        return DiffusionOutput(
            artifacts=artifacts,
            primary_artifact=primary_name,
            preview_artifact=header["preview"],
            diffusion_output=primary.data,
            log_probs=log_probs,
            stop_reason=self._map_stop_reason(_finish_reason(request_output, default="stop")),
            num_preempted=self._extract_num_preempted(request_output),
            extra_fields=fields,
        )
