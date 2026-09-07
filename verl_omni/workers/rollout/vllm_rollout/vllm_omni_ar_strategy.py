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
import json
import logging
import os
import tempfile
from argparse import Namespace
from typing import Any, Optional

import torch
import yaml
from verl.utils.device import get_visible_devices_keyword
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
from vllm import SamplingParams
from vllm_omni.lora.request import LoRARequest

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import OmniStrategyBase

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_WORKER_EXTENSION = "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"


def _drop_none_mapping_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_mapping_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_mapping_values(item) for item in value]
    return value


class ARStrategy(OmniStrategyBase):
    """Concrete AR/thinker strategy.

    Token-centric I/O: converts configs to ``RolloutConfig``/``OmniModelConfig``,
    writes the deploy config, prepares token-id prompts, and produces
    ``TokenOutput`` from :meth:`process_output`.
    """

    rollout_config_cls = RolloutConfig
    model_config_cls = OmniModelConfig

    def validate_configs(self) -> None:
        if self.server.config.max_model_len is None:
            self.server.config.max_model_len = self.server.config.prompt_length + self.server.config.response_length

    def apply_quantization(self) -> tuple[str | None, dict[str, Any]]:
        return vLLMHttpServer._apply_quantization(self.server)

    def override_generation_config(self) -> dict[str, Any]:
        return vLLMHttpServer._get_override_generation_config(self.server)

    def worker_extension_cls(self, device_type: str) -> str:
        return _WORKER_EXTENSION

    def preprocess_engine_kwargs(self, engine_kwargs: dict[str, Any]) -> None:
        super().preprocess_engine_kwargs(engine_kwargs)
        engine_kwargs.pop("custom_pipeline", None)
        # TODO (mike): drop this later. It should be inferred from the model config.
        pipeline_name = engine_kwargs.pop("pipeline_name", None)
        pipeline_mode = engine_kwargs.pop("pipeline_mode", "thinker_only")

        adapter_cls = OmniRolloutPipelineBase.get_class(pipeline_name)
        if adapter_cls is not None:
            self._write_deploy_config(engine_kwargs, pipeline_name, adapter_cls, pipeline_mode)
            self.server._rollout_flags = adapter_cls.rollout_flags(pipeline_mode=pipeline_mode)
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

    def _write_deploy_config(
        self, engine_kwargs: dict[str, Any], pipeline_name: str, adapter_cls: Any, pipeline_mode: str
    ) -> None:
        """Write a deploy config YAML from the adapter's stage topology."""
        adapter_cls.ensure_pipeline_registered(pipeline_mode)
        stages = adapter_cls.build_stage_configs(pipeline_mode=pipeline_mode)
        pipeline_id = adapter_cls.get_pipeline_id(pipeline_mode)

        device_control_env = get_visible_devices_keyword()
        visible_devices = os.environ.get(device_control_env, "")
        tp_size = self.server.config.tensor_model_parallel_size

        deploy_dict: dict[str, object] = {"pipeline": pipeline_id}

        if visible_devices:
            device_count = len([device for device in visible_devices.split(",") if device.strip()])
            devices = ",".join(str(device_id) for device_id in range(device_count))
            stage_ids = [stage.stage_id for stage in stages]
            deploy_dict["stages"] = [
                {
                    "stage_id": stage_id,
                    "devices": devices,
                    "tensor_parallel_size": tp_size,
                    "text_encoder_tp_size": getattr(self.server.config, "text_encoder_tp_size", 1),
                    "engine_extras": adapter_cls.get_stage_engine_extras(stage_id, pipeline_mode=pipeline_mode),
                }
                for stage_id in stage_ids
            ]
        else:
            raise RuntimeError(
                f"Environment variable `{device_control_env}` is not set, cannot generate deploy config."
            )

        yaml_str = yaml.dump(deploy_dict).strip()
        logger.info("Generated deploy config:\n%s", yaml_str)
        self.server._temp_deploy_ctx = tempfile.TemporaryDirectory(prefix="verl_omni_deploy_")
        deploy_path = os.path.join(self.server._temp_deploy_ctx.name, f"{pipeline_name}.yaml")
        try:
            with open(deploy_path, "w") as file:
                file.write(yaml_str)
        except OSError as error:
            self.server._temp_deploy_ctx.cleanup()
            self.server._temp_deploy_ctx = None
            raise RuntimeError(f"Failed to write deploy config to {deploy_path}: {error}") from error
        engine_kwargs["deploy_config"] = deploy_path

    def prepare_engine_args(self, engine_args: dict[str, Any], args: Namespace) -> None:
        for timeout_key in ("stage_init_timeout", "init_timeout"):
            timeout_value = getattr(args, timeout_key, None)
            if timeout_value is not None:
                engine_args[timeout_key] = int(timeout_value)
        engine_args["logprobs_mode"] = getattr(self.server.config, "logprobs_mode", "processed_logprobs")
        if isinstance(engine_args.get("compilation_config"), dict):
            engine_args["compilation_config"] = _drop_none_mapping_values(engine_args["compilation_config"])

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
    ) -> tuple[dict[str, Any], SamplingParams]:
        if multi_modal_data:
            processor = getattr(self.server.model_config, "processor", None)
            if processor is not None and hasattr(processor, "dedup_pad_tokens"):
                prompt_ids = processor.dedup_pad_tokens(prompt_ids)
        max_possible_tokens = self.server.config.max_model_len - len(prompt_ids)
        if max_possible_tokens <= 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) meets or exceeds the model's maximum context length "
                f"({self.server.config.max_model_len}), leaving no space for generation."
            )

        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.server.config.response_length,
                self.server.config.prompt_length + self.server.config.response_length - len(prompt_ids),
            )
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        logprobs = sampling_params.pop("logprobs", None)
        if logprobs is True:
            sampling_params["logprobs"] = 0
        elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
            sampling_params["logprobs"] = logprobs
        else:
            sampling_params["logprobs"] = None
        sampling_params.setdefault("repetition_penalty", getattr(self.server.config, "repetition_penalty", 1.0))
        params = SamplingParams(max_tokens=max_tokens, **sampling_params)

        prompt = {"prompt_token_ids": prompt_ids}
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs
        return prompt, params

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
                sampling_params_list=params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
        )

    def process_output(self, final_res: Any, params: SamplingParams, sampling_params: dict[str, Any]) -> TokenOutput:
        if final_res is None:
            raise RuntimeError("AR mode: vLLM-Omni engine yielded no output for the prompt.")

        req_output = getattr(final_res, "request_output", None) or final_res
        if not req_output.outputs:
            raise RuntimeError("AR mode expects outputs with token IDs, but got None or empty.")

        extra_fields = {"global_steps": self.server.global_steps}
        token_ids = req_output.outputs[0].token_ids
        log_probs = None
        if params.logprobs is not None:
            log_probs = [
                logprobs[token_ids[index]].logprob for index, logprobs in enumerate(req_output.outputs[0].logprobs)
            ]

        finish_reason = req_output.outputs[0].finish_reason
        stop_reason = self._map_stop_reason(finish_reason)
        num_preempted = self._extract_num_preempted(req_output)

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )
