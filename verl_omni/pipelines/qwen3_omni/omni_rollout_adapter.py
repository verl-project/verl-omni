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
"""Qwen3-Omni vLLM-Omni rollout adapter."""

from typing import Any

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

__all__ = ["Qwen3OmniRolloutAdapter"]


@OmniRolloutPipelineBase.register("qwen3_omni_moe")
class Qwen3OmniRolloutAdapter(OmniRolloutPipelineBase):
    """Rollout topology adapter for Qwen3-Omni."""

    @classmethod
    def get_pipeline_model_type(cls) -> str:
        return "qwen3_omni_moe"

    @classmethod
    def get_stage_config(cls, pipeline_mode: str = "thinker_only", num_gpus: int = 8) -> list[dict[str, Any]]:
        if pipeline_mode != "thinker_only":
            raise NotImplementedError(
                f"Qwen3-Omni rollout pipeline_mode={pipeline_mode!r} is not implemented yet. "
                "Only thinker_only is supported for the initial DPO migration."
            )

        return [
            {
                "stage_id": 0,
                "runtime": {"devices": ",".join(str(i) for i in range(num_gpus))},
                "engine_args": {
                    "model_stage": "thinker",
                    "model_arch": "Qwen3OmniMoeThinkerForConditionalGeneration",
                    "worker_type": "ar",
                    "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                    "engine_output_type": "text",
                    "hf_config_name": "thinker_config",
                    "tensor_parallel_size": num_gpus,
                    "distributed_executor_backend": "mp",
                    "dtype": "bfloat16",
                    "load_format": "safetensors",
                    "trust_remote_code": True,
                    "enable_prefix_caching": False,
                    "enforce_eager": True,
                    "enable_lora": True,
                    "max_lora_rank": 64,
                    "max_loras": 1,
                    "enable_sleep_mode": True,
                    "logprobs_mode": "processed_logprobs",
                    "disable_log_stats": True,
                },
                "final_output": True,
                "final_output_type": "text",
                "is_comprehension": True,
            }
        ]

    @classmethod
    def get_deploy_config(cls, pipeline_mode: str = "thinker_only", num_gpus: int = 8) -> dict[str, Any]:
        return {"stage_args": cls.get_stage_config(pipeline_mode=pipeline_mode, num_gpus=num_gpus)}
