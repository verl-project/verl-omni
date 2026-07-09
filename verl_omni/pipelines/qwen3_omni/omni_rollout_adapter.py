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
"""Qwen3-Omni rollout pipeline adapter.

Generates per-stage ``engine_args`` topology defaults for running
Qwen3-Omni as a multi-stage pipeline in vLLM-Omni.
"""

import logging

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

logger = logging.getLogger(__name__)


@OmniRolloutPipelineBase.register("qwen3_omni_moe")
class Qwen3OmniRolloutAdapter(OmniRolloutPipelineBase):
    """Rollout pipeline topology adapter for Qwen3-Omni.

    Registered under ``model_type="qwen3_omni_moe"``.  Supports three
    pipeline modes: ``"thinker_only"`` (AR text), ``"thinker_talker"``
    (speech codec tokens), and ``"full"`` (audio waveform).
    """

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list[dict]:
        """Return per-stage model-topology defaults for vLLM-Omni.

        Args:
            pipeline_mode: ``"thinker_only"`` | ``"thinker_talker"`` | ``"full"``.

        Returns:
            list[dict]: One topology dict per pipeline stage.
        """
        thinker_engine = {
            "model_stage": "thinker",
            "model_arch": "Qwen3OmniMoeThinkerForConditionalGeneration",
            "worker_type": "ar",
            "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
            "hf_config_name": "thinker_config",
        }
        thinker = {
            "stage_id": 0,
            "engine_args": thinker_engine,
            "final_output": True,
            "final_output_type": "text",
        }
        thinker_with_hidden = {
            "stage_id": 0,
            "engine_args": {**thinker_engine, "return_hidden_states": True},
            "final_output": False,
        }
        talker = {
            "stage_id": 1,
            "engine_args": {
                "model_stage": "talker",
                "model_arch": "Qwen3OmniMoeTalkerForConditionalGeneration",
                "worker_type": "codec",
                "hf_config_name": "talker_config",
            },
        }
        code2wav = {
            "stage_id": 2,
            "engine_args": {
                "model_stage": "code2wav",
                "model_arch": "Qwen3OmniMoeCode2WavForConditionalGeneration",
                "worker_type": "waveform",
            },
        }

        if pipeline_mode == "thinker_only":
            return [thinker]
        elif pipeline_mode == "thinker_talker":
            return [
                thinker_with_hidden,
                {**talker, "final_output": True, "final_output_type": "codec"},
            ]
        elif pipeline_mode == "full":
            return [
                thinker_with_hidden,
                talker,
                {**code2wav, "final_output": True, "final_output_type": "waveform"},
            ]
        else:
            raise ValueError(
                f"Unknown pipeline_mode={pipeline_mode!r}. Expected one of: 'thinker_only', 'thinker_talker', 'full'."
            )
