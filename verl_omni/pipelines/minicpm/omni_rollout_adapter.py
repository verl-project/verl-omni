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
"""MiniCPM-o 4.5 rollout pipeline adapter (thinker-only, text output).

vLLM-Omni ships only the 3-stage MiniCPM-o 4.5 pipeline whose stage 0 emits
``engine_output_type="latent"`` for the Talker. RL training needs a single
text-output stage, so this adapter clones stage 0 at runtime with
``engine_output_type="text"`` and registers the clone. Upstream a
``MINICPMO_4_5_THINKER_ONLY_PIPELINE`` to vLLM-Omni and delete this clone.
"""

from dataclasses import replace

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

MINICPMO_4_5_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5_thinker_only",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    # Same 4.5 routing contract as the 3-stage pipeline: the shared
    # ``MiniCPMO`` architecture name must not also capture 2.6 checkpoints.
    hf_architectures=("MiniCPMO", "MiniCPMO45OmniForConditionalGeneration"),
    hf_config_predicate=lambda config: str(getattr(config, "version", "")) == "4.5",
    # A fresh PipelineConfig carries no duplex_runtime_extension /
    # duplex_control_enabled / default_deploy_config_name, so deploy-config
    # resolution by field value cannot pull the 3-stage minicpmo_4_5.yaml.
    stages=(
        replace(
            MINICPMO_4_5_PIPELINE.stages[0],
            engine_output_type="text",
        ),
    ),
)


@OmniRolloutPipelineBase.register("minicpmo_4_5")
class MiniCPMORolloutAdapter(OmniRolloutPipelineBase):
    """Thinker-only rollout topology for MiniCPM-o 4.5 (image+audio in, text out).

    Registered under ``model_type="minicpmo_4_5"`` (the recipe's
    ``pipeline_name``). ``thinker_only`` is the only implemented mode; the
    Talker / Code2Wav stages are out of scope for RL training.

    AVQA training keeps both encoders live: ``get_engine_hf_overrides``
    returns no ``init_audio=False`` (the apm Whisper tower must build), and
    the image multimodal limit stays above zero so vpm/resampler build.
    """

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        """Return the single thinker stage cloned from vLLM-Omni's pipeline."""
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"MiniCPMORolloutAdapter implements thinker_only only, got {pipeline_mode!r}. "
                "Talker / Code2Wav rollout is not part of MiniCPM-o RL training."
            )
        stages = list(MINICPMO_4_5_THINKER_ONLY_PIPELINE.stages)
        assert len(stages) == 1, (
            f"Expected 1 stage in the thinker-only pipeline, got {len(stages)}. "
            "The runtime clone of MINICPMO_4_5_PIPELINE.stages[0] is wrong."
        )
        return stages

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        """Return the clone's model_type, not the 3-stage ``minicpmo_4_5`` id."""
        return MINICPMO_4_5_THINKER_ONLY_PIPELINE.model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "thinker_only") -> None:
        """Register the runtime-cloned thinker-only pipeline in vLLM-Omni."""
        register_pipeline(MINICPMO_4_5_THINKER_ONLY_PIPELINE)

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "thinker_only") -> dict:
        """Keep vision and audio towers enabled — AVQA prompts carry both."""
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        """Pin stage 0 to the plain vLLM LLM class, on the sync scheduler.

        ``MiniCPMO45OmniLLMForConditionalGeneration`` is a standard vLLM LLM
        class: normal logprob support (the omni wrapper hardcodes
        ``logprobs_tensors=None``) and thinker-LLM-only weights, which the
        merged-LoRA sync path (`llm.` → `thinker.` remap) targets.

        ``async_scheduling=False`` mirrors the upstream MiniCPM-o deploy
        profiles: vllm-omni's AR async scheduler never forwards ``is_stale``
        to the base scheduler, and frames that escape its drain predicates
        after a zeroing event decrement an already-zero
        ``num_output_placeholders`` (assert in vllm's async_scheduler.py) —
        hit minutes into an RL rollout once KV-cache pressure starts
        preempting. This key is engine-owned and flows from engine_extras
        into the stage SchedulerConfig.
        """
        if pipeline_mode == "thinker_only" and stage_id == 0:
            return {
                "model_arch": "MiniCPMO45OmniLLMForConditionalGeneration",
                "async_scheduling": False,
            }
        return {}
