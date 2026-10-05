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
"""MiniCPM-o 4.5 rollout pipeline adapter (thinker-only, text output, token-native replay).

vLLM-Omni ships only the 3-stage MiniCPM-o 4.5 pipeline, whose stage 0 emits
``engine_output_type="latent"`` for the Talker. RL training needs a single
text-output stage, so this adapter clones stage 0 at runtime with
``engine_output_type="text"`` and registers the clone.
"""

from dataclasses import replace
from functools import wraps

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

from verl_omni.pipelines.minicpm.vllm_plugin import assert_entry_point_installed
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

MINICPM_PROMPT_KEY = "minicpm_prompt"
_MINICPM_PROCESSED_PROMPT_KEY = "_verl_minicpm_processed_prompt"


def _install_token_native_multimodal_replay() -> None:
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMO45OmniLLMMultiModalProcessor,
    )

    processor_cls = MiniCPMO45OmniLLMMultiModalProcessor
    if hasattr(processor_cls, "_verl_original_cached_apply_hf_processor"):
        return
    original = processor_cls._cached_apply_hf_processor

    @wraps(original)
    def apply_with_preserved_tokens(self, inputs, timing_ctx):
        replay = inputs.hf_processor_mm_kwargs.get(_MINICPM_PROCESSED_PROMPT_KEY)
        if replay is None:
            return original(self, inputs, timing_ctx)
        if not isinstance(inputs.prompt, list):
            raise TypeError("MiniCPM-o replay requires token IDs as the serving prompt.")
        processor_kwargs = dict(inputs.hf_processor_mm_kwargs)
        processor_kwargs.pop(_MINICPM_PROCESSED_PROMPT_KEY)
        source_inputs = ProcessorInputs(
            prompt=replay["source_ids"],
            mm_data_items=inputs.mm_data_items,
            mm_uuid_items=inputs.mm_uuid_items,
            hf_processor_mm_kwargs=processor_kwargs,
            tokenization_kwargs=inputs.tokenization_kwargs,
        )
        source_ids, mm_info, is_update_applied = original(self, source_inputs, timing_ctx)
        source_ids, _ = self._maybe_apply_prompt_updates(
            inputs.mm_data_items,
            source_ids,
            mm_info.kwargs,
            mm_info.prompt_updates,
            is_update_applied,
        )
        expanded_ids = replay["expanded_ids"]
        if source_ids != expanded_ids:
            raise ValueError("MiniCPM-o serving expansion differs from the actor's processed prompt.")
        if inputs.prompt[: len(expanded_ids)] != expanded_ids:
            raise ValueError("MiniCPM-o serving prompt no longer starts with the processed actor prompt.")
        return list(inputs.prompt), mm_info, True

    processor_cls._verl_original_cached_apply_hf_processor = original
    processor_cls._cached_apply_hf_processor = apply_with_preserved_tokens


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
    """Rollout pipeline topology adapter for MiniCPM-o 4.5.

    Registered under ``model_type="minicpmo_4_5"``. Stage topology is a runtime
    clone of vLLM-Omni's ``MINICPMO_4_5_PIPELINE`` stage 0, with
    ``engine_output_type`` flipped from ``latent`` (the Talker bridge) to
    ``text``.

    ``thinker_only`` is the only mode this adapter implements: MiniCPM-o's
    Talker and Code2Wav stages are not part of this integration, and training
    them would need their own adapter.
    """

    supports_async_chunk = False

    @classmethod
    def _pipeline(cls, pipeline_mode="thinker_only"):
        if pipeline_mode != "thinker_only":
            raise ValueError("MiniCPM-o simplex training supports pipeline_mode=thinker_only only.")
        return MINICPMO_4_5_THINKER_ONLY_PIPELINE

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="thinker_only"):
        cls.build_stage_configs(pipeline_mode)
        return [0]

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        """Return the single thinker stage cloned from vLLM-Omni's pipeline.

        Args:
            pipeline_mode (str): Pipeline mode selector; ``thinker_only`` only.

        Returns:
            list: The one text-output stage this adapter supports.

        Raises:
            ValueError: When a Talker / Code2Wav mode is requested.
        """
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"MiniCPMORolloutAdapter implements thinker_only only, got {pipeline_mode!r}. "
                "MiniCPM-o's Talker / Code2Wav rollout is not implemented by this integration."
            )
        stages = list(MINICPMO_4_5_THINKER_ONLY_PIPELINE.stages)
        # Guard against upstream changes that silently add stages.
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
        """Register the runtime-cloned thinker-only pipeline in vLLM-Omni.

        The entry-point assertion runs first, before the engine cores spawn, so a
        stale plugin install is reported here rather than downstream.
        """
        # TODO (mike): drop with the vllm_plugin module at the pin bump — the guard
        # fails once the plugin entry point is legitimately gone.
        assert_entry_point_installed()
        register_pipeline(MINICPMO_4_5_THINKER_ONLY_PIPELINE)

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "thinker_only") -> dict:
        """Keep both encoders enabled — prompts may carry vision and audio."""
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        """Return per-stage engine overrides for *pipeline_mode*.

        Args:
            stage_id: Zero-based pipeline stage index.
            pipeline_mode (str): Pipeline mode selector; ``thinker_only`` only.

        Returns:
            dict: Engine kwargs for the stage, empty when nothing is overridden.
        """
        if pipeline_mode == "thinker_only" and stage_id == 0:
            return {
                # A plain vLLM LLM class: logprob support (the omni wrapper hardcodes
                # logprobs_tensors=None) and the thinker-LLM-only weight names the
                # merged-LoRA sync targets.
                "model_arch": "MiniCPMO45OmniLLMForConditionalGeneration",
                # TODO (mike): drop once vllm-omni's AR async scheduler forwards is_stale.
                # It drops the flag, so frames escaping its drain predicates after a
                # zeroing event trip an assert in vllm's async_scheduler.py once KV-cache
                # pressure starts preempting; upstream MiniCPM-o deploy profiles pin this too.
                "async_scheduling": False,
            }
        return {}

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        processor_kwargs = dict(mm_processor_kwargs or {})
        replay = processor_kwargs.pop(MINICPM_PROMPT_KEY, None)
        if replay is None:
            if multi_modal_data:
                raise ValueError("MiniCPM-o multimodal rollout requires the minicpm_simplex_agent prompt contract.")
            effective_ids = list(prompt_ids)
        else:
            prefix = replay["expanded_ids"]
            if list(prompt_ids[: len(prefix)]) != prefix:
                raise ValueError("MiniCPM-o rollout/teacher prefix differs from the actor's processed prompt.")
            if multi_modal_data:
                _install_token_native_multimodal_replay()
                processor_kwargs[_MINICPM_PROCESSED_PROMPT_KEY] = replay
                effective_ids = list(prompt_ids)
            else:
                effective_ids = [*replay["source_ids"], *prompt_ids[len(prefix) :]]
        return {"prompt_token_ids": effective_ids, "mm_processor_kwargs": processor_kwargs}


# Backward-compatible name used by the simplex OPD agent loop and tests.
MiniCPMRolloutAdapter = MiniCPMORolloutAdapter
