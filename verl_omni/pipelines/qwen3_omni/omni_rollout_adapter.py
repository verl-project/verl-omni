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

Provides the per-stage pipeline topology for Qwen3-Omni by delegating to
vLLM-Omni's frozen :class:`~vllm_omni.config.stage_config.PipelineConfig`
definitions — no duplication of what vLLM-Omni already owns.
"""

from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
    QWEN3_OMNI_THINKER_ONLY_PIPELINE,
)

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase


@OmniRolloutPipelineBase.register("qwen3_omni_moe")
class Qwen3OmniRolloutAdapter(OmniRolloutPipelineBase):
    """Rollout pipeline topology adapter for Qwen3-Omni.

    Registered under ``model_type="qwen3_omni_moe"``.  Stage topology
    comes unchanged from vLLM-Omni's ``QWEN3_OMNI_PIPELINE`` and
    ``QWEN3_OMNI_THINKER_ONLY_PIPELINE``.

    Three pipeline modes map to subsets of the full 3-stage pipeline
    (thinker → talker → code2wav):

    - ``thinker_only`` — stage 0 (text output).
    - ``thinker_talker`` — stages 0-1 (codec output).
    - ``full`` — stages 0-2 (audio waveform output).
    """

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        """Return per-stage :class:`~vllm_omni.config.stage_config.StagePipelineConfig` objects.

        Args:
            pipeline_mode (str): Pipeline mode selector. One of
                ``thinker_only``, ``thinker_talker``, ``full``.

        Returns:
            list: Per-stage pipeline topology objects from vLLM-Omni.

        ================= ============================================
        pipeline_mode     stages returned
        ================= ============================================
        ``thinker_only`` stage 0 from the thinker-only variant
        ``thinker_talker`` stages 0-1 from the full pipeline
        ``full``         stages 0-2 from the full pipeline
        ================= ============================================

        Stages are returned **as-is** from vLLM-Omni.  Call
        :meth:`rollout_flags` to get model-specific rollout flags per
        stage.
        """
        if pipeline_mode == "thinker_only":
            stages = list(QWEN3_OMNI_THINKER_ONLY_PIPELINE.stages)
            # Guard against upstream changes that silently add stages.
            assert len(stages) == 1, (
                f"Expected 1 stage in thinker-only pipeline, got {len(stages)}. "
                "vLLM-Omni may have changed the pipeline definition."
            )
            return stages
        if pipeline_mode == "thinker_talker":
            return list(QWEN3_OMNI_PIPELINE.stages[:2])
        if pipeline_mode == "full":
            return list(QWEN3_OMNI_PIPELINE.stages)
        raise ValueError(
            f"Unknown pipeline_mode={pipeline_mode!r}. Expected one of: 'thinker_only', 'thinker_talker', 'full'."
        )

    @classmethod
    def rollout_flags(cls, pipeline_mode="thinker_only"):
        """Return per-stage rollout flags for *pipeline_mode*.

        Args:
            pipeline_mode (str): Pipeline mode selector.  One of
                ``thinker_only``, ``thinker_talker``, ``full``.

        Returns:
            dict[int, dict]: Per-stage flags mapping stage IDs to
            ``{return_hidden_states, final_output, final_output_type}``.
            Empty dict for ``thinker_only``.

        ================= ==================================================
        pipeline_mode     flags
        ================= ==================================================
        ``thinker_only`` ``{}`` — no flags needed.
        ``thinker_talker`` stage 0: ``return_hidden_states``, non-terminal.
                            stage 1: terminal ``codec`` output.
        ``full``         stage 0: ``return_hidden_states``, non-terminal.
        ================= ==================================================

        Each per-stage dict contains:

        - ``return_hidden_states`` (bool) — whether the stage should
          emit hidden states for downstream consumers.
        - ``final_output`` (bool) — terminal output stage.
        - ``final_output_type`` (str | None) — output modality
          (``text``, ``codec``, ``audio``).
        """
        if pipeline_mode == "thinker_only":
            return {}
        if pipeline_mode == "thinker_talker":
            return {
                0: {"return_hidden_states": True, "final_output": False, "final_output_type": None},
                1: {"final_output": True, "final_output_type": "codec"},
            }
        if pipeline_mode == "full":
            return {
                0: {"return_hidden_states": True, "final_output": False, "final_output_type": None},
            }
        raise ValueError(
            f"Unknown pipeline_mode={pipeline_mode!r}. Expected one of: 'thinker_only', 'thinker_talker', 'full'."
        )
