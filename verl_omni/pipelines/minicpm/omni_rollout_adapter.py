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
"""MiniCPM-o 4.5 rollout pipeline adapter (thinker-only and thinker-talker).

vLLM-Omni ships only the 3-stage MiniCPM-o 4.5 pipeline, whose stage 0 emits
``engine_output_type="latent"`` for the Talker. Thinker-only RL needs a single
text-output stage, so that mode clones stage 0 at runtime with
``engine_output_type="text"`` and registers the clone.

``thinker_talker`` (talker-stage training, verl-omni#565) keeps the original
stage-0 latent bridge and promotes the Talker to the final codec-output stage:
the Thinker generates the text response on-policy, the Talker speaks it, and
the batch's response tokens are the codec frames themselves (the real prompt
travels in the per-sample replay payload — see
``verl_omni.pipelines.minicpm.talker_forward``).
"""

from dataclasses import replace
from typing import Any

import torch
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

from verl_omni.pipelines.minicpm.prompt_parity import resolve_media_tokens
from verl_omni.pipelines.minicpm.talker_forward import MINICPM_TALKER_REPLAY_KEY
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

MINICPMO_4_5_TALKER_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5_talker",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    hf_architectures=("MiniCPMO", "MiniCPMO45OmniForConditionalGeneration"),
    hf_config_predicate=lambda config: str(getattr(config, "version", "")) == "4.5",
    stages=(
        # Stage 0: the thinker, final as text. Both stages are final: the
        # engine runs the serial bridge (thinker -> llm2tts -> talker) and the
        # rollout adapter combines the text response with the codec frames.
        # This requires async_chunk=false (see below): the engine's async-chunk
        # prewarm otherwise submits unconditioned placeholder requests to the
        # talker stage, whose preprocess requires the bridge tensors.
        MINICPMO_4_5_PIPELINE.stages[0],
        # Stage 1: the talker, promoted to the pipeline's final codec output.
        replace(
            MINICPMO_4_5_PIPELINE.stages[1],
            final_output=True,
            final_output_type="codec",
        ),
    ),
)


@OmniRolloutPipelineBase.register("minicpmo_4_5")
class MiniCPMORolloutAdapter(OmniRolloutPipelineBase):
    # thinker_talker rejects async_chunk: the engine's prewarm submits
    # unconditioned placeholder requests to the talker stage (its preprocess
    # requires the thinker bridge tensors). thinker_only keeps the default
    # behavior; the thinker_talker recipe sets async_chunk=false explicitly.
    supports_async_chunk = True
    """Rollout pipeline topology adapter for MiniCPM-o 4.5.

    Registered under ``model_type="minicpmo_4_5"``. Stage topology comes from
    vLLM-Omni's ``MINICPMO_4_5_PIPELINE``:

    - ``thinker_only`` — stage 0 cloned with ``engine_output_type="text"``.
    - ``thinker_talker`` — stages 0-1 (Thinker latent bridge + Talker codec
      output); the Talker is the trained policy stage.
    """

    @classmethod
    def _check_mode(cls, pipeline_mode):
        if pipeline_mode not in ("thinker_only", "thinker_talker"):
            raise ValueError(
                f"MiniCPMORolloutAdapter implements thinker_only / thinker_talker only, got {pipeline_mode!r}."
            )

    @classmethod
    def validate_async_chunk(cls, pipeline_mode: str, async_chunk: bool | None) -> bool:
        """Resolve async_chunk per pipeline mode.

        thinker_only keeps the engine default (absent → True is valid).
        thinker_talker requires serial execution: an absent value defaults
        to False (the engine's async-chunk prewarm submits unconditioned
        placeholder requests to the talker stage), and an explicit True
        is rejected.
        """
        if pipeline_mode != "thinker_talker":
            return async_chunk if async_chunk is not None else True
        if async_chunk is True:
            raise ValueError(
                "MiniCPMORolloutAdapter thinker_talker mode requires async_chunk=false: "
                "the engine's async-chunk prewarm submits unconditioned placeholder requests "
                "to the talker stage (its preprocess requires the thinker bridge tensors)."
            )
        return False

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        """Return the pipeline stages for *pipeline_mode*.

        Args:
            pipeline_mode (str): ``thinker_only`` or ``thinker_talker``.

        Returns:
            list: The per-stage pipeline topology objects.
        """
        cls._check_mode(pipeline_mode)
        if pipeline_mode == "thinker_only":
            stages = list(MINICPMO_4_5_THINKER_ONLY_PIPELINE.stages)
            # Guard against upstream changes that silently add stages.
            assert len(stages) == 1, (
                f"Expected 1 stage in the thinker-only pipeline, got {len(stages)}. "
                "The runtime clone of MINICPMO_4_5_PIPELINE.stages[0] is wrong."
            )
            return stages
        stages = list(MINICPMO_4_5_TALKER_PIPELINE.stages)
        assert len(stages) == 2, (
            f"Expected 2 stages in the talker pipeline, got {len(stages)}. "
            "The runtime clone of MINICPMO_4_5_PIPELINE.stages[:2] is wrong."
        )
        return stages

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        """Return the runtime clone's model_type for *pipeline_mode*."""
        cls._check_mode(pipeline_mode)
        if pipeline_mode == "thinker_only":
            return MINICPMO_4_5_THINKER_ONLY_PIPELINE.model_type
        return MINICPMO_4_5_TALKER_PIPELINE.model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "thinker_only") -> None:
        """Register the runtime-cloned pipeline in vLLM-Omni."""
        cls._check_mode(pipeline_mode)
        if pipeline_mode == "thinker_only":
            register_pipeline(MINICPMO_4_5_THINKER_ONLY_PIPELINE)
        else:
            register_pipeline(MINICPMO_4_5_TALKER_PIPELINE)

    @classmethod
    def rollout_flags(cls, pipeline_mode="thinker_only") -> dict[int, dict]:
        """Return per-stage rollout flags for *pipeline_mode*.

        In ``thinker_talker`` mode the Thinker's output is an internal latent
        (not a final output) and the Talker emits the final codec frames.
        """
        cls._check_mode(pipeline_mode)
        if pipeline_mode == "thinker_talker":
            return {
                0: {"return_hidden_states": True, "final_output": False, "final_output_type": None},
                1: {"final_output": True, "final_output_type": "codec"},
            }
        return {}

    @classmethod
    def policy_stage_id(cls, pipeline_mode: str = "thinker_only") -> int:
        """The Talker owns the trained policy in ``thinker_talker`` mode."""
        cls._check_mode(pipeline_mode)
        if pipeline_mode == "thinker_talker":
            return 1
        return 0

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="thinker_only") -> list[int] | None:
        """Sync only the Talker engine; the frozen Thinker never changes.

        ``None`` keeps the default (sync all stages) for thinker_only, where
        the single LLM stage is the merged-LoRA target.
        """
        cls._check_mode(pipeline_mode)
        if pipeline_mode == "thinker_talker":
            return [1]
        return None

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode="thinker_only") -> dict:
        """Keep both encoders enabled — prompts may carry vision and audio."""
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode="thinker_only", model_config=None) -> dict:
        """Return per-stage engine overrides for *pipeline_mode*.

        Args:
            stage_id: Zero-based pipeline stage index.
            pipeline_mode (str): ``thinker_only`` or ``thinker_talker``.

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
        if pipeline_mode == "thinker_talker" and stage_id == 0:
            # The latent-bridge stage needs the same async-scheduling pin.
            return {"async_scheduling": False}
        return {}

    @classmethod
    def combine_engine_outputs(cls, outputs: list, prompt: dict) -> tuple[Any, dict[str, Any]]:
        """Combine the Thinker's text response with the Talker's codec frames.

        In ``thinker_talker`` rollouts the engine returns the per-stage
        outputs; the Talker's completion carries the codec trajectory and the
        Thinker's completion the spoken text ids.
        """
        thinker_outputs = [output for output in outputs if getattr(output, "stage_id", 0) == 0]
        talker_outputs = [output for output in outputs if getattr(output, "stage_id", 1) == 1]
        if not talker_outputs:
            raise RuntimeError("MiniCPM-o talker rollout produced no stage-1 (Talker) output.")
        talker_output = talker_outputs[-1]
        if len(talker_output.outputs) != 1:
            raise RuntimeError(
                f"MiniCPM-o stage 1 must return exactly one completion, got {len(talker_output.outputs)}."
            )
        completion = talker_output.outputs[0]
        codec_ids = list(getattr(completion, "token_ids", None) or [])
        if not codec_ids:
            raise RuntimeError("MiniCPM-o talker rollout returned an empty codec trajectory.")

        text_response_ids: list[int] = []
        if thinker_outputs:
            thinker_output = thinker_outputs[-1]
            if thinker_output.outputs:
                text_response_ids = [int(token) for token in (thinker_output.outputs[0].token_ids or [])]

        fields = {
            "tts_audio_codes": torch.as_tensor(codec_ids, dtype=torch.long).detach().cpu(),
            "tts_text_response_ids": text_response_ids,
        }
        return talker_output, fields

    @classmethod
    def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length: int):
        """Map the Talker rollout to codec policy tokens plus the replay payload.

        The batch's response tokens become the codec frames; the real prompt
        and the Thinker's text response travel in the per-sample replay
        payload, which the talker training adapter unpacks to rebuild the
        thinker conditioning.
        """
        extra = output.extra_fields
        if "tts_audio_codes" not in extra or "tts_text_response_ids" not in extra:
            raise RuntimeError("MiniCPM-o talker rollout did not return codec codes and text ids.")
        if MINICPM_TALKER_REPLAY_KEY in extra:
            raise RuntimeError(
                f"MiniCPM-o talker rollout unexpectedly returned reserved field {MINICPM_TALKER_REPLAY_KEY!r}."
            )
        codes = torch.as_tensor(extra["tts_audio_codes"], dtype=torch.long).reshape(-1)
        text_response_ids = [int(token) for token in extra["tts_text_response_ids"]]
        codes = codes[:response_length]
        policy_ids = codes.tolist()
        if not policy_ids:
            raise RuntimeError("MiniCPM-o talker rollout returned an empty codec trajectory.")
        if output.response_logprobs is not None:
            if len(output.response_logprobs) < len(policy_ids):
                raise RuntimeError("MiniCPM-o talker rollout logprobs are shorter than the policy trajectory.")
            output.response_logprobs = output.response_logprobs[: len(policy_ids)]

        # The replayed thinker sequence: the processor-expanded prompt ids
        # followed by the Thinker's text response. The conditioning span
        # follows the official chat convention: the text between the last
        # <|tts_bos|> and the last <|tts_eos|> marker (or the end when the
        # model did not emit the closing marker).
        prompt_ids = [int(token) for token in (output.prompt_ids or [])]
        full_text_ids = prompt_ids + text_response_ids
        tts_bos = tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        tts_eos = tokenizer.convert_tokens_to_ids("<|tts_eos|>")
        bos_positions = [i for i, token in enumerate(full_text_ids) if token == tts_bos]
        span_start = (bos_positions[-1] + 1) if bos_positions else len(prompt_ids)
        eos_after_start = [i for i in range(span_start, len(full_text_ids)) if full_text_ids[i] == tts_eos]
        span_end = eos_after_start[0] if eos_after_start else len(full_text_ids)
        if not 0 < span_start < span_end <= len(full_text_ids):
            raise RuntimeError(
                f"MiniCPM-o talker replay span is invalid: [{span_start}, {span_end}) "
                f"over {len(full_text_ids)} text tokens."
            )
        extra[MINICPM_TALKER_REPLAY_KEY] = {
            "text_ids": full_text_ids,
            "response_start": span_start,
            "response_end": span_end,
            "audio_codes": codes.tolist(),
        }
        del extra["tts_audio_codes"]
        del extra["tts_text_response_ids"]
        output.prompt_ids = [0]
        output.response_ids = policy_ids
        output.response_mask = [1] * len(policy_ids)
        return output

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        """Collapse the request's expanded media spans into the slots vLLM-Omni re-expands.

        Rollout and teacher requests both carry processor-expanded ids; a teacher
        request also carries the student's response after the prompt. Only as many
        spans as the request has media are collapsed, so the response passes
        through verbatim.

        Args:
            prompt_ids: Processor-expanded ids, optionally followed by response ids.
            model_config: The server's ``OmniModelConfig`` carrying the processor.
            multi_modal_data: Engine media keyed by ``image`` / ``audio``.
            mm_processor_kwargs: Processor options forwarded to the engine.

        Returns:
            The engine prompt, or None for a text-only request.
        """
        if not multi_modal_data:
            return None
        prompt = {
            "prompt_token_ids": resolve_media_tokens(model_config.processor).collapse_ids_to_slots(
                prompt_ids,
                num_images=len(multi_modal_data.get("image") or []),
                num_audios=len(multi_modal_data.get("audio") or []),
            )
        }
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs
        return prompt
