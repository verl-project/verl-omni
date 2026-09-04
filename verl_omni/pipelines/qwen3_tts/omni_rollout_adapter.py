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
"""Qwen3-TTS two-stage rollout adapter."""

import hashlib
from dataclasses import replace
from functools import lru_cache

import torch
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_tts.rollout_utils import QWEN3_TTS_REPLAY_KEY, align_audio_codes
from verl_omni.pipelines.qwen3_tts.talker_forward import (
    TEXT_PROMPT_TRAILER_TOKENS,
    build_assistant_text,
    load_speaker_xvector,
    require_auto_language,
)

_PIPELINE_ID = "qwen3_tts_rl"
QWEN3_TTS_RL_PIPELINE = PipelineConfig(
    model_type=_PIPELINE_ID,
    model_arch=QWEN3_TTS_PIPELINE.model_arch,
    stages=(
        replace(QWEN3_TTS_PIPELINE.stages[0], final_output=True, final_output_type="latent"),
        QWEN3_TTS_PIPELINE.stages[1],
    ),
)


@lru_cache(maxsize=4)
def _load_speaker_vector(path: str) -> list[float]:
    return load_speaker_xvector(path).reshape(-1).tolist()


@OmniRolloutPipelineBase.register(_PIPELINE_ID)
class Qwen3TTSRolloutAdapter(OmniRolloutPipelineBase):
    supports_async_chunk = False

    @classmethod
    def _check_mode(cls, pipeline_mode):
        if pipeline_mode != "full":
            raise ValueError("Qwen3-TTS RL supports only pipeline_mode='full'.")

    @classmethod
    def build_stage_configs(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return list(QWEN3_TTS_RL_PIPELINE.stages)

    @classmethod
    def get_pipeline_id(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return _PIPELINE_ID

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        register_pipeline(QWEN3_TTS_RL_PIPELINE)

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="full"):
        """Sync actor weights only to stage 0; stage 1 is the frozen decoder."""
        cls._check_mode(pipeline_mode)
        return [0]

    @classmethod
    def get_stage_engine_extras(cls, stage_id, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        if stage_id == 0:
            return {}
        if stage_id == 1:
            return {"max_model_len": 65536, "max_num_batched_tokens": 65536}
        raise ValueError(f"Qwen3-TTS has no rollout stage {stage_id}.")

    @classmethod
    def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
        """Map the 16-codebook rollout to codec-0 policy tokens and replay fields."""
        extra = output.extra_fields
        if "tts_audio_codes" not in extra or "tts_text" not in extra:
            raise RuntimeError("Qwen3-TTS rollout did not return codec codes and text.")
        if QWEN3_TTS_REPLAY_KEY in extra:
            raise RuntimeError(f"Qwen3-TTS rollout unexpectedly returned reserved field {QWEN3_TTS_REPLAY_KEY!r}.")
        codes, text = extra["tts_audio_codes"], extra["tts_text"]
        if not isinstance(text, str):
            raise TypeError(f"Qwen3-TTS rollout text must be a string, got {type(text).__name__}.")
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[-1] != 16:
            raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
        codes = codes[:response_length]
        policy_ids = codes[:, 0].tolist()
        if not policy_ids:
            raise RuntimeError("Qwen3-TTS rollout returned an empty codec trajectory.")
        if output.response_logprobs is not None:
            if len(output.response_logprobs) < len(policy_ids):
                raise RuntimeError("Qwen3-TTS rollout logprobs are shorter than the policy trajectory.")
            output.response_logprobs = output.response_logprobs[: len(policy_ids)]
        text_ids = tokenizer(build_assistant_text(text), return_tensors="pt", padding=False)["input_ids"]
        text_ids = torch.as_tensor(text_ids, dtype=torch.long)
        if text_ids.ndim == 1:
            text_ids = text_ids.unsqueeze(0)
        if text_ids.ndim != 2 or text_ids.shape[1] <= TEXT_PROMPT_TRAILER_TOKENS:
            raise ValueError("Qwen3-TTS assistant text tokenization returned an invalid sequence.")
        extra[QWEN3_TTS_REPLAY_KEY] = {
            "text_ids": text_ids[:, :-TEXT_PROMPT_TRAILER_TOKENS].reshape(-1).tolist(),
            "audio_codes": codes,
        }
        del extra["tts_audio_codes"]
        del extra["tts_text"]
        output.prompt_ids = [0]
        output.response_ids = policy_ids
        output.response_mask = [1] * len(policy_ids)
        return output

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        """Build the Base-task prompt and fixed-speaker conditioning for vLLM-Omni."""
        text = model_config.tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Qwen3-TTS received an empty text prompt.")
        speaker_path = model_config.override_config.get("tts_spk_embed_path")
        if not speaker_path:
            raise ValueError("Qwen3-TTS GRPO requires tts_spk_embed_path for the validated non-streaming replay.")
        language = require_auto_language(model_config.override_config.get("tts_language"))
        additional_information = {
            "task_type": ["Base"],
            "text": [text],
            "language": [language],
            "non_streaming_mode": [True],
            "x_vector_only_mode": [True],
            "voice_clone_prompt": [{"ref_spk_embedding": _load_speaker_vector(speaker_path)}],
        }
        assistant_ids = model_config.tokenizer(build_assistant_text(text), padding=False)["input_ids"]
        assistant_ids = torch.as_tensor(assistant_ids).reshape(-1).tolist()
        prompt_length = len(assistant_ids) + 2
        identity = "\0".join((text, str(language), str(speaker_path))).encode()
        return {
            "prompt_token_ids": [1] * prompt_length,
            "additional_information": additional_information,
            "cache_salt": hashlib.sha256(identity).hexdigest(),
        }

    @classmethod
    def combine_engine_outputs(cls, outputs, prompt):
        """Combine stage-0 policy tokens with codec and waveform outputs."""
        policy_outputs = [output for output in outputs if output.stage_id == 0]
        decoder_outputs = [output for output in outputs if output.stage_id == 1]
        if not policy_outputs:
            raise RuntimeError("Qwen3-TTS rollout produced no stage-0 policy output.")
        if not decoder_outputs:
            raise RuntimeError("Qwen3-TTS rollout produced no stage-1 decoder output.")

        policy_output = policy_outputs[-1]
        decoder_output = decoder_outputs[-1]
        if len(policy_output.outputs) != 1:
            raise RuntimeError(
                f"Qwen3-TTS stage 0 must return exactly one completion, got {len(policy_output.outputs)}."
            )
        token_ids = list(policy_output.outputs[0].token_ids)
        if not token_ids:
            raise RuntimeError("Qwen3-TTS rollout returned an empty stage-0 policy trajectory.")

        try:
            audio_codes = torch.as_tensor(policy_output.multimodal_output["codes"]["audio"]).detach().cpu()
            waveform = torch.as_tensor(decoder_output.multimodal_output["audio"]).detach().cpu().float().reshape(-1)
            sample_rate = torch.as_tensor(decoder_output.multimodal_output["sr"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError("Qwen3-TTS rollout output does not match the pinned two-stage contract.") from error
        if waveform.numel() == 0:
            raise RuntimeError("Qwen3-TTS stage 1 returned an empty waveform.")
        if sample_rate.numel() != 1:
            raise RuntimeError("Qwen3-TTS stage 1 must return one scalar sample rate.")
        sample_rate_value = float(sample_rate.item())
        if sample_rate_value <= 0 or not sample_rate_value.is_integer():
            raise RuntimeError(f"Qwen3-TTS stage 1 returned an invalid sample rate: {sample_rate_value!r}.")

        fields = {
            "tts_audio_codes": align_audio_codes(audio_codes, token_ids),
            "tts_text": prompt["additional_information"]["text"][0],
            "audio": waveform,
            "audio_sample_rate": int(sample_rate_value),
        }
        return policy_output, fields
