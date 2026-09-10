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
"""Qwen3-TTS talker actor adapter."""

import logging
import types
from collections.abc import Mapping
from typing import Any

import torch
from transformers import AutoModelForTextToWaveform
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.pipelines.qwen3_tts.rollout_utils import QWEN3_TTS_REPLAY_KEY
from verl_omni.pipelines.qwen3_tts.talker_forward import (
    load_speaker_xvector,
    require_auto_language,
    tts_actor_logits,
)


def _speaker_embedding(model, batch_size, device, dtype):
    return model._verl_tts_speaker_embedding.to(device=device, dtype=dtype).expand(batch_size, -1)


def _qwen3_tts_forward(
    self,
    input_ids=None,
    attention_mask=None,
    tts_text_ids=None,
    tts_audio_codes=None,
    response_len=None,
    text_len=None,
    **kwargs,
):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    required_inputs = (input_ids, attention_mask, tts_text_ids, tts_audio_codes, response_len, text_len)
    if any(value is None for value in required_inputs):
        raise RuntimeError("Qwen3-TTS forward is missing exact rollout codec fields.")
    speaker = _speaker_embedding(self, input_ids.shape[0], input_ids.device, next(self.talker.parameters()).dtype)
    return CausalLMOutputWithPast(
        logits=tts_actor_logits(
            self,
            input_ids,
            attention_mask,
            tts_text_ids,
            tts_audio_codes,
            response_len,
            text_len,
            speaker,
        )
    )


def _get_input_embeddings(self):
    return self.talker.model.codec_embedding


def _set_input_embeddings(self, value):
    self.talker.model.codec_embedding = value


@OmniModelBase.register("Qwen3TTSForConditionalGeneration", stage="talker")
class Qwen3TTSTalkerAdapter(OmniModelBase):
    auto_model_class = AutoModelForTextToWaveform

    @classmethod
    def register_auto_classes(cls) -> None:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        from transformers import AutoConfig, AutoModelForTextToWaveform

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
        AutoModelForTextToWaveform.register(
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
            exist_ok=True,
        )

    @classmethod
    def get_strip_modules(cls, model_config):
        return ["speaker_encoder", "speech_tokenizer", "code2wav"]

    @classmethod
    def configure_model(cls, module, model_config):
        if getattr(model_config, "use_remove_padding", False):
            raise ValueError("Qwen3-TTS Talker training requires actor_rollout_ref.model.use_remove_padding=false.")
        module = super().configure_model(module, model_config)
        module.config.tts_spk_embed_path = model_config.override_config.get("tts_spk_embed_path")
        module.config.tts_language = require_auto_language(model_config.override_config.get("tts_language"))
        if not module.config.tts_spk_embed_path:
            raise ValueError("Qwen3-TTS GRPO requires tts_spk_embed_path for the validated non-streaming replay.")
        module._verl_tts_speaker_embedding = load_speaker_xvector(module.config.tts_spk_embed_path)
        module.forward = types.MethodType(_qwen3_tts_forward, module)
        module.get_input_embeddings = types.MethodType(_get_input_embeddings, module)
        module.set_input_embeddings = types.MethodType(_set_input_embeddings, module)
        module._no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]
        trainable = 0
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name.startswith(("talker.model.", "talker.codec_head.")))
            trainable += int(parameter.requires_grad)
        logging.getLogger(__name__).info("Qwen3-TTS talker adapter enabled %d trainable parameter tensors", trainable)
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        return None

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Qwen3-TTS tokenizer must define either pad_token_id or eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model_config.hf_config.talker_config.tie_word_embeddings = False
        return tokenizer

    @classmethod
    def prepare_model_inputs(cls, model_inputs, micro_batch, model_config):
        del model_config
        # The online V1 trainer retains AgentLoopOutput.extra_fields as one mapping per sample.
        sample_extra_fields = tu.get(micro_batch, "extra_fields")
        if sample_extra_fields is None:
            raise RuntimeError(f"Qwen3-TTS actor inputs require the {QWEN3_TTS_REPLAY_KEY!r} replay payload.")
        if not isinstance(sample_extra_fields, list) or any(
            not isinstance(item, Mapping) for item in sample_extra_fields
        ):
            raise TypeError("Qwen3-TTS actor extra_fields must be a list of mappings.")
        if len(sample_extra_fields) != model_inputs["input_ids"].shape[0]:
            raise RuntimeError("Qwen3-TTS actor extra_fields do not match the actor batch size.")
        if any(QWEN3_TTS_REPLAY_KEY not in item for item in sample_extra_fields):
            raise RuntimeError(f"Qwen3-TTS actor inputs require the {QWEN3_TTS_REPLAY_KEY!r} replay payload.")

        fields = [item[QWEN3_TTS_REPLAY_KEY] for item in sample_extra_fields]
        if any(not isinstance(item, Mapping) for item in fields):
            raise TypeError(f"Qwen3-TTS {QWEN3_TTS_REPLAY_KEY!r} must be a list of mappings.")
        if any("text_ids" not in item or "audio_codes" not in item for item in fields):
            raise RuntimeError("Qwen3-TTS replay payloads must contain text_ids and audio_codes.")

        texts = [torch.as_tensor(item["text_ids"], dtype=torch.long).reshape(-1) for item in fields]
        codes = [torch.as_tensor(item["audio_codes"], dtype=torch.long) for item in fields]
        if any(item.numel() < 3 for item in texts):
            raise ValueError("Qwen3-TTS actor text fields must contain at least three prefix tokens.")
        if any(item.ndim != 2 or item.shape[-1] != 16 for item in codes):
            raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
        if any(item.shape[0] == 0 for item in codes):
            raise ValueError("Qwen3-TTS actor codec fields must contain at least one frame.")
        device = model_inputs["input_ids"].device
        text_buffer = torch.zeros((len(fields), max(item.numel() for item in texts)), dtype=torch.long, device=device)
        code_buffer = torch.zeros(
            (len(fields), max(item.shape[0] for item in codes), 16), dtype=torch.long, device=device
        )
        text_lens = torch.empty(len(fields), dtype=torch.long, device=device)
        response_lens = torch.empty_like(text_lens)
        for index, (text, code) in enumerate(zip(texts, codes, strict=True)):
            text_buffer[index, : text.numel()] = text.to(device)
            code_buffer[index, : code.shape[0]] = code.to(device)
            text_lens[index], response_lens[index] = text.numel(), code.shape[0]
        model_inputs.update(
            {
                "tts_text_ids": text_buffer,
                "tts_audio_codes": code_buffer,
                "text_len": text_lens,
                "response_len": response_lens,
            }
        )
        return model_inputs
