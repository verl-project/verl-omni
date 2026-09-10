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

from __future__ import annotations

import types
from typing import Any

import torch

from verl_omni.pipelines.model_base import OmniModelBase


class _MiniCPMAutoModel:
    """Load the released remote model with Transformers 5 initialization."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        if not kwargs.get("trust_remote_code"):
            raise ValueError("MiniCPM-o requires trust_remote_code=true after reviewing the checkpoint code.")
        config = kwargs["config"]
        config.init_tts = False
        model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], pretrained_model_name_or_path)
        if not getattr(model_cls, "_verl_post_init", False):
            original_init = model_cls.__init__

            def _init(self, *args, **init_kwargs):
                original_init(self, *args, **init_kwargs)
                if not hasattr(self, "all_tied_weights_keys"):
                    self.post_init()

            model_cls.__init__ = _init
            model_cls._verl_post_init = True
        return model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)


def _bounds_are_empty(bounds) -> bool:
    if isinstance(bounds, torch.Tensor):
        return bounds.numel() == 0
    if isinstance(bounds, list | tuple):
        return all(_bounds_are_empty(item) for item in bounds)
    return bounds is None


def _pad_audio_features(features: list[torch.Tensor]) -> torch.Tensor:
    normalized = [
        feature.squeeze(0) if feature.ndim == 3 and feature.shape[0] == 1 else feature for feature in features
    ]
    if any(feature.ndim != 2 for feature in normalized):
        shapes = [tuple(feature.shape) for feature in normalized]
        raise ValueError(f"MiniCPM-o audio features must have shape (80, frames), got {shapes}.")
    max_frames = max(feature.shape[-1] for feature in normalized)
    padded = [torch.nn.functional.pad(feature, (0, max_frames - feature.shape[-1])) for feature in normalized]
    return torch.stack(padded)


def _offset_media_bounds(data: dict[str, Any], attention_mask: torch.Tensor | None) -> None:
    if attention_mask is None:
        return
    leading_padding = attention_mask.long().argmax(dim=-1)
    for key in ("image_bound", "audio_bounds", "spk_bounds"):
        bounds = data.get(key)
        if not isinstance(bounds, list) or len(bounds) != len(leading_padding):
            continue
        data[key] = [
            (
                bound + leading_padding[index].to(bound.device)
                if isinstance(bound, torch.Tensor) and bound.numel()
                else bound
            )
            for index, bound in enumerate(bounds)
        ]


def _minicpmo_whisper_attention_forward(self, *args, **kwargs):
    past_key_values = kwargs.pop("past_key_value", kwargs.get("past_key_values"))
    kwargs["past_key_values"] = past_key_values
    output = self._verl_minicpmo_original_forward(*args, **kwargs)
    if len(output) == 2:
        return output[0], output[1], past_key_values
    return output


def _minicpmo_get_vllm_embedding(self, data):
    vision_hidden_states = self.get_vision_embedding(data)
    vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])
    if hasattr(self.llm.config, "scale_emb"):
        vllm_embedding = vllm_embedding * self.llm.config.scale_emb

    vision_hidden_states = [
        item.to(dtype=vllm_embedding.dtype) if isinstance(item, torch.Tensor) else item for item in vision_hidden_states
    ]
    batch_embeddings = []
    for index, current_embedding in enumerate(vllm_embedding):
        current_vision = vision_hidden_states[index]
        image_bounds = data["image_bound"][index]
        if len(current_vision) > 0 and len(image_bounds) > 0:
            image_indices = torch.cat(
                [
                    torch.arange(bound[0], bound[1], dtype=torch.long, device=vllm_embedding.device)
                    for bound in image_bounds
                ]
            )
            current_embedding = current_embedding.index_copy(
                0,
                image_indices,
                current_vision.reshape(-1, current_vision.shape[-1]),
            )
        elif len(current_vision) > 0 and self.training:
            current_embedding = current_embedding + current_vision[0].mean() * 0
        batch_embeddings.append(current_embedding)
    return torch.stack(batch_embeddings), vision_hidden_states


def _minicpmo_get_audio_embedding(self, data, chunk_length=-1, dummy=True):
    audio_features = data.get("audio_features")
    audio_feature_lens = data.get("audio_feature_lens", [])
    if not isinstance(audio_features, torch.Tensor) or audio_features.ndim != 3:
        return self._verl_minicpmo_original_get_audio_embedding(data, chunk_length, dummy)

    active_indices = [
        index
        for index, lengths in enumerate(audio_feature_lens)
        if isinstance(lengths, torch.Tensor) and lengths.numel() > 0
    ]
    if len(active_indices) == len(audio_feature_lens):
        return self._verl_minicpmo_original_get_audio_embedding(data, chunk_length, dummy)
    if not active_indices:
        return [[] for _ in audio_feature_lens]

    active_data = dict(data)
    active_data["audio_features"] = audio_features[active_indices]
    active_data["audio_feature_lens"] = [audio_feature_lens[index] for index in active_indices]
    active_embeddings = self._verl_minicpmo_original_get_audio_embedding(active_data, chunk_length, dummy)
    result = [[] for _ in audio_feature_lens]
    for index, embeddings in zip(active_indices, active_embeddings, strict=True):
        result[index] = embeddings
    return result


def _minicpmo_get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
    audio_features = data.get("audio_features")
    if audio_features is None or (isinstance(audio_features, torch.Tensor) and audio_features.numel() == 0):
        return input_embeddings
    if isinstance(audio_features, list | tuple) and not audio_features:
        return input_embeddings
    return self._verl_minicpmo_original_get_omni_embedding(
        data,
        input_embeddings,
        chunk_length=chunk_length,
        stream_input=stream_input,
    )


def _minicpmo_forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **kwargs,
):
    if input_ids.ndim != 2:
        raise ValueError("MiniCPM-o replay requires padded (batch, sequence) input_ids.")
    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    data_keys = {
        "pixel_values",
        "image_sizes",
        "tgt_sizes",
        "image_bound",
        "audio_features",
        "audio_feature_lens",
        "audio_bounds",
        "spk_bounds",
        "vision_hidden_states",
    }
    data = {"input_ids": input_ids, "position_ids": position_ids}
    kwargs.pop("inputs_embeds", None)
    for key in data_keys:
        if key in kwargs:
            data[key] = kwargs.pop(key)

    audio_features = data.get("audio_features")
    if isinstance(audio_features, list) and audio_features:
        prototype = next((feature for feature in audio_features if isinstance(feature, torch.Tensor)), None)
        if prototype is not None:
            data["audio_features"] = _pad_audio_features(
                [
                    feature if isinstance(feature, torch.Tensor) else prototype.new_zeros((prototype.shape[-2], 1))
                    for feature in audio_features
                ]
            )

    data.setdefault("pixel_values", [[] for _ in range(batch_size)])
    data.setdefault("tgt_sizes", [[] for _ in range(batch_size)])
    data.setdefault("image_bound", [torch.empty((0, 2), dtype=torch.long, device=device) for _ in range(batch_size)])
    data.setdefault("audio_features", [])
    data.setdefault("audio_feature_lens", [])
    data.setdefault("audio_bounds", [torch.empty((0, 2), dtype=torch.long, device=device) for _ in range(batch_size)])
    data.setdefault("spk_bounds", [torch.empty((0, 2), dtype=torch.long, device=device) for _ in range(batch_size)])

    if _bounds_are_empty(data["image_bound"]):
        data["pixel_values"] = [[] for _ in range(batch_size)]
        data["tgt_sizes"] = [[] for _ in range(batch_size)]
        data["image_bound"] = [torch.empty((0, 2), dtype=torch.long, device=device) for _ in range(batch_size)]
        data["vision_hidden_states"] = [[] for _ in range(batch_size)]
    if _bounds_are_empty(data["audio_bounds"]):
        data["audio_features"] = []
        data["audio_feature_lens"] = []
        data["audio_bounds"] = [torch.empty((0, 2), dtype=torch.long, device=device) for _ in range(batch_size)]

    _offset_media_bounds(data, attention_mask)

    if data["position_ids"] is None:
        if attention_mask is None:
            data["position_ids"] = torch.arange(input_ids.shape[-1], device=device).expand(batch_size, -1)
        else:
            data["position_ids"] = attention_mask.long().cumsum(-1) - 1
            data["position_ids"].masked_fill_(attention_mask == 0, 0)

    return self._verl_minicpmo_original_forward(
        data,
        attention_mask=attention_mask,
        **kwargs,
    )


@OmniModelBase.register("MiniCPMO", stage="thinker")
class MiniCPMThinkerAdapter(OmniModelBase):
    """Algorithm-independent, bounded MiniCPM-o 4.5 thinker replay."""

    auto_model_class = _MiniCPMAutoModel

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["tts"]

    @classmethod
    def prepare_model_inputs(cls, model_inputs, micro_batch, model_config):
        if model_config.use_remove_padding:
            raise ValueError("MiniCPM-o requires model.use_remove_padding=false to preserve per-sample media bounds.")
        return model_inputs

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)
        module._verl_minicpmo_original_forward = module.forward
        module.forward = types.MethodType(_minicpmo_forward, module)
        module.config.stream_input = False
        module.get_vllm_embedding = types.MethodType(_minicpmo_get_vllm_embedding, module)
        for name in ("vpm", "resampler", "apm", "audio_projection_layer"):
            encoder = getattr(module, name, None)
            if encoder is not None:
                encoder.requires_grad_(False)
                # Conditional encoders must not introduce per-embedding FSDP collectives.
                for child in encoder.modules():
                    if isinstance(child, torch.nn.Embedding):
                        weight = child.weight.detach()
                        del child.weight
                        child.register_buffer("weight", weight)
        if hasattr(module, "apm"):
            for layer in module.apm.layers:
                attention = layer.self_attn
                attention._verl_minicpmo_original_forward = attention.forward
                attention.forward = types.MethodType(_minicpmo_whisper_attention_forward, attention)
        if hasattr(module, "get_audio_embedding"):
            module._verl_minicpmo_original_get_audio_embedding = module.get_audio_embedding
            module.get_audio_embedding = types.MethodType(_minicpmo_get_audio_embedding, module)
        if hasattr(module, "get_omni_embedding"):
            module._verl_minicpmo_original_get_omni_embedding = module.get_omni_embedding
            module.get_omni_embedding = types.MethodType(_minicpmo_get_omni_embedding, module)
        module.get_input_embeddings = module.llm.get_input_embeddings
        module.set_input_embeddings = module.llm.set_input_embeddings
        module.prepare_inputs_for_generation = module.llm.prepare_inputs_for_generation
        module._no_split_modules = ["Qwen3DecoderLayer"]
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        from .processor import render_minicpmo_messages

        processor.chat_template = processor.tokenizer.chat_template
        processor.apply_chat_template = types.MethodType(render_minicpmo_messages, processor)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        from transformers import AutoTokenizer

        if str(getattr(model_config.hf_config, "version", "")) != "4.5":
            raise ValueError("The MiniCPM-o simplex adapter supports checkpoint version 4.5 only.")
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
