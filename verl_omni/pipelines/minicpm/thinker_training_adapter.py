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
"""MiniCPM thinker training adapter.

MiniCPM-V/o checkpoints are loaded through their Hugging Face remote-code
``AutoModel`` entrypoint.  For offline DPO we keep only the multimodal
understanding path and remove inference-only audio generation modules before
FSDP wrapping.
"""

from __future__ import annotations

import types
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

_MINICPM_ARCHITECTURES = ("MiniCPMO",)
_MINICPM_NO_SPLIT_MODULES = [
    "Qwen3DecoderLayer",
    "MiniCPMODecoderLayer",
    "MiniCPMWhisperEncoder",
]
# Keys consumed by MiniCPMO.forward(data, **kwargs) / get_vllm_embedding / get_omni_embedding.
_MINICPM_DATA_KEYS = (
    "input_ids",
    "position_ids",
    "pixel_values",
    "tgt_sizes",
    "image_bound",
    "audio_features",
    "audio_feature_lens",
    "audio_bounds",
    "spk_bounds",
    "vision_hidden_states",
)
_MINICPM_REQUIRED_DATA_KEYS = (
    "input_ids",
    "position_ids",
    "pixel_values",
    "tgt_sizes",
    "image_bound",
    "audio_bounds",
)


def _register_minicpm_architectures(cls):
    for architecture in _MINICPM_ARCHITECTURES:
        OmniModelBase.register(architecture, stage="thinker")(cls)
    return cls


def _first_existing_attr(module, names: list[str]):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _batch_size_from_input_ids(input_ids) -> int:
    if hasattr(input_ids, "shape") and len(input_ids.shape) > 0:
        return int(input_ids.shape[0])
    return len(input_ids)


def split_minicpm_forward_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split HF-style kwargs into MiniCPMO ``data`` plus LLM kwargs.

    Remote ``MiniCPMO.forward(self, data, **kwargs)`` reads ``input_ids``,
    ``position_ids``, and media tensors from ``data``, then calls
    ``self.llm(..., **kwargs)``. verl's FSDP engine instead unpacks
    ``input_ids=`` / ``pixel_values=`` at the top level.
    """
    kwargs = dict(kwargs)
    if "data" in kwargs:
        data = dict(kwargs.pop("data"))
        llm_kwargs = {key: value for key, value in kwargs.items() if key not in _MINICPM_DATA_KEYS}
        for key in _MINICPM_DATA_KEYS:
            if key in kwargs and key not in data:
                data[key] = kwargs[key]
    else:
        data = {}
        llm_kwargs = {}
        for key, value in kwargs.items():
            if key in _MINICPM_DATA_KEYS:
                data[key] = value
            else:
                llm_kwargs[key] = value

    if "input_ids" not in data:
        raise TypeError(
            "MiniCPMO.forward requires a `data` dict with `input_ids`, or top-level `input_ids`. "
            f"Received keys: {sorted(kwargs)}."
        )
    if "position_ids" not in data:
        raise TypeError("MiniCPMO.forward requires `position_ids` in `data` or as a keyword argument.")

    batch_size = _batch_size_from_input_ids(data["input_ids"])
    image_bound = data.get("image_bound")
    if image_bound is not None and batch_size == 1 and len(image_bound) > 1:
        raise ValueError(
            "MiniCPMO.forward expects per-sample sequences in data['input_ids'], but the batch "
            f"was packed to shape {tuple(data['input_ids'].shape)} while image_bound has "
            f"{len(image_bound)} samples. Set actor_rollout_ref.model.use_remove_padding=false."
        )
    data.setdefault("pixel_values", [[] for _ in range(batch_size)])
    data.setdefault("tgt_sizes", [[] for _ in range(batch_size)])
    data.setdefault("image_bound", [[] for _ in range(batch_size)])
    data.setdefault("audio_bounds", [[] for _ in range(batch_size)])
    from verl_omni.utils.dataset.minicpm_transform import (
        _batch_audio_feature_lens,
        _normalize_audio_features,
        _sample_pixel_slices,
        _sample_tgt_sizes,
    )

    pixel_values = data["pixel_values"]
    if isinstance(pixel_values, (list | tuple)):
        data["pixel_values"] = [_sample_pixel_slices(sample) for sample in pixel_values]
    else:
        data["pixel_values"] = [_sample_pixel_slices(pixel_values)]
    tgt_sizes = data["tgt_sizes"]
    if isinstance(tgt_sizes, (list, tuple)):
        data["tgt_sizes"] = [
            _sample_tgt_sizes(sample, n_slices=len(slices), device=data["input_ids"].device)
            for sample, slices in zip(tgt_sizes, data["pixel_values"], strict=False)
        ]
    data["audio_features"] = _normalize_audio_features(data.get("audio_features"))
    if data["audio_features"] == []:
        data["audio_feature_lens"] = []
    else:
        data["audio_feature_lens"] = _batch_audio_feature_lens(data.get("audio_feature_lens"), data["input_ids"].device)
    missing = [key for key in _MINICPM_REQUIRED_DATA_KEYS if key not in data]
    if missing:
        raise TypeError(f"MiniCPMO.forward data dict is missing required keys: {missing}.")
    return data, llm_kwargs


@_register_minicpm_architectures
class MiniCPMThinkerAdapter(OmniModelBase):
    """Training adapter for MiniCPM multimodal understanding-only DPO."""

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return [
            "talker",
            "tts",
            "audio_decoder",
            "audio_generator",
            "audio_head",
            "audio_detokenizer",
            "codec",
            "code2wav",
            "code_predictor",
            "snac",
            "vocoder",
        ]

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        from transformers import AutoModel

        from verl_omni.models.transformers.remote_code_compat import patch_remote_auto_model_init

        patch_remote_auto_model_init(
            model_config.local_path,
            trust_remote_code=model_config.trust_remote_code,
            config=model_config.hf_config,
        )

        return AutoModel.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            config=model_config.hf_config,
            trust_remote_code=model_config.trust_remote_code,
        )

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)

        # Keep MiniCPMO.forward so vpm/resampler/apm stay in the training graph.
        # Wrap it so verl's `module(**hf_kwargs)` becomes `forward(data, **llm_kwargs)`.
        original_forward = module.__class__.forward

        def _forward(self, data=None, **kwargs):
            payload = kwargs if data is None else {"data": data, **kwargs}
            packed_data, llm_kwargs = split_minicpm_forward_kwargs(payload)
            return original_forward(self, packed_data, **llm_kwargs)

        module.forward = types.MethodType(_forward, module)

        trainable_component = _first_existing_attr(
            module,
            ["llm", "language_model", "model", "base_model", "text_model"],
        )
        if trainable_component is not None:
            if hasattr(trainable_component, "get_input_embeddings"):
                module.get_input_embeddings = trainable_component.get_input_embeddings
            if hasattr(trainable_component, "set_input_embeddings"):
                module.set_input_embeddings = trainable_component.set_input_embeddings
            if hasattr(trainable_component, "prepare_inputs_for_generation"):
                module.prepare_inputs_for_generation = trainable_component.prepare_inputs_for_generation

        module._no_split_modules = _MINICPM_NO_SPLIT_MODULES
        return module

    @classmethod
    def prepare_model_inputs(cls, model_inputs: dict[str, Any], micro_batch, model_config) -> dict[str, Any]:
        del micro_batch, model_config
        data, llm_kwargs = split_minicpm_forward_kwargs(dict(model_inputs))
        return {"data": data, **llm_kwargs}

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MiniCPM AutoProcessor from {model_path}. "
                "A tokenizer cannot substitute for the multimodal processor."
            ) from exc
        if getattr(processor, "tokenizer", None) is None:
            processor.tokenizer = cls.configure_tokenizer(model_path, model_config)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
