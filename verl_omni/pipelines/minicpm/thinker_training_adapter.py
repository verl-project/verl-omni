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

import torch

from verl_omni.pipelines.model_base import OmniModelBase

_MINICPM_NO_SPLIT_MODULES = ["Qwen3DecoderLayer", "MiniCPMODecoderLayer"]
# Keys routed into MiniCPMO's ``data`` dict instead of ``self.llm(**kwargs)``.
# Most are consumed by MiniCPMO.forward / get_vllm_embedding /
# get_omni_embedding; ``image_sizes`` is processor-emitted metadata no
# MiniCPM-o forward consumes (the remote chat() likewise pops it) — it is
# classified here so it can never reach the Qwen3 decoder, which rejects
# unknown kwargs.
_MINICPM_DATA_KEYS = (
    "input_ids",
    "position_ids",
    "pixel_values",
    "tgt_sizes",
    "image_sizes",
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
# MiniCPMO.forward binds these before ``self.llm(..., **kwargs)``. The adapter wrap
# must not forward engine copies or the LLM call raises TypeError.
_MINICPM_LLM_BOUND_KEYS = ("input_ids", "position_ids", "inputs_embeds")


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
    ``input_ids=`` / ``pixel_values=`` at the top level. Keys that MiniCPMO
    already binds on the LLM call (``inputs_embeds``, ``input_ids``,
    ``position_ids``) are dropped from ``llm_kwargs``.
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
            f"{len(image_bound)} samples. A packed (use_remove_padding=true) batch reached the "
            "model without going through MiniCPMThinkerAdapter.prepare_model_inputs."
        )
    data.setdefault("pixel_values", [[] for _ in range(batch_size)])
    data.setdefault("tgt_sizes", [[] for _ in range(batch_size)])
    data.setdefault("image_bound", [[] for _ in range(batch_size)])
    data.setdefault("audio_bounds", [[] for _ in range(batch_size)])
    from verl_omni.pipelines.minicpm.media_inputs import (
        batch_audio_feature_lens,
        normalize_audio_features,
        sample_pixel_slices,
        sample_tgt_sizes,
    )

    pixel_values = data["pixel_values"]
    if isinstance(pixel_values, (list | tuple)):
        data["pixel_values"] = [sample_pixel_slices(sample) for sample in pixel_values]
    else:
        data["pixel_values"] = [sample_pixel_slices(pixel_values)]
    tgt_sizes = data["tgt_sizes"]
    if isinstance(tgt_sizes, (list | tuple)):
        data["tgt_sizes"] = [
            sample_tgt_sizes(sample, n_slices=len(slices), device=data["input_ids"].device)
            for sample, slices in zip(tgt_sizes, data["pixel_values"], strict=False)
        ]
    data["audio_features"] = normalize_audio_features(data.get("audio_features"))
    if data["audio_features"] == []:
        data["audio_feature_lens"] = []
    else:
        data["audio_feature_lens"] = batch_audio_feature_lens(data.get("audio_feature_lens"), data["input_ids"].device)
    missing = [key for key in _MINICPM_REQUIRED_DATA_KEYS if key not in data]
    if missing:
        raise TypeError(f"MiniCPMO.forward data dict is missing required keys: {missing}.")
    for key in _MINICPM_LLM_BOUND_KEYS:
        llm_kwargs.pop(key, None)
    return data, llm_kwargs


class MiniCPMO:
    """HF ``architectures[0]`` loader for MiniCPM-o remote code.

    ``OmniFSDPEngine._build_module`` calls ``from_pretrained`` on this class.
    The checkpoint's remote ``MiniCPMO`` type is resolved through ``AutoModel``.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from transformers import AutoModel

        from verl_omni.models.transformers.minicpm_o import (
            patch_remote_auto_model_init,
            patch_remote_siglip_flash_attn_support,
        )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        patch_remote_auto_model_init(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            config=kwargs.get("config"),
        )
        patch_remote_siglip_flash_attn_support(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return AutoModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def _media_in_data(data: dict[str, Any]) -> bool:
    pixel_values = data.get("pixel_values")
    has_pixels = bool(pixel_values) and any(len(sample) for sample in pixel_values)
    audio_features = data.get("audio_features")
    has_audio = audio_features is not None and len(audio_features) > 0
    return has_pixels or has_audio


def _is_packed_batch(data: dict[str, Any]) -> bool:
    """rmpad layout: flattened ``[1, total]`` ids with resetting position_ids."""
    input_ids = data.get("input_ids")
    position_ids = data.get("position_ids")
    if input_ids is None or position_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return False
    resets = int((position_ids.reshape(-1) == 0).sum().item())
    return resets > 1


def _apply_media_bounds(data: dict[str, Any], model_config) -> None:
    """Derive ``image_bound`` / ``audio_bounds`` from the (already expanded) ids.

    Cross-checks span counts against the media-feature counts so a train/rollout
    expansion mismatch fails here instead of scattering wrong embeddings.
    Text-only batches keep the empty defaults from the split.
    """
    if not _media_in_data(data):
        return
    processor = getattr(model_config, "processor", None)
    if processor is None:
        raise RuntimeError(
            "MiniCPM media batches require model_config.processor to derive "
            "image_bound/audio_bounds; the model config did not load a processor."
        )
    from verl_omni.pipelines.minicpm.prompt_parity import resolve_media_tokens

    tokens = resolve_media_tokens(processor)

    def _counts_match(spans: list[list[int]], expected: int, kind: str) -> None:
        if len(spans) != expected:
            raise ValueError(
                f"MiniCPM {kind} parity failure: ids expanded into {len(spans)} spans but the "
                f"processor features describe {expected}. The rollout and actor renderings "
                "disagree; compare slot expansion on both sides before training."
            )

    if _is_packed_batch(data):
        image_bounds, audio_bounds = tokens.derive_media_bounds(data["input_ids"].reshape(-1))
        _counts_match(image_bounds, sum(len(sample) for sample in data["pixel_values"]), "image")
        _counts_match(audio_bounds, sum(len(sample) for sample in data["audio_feature_lens"]), "audio")
        data["image_bound"] = [image_bounds]
        data["audio_bounds"] = [audio_bounds]
    else:
        per_row_image, per_row_audio = [], []
        for row in range(data["input_ids"].shape[0]):
            image_bounds, audio_bounds = tokens.derive_media_bounds(data["input_ids"][row])
            per_row_image.append(image_bounds)
            per_row_audio.append(audio_bounds)
        _counts_match(
            [span for spans in per_row_image for span in spans],
            sum(len(sample) for sample in data["pixel_values"]),
            "image",
        )
        _counts_match(
            [span for spans in per_row_audio for span in spans],
            sum(len(sample) for sample in data["audio_feature_lens"]),
            "audio",
        )
        data["image_bound"] = per_row_image
        data["audio_bounds"] = per_row_audio


def _merge_packed_media(data: dict[str, Any]) -> None:
    """Fold per-sample media into one pseudo-sample for the flattened batch.

    The remote embedders keep the batch-row dimension: ``get_vision_embedding``
    iterates ``data["pixel_values"]`` per row (each row a list of that row's
    slices), and ``get_vllm_embedding`` / ``get_omni_embedding`` index
    ``data["image_bound"][i]`` / ``data["audio_bounds"][i]`` per row. With the
    packed layout ``bs == 1`` and that single row is the concatenation of all
    samples — so pixel_values and the bounds must be ONE pseudo-row holding
    every slice/span (sample-major, media order within, matching the id scan),
    not flattened past the row dimension.
    """
    data["pixel_values"] = [[slice_ for sample in data["pixel_values"] for slice_ in sample]]
    tgt_sizes = [sample for sample in data["tgt_sizes"] if int(sample.numel()) > 0]
    data["tgt_sizes"] = [torch.cat(tgt_sizes, dim=0)] if tgt_sizes else [torch.zeros(0, 2, dtype=torch.int32)]
    data["audio_feature_lens"] = [
        [lens for sample in data["audio_feature_lens"] for lens in torch.as_tensor(sample).reshape(-1).tolist()]
    ]
    data["image_bound"] = [[span for spans in data["image_bound"] for span in spans]]
    data["audio_bounds"] = [[span for spans in data["audio_bounds"] for span in spans]]


@OmniModelBase.register("MiniCPMO", stage="thinker")
class MiniCPMThinkerAdapter(OmniModelBase):
    """Training adapter for MiniCPM multimodal understanding."""

    auto_model_class = MiniCPMO

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["tts"]

    @classmethod
    def configure_model(cls, module, model_config):
        from verl_omni.models.transformers.minicpm_o import (
            patch_minicpm_get_vision_embedding,
            patch_minicpm_get_vllm_embedding,
            patch_remote_whisper_self_attn,
        )

        version = str(getattr(getattr(module, "config", None), "version", ""))
        if version != "4.5":
            raise ValueError(
                f"MiniCPMThinkerAdapter supports MiniCPM-o 4.5 checkpoints only; "
                f"config.version={version!r}. MiniCPM-o 2.6 is rejected by vLLM-Omni."
            )
        module = super().configure_model(module, model_config)
        patch_remote_whisper_self_attn(module)
        patch_minicpm_get_vision_embedding(module)
        patch_minicpm_get_vllm_embedding(module)

        # Keep MiniCPMO.forward so vpm/resampler/apm still produce multimodal
        # embeddings; wrap it so verl's `module(**hf_kwargs)` becomes
        # `forward(data, **llm_kwargs)`.
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
    def get_fsdp_ignored_module_names(cls, model_config) -> list[str]:
        return ["apm"]

    @classmethod
    def prepare_model_inputs(cls, model_inputs: dict[str, Any], micro_batch, model_config) -> dict[str, Any]:
        del micro_batch
        model_inputs = dict(model_inputs)
        # The processor emits image_bound/audio_bounds in its own per-sample
        # batch layout; verl's rmpad flattening invalidates those coordinates
        # (and they trip the packed-batch tripwire in the split below). Bounds
        # are re-derived from the actual training ids in _apply_media_bounds —
        # the ids are the single source of truth — so the stale copies are
        # dropped instead of trusted.
        model_inputs.pop("image_bound", None)
        model_inputs.pop("audio_bounds", None)
        # image_sizes is processor-emitted metadata that neither MiniCPMO.forward
        # nor the LLM consumes; the remote chat() pops it before generate too.
        model_inputs.pop("image_sizes", None)
        data, llm_kwargs = split_minicpm_forward_kwargs(model_inputs)
        _apply_media_bounds(data, model_config)
        if _is_packed_batch(data):
            _merge_packed_media(data)
            # FA2 derives cu_seqlens from position_ids; a padded attention_mask
            # would contradict the packed layout.
            llm_kwargs.pop("attention_mask", None)
        return {"data": data, **llm_kwargs}

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        from transformers import AutoProcessor

        from verl_omni.pipelines.minicpm.prompt_parity import bind_minicpm_processor

        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MiniCPM AutoProcessor from {model_path}. "
                "A tokenizer cannot substitute for the multimodal processor."
            ) from exc
        if getattr(processor, "tokenizer", None) is None:
            processor.tokenizer = cls.configure_tokenizer(model_path, model_config)
        return bind_minicpm_processor(processor)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
