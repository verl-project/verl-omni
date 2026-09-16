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
"""MiniCPM-o Thinker training adapter.

Implements ``OmniModelBase`` for thinker-stage training of MiniCPM-o:
remote-code loading with the training-path config invariants, sub-module
stripping, forward redirection to the multimodal understanding path, and
media normalization for the packed (rmpad) layout.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from verl_omni.pipelines.model_base import OmniModelBase

# Keys routed into MiniCPMO's ``data`` dict instead of ``self.llm(**kwargs)``.
# ``image_sizes`` is processor metadata no forward consumes; classifying it
# here keeps it off the Qwen3 decoder, which rejects unknown kwargs.
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
# MiniCPMO.forward binds these on the LLM call; forwarding engine copies raises.
_MINICPM_LLM_BOUND_KEYS = ("input_ids", "position_ids", "inputs_embeds")


class MiniCPMO:
    """``auto_model_class`` loader: patches MiniCPM-o's remote code, then loads through ``AutoModel``."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load MiniCPM-o, applying the compatibility patches before ``AutoModel`` builds it.

        Args:
            pretrained_model_name_or_path: Local path to the model checkpoint.
            *args: Forwarded to ``AutoModel.from_pretrained``.
            **kwargs: Forwarded to ``AutoModel.from_pretrained``; ``config``
                and ``trust_remote_code`` also drive the compatibility patches.

        Returns:
            The loaded remote-code model.
        """
        from transformers import AutoModel

        from verl_omni.models.transformers.minicpm_o import (
            patch_remote_auto_model_init,
            patch_remote_siglip_flash_attn_support,
        )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        config = kwargs.get("config")
        # Training-path invariants, set before the remote __init__ reads them.
        if config is not None:
            config.init_tts = False
            config.use_cache = False
            config.stream_input = False
        # Remote code omits post_init(), which transformers >= 5 calls before from_pretrained returns.
        patch_remote_auto_model_init(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            config=config,
        )
        # The remote vision tower declares only the 4.x FA2 support flag, which transformers >= 5 rejects.
        patch_remote_siglip_flash_attn_support(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            config=config,
        )
        return AutoModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


@OmniModelBase.register("MiniCPMO", stage="thinker")
class MiniCPMThinkerAdapter(OmniModelBase):
    """Thinker-stage training adapter for MiniCPM-o.

    Handles model setup that is required before verl's FSDP engine loads and
    wraps the model: sub-module stripping, remote-code patching, forward
    redirection for verl's call convention, frozen-tower FSDP2 handling, and
    processor/tokenizer configuration shared with the rollout.
    """

    auto_model_class = MiniCPMO

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["tts"]

    @classmethod
    def configure_model(cls, module, model_config):
        """Strip the audio-generation stage and install the training hooks.

        Args:
            module: The loaded MiniCPM-o model before FSDP wrapping.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured module with the TTS stage stripped, remote-code
            patches installed, and the text accessors re-exposed.
        """
        # Base strips each submodule named by get_strip_modules (the TTS stage).
        module = super().configure_model(module, model_config)
        # The media towers are batch-sensitive; each patch fixes one deviation.
        _apply_remote_code_patches(module)
        # verl calls module(**hf_kwargs); the remote forward takes (data, **kwargs).
        _wrap_forward_for_verl(module)
        # The inner text model carries the embedding and generation hooks.
        module.get_input_embeddings = module.llm.get_input_embeddings
        module.set_input_embeddings = module.llm.set_input_embeddings
        module.prepare_inputs_for_generation = module.llm.prepare_inputs_for_generation
        # Wrap the inner decoder layers, so each becomes its own FSDP unit.
        module._no_split_modules = ["Qwen3DecoderLayer", "MiniCPMODecoderLayer"]
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the MiniCPM-o processor and bind the RL parity behaviors.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured processor, upgraded in place.
        """
        from transformers import AutoProcessor

        from verl_omni.pipelines.minicpm.prompt_parity import bind_minicpm_processor

        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MiniCPM AutoProcessor from {model_path}. "
                "A tokenizer cannot substitute for the multimodal processor."
            ) from exc
        # Some checkpoints ship the processor without a tokenizer; the parity bind needs one.
        if getattr(processor, "tokenizer", None) is None:
            processor.tokenizer = cls.configure_tokenizer(model_path, model_config)
        # Binds __call__ / apply_chat_template / dedup_pad_tokens; see prompt_parity.
        return bind_minicpm_processor(processor)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer and demote the checkpoint's special answer tags.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured tokenizer, with the two tags demoted to plain
            added tokens (ids and atomic encoding unchanged).
        """
        from transformers import AutoTokenizer

        from verl_omni.models.transformers.minicpm_o import keep_answer_tags_when_decoding

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        # The reward manager decodes with skip_special_tokens=True, which would strip
        # the checkpoint's special <answer> tags and zero every choice-reward score.
        keep_answer_tags_when_decoding(tokenizer)
        return tokenizer

    @classmethod
    def get_fsdp_ignored_module_names(cls, model_config) -> list[str]:
        """Frozen encoder towers left unsharded: Whisper audio, vision, resampler."""
        return ["apm", "vpm", "resampler"]

    @classmethod
    def prepare_model_inputs(cls, model_inputs: dict[str, Any], micro_batch, model_config) -> dict[str, Any]:
        """Adapt the engine's language-model inputs to MiniCPMO's ``data`` dict.

        Args:
            model_inputs: Standard language-model inputs prepared by verl.
            micro_batch: The micro batch being prepared (unused).
            model_config: The ``OmniModelConfig`` carrying the processor.

        Returns:
            The model call kwargs: MiniCPMO's ``data`` dict plus any LLM kwargs.
        """
        model_inputs = dict(model_inputs)
        # rmpad flattening invalidates the processor's per-sample bounds coordinates;
        # _apply_media_bounds re-derives them from the ids instead of trusting them.
        model_inputs.pop("image_bound", None)
        model_inputs.pop("audio_bounds", None)
        # Processor metadata no forward consumes (the remote chat() pops it too).
        model_inputs.pop("image_sizes", None)
        # Split into the remote forward's data dict plus the inner LLM's kwargs.
        data, llm_kwargs = split_minicpm_forward_kwargs(model_inputs)
        # Re-derive the media bounds from the ids and cross-check the span counts.
        _apply_media_bounds(data, model_config)
        if _is_packed_batch(data):
            # Fold the per-sample media into MiniCPMO's single-row packed layout.
            _merge_packed_media(data)
            # FA2 derives cu_seqlens from position_ids; a padded attention_mask
            # would contradict the packed layout.
            llm_kwargs.pop("attention_mask", None)
        return {"data": data, **llm_kwargs}


def _apply_remote_code_patches(module) -> None:
    """Install the compatibility shims the remote-code forward paths need."""
    from verl_omni.models.transformers.minicpm_o import (
        patch_minicpm_get_audio_embedding,
        patch_minicpm_get_omni_embedding,
        patch_minicpm_get_vision_embedding,
        patch_minicpm_get_vllm_embedding,
        patch_remote_whisper_self_attn,
    )

    # Remote Whisper self-attn returns a 2-tuple the remote encoder layer unpacks as 3.
    patch_remote_whisper_self_attn(module)
    # Batched vision/resampler bf16 kernels deviate from the bs==1 realization.
    patch_minicpm_get_vision_embedding(module)
    # The remote in-place scatter fails once PEFT makes the embeddings a leaf.
    patch_minicpm_get_vllm_embedding(module)
    # The remote batched audio mask under-masks by ~2x.
    patch_minicpm_get_audio_embedding(module)
    # Remote get_omni_embedding splices audio for the last row only.
    patch_minicpm_get_omni_embedding(module)


def _wrap_forward_for_verl(module) -> None:
    """Adapt verl's ``module(**hf_kwargs)`` call to the remote ``forward(data, **kwargs)``."""
    original_forward = module.__class__.forward

    def _forward(self, data=None, **kwargs):
        payload = kwargs if data is None else {"data": data, **kwargs}
        packed_data, llm_kwargs = split_minicpm_forward_kwargs(payload)
        return original_forward(self, packed_data, **llm_kwargs)

    module.forward = types.MethodType(_forward, module)


def split_minicpm_forward_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the engine's HF-style kwargs into MiniCPMO's ``data`` dict plus LLM kwargs.

    Args:
        kwargs: The engine's top-level model inputs, optionally including a
            nested ``data`` dict.

    Returns:
        ``(data, llm_kwargs)``: the remote forward's first argument, and the
        kwargs forwarded to the inner LLM call.
    """
    kwargs = dict(kwargs)
    data, llm_kwargs = _split_data_and_llm_kwargs(kwargs)
    if "input_ids" not in data:
        raise TypeError(
            "MiniCPMO.forward requires a `data` dict with `input_ids`, or top-level `input_ids`. "
            f"Received keys: {sorted(kwargs)}."
        )
    if "position_ids" not in data:
        raise TypeError("MiniCPMO.forward requires `position_ids` in `data` or as a keyword argument.")

    _reject_unprepared_packed_batch(data)
    _fill_missing_media_defaults(data)
    _normalize_media_containers(data)

    missing = [key for key in _MINICPM_REQUIRED_DATA_KEYS if key not in data]
    if missing:
        raise TypeError(f"MiniCPMO.forward data dict is missing required keys: {missing}.")
    # The remote forward binds these itself; forwarding engine copies raises TypeError.
    for key in _MINICPM_LLM_BOUND_KEYS:
        llm_kwargs.pop(key, None)
    return data, llm_kwargs


def _split_data_and_llm_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition kwargs: a nested ``data`` dict wins; top-level media keys fold in where absent."""
    data = dict(kwargs.pop("data", {}))
    llm_kwargs = {}
    for key, value in kwargs.items():
        if key not in _MINICPM_DATA_KEYS:
            llm_kwargs[key] = value
        elif key not in data:
            data[key] = value
    return data, llm_kwargs


def _reject_unprepared_packed_batch(data: dict[str, Any]) -> None:
    """Reject a packed batch whose per-sample bounds were never re-derived."""
    image_bound = data.get("image_bound")
    if image_bound is None or len(image_bound) <= 1:
        return
    if _batch_size_from_input_ids(data["input_ids"]) != 1:
        return
    raise ValueError(
        "MiniCPMO.forward expects per-sample sequences in data['input_ids'], but the batch "
        f"was packed to shape {tuple(data['input_ids'].shape)} while image_bound has "
        f"{len(image_bound)} samples. A packed (use_remove_padding=true) batch reached the "
        "model without going through MiniCPMThinkerAdapter.prepare_model_inputs."
    )


def _fill_missing_media_defaults(data: dict[str, Any]) -> None:
    """Give absent media keys the per-sample empty-list shape the remote forward expects."""
    batch_size = _batch_size_from_input_ids(data["input_ids"])
    for key in ("pixel_values", "tgt_sizes", "image_bound", "audio_bounds"):
        data.setdefault(key, [[] for _ in range(batch_size)])


def _normalize_media_containers(data: dict[str, Any]) -> None:
    """Reshape collated media into the per-sample containers MiniCPMO.forward expects."""
    from verl_omni.pipelines.minicpm.media_inputs import (
        batch_audio_feature_lens,
        normalize_audio_features,
        sample_pixel_slices,
        sample_tgt_sizes,
    )

    device = data["input_ids"].device
    # Per-sample slice lists, including the collated single-batch form.
    pixel_values = data["pixel_values"]
    if isinstance(pixel_values, (list | tuple)):
        data["pixel_values"] = [sample_pixel_slices(sample) for sample in pixel_values]
    else:
        data["pixel_values"] = [sample_pixel_slices(pixel_values)]
    tgt_sizes = data["tgt_sizes"]
    if isinstance(tgt_sizes, (list | tuple)):
        data["tgt_sizes"] = [sample_tgt_sizes(sample, device=device) for sample in tgt_sizes]
    # Mel features stack per clip; the lens must stay 1-D tensors for the tower's hstack.
    data["audio_features"] = normalize_audio_features(data.get("audio_features"))
    if data["audio_features"] == []:
        # An empty-features/nonnull-lens inconsistency is laundered into "no audio"
        # here; the dangerous direction is caught fail-closed in _apply_media_bounds.
        data["audio_feature_lens"] = []
    else:
        data["audio_feature_lens"] = batch_audio_feature_lens(data.get("audio_feature_lens"), device)


def _apply_media_bounds(data: dict[str, Any], model_config) -> None:
    """Derive ``image_bound`` / ``audio_bounds`` from the expanded ids, cross-checking the span counts."""
    if not _media_in_data(data):
        _reject_media_ids_without_features(data, model_config)
        return
    tokens = _resolve_media_tokens(model_config)
    if _is_packed_batch(data):
        _apply_packed_media_bounds(data, tokens)
    else:
        _apply_per_row_media_bounds(data, tokens)


def _reject_media_ids_without_features(data: dict[str, Any], model_config) -> None:
    """Fail closed when the ids carry media spans but no features survived into the batch."""
    processor = getattr(model_config, "processor", None)
    if processor is None:
        return
    from verl_omni.pipelines.minicpm.prompt_parity import resolve_media_tokens

    if not resolve_media_tokens(processor).has_media_tokens(data["input_ids"]):
        return
    raise ValueError(
        "MiniCPM training ids carry expanded media spans but no media features "
        "survived into the batch (pixel_values/audio_features are empty). The "
        "processor output was wiped after the call — refusing to train text-only "
        "over media positions. Inspect multi_modal_inputs for None values."
    )


def _resolve_media_tokens(model_config):
    """The configured processor's media-token scanner; requires a loaded processor."""
    processor = getattr(model_config, "processor", None)
    if processor is None:
        raise RuntimeError(
            "MiniCPM media batches require model_config.processor to derive "
            "image_bound/audio_bounds; the model config did not load a processor."
        )
    from verl_omni.pipelines.minicpm.prompt_parity import resolve_media_tokens

    return resolve_media_tokens(processor)


def _apply_packed_media_bounds(data: dict[str, Any], tokens) -> None:
    """One flattened row: every sample's spans concatenate under it."""
    image_bounds, audio_bounds = tokens.derive_media_bounds(data["input_ids"].reshape(-1))
    _require_span_count(image_bounds, sum(len(sample) for sample in data["pixel_values"]), "image")
    _require_span_count(audio_bounds, sum(len(sample) for sample in data["audio_feature_lens"]), "audio")
    data["image_bound"] = [image_bounds]
    data["audio_bounds"] = [audio_bounds]


def _apply_per_row_media_bounds(data: dict[str, Any], tokens) -> None:
    """Padded layout: one span list per row, checked against the per-sample media lists."""
    per_row_image, per_row_audio = [], []
    for row in range(data["input_ids"].shape[0]):
        image_bounds, audio_bounds = tokens.derive_media_bounds(data["input_ids"][row])
        per_row_image.append(image_bounds)
        per_row_audio.append(audio_bounds)
    _require_span_count(
        [span for spans in per_row_image for span in spans],
        sum(len(sample) for sample in data["pixel_values"]),
        "image",
    )
    _require_span_count(
        [span for spans in per_row_audio for span in spans],
        sum(len(sample) for sample in data["audio_feature_lens"]),
        "audio",
    )
    data["image_bound"] = per_row_image
    data["audio_bounds"] = per_row_audio


def _require_span_count(spans: list[list[int]], expected: int, kind: str) -> None:
    """Fail when the ids' expanded spans and the processor's features disagree."""
    if len(spans) == expected:
        return
    raise ValueError(
        f"MiniCPM {kind} parity failure: ids expanded into {len(spans)} spans but the "
        f"processor features describe {expected}. The rollout and actor renderings "
        "disagree; compare slot expansion on both sides before training."
    )


def _merge_packed_media(data: dict[str, Any]) -> None:
    """Fold the per-sample media into one pseudo-row, sample-major to match the id scan."""
    # Per-example slice counts for the patched vision tower's re-split;
    # must be taken before the fold flattens the structure away.
    data["packed_vision_slices"] = [len(sample) for sample in data["pixel_values"]]
    data["pixel_values"] = [[slice_ for sample in data["pixel_values"] for slice_ in sample]]
    tgt_sizes = [sample for sample in data["tgt_sizes"] if int(sample.numel()) > 0]
    data["tgt_sizes"] = [torch.cat(tgt_sizes, dim=0)] if tgt_sizes else [torch.zeros(0, 2, dtype=torch.int32)]
    data["audio_feature_lens"] = [
        [lens for sample in data["audio_feature_lens"] for lens in torch.as_tensor(sample).reshape(-1).tolist()]
    ]
    data["image_bound"] = [[span for spans in data["image_bound"] for span in spans]]
    data["audio_bounds"] = [[span for spans in data["audio_bounds"] for span in spans]]


def _media_in_data(data: dict[str, Any]) -> bool:
    """True when the split left any pixels or audio features in the batch."""
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


def _batch_size_from_input_ids(input_ids) -> int:
    """Row count of the id tensor, tolerating nested sequence inputs."""
    if hasattr(input_ids, "shape") and len(input_ids.shape) > 0:
        return int(input_ids.shape[0])
    return len(input_ids)
