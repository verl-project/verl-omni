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
"""Runtime shims for MiniCPM-o Hugging Face remote-code models.

Each patch documents its own trigger. Two fix class-level problems and run before
``from_pretrained`` (the auto-model init, which routes through the private
``_wrap_init_with_post_init``, and SigLIP's FA flag); the rest patch a method on
the loaded module.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "patch_minicpm_answer_tags",
    "patch_minicpm_auto_model_init",
    "patch_minicpm_get_audio_embedding",
    "patch_minicpm_get_omni_embedding",
    "patch_minicpm_get_vision_embedding",
    "patch_minicpm_get_vllm_embedding",
    "patch_minicpm_siglip_flash_attn_support",
    "patch_minicpm_whisper_self_attn",
]


def patch_minicpm_auto_model_init(model_path: str, config: Any = None) -> None:
    """Wrap a remote auto-model class so ``post_init()`` runs when missing.

    The remote code predates ``post_init()``, which transformers 5 reads before
    ``from_pretrained`` returns; without the wrap the load raises on the missing
    ``all_tied_weights_keys``.

    Args:
        model_path: Local path to the model checkpoint.
        config: Pre-resolved config, or None to load it here.
    """
    from transformers import AutoConfig
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    resolved_config = config
    if resolved_config is None:
        resolved_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model_cls = get_class_from_dynamic_module(
        resolved_config.auto_map["AutoModel"],
        model_path,
        trust_remote_code=True,  # the checkpoint defines its classes in remote code
    )
    _wrap_init_with_post_init(model_cls)


def patch_minicpm_siglip_flash_attn_support(model_path: str, config: Any = None) -> None:
    """Alias the remote SigLIP's FA2 support flag to the name transformers 5 reads.

    The remote ``modeling_navit_siglip.py`` implements FA2 natively but declares only
    the pre-5 flag name, which transformers 5's init-time dispatch rejects, so the
    vision tower cannot be constructed with the pinned ``flash_attention_2``.

    Args:
        model_path: Local path to the model checkpoint.
        config: Pre-resolved config, or None to load it here.
    """
    from transformers import AutoConfig, PreTrainedModel
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    marker = "_verl_omni_siglip_fa2_aliased"
    resolved_config = config
    if resolved_config is None:
        resolved_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(
        resolved_config.auto_map["AutoModel"],
        model_path,
        trust_remote_code=True,  # the checkpoint defines its classes in remote code
    )
    main_module = sys.modules[model_cls.__module__]

    # The main modeling module binds SiglipVisionTransformer via its remote
    # import, so its namespace holds the exact class object the model
    # constructs — alias flagged classes found there.
    for value in vars(main_module).values():
        if (
            isinstance(value, type)
            and issubclass(value, PreTrainedModel)
            and getattr(value, "_supports_flash_attn_2", False)
            and "_supports_flash_attn" not in value.__dict__
            and not getattr(value, marker, False)
        ):
            value._supports_flash_attn = True
            setattr(value, marker, True)
            logger.debug(
                "Aliased %s._supports_flash_attn_2 to the _supports_flash_attn name transformers 5 reads.",
                value.__name__,
            )


def patch_minicpm_whisper_self_attn(module) -> None:
    """Patch MiniCPM-o ``apm`` Whisper self-attn after remote-code ``from_pretrained``."""
    for layer in module.apm.layers:
        _wrap_whisper_attn_forward(layer.self_attn)


def patch_minicpm_get_vision_embedding(module) -> None:
    """Run the vision tower one sample (or one packed example) at a time.

    Batched bf16 kernels flip reduction order at small batch sizes, which the LLM
    amplifies into interior logprob deviations, so each sample's slice group runs
    the tower alone. The tower is frozen, so no_grad is free.
    """
    marker = "_verl_omni_get_vision_embedding_patched"
    if getattr(module, marker, False):
        return
    original = module.get_vision_embedding

    def get_vision_embedding(self, data, _original=original):
        if isinstance(data, dict) and "vision_hidden_states" in data:
            return _original(data)
        pixel_values = data.get("pixel_values") if isinstance(data, dict) else None
        if not _has_pixel_slices(pixel_values):
            return [[] for _ in (pixel_values or [])]
        import torch

        def _row(values, tgt_size):
            row_data = dict(data)
            row_data["pixel_values"] = [values]
            row_data["tgt_sizes"] = [tgt_size]
            return _original(row_data)

        with torch.no_grad():
            packed_counts = data.get("packed_vision_slices") if isinstance(data, dict) else None
            if packed_counts is not None:
                # Packed pseudo-row: re-split per example so each example's
                # group runs the tower alone; outputs concatenate in flat
                # span order (sample-major, matching the id scan).
                flat_slices = pixel_values[0]
                tgt_sizes = data["tgt_sizes"][0]
                outputs = []
                offset = 0
                for count in packed_counts:
                    if count:
                        outputs.append(
                            _row(flat_slices[offset : offset + count], tgt_sizes[offset : offset + count])[0]
                        )
                    offset += count
                stacked = [out for out in outputs if isinstance(out, torch.Tensor) and out.numel()]
                return [torch.cat(stacked, dim=0) if stacked else torch.zeros(0, 1, 1)]
            rows = []
            for values, tgt_size in zip(pixel_values, data["tgt_sizes"], strict=False):
                rows.append(_row(values, tgt_size)[0] if values else [])
            return rows

    module.get_vision_embedding = types.MethodType(get_vision_embedding, module)
    setattr(module, marker, True)


def patch_minicpm_get_vllm_embedding(module) -> None:
    """Clone text embeddings before the vision scatter.

    The remote ``get_vllm_embedding`` scatters in place on the
    embed_tokens output, which PEFT's ``enable_input_require_grads``
    turns into a leaf — the in-place view op fails. Out-of-place
    ``scatter`` on cloned rows instead.
    """
    marker = "_verl_omni_get_vllm_embedding_patched"
    if getattr(module, marker, False):
        return

    def get_vllm_embedding(self, data):
        import torch

        vision_hidden_states = self.get_vision_embedding(data)
        vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])
        llm_config = getattr(self.llm, "config", None)
        if llm_config is not None and hasattr(llm_config, "scale_emb"):
            vllm_embedding = vllm_embedding * llm_config.scale_emb

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        rows = []
        batch_size = len(data["input_ids"])
        image_bound = data.get("image_bound") or [[] for _ in range(batch_size)]
        for i in range(batch_size):
            row = vllm_embedding[i].clone()
            cur_vs_hs = vision_hidden_states[i] if i < len(vision_hidden_states) else []
            if len(cur_vs_hs) > 0:
                cur_image_bound = image_bound[i]
                if len(cur_image_bound) > 0:
                    # Spans of different token counts (per-slice grids) are unequal-
                    # length aranges; cat keeps them in span order, stack would raise.
                    image_indices = torch.cat(
                        [torch.arange(int(bound[0]), int(bound[1]), dtype=torch.long) for bound in cur_image_bound]
                    ).to(vllm_embedding.device)
                    src = cur_vs_hs.view(-1, cur_vs_hs.shape[-1]).to(device=row.device, dtype=row.dtype)
                    index = image_indices.view(-1, 1).repeat(1, row.shape[-1])
                    row = row.scatter(0, index, src)
                elif self.training:
                    row = row + cur_vs_hs[0].mean() * 0
            rows.append(row)
        return torch.stack(rows, dim=0), vision_hidden_states

    module.get_vllm_embedding = types.MethodType(get_vllm_embedding, module)
    setattr(module, marker, True)


def patch_minicpm_get_audio_embedding(module) -> None:
    """Run the Whisper tower one exact-length clip at a time.

    The remote method pads every clip to one ``max_frames`` and masks with
    mel-frame lengths against post-conv2 positions (under-masking by ~2x), so
    batched clips deviate from the single-clip output. Audio-free training
    batches skip the frozen tower and return one zero token.
    """
    marker = "_verl_omni_get_audio_embedding_patched"
    if getattr(module, marker, False):
        return
    original = module.get_audio_embedding

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True, **kwargs):
        # Mirrors the remote get_omni_embedding emptiness test (post-split
        # normalization empties are exactly []); no `or` — features may be a
        # tensor and tensor truthiness is ambiguous.
        features = data.get("audio_features", [])
        if features is None:
            features = []
        if self.training and len(features) == 0:
            import torch

            weight = self.apm.conv1.weight
            hidden = getattr(getattr(self.llm, "config", None), "hidden_size", 1)
            return [torch.zeros(1, hidden, 1, device=weight.device, dtype=weight.dtype)]
        if len(features) == 0:
            return original(data, chunk_length=chunk_length, dummy=dummy, **kwargs)

        import torch

        # Per-clip execution needs the stacked (n_clips, 80, frames) tensor
        # the split normalization produces; any other container falls back
        # to the remote batched path with a warning instead of guessing.
        if not isinstance(features, torch.Tensor):
            logger.warning(
                "MiniCPM per-clip audio path received %s audio_features; delegating the whole "
                "batch to the remote batched path.",
                type(features).__name__,
            )
            return original(data, chunk_length=chunk_length, dummy=dummy, **kwargs)

        lens_raw = data.get("audio_feature_lens") or []
        flat_lens = [int(length) for row in lens_raw for length in torch.as_tensor(row).reshape(-1).tolist()]
        if len(flat_lens) != len(features):
            raise ValueError(f"Audio clips ({len(features)}) and audio_feature_lens ({len(flat_lens)}) disagree.")
        with torch.no_grad():
            clip_embeddings = []
            for index, length in enumerate(flat_lens):
                clip_data = dict(data)
                clip_data["audio_features"] = features[index : index + 1, :, :length].contiguous()
                # The remote hstacks per-sample 1-D tensors; nested python
                # lists raise TypeError inside get_audio_embedding.
                clip_data["audio_feature_lens"] = [torch.tensor([length], dtype=torch.long, device=features.device)]
                clip_embeddings.append(original(clip_data, chunk_length=chunk_length, dummy=dummy, **kwargs)[0][0])
        grouped = []
        offset = 0
        for row in lens_raw:
            count = len(torch.as_tensor(row).reshape(-1).tolist())
            grouped.append(clip_embeddings[offset : offset + count])
            offset += count
        return grouped

    module.get_audio_embedding = types.MethodType(get_audio_embedding, module)
    setattr(module, marker, True)


def patch_minicpm_get_omni_embedding(module) -> None:
    """Splice audio embeddings per row in the non-streaming omni path.

    The remote ``get_omni_embedding`` dedents its audio splice out of the
    ``for i in range(bs)`` loop — only the LAST row of a bs>1 batch is
    ever written; earlier rows keep placeholder embeddings. Inert on the
    packed bs==1 path the recipe ships. The patched path re-implements
    the splice per row (both remote layouts, the length-mismatch check,
    clone-before-write); streaming and audio-free batches delegate.
    """
    marker = "_verl_omni_get_omni_embedding_patched"
    if getattr(module, marker, False):
        return
    original = module.get_omni_embedding

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False, **kwargs):
        config_stream = bool(getattr(getattr(self, "config", None), "stream_input", False))
        features = data.get("audio_features", [])
        if stream_input or config_stream or len(features) == 0:
            return original(data, input_embeddings, chunk_length=chunk_length, stream_input=stream_input, **kwargs)

        audio_embeddings = self.get_audio_embedding(data, chunk_length)
        import torch

        if len(audio_embeddings) != len(input_embeddings):
            raise ValueError(
                f"Audio embeddings cover {len(audio_embeddings)} rows but the batch has {len(input_embeddings)}."
            )
        audio_bounds = data["audio_bounds"]
        result = input_embeddings.clone()
        for row, (audio_embs, bounds) in enumerate(zip(audio_embeddings, audio_bounds, strict=False)):
            one_to_one_match = len(audio_embs) == len(bounds) and all(
                embs.shape[0] == int(bound[1] - bound[0]) for embs, bound in zip(audio_embs, bounds, strict=False)
            )
            if one_to_one_match:
                for embs, bound in zip(audio_embs, bounds, strict=False):
                    result[row, int(bound[0]) : int(bound[1])] = embs.to(device=result.device, dtype=result.dtype)
                continue
            flat_audio_embs = torch.cat(audio_embs, dim=0).to(device=result.device, dtype=result.dtype)
            total_bound_len = sum(int(bound[1] - bound[0]) for bound in bounds)
            if flat_audio_embs.shape[0] != total_bound_len:
                raise ValueError(f"Audio total length mismatch: {flat_audio_embs.shape[0]} != {total_bound_len}")
            offset = 0
            for bound in bounds:
                audio_len = int(bound[1] - bound[0])
                result[row, int(bound[0]) : int(bound[1])] = flat_audio_embs[offset : offset + audio_len]
                offset += audio_len
        return result

    module.get_omni_embedding = types.MethodType(get_omni_embedding, module)
    setattr(module, marker, True)


def patch_minicpm_answer_tags(tokenizer) -> bool:
    """Demote ``<answer>`` / ``</answer>`` from special to plain added tokens.

    The checkpoint registers them ``special: true``, and verl's reward decode skips
    special tokens, so the two tags are stripped and every choice-reward score is
    zeroed. Only the decode skip filter consults the flag — ids, atomic encoding, and
    generation are unchanged.

    Args:
        tokenizer: The tokenizer whose decode must keep the tags.

    Returns:
        True when the tags were demoted; False when they were already plain.
    """
    answer_tags = ("<answer>", "</answer>")
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    needs_demotion = any(
        getattr(decoder.get(tokenizer.convert_tokens_to_ids(token)), "special", False) for token in answer_tags
    )
    if not needs_demotion:
        return False

    from tokenizers import Tokenizer

    backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise RuntimeError(
            "MiniCPM-o answer tags are special tokens but the tokenizer has no fast backend "
            "to demote them on; skip_special_tokens=True would strip the tags and zero the "
            "choice reward."
        )
    data = json.loads(backend.to_str())
    demoted = False
    for entry in data.get("added_tokens", []):
        if entry.get("content") in answer_tags and entry.get("special", False):
            entry["special"] = False
            demoted = True
    if not demoted:
        return False
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(data))
    return demoted


def _wrap_init_with_post_init(model_cls: type) -> None:
    """Ensure ``model_cls.__init__`` ends with ``post_init()`` when needed."""
    marker = "_verl_omni_post_init_patched"
    if getattr(model_cls, marker, False):
        return

    original_init = model_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "post_init"):
            self.post_init()

    model_cls.__init__ = patched_init
    setattr(model_cls, marker, True)
    logger.debug(
        "Patched %s.__init__ to call post_init() after the remote __init__.",
        model_cls.__name__,
    )


def _pad_whisper_self_attn_output(output, past_key_values=None):
    """Normalize WhisperAttention output to the 3-tuple MiniCPM remote code unpacks."""
    if not isinstance(output, tuple):
        return output, None, past_key_values
    if len(output) == 2:
        hidden_states, attn_weights = output
        return hidden_states, attn_weights, past_key_values
    return output


def _wrap_whisper_attn_forward(attn_module) -> None:
    """Pad WhisperAttention's 2-tuple to the 3-tuple the remote encoder layer unpacks."""
    marker = "_verl_omni_whisper_attn_return3"
    if getattr(attn_module, marker, False):
        return

    original_forward = attn_module.forward

    def _forward(*args, _original=original_forward, **kwargs):
        # The remote layer passes ``past_key_value`` (singular); current
        # WhisperAttention takes the plural, and returns two values not three.
        past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if "past_key_value" in kwargs and "past_key_values" not in kwargs:
            kwargs["past_key_values"] = kwargs.pop("past_key_value")
        return _pad_whisper_self_attn_output(_original(*args, **kwargs), past_key_values)

    attn_module.forward = _forward
    setattr(attn_module, marker, True)


def _has_pixel_slices(pixel_values) -> bool:
    if pixel_values is None or pixel_values == []:
        return False
    if isinstance(pixel_values, (list | tuple)):
        return any(_has_pixel_slices(sample) for sample in pixel_values)
    return True
