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

Keep MiniCPM-o remote-code patches in this file; each patch documents its
own trigger. The ``patch_remote_*`` entry points run before
``from_pretrained``, the rest after the model is loaded.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

_POST_INIT_PATCHED_ATTR = "_verl_omni_post_init_patched"
_WHISPER_ATTN_PATCHED_ATTR = "_verl_omni_whisper_attn_return3"
_VISION_EMB_PATCH_ATTR = "_verl_omni_get_vision_embedding_patched"
_VLLM_EMB_PATCH_ATTR = "_verl_omni_get_vllm_embedding_patched"
_SIGLIP_FA2_PATCHED_ATTR = "_verl_omni_siglip_fa2_aliased"
_AUDIO_DUMMY_PATCH_ATTR = "_verl_omni_get_audio_embedding_patched"
# Back-compat alias for tests that reset the post_init wrap.
_PATCHED_ATTR = _POST_INIT_PATCHED_ATTR


def _needs_transformers5_compat() -> bool:
    try:
        import transformers

        return int(transformers.__version__.split(".", 1)[0]) >= 5
    except Exception:
        return False


def patch_remote_auto_model_init(
    model_path: str,
    *,
    trust_remote_code: bool,
    config: Any = None,
    auto_class_name: str = "AutoModel",
) -> None:
    """Wrap a remote auto-model class so ``post_init()`` runs when missing.

    The remote code (transformers ~4.10) omits ``post_init()``, which
    transformers >= 5 requires to set ``all_tied_weights_keys``.
    Call before ``AutoModel.from_pretrained``.
    """
    if not _needs_transformers5_compat() or not trust_remote_code:
        return

    from transformers import AutoConfig
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    resolved_config = config
    if resolved_config is None:
        resolved_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    auto_map = getattr(resolved_config, "auto_map", None)
    if not auto_map or auto_class_name not in auto_map:
        return

    model_cls = get_class_from_dynamic_module(
        auto_map[auto_class_name],
        model_path,
        trust_remote_code=trust_remote_code,
    )
    wrap_model_init_with_post_init(model_cls)


def wrap_model_init_with_post_init(model_cls: type) -> None:
    """Ensure ``model_cls.__init__`` ends with ``post_init()`` when needed."""
    if getattr(model_cls, _POST_INIT_PATCHED_ATTR, False):
        return

    original_init = model_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "post_init"):
            self.post_init()

    model_cls.__init__ = patched_init
    setattr(model_cls, _POST_INIT_PATCHED_ATTR, True)
    logger.debug(
        "Patched %s.__init__ to call post_init() for transformers >= 5 compatibility.",
        model_cls.__name__,
    )


def patch_remote_siglip_flash_attn_support(model_path: str, *, trust_remote_code: bool, config: Any = None) -> None:
    """Alias the remote SigLIP's transformers-4.x FA2 flag to the >= 5 name.

    The remote ``modeling_navit_siglip.py`` implements FA2 natively but
    declares the 4.x flag ``_supports_flash_attn_2``; transformers 5's
    init-time dispatch hard-rejects ``flash_attention_2`` without the
    renamed flag. The alias copies the remote's own declaration only.

    The class is resolved through the auto_map ``AutoModel`` entry — the
    entry the loader itself uses. For local model paths, requesting the
    siglip module directly lands in a different dynamic-module cache dir
    than ``from_pretrained``, and the alias would sit on an orphaned class
    copy the model never imports.

    Call before ``AutoModel.from_pretrained``. No-op on transformers < 5.
    """
    if not _needs_transformers5_compat() or not trust_remote_code:
        return

    from transformers import AutoConfig, PreTrainedModel
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    resolved_config = config
    if resolved_config is None:
        resolved_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    auto_map = getattr(resolved_config, "auto_map", None) or {}
    main_class_ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if main_class_ref is None:
        return

    model_cls = get_class_from_dynamic_module(main_class_ref, model_path, trust_remote_code=trust_remote_code)
    main_module = sys.modules.get(model_cls.__module__)
    if main_module is None:
        return

    # The main modeling module binds SiglipVisionTransformer via its remote
    # import, so its namespace holds the exact class object the model
    # constructs — alias flagged classes found there.
    for value in vars(main_module).values():
        if (
            isinstance(value, type)
            and issubclass(value, PreTrainedModel)
            and getattr(value, "_supports_flash_attn_2", False)
            and "_supports_flash_attn" not in value.__dict__
            and not getattr(value, _SIGLIP_FA2_PATCHED_ATTR, False)
        ):
            value._supports_flash_attn = True
            setattr(value, _SIGLIP_FA2_PATCHED_ATTR, True)
            logger.debug(
                "Aliased %s._supports_flash_attn_2 to the transformers>=5 _supports_flash_attn name.",
                value.__name__,
            )


def _pad_whisper_self_attn_output(output, past_key_values=None):
    """Normalize WhisperAttention output to the 3-tuple MiniCPM remote code unpacks."""
    if not isinstance(output, tuple):
        return output, None, past_key_values
    if len(output) == 2:
        hidden_states, attn_weights = output
        return hidden_states, attn_weights, past_key_values
    return output


def wrap_whisper_self_attn_forward(attn_module) -> None:
    """Make ``self_attn`` always return ``(hidden_states, attn_weights, past_key_values)``.

    The remote ``MiniCPMWhisperEncoderLayer`` (4.x era) unpacks three values
    and passes ``past_key_value`` (singular); current WhisperAttention
    returns two and takes ``past_key_values``.
    """
    if attn_module is None or getattr(attn_module, _WHISPER_ATTN_PATCHED_ATTR, False):
        return

    original_forward = attn_module.forward

    def _forward(*args, _original=original_forward, **kwargs):
        past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if "past_key_value" in kwargs and "past_key_values" not in kwargs:
            kwargs["past_key_values"] = kwargs.pop("past_key_value")
        return _pad_whisper_self_attn_output(_original(*args, **kwargs), past_key_values)

    attn_module.forward = _forward
    setattr(attn_module, _WHISPER_ATTN_PATCHED_ATTR, True)


def patch_remote_whisper_self_attn(module) -> None:
    """Patch MiniCPM-o ``apm`` Whisper self-attn after remote-code ``from_pretrained``.

    Walks ``module.apm.layers[*].self_attn``. No-op when ``apm`` is missing.
    Call this after the model is loaded.
    """
    apm = getattr(module, "apm", None)
    layers = getattr(apm, "layers", None) if apm is not None else None
    if not layers:
        return
    for layer in layers:
        wrap_whisper_self_attn_forward(getattr(layer, "self_attn", None))


def _has_pixel_slices(pixel_values) -> bool:
    if pixel_values is None or pixel_values == []:
        return False
    if isinstance(pixel_values, (list | tuple)):
        return any(_has_pixel_slices(sample) for sample in pixel_values)
    return True


def patch_minicpm_get_vision_embedding(module) -> None:
    """Run the vision tower once per sample, under no_grad.

    The remote method batches every sample's slices through ``vpm`` +
    ``resampler``; batched bf16 kernels flip their reduction order at
    small batch sizes, and the LLM amplifies the ulp difference into
    interior logprob deviations. Each sample's slice group — re-split per
    example for the packed pseudo-row via the ``packed_vision_slices``
    stash — runs the tower alone (the bs==1 realization), outputs
    concatenated in flat span order. The tower is frozen under LoRA, so
    it stays out of autograd and empty rows skip the encoder.
    """
    original = getattr(module, "get_vision_embedding", None)
    if original is None or getattr(module, _VISION_EMB_PATCH_ATTR, False):
        return

    def get_vision_embedding(self, data, _original=original):
        del self
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
    setattr(module, _VISION_EMB_PATCH_ATTR, True)


def _embed_tokens_module(module):
    llm = getattr(module, "llm", None)
    if llm is None:
        return None
    model = getattr(llm, "model", llm)
    return getattr(model, "embed_tokens", None)


def patch_minicpm_get_vllm_embedding(module) -> None:
    """Clone text embeddings before the vision scatter.

    The remote ``get_vllm_embedding`` scatters in place on the
    embed_tokens output, which PEFT's ``enable_input_require_grads``
    turns into a leaf — the in-place view op fails. Out-of-place
    ``scatter`` on cloned rows instead.
    """
    if getattr(module, _VLLM_EMB_PATCH_ATTR, False):
        return
    if _embed_tokens_module(module) is None or not hasattr(module, "get_vision_embedding"):
        return

    def get_vllm_embedding(self, data):
        import torch

        vision_hidden_states = self.get_vision_embedding(data)
        vllm_embedding = _embed_tokens_module(self)(data["input_ids"])
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
                    image_indices = torch.stack(
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
    setattr(module, _VLLM_EMB_PATCH_ATTR, True)


def patch_minicpm_get_audio_embedding(module) -> None:
    """Run the Whisper tower once per clip on its exact-length slice.

    The remote method pads all clips to one ``max_frames`` and masks with
    mel-frame lengths compared against post-conv2 positions — the mask
    under-masks by ~2x, and even equal-length batched clips deviate from
    single-clip outputs. One exact-length ``[1, 80, len]`` clip per call
    leaves no padded frames and reproduces the bs==1 realization,
    regrouped into the remote's per-row layout. Audio-free training
    batches skip the frozen tower and return one zero token, preserving
    ``get_omni_embedding``'s ``audio_embeddings[0].mean() * 0`` anchor.
    """
    apm = getattr(module, "apm", None)
    llm = getattr(module, "llm", None)
    original = getattr(module, "get_audio_embedding", None)
    if apm is None or llm is None or original is None or getattr(module, _AUDIO_DUMMY_PATCH_ATTR, False):
        return

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
    setattr(module, _AUDIO_DUMMY_PATCH_ATTR, True)


_OMNI_EMB_PATCH_ATTR = "_verl_omni_get_omni_embedding_patched"


def patch_minicpm_get_omni_embedding(module) -> None:
    """Splice audio embeddings per row in the non-streaming omni path.

    The remote ``get_omni_embedding`` dedents its audio splice out of the
    ``for i in range(bs)`` loop — only the LAST row of a bs>1 batch is
    ever written; earlier rows keep placeholder embeddings. Inert on the
    packed bs==1 path the recipe ships. The patched path re-implements
    the splice per row (both remote layouts, the length-mismatch check,
    clone-before-write); streaming and audio-free batches delegate.
    """
    original = getattr(module, "get_omni_embedding", None)
    if original is None or getattr(module, _OMNI_EMB_PATCH_ATTR, False):
        return

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
    setattr(module, _OMNI_EMB_PATCH_ATTR, True)


_ANSWER_TAG_TOKENS = ("<answer>", "</answer>")


def keep_answer_tags_when_decoding(tokenizer) -> bool:
    """Demote ``<answer>`` / ``</answer>`` from special to plain added tokens.

    The checkpoint registers them ``special: true``; verl's reward decode
    uses ``skip_special_tokens=True`` and strips exactly the two tags,
    zeroing every choice-reward score. Only the decode skip filter
    consults the flag — ids, atomic encoding, and generation are
    unchanged. The flag lives in the Rust backend's registry, so the fix
    rebuilds it with the two flags flipped. Returns whether anything was
    demoted; already-plain tags are a no-op, an unfixable tokenizer
    raises rather than silently zeroing rewards.
    """
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    needs_demotion = any(
        getattr(decoder.get(tokenizer.convert_tokens_to_ids(token)), "special", False) for token in _ANSWER_TAG_TOKENS
    )
    if not needs_demotion:
        return False

    import json

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
        if entry.get("content") in _ANSWER_TAG_TOKENS and entry.get("special", False):
            entry["special"] = False
            demoted = True
    if not demoted:
        return False
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(data))
    return demoted
