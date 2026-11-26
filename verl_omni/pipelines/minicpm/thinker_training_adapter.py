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

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.utils.dataset.duplex.policy_measure import resolve_forbidden_token_ids, two_stage_logprobs

# Keys routed into MiniCPMO's ``data`` dict instead of ``self.llm(**kwargs)``.
_MINICPM_DATA_KEYS = (
    "input_ids",
    "position_ids",
    "pixel_values",
    "tgt_sizes",
    "image_sizes",  # processor metadata no MiniCPM-o forward consumes
    "image_bound",
    "audio_features",
    "audio_feature_lens",
    "audio_bounds",
    "spk_bounds",
    "vision_hidden_states",
)
# Response rows per chunk of the two-stage measure: each chunk holds four (rows, V) fp32 copies at peak.
_DUPLEX_LOGPROB_CHUNK_ROWS = 512


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
            patch_minicpm_auto_model_init,
            patch_minicpm_siglip_flash_attn_support,
        )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        config = kwargs.get("config")
        if not trust_remote_code:
            raise ValueError(
                "MiniCPM-o checkpoints define their classes in remote code, so "
                "actor_rollout_ref.model.trust_remote_code must be True."
            )
        # Training-path invariants, set before the remote __init__ reads them.
        if config is not None:
            config.init_tts = False
            config.use_cache = False
            config.stream_input = False
        # Remote code omits post_init(), which transformers 5 calls before from_pretrained returns.
        patch_minicpm_auto_model_init(pretrained_model_name_or_path, config=config)
        # The remote vision tower declares only the pre-5 FA2 support flag.
        patch_minicpm_siglip_flash_attn_support(pretrained_model_name_or_path, config=config)
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if hasattr(model, "resampler"):
            # Transformers initializes non-persistent buffers it cannot load to zero.
            resampler = model.resampler
            resampler._set_2d_pos_cache(resampler.max_size, device=resampler.pos_embed.device)
        return model


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
        # Whole-model dtype casts must not quantize the non-persistent RoPE frequencies.
        _keep_rotary_frequencies_dtype(module)
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
        # Binds __call__ / apply_chat_template; see prompt_parity.
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

        from verl_omni.models.transformers.minicpm_o import patch_minicpm_answer_tags

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        # The reward manager decodes with skip_special_tokens=True, which would strip
        # the checkpoint's special <answer> tags and zero every choice-reward score.
        patch_minicpm_answer_tags(tokenizer)
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
            micro_batch: The micro batch being prepared; a duplex batch (one
                carrying the namespaced ``duplex_replay`` key) takes the
                explicit-bounds path instead of the marker scan — the engine's
                duplex layout uses bare audio placeholders without
                ``<|audio_start|>`` markers, so the bounds must arrive with
                the replay instead of being re-derived from markers.
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
        duplex_replays = _duplex_replay_payloads(micro_batch)
        if duplex_replays is not None:
            # Duplex replay: media arrive with explicit per-unit bounds.
            _apply_duplex_media(data, duplex_replays)
        else:
            # Re-derive the media bounds from the ids and cross-check the span counts.
            _apply_media_bounds(data, model_config)
        if _is_packed_batch(data):
            # Fold the per-sample media into MiniCPMO's single-row packed layout.
            _merge_packed_media(data)
            # FA2 derives cu_seqlens from position_ids; a padded attention_mask
            # would contradict the packed layout.
            llm_kwargs.pop("attention_mask", None)
        return {"data": data, **llm_kwargs}

    @classmethod
    def prepare_model_outputs(cls, model_output, output, output_args, micro_batch, model_config):
        """Score a duplex batch's masked response tokens under the engine's two-stage joint.

        Runs after verl's own ``prepare_model_outputs`` on every actor forward
        (``old_log_probs``, the update forward's ``log_probs``, ``ref_log_prob``):
        each masked response position's full-vocabulary log-probability gives
        way to ``two_stage_logprobs`` on the same predicting logits, so the
        training measure equals the engine's recorded one. Mask-0 positions
        keep verl's value and ``entropy`` stays as verl computed it. A
        non-duplex batch comes back unchanged.

        Args:
            model_output: verl's prepared output; ``log_probs`` is nested
                ``(bsz,)`` with one entry per ``prompt + response`` position
                of each sample.
            output: The module's raw output; ``.logits`` is
                ``(bsz, max_seq_len, V)``.
            output_args: verl's dense-branch arguments:
                ``input_ids_rmpad_rolled`` ``(total_nnz,)`` and ``temperature``
                ``(bsz,)``.
            micro_batch: The micro-batch: nested ``input_ids`` and
                ``response_mask``, plus the ``duplex_replay`` payloads.
            model_config: The ``OmniModelConfig`` carrying the processor.

        Returns:
            ``model_output`` with ``log_probs`` replaced at masked response
            positions, re-nested to verl's layout.

        Raises:
            NotImplementedError: On a packed (``use_remove_padding``) batch —
                the layout sequence parallelism requires in verl; the recipe
                ships the dense one.
            RuntimeError: When ``model_config`` carries no processor/tokenizer.
            ValueError: When the tokenizer carries no ``bad_token_ids``.
            RecordSetError: When the tokenizer cannot resolve the forbidden set.
        """
        from verl.utils import tensordict_utils as tu

        if _duplex_replay_payloads(micro_batch) is None:
            return model_output
        if tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True):
            raise NotImplementedError(
                "MiniCPM-o duplex log-probabilities are implemented for the dense layout only "
                "(use_remove_padding=false, ulysses_sequence_parallel_size=1); a packed batch reached "
                "prepare_model_outputs."
            )
        logits = output.logits
        chunk_eos_id, forbidden_index = cls._resolve_duplex_forbidden_ids(model_config, logits.device)
        cu_seqlens = micro_batch["input_ids"].offsets().to(logits.device)
        response_lengths = micro_batch["response_mask"].offsets().diff().to(logits.device)
        # Response position r of sample i is predicted by the logit one token earlier: packed row
        # off_i + seq_len_i − resp_len_i − 1 + r (response_from_nested's shift), padded column
        # seq_len_i − resp_len_i − 1 + r.
        packed_starts = cu_seqlens[1:] - response_lengths - 1
        padded_starts = cu_seqlens.diff() - response_lengths - 1
        in_response = torch.arange(int(response_lengths.max()), device=logits.device) < response_lengths[:, None]
        samples, response_index = in_response.nonzero(as_tuple=True)
        response_logits = logits[samples, padded_starts[samples] + response_index]
        packed_rows = packed_starts[samples] + response_index
        action_ids = output_args["input_ids_rmpad_rolled"][packed_rows]
        temperature = torch.as_tensor(output_args["temperature"], device=logits.device)
        if temperature.ndim == 1 and temperature.numel() == response_lengths.shape[0]:
            temperature = temperature[samples]
        apply = micro_batch["response_mask"].values() != 0
        joint = torch.cat(
            [
                checkpoint(
                    _two_stage_chunk,
                    response_logits[start : start + _DUPLEX_LOGPROB_CHUNK_ROWS],
                    action_ids[start : start + _DUPLEX_LOGPROB_CHUNK_ROWS],
                    chunk_eos_id=chunk_eos_id,
                    forbidden_index=forbidden_index,
                    temperature=temperature[start : start + _DUPLEX_LOGPROB_CHUNK_ROWS],
                    use_reentrant=False,
                )
                for start in range(0, apply.shape[0], _DUPLEX_LOGPROB_CHUNK_ROWS)
            ]
        )
        verl_log_probs = model_output["log_probs"].values()
        model_output["log_probs"] = torch.nested.nested_tensor_from_jagged(
            verl_log_probs.index_copy(0, packed_rows, torch.where(apply, joint, verl_log_probs[packed_rows])),
            micro_batch["input_ids"].offsets(),
        )
        return model_output

    @staticmethod
    def _resolve_duplex_forbidden_ids(model_config, device) -> tuple[int, torch.Tensor]:
        """``(chunk_eos_id, forbidden_index)`` for ``two_stage_logprobs``, from the run's tokenizer."""
        tokenizer = getattr(getattr(model_config, "processor", None), "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "MiniCPM-o duplex log-probabilities need model_config.processor.tokenizer to resolve "
                "the engine's forbidden set; the model config did not load a processor."
            )
        if getattr(tokenizer, "bad_token_ids", None) is None:
            raise ValueError(
                "the checkpoint tokenizer carries no bad_token_ids; the engine's forbidden set "
                "(tts_pad + bad_token_ids + chunk_eos) cannot be resolved"
            )
        forbidden_ids = resolve_forbidden_token_ids(tokenizer)
        return forbidden_ids[-1], torch.tensor(forbidden_ids, dtype=torch.long, device=device)


def _unwrap_non_tensor(value: Any) -> Any:
    """``NonTensorData.data`` when wrapped; otherwise the value itself."""
    from tensordict import NonTensorData

    if isinstance(value, NonTensorData):
        return value.data
    return value


def _as_entries(payloads: Any) -> list[Any]:
    """One list of unwrapped values, whether the column is a stack, a slice, or a mapping."""
    payloads = _unwrap_non_tensor(payloads)
    if isinstance(payloads, list | tuple) and not isinstance(payloads, dict):
        return [_unwrap_non_tensor(entry) for entry in payloads]
    if hasattr(payloads, "tolist") and not isinstance(payloads, dict):
        return [_unwrap_non_tensor(entry) for entry in payloads.tolist()]
    return [payloads]


def _lookup_column(micro_batch: Any, key: str) -> Any:
    """A micro-batch column, including one nested under ``non_tensor_batch``."""
    from collections.abc import Mapping

    if isinstance(micro_batch, Mapping):
        if key in micro_batch:
            return micro_batch[key]
        nested = micro_batch.get("non_tensor_batch")
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
        return None
    keys = micro_batch.keys() if hasattr(micro_batch, "keys") else ()
    if key in keys:
        from verl.utils.tensordict_utils import get_non_tensor_data

        return get_non_tensor_data(micro_batch, key, None)
    nested = micro_batch.get("non_tensor_batch") if hasattr(micro_batch, "get") else None
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return None


def _duplex_replay_payloads(micro_batch: Any) -> list[dict[str, Any]] | None:
    """The micro-batch's duplex replay payloads, or ``None`` when not duplex.

    The replay is keyed ``duplex_replay``. verl sometimes spreads that key as
    its own column and sometimes keeps the agent loop's ``extra_fields``
    mapping as one column, with the replay inside each sample.
    """
    from collections.abc import Mapping

    from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY

    if micro_batch is None:
        return None
    column = _lookup_column(micro_batch, DUPLEX_REPLAY_KEY)
    if column is None:
        extra = _lookup_column(micro_batch, "extra_fields")
        if extra is None:
            return None
        entries = []
        saw_replay = False
        saw_other = False
        for item in _as_entries(extra):
            if isinstance(item, Mapping) and DUPLEX_REPLAY_KEY in item:
                saw_replay = True
                entries.append(_unwrap_non_tensor(item[DUPLEX_REPLAY_KEY]))
            else:
                saw_other = True
        if saw_replay and saw_other:
            raise ValueError(
                "duplex extra_fields column is missing duplex_replay on some samples; "
                "the batch mixes duplex and non-duplex rows"
            )
        if not saw_replay:
            return None
    else:
        entries = _as_entries(column)
    if not all(isinstance(entry, dict) and entry for entry in entries):
        raise ValueError(
            f"{DUPLEX_REPLAY_KEY!r} entries must be non-empty replay payload mappings; "
            "got " + ", ".join(type(entry).__name__ for entry in entries)
        )
    return entries


def _two_stage_chunk(logits, token_ids, *, chunk_eos_id, forbidden_index, temperature):
    """One row-chunk of the two-stage joint on fp32 logits; the caller checkpoints it."""
    return two_stage_logprobs(
        logits.float(), token_ids, chunk_eos_id=chunk_eos_id, forbidden_index=forbidden_index, temperature=temperature
    )


def _apply_duplex_media(data: dict[str, Any], replays: list[dict[str, Any]]) -> None:
    """Merge per-sample duplex media (features + explicit bounds) into ``data``.

    Per-sample bounds are full-row (prompt + response) coordinates; a packed
    batch (rmpad flattening) shifts every sample after the first by its
    predecessor's row length, so bounds are re-based on the packed offsets
    derived from the position-id resets before ``_merge_packed_media`` folds
    the per-sample containers. Degenerate rows (zero-unit payloads from the
    failure-tolerance path) contribute empty containers.

    ``audio_clip_kinds`` is the duplex-only marker the audio patch routes on:
    one entry per stacked clip in row order — ``"reference"`` for the
    offline-encoded reference block a replay carries (``ref_audio``),
    ``"unit"`` for the streaming-encoded unit clips in feed order.
    """
    import numpy as np

    packed = _is_packed_batch(data)
    row_count = _batch_size_from_input_ids(data["input_ids"])
    batch_size = len(replays)
    if (packed and row_count != 1) or (not packed and batch_size != row_count):
        raise ValueError(
            f"duplex batch carries {batch_size} replay payloads over {row_count} id rows "
            f"({'packed' if packed else 'padded'} layout); the replay column and the ids disagree"
        )
    packed_offsets = [0] * batch_size
    if packed:
        position_ids = data["position_ids"].reshape(-1)
        reset_positions = (position_ids == 0).nonzero(as_tuple=True)[0].tolist()
        if len(reset_positions) != batch_size or reset_positions[0] != 0:
            raise ValueError(
                "duplex packed batch has ambiguous sample boundaries: "
                f"{len(reset_positions)} position-id resets for {batch_size} replay payloads"
            )
        packed_offsets = reset_positions
    audio_features: list[list] = []
    audio_feature_lens: list[list] = []
    audio_bounds: list[list] = []
    audio_clip_kinds: list[str] = []
    for sample, payload in enumerate(replays):
        mels = payload.get("unit_mels")
        mel_lens = payload.get("unit_mel_lens")
        bounds = payload.get("audio_bounds")
        if mels is None or mel_lens is None:
            if not bounds:
                # Degenerate episode (the failure-tolerance path's zero-unit
                # payload): no units to embed, so the row contributes empty
                # media containers instead of failing the micro-batch.
                audio_features.append([])
                audio_feature_lens.append([])
                audio_bounds.append([])
                continue
            raise ValueError(
                "a duplex replay payload is missing its mel features but carries "
                "audio bounds; the rollout adapter must attach them before training"
            )
        if not bounds:
            raise ValueError(
                "a duplex replay payload carries mel features but no audio bounds; the spans and the clips disagree"
            )
        offset = packed_offsets[sample]
        shifted = [[int(start) + offset, int(end) + offset] for start, end in bounds]
        device = data["input_ids"].device
        audio_features.append([torch.as_tensor(np.asarray(mel), dtype=torch.float32, device=device) for mel in mels])
        audio_feature_lens.append([torch.tensor([int(value)], dtype=torch.long, device=device) for value in mel_lens])
        audio_bounds.append(shifted)
        kinds = ["unit"] * len(mels)
        if payload.get("ref_audio") is not None and kinds:
            kinds[0] = "reference"
        audio_clip_kinds.extend(kinds)
    _require_span_count(
        [span for spans in audio_bounds for span in spans],
        sum(len(sample) for sample in audio_feature_lens),
        "audio",
    )
    data["audio_features"] = audio_features
    data["audio_feature_lens"] = audio_feature_lens
    data["audio_bounds"] = audio_bounds
    data["audio_clip_kinds"] = audio_clip_kinds
    data.setdefault("pixel_values", [[] for _ in range(batch_size)])
    data.setdefault("tgt_sizes", [[] for _ in range(batch_size)])
    data.setdefault("image_bound", [[] for _ in range(batch_size)])


def _keep_rotary_frequencies_dtype(module) -> None:
    """Make the inner LLM's rotary embeddings ignore dtype casts of their buffers."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    for child in module.llm.modules():
        # Guarded so a repeated configure_model pass cannot wrap the wrapper and recurse.
        if isinstance(child, Qwen3RotaryEmbedding) and not hasattr(child, "_verl_omni_original_apply"):
            child._verl_omni_original_apply = child._apply
            child._apply = types.MethodType(_rotary_apply_keeping_dtype, child)


def _rotary_apply_keeping_dtype(self, fn, recurse=True):
    def keep_dtype(tensor):
        converted = fn(tensor)
        if converted.dtype != tensor.dtype:
            return tensor.to(device=converted.device)
        return converted

    return self._verl_omni_original_apply(keep_dtype, recurse=recurse)


def _apply_remote_code_patches(module) -> None:
    """Install the compatibility shims the remote-code forward paths need."""
    from verl_omni.models.transformers.minicpm_o import (
        patch_minicpm_get_audio_embedding,
        patch_minicpm_get_omni_embedding,
        patch_minicpm_get_vision_embedding,
        patch_minicpm_get_vllm_embedding,
        patch_minicpm_whisper_self_attn,
    )

    # Remote Whisper self-attn returns a 2-tuple the remote encoder layer unpacks as 3.
    patch_minicpm_whisper_self_attn(module)
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

    # The remote forward binds inputs_embeds itself; a forwarded copy would collide.
    llm_kwargs.pop("inputs_embeds", None)
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


def _is_flat_slice_pack(values: list, batch_size: int) -> bool:
    """Whether a per-sample-looking list is one sample's flat ``(C, H, W)`` pack.

    One image's slices arrive as a bare list of 3-D arrays; only a length
    disagreement with the row count distinguishes it from one slice per row.
    """
    if not values or len(values) == batch_size:
        return False
    first = values[0]
    return isinstance(first, torch.Tensor | np.ndarray) and first.ndim == 3


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
        unwrap_collated,
    )

    device = data["input_ids"].device
    batch_size = _batch_size_from_input_ids(data["input_ids"])
    # Unwrap DataProto's ragged collation first: a padded bs>1 batch arrives as an
    # object-dtype ndarray, which the isinstance checks below would otherwise treat
    # as one sample and flatten every row's slices onto batch row 0. A flat list of
    # (C, H, W) slices is likewise one sample's pack whenever the counts disagree.
    pixel_values = unwrap_collated(data["pixel_values"])
    if isinstance(pixel_values, (list | tuple)) and not _is_flat_slice_pack(pixel_values, batch_size):
        data["pixel_values"] = [sample_pixel_slices(sample) for sample in pixel_values]
    else:
        data["pixel_values"] = [sample_pixel_slices(pixel_values)]
    tgt_sizes = unwrap_collated(data["tgt_sizes"])
    if isinstance(tgt_sizes, (list | tuple)):
        data["tgt_sizes"] = [sample_tgt_sizes(sample, device=device) for sample in tgt_sizes]
    else:
        data["tgt_sizes"] = [sample_tgt_sizes(tgt_sizes, device=device)]
    # Mel features stack per clip; the lens must stay 1-D tensors for the tower's hstack.
    audio_features = normalize_audio_features(data.get("audio_features"))
    data["audio_features"] = audio_features
    if isinstance(audio_features, torch.Tensor):
        data["audio_feature_lens"] = batch_audio_feature_lens(data.get("audio_feature_lens"), device)
    else:
        # The empty list means no audio; an empty-features/nonnull-lens inconsistency
        # is laundered into "no audio" here — the dangerous direction is caught
        # fail-closed in _apply_media_bounds.
        data["audio_feature_lens"] = []


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
    # Per-example slice/clip counts for the patched towers' re-split (vision runs
    # per example; audio resets its streaming history per example); taken before the fold.
    data["packed_vision_slices"] = [len(sample) for sample in data["pixel_values"]]
    data["packed_audio_clips"] = [len(sample) for sample in data["audio_feature_lens"]]
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
