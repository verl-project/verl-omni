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
"""MiniCPM-o talker (TTS) training adapter.

Trains the talker on speech codec tokens while the thinker stays frozen.
``prepare_model_inputs`` unpacks the per-sample replay payload (the thinker's
text sequence plus the codec frames); the wrapped forward teacher-forces the
codec frames and scatters the codec distributions onto the response positions
of verl's batch. The replay contract follows the Qwen3-TTS adapter: the
batch's response tokens are the codec frames, the prompt is a dummy token, and
the real prompt travels in the replay payload.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import OmniModelBase

__all__ = ["MiniCPMTalkerAdapter", "MiniCPMOWithTTS"]


# --- Teacher-forced talker forward (mirrors MiniCPMO._generate_speech_non_streaming) ---
# conditioning = emb_text(text[span]) + projector_semantic(hidden[span]) (optionally
# L2-normalized); input_embeds = [conditioning, text_eos, audio_bos, codec[:-1]];
# logits = head_code[q](backbone(input_embeds)), scattered back onto the batch's
# response positions so the standard logprob machinery needs no shifting.

MINICPM_TALKER_REPLAY_KEY = "minicpm_talker_replay"


@dataclass
class TalkerTokens:
    """The scalar talker constants the teacher-forced forward needs."""

    text_eos_token_id: int
    audio_bos_token_id: int
    num_vq: int
    num_audio_tokens: int
    hidden_size: int
    normalize_projected_hidden: bool

    @classmethod
    def from_config(cls, config) -> TalkerTokens:
        tts = config.tts_config
        return cls(
            int(tts.text_eos_token_id),
            int(tts.audio_bos_token_id),
            int(tts.num_vq),
            int(tts.num_audio_tokens),
            int(tts.hidden_size),
            bool(tts.normalize_projected_hidden),
        )


@dataclass
class TalkerBatch:
    """A padded, teacher-forced talker input batch.

    ``input_embeds`` rows are ``[conditioning, text_eos, audio_bos,
    codec_0..codec_{T-2}]``; position ``cond_len + 1 + j`` predicts codec
    token ``j``.
    """

    input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    cond_lens: list[int]
    codec_lens: list[int]


def build_conditioning(model, text_tokens: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """The ``hidden_text_merge`` conditioning (official math, batched).

    Args:
        model: The MiniCPMO module (uses ``tts.emb_text`` / ``tts.projector_semantic``).
        text_tokens: ``[batch, span]`` text ids the thinker produced.
        hidden_states: ``[batch, span, llm_hidden]`` thinker states at
            ``tts_proj_layer``, for the same span.
    """
    llm_embeds = model.tts.emb_text(text_tokens)
    hidden_embeds = model.tts.projector_semantic(hidden_states)
    if model.tts.config.normalize_projected_hidden:
        hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
    return llm_embeds + hidden_embeds


def build_talker_batch(
    model,
    text_ids: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    audio_codes: list[torch.Tensor],
    tokens: TalkerTokens,
) -> TalkerBatch:
    """Build the padded teacher-forced input for a per-sample batch.

    Args:
        model: The MiniCPMO module.
        text_ids: Per-sample response-span text ids ``[span_i]`` — the TTS
            region (the caller slices the full thinker sequence first).
        hidden_states: Per-sample thinker states for the same span
            ``[span_i, llm_hidden]`` at ``tts_proj_layer``.
        audio_codes: Per-sample codec tokens ``[codec_len_i]``.
        tokens: The scalar talker constants.
    """
    if len(text_ids) != len(hidden_states) or len(text_ids) != len(audio_codes):
        raise ValueError(
            "MiniCPM talker teacher forcing requires matching text / hidden / codec batches: "
            f"{len(text_ids)} / {len(hidden_states)} / {len(audio_codes)}."
        )
    if tokens.num_vq != 1:
        raise NotImplementedError(
            "MiniCPM-o 4.5 talker training is validated for the single-codebook "
            f"(num_vq=1) layout only, got num_vq={tokens.num_vq}."
        )
    device = next(model.tts.parameters()).device
    dtype = model.tts.emb_text.weight.dtype

    rows = []
    cond_lens = []
    codec_lens = []
    text_eos = model.tts.emb_text(torch.tensor([tokens.text_eos_token_id], device=device, dtype=torch.long))
    audio_bos = model.tts.emb_text(torch.tensor([tokens.audio_bos_token_id], device=device, dtype=torch.long))
    # The TTS region is the assistant's text response: everything after the
    # prompt. The rollout replays the full thinker sequence, so the span is
    # [prompt_len, text_len) — derived here from the replayed ids.
    for text, hidden, codes in zip(text_ids, hidden_states, audio_codes, strict=True):
        text = text.reshape(-1).to(device=device, dtype=torch.long)
        hidden = hidden.reshape(text.numel(), -1).to(device=device, dtype=dtype)
        codes = codes.reshape(-1).to(device=device, dtype=torch.long)
        if codes.numel() < 1:
            raise ValueError("MiniCPM talker replay rows must contain at least one codec frame.")
        if codes.min() < 0 or codes.max() >= tokens.num_audio_tokens:
            raise ValueError("MiniCPM talker codec ids are outside the codec vocabulary.")
        conditioning = build_conditioning(model, text.unsqueeze(0), hidden.unsqueeze(0))[0]
        codec_embeds = model.tts.emb_code[0](codes)
        rows.append(torch.cat([conditioning, text_eos, audio_bos, codec_embeds[:-1]], dim=0))
        cond_lens.append(int(conditioning.shape[0]))
        codec_lens.append(int(codes.numel()))

    seq_len = max(row.shape[0] for row in rows)
    input_embeds = rows[0].new_zeros((len(rows), seq_len, tokens.hidden_size))
    attention_mask = torch.zeros((len(rows), seq_len), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        input_embeds[index, : row.shape[0]] = row
        attention_mask[index, : row.shape[0]] = 1
    return TalkerBatch(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cond_lens=cond_lens,
        codec_lens=codec_lens,
    )


def talker_logits(model, batch: TalkerBatch) -> torch.Tensor:
    """Backbone forward + the per-codebook ``head_code`` heads.

    Returns ``[batch, seq, num_audio_tokens]`` for the validated
    single-codebook layout.
    """
    position_ids = torch.arange(batch.input_embeds.shape[1], device=batch.input_embeds.device)
    position_ids = position_ids.long().unsqueeze(0).expand(batch.input_embeds.shape[0], -1)
    outputs = model.tts.model(
        inputs_embeds=batch.input_embeds,
        attention_mask=batch.attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden = outputs.last_hidden_state
    # The backbone is a bare LlamaModel; the weight-norm heads produce the
    # logits, exactly like tts.generate()'s decode loop.
    return model.tts.head_code[0](hidden)


def minicpm_talker_actor_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    text_ids: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    audio_codes: list[torch.Tensor],
) -> torch.Tensor:
    """Teacher-forced talker logits aligned to the actor's response positions.

    Args:
        model: The MiniCPMO module.
        input_ids: verl's padded batch tokens — a dummy prompt followed by the
            codec frames as the response.
        attention_mask: The batch attention mask over *input_ids*.
        text_ids: Per-sample response-span text ids (the TTS region).
        hidden_states: Per-sample thinker states for the same span.
        audio_codes: Per-sample codec tokens.

    Returns:
        ``[batch, output_len, num_audio_tokens]`` — zeros outside the response
        region, the codec-frame distributions inside it.
    """
    tokens = TalkerTokens.from_config(model.config)
    batch = build_talker_batch(model, text_ids, hidden_states, audio_codes, tokens)
    logits = talker_logits(model, batch)

    batch_size, output_len = input_ids.shape
    if logits.shape[2] != tokens.num_audio_tokens:
        raise RuntimeError(f"Talker head produced vocab {logits.shape[2]}, expected {tokens.num_audio_tokens}.")
    aligned = logits.new_zeros((batch_size, output_len, tokens.num_audio_tokens))
    import logging

    for index in range(batch_size):
        response_len = int(batch.codec_lens[index])
        # The codec frames are the trailing valid tokens of the padded batch.
        valid_positions = attention_mask[index].nonzero(as_tuple=True)[0]
        response_start = int(valid_positions[-response_len]) if len(valid_positions) >= response_len else -1
        if response_start < 1 or response_start + response_len > output_len:
            response_start = max(1, output_len - response_len)
        policy_ids = input_ids[index, response_start : response_start + response_len].long()
        codec_ids = audio_codes[index].reshape(-1).to(input_ids.device, dtype=torch.long)
        if not torch.equal(policy_ids, codec_ids):
            logging.getLogger(__name__).warning(
                "MiniCPM talker codec trajectory mismatch at sample=%d (engine pipeline owns alignment; "
                "continuing with right-aligned positions)",
                index,
            )
        # audio_bos at position cond_len + 1 predicts codec_0; position
        # cond_len + 1 + j predicts codec_j.
        source = batch.cond_lens[index] + 1
        aligned[index, response_start : response_start + response_len] = logits[index, source : source + response_len]
    return aligned


class MiniCPMOWithTTS:
    """``auto_model_class`` loader: the thinker loader with TTS weights kept.

    Identical to the thinker adapter's ``MiniCPMO`` loader except
    ``config.init_tts`` stays ``True`` so the talker weights load.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
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
            config.init_tts = True
            config.use_cache = False
            config.stream_input = False
            # Transformers 5.x strips keys with no default from sub-configs
            # (top_p / top_k / repetition_penalty / tts_proj_layer); the remote
            # MiniCPMTTS.__init__ reads them unconditionally, so restore them.
            tts = getattr(config, "tts_config", None)
            if tts is not None:
                for key, value in (
                    ("top_p", 0.85),
                    ("top_k", 25),
                    ("repetition_penalty", 1.05),
                    ("tts_proj_layer", -1),
                ):
                    if not hasattr(tts, key):
                        setattr(tts, key, value)
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


def _minicpm_talker_forward(
    self,
    tts_policy_ids=None,
    tts_policy_attention_mask=None,
    attention_mask=None,
    data=None,
    tts_text_ids=None,
    tts_text_lens=None,
    tts_response_starts=None,
    tts_response_ends=None,
    tts_audio_codes=None,
    tts_codec_lens=None,
    **llm_kwargs,
):
    """Two-stage forward: thinker (frozen, no grad) → talker (trainable)."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    required = (
        tts_policy_ids,
        attention_mask,
        data,
        tts_text_ids,
        tts_text_lens,
        tts_response_starts,
        tts_response_ends,
        tts_audio_codes,
        tts_codec_lens,
    )
    missing = [
        name
        for name, value in zip(
            (
                "tts_policy_ids",
                "attention_mask",
                "data",
                "tts_text_ids",
                "tts_text_lens",
                "tts_response_starts",
                "tts_response_ends",
                "tts_audio_codes",
                "tts_codec_lens",
            ),
            required,
            strict=True,
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(f"MiniCPM talker forward is missing fields: {missing}. present={sorted(llm_kwargs)}")

    # Stage 1: frozen thinker forward over the text sequence. The captured
    # class-level forward is called as a plain function; drop the embed-style
    # inputs the engine passes explicitly (we always need output_hidden_states).
    tts_proj_layer = int(self.config.tts_config.tts_proj_layer)
    for key in ("input_ids", "position_ids", "inputs_embeds", "output_hidden_states"):
        llm_kwargs.pop(key, None)
    with torch.no_grad():
        thinker_outputs = _TALKER_STATE["original_forward"](self, data=data, output_hidden_states=True, **llm_kwargs)
        hidden_states = thinker_outputs.hidden_states[tts_proj_layer]  # [b, text_len, llm_hidden]

    # Stage 2: per-sample response spans and the teacher-forced talker.
    batch_size = tts_policy_ids.shape[0]
    text_rows, hidden_rows, code_rows = [], [], []
    for index in range(batch_size):
        start = int(tts_response_starts[index])
        end = int(tts_response_ends[index])
        codec_len = int(tts_codec_lens[index])
        if not 0 <= start < end <= int(tts_text_lens[index]):
            raise ValueError(f"MiniCPM talker replay span is invalid for sample {index}: [{start}, {end}).")
        text_rows.append(tts_text_ids[index, start:end])
        hidden_rows.append(hidden_states[index, start:end, :])
        # Slice the padding off: the padded row would teacher-force zeros
        # beyond the sample's true codec trajectory.
        code_rows.append(tts_audio_codes[index, :codec_len])

    logits = minicpm_talker_actor_logits(
        self,
        input_ids=tts_policy_ids,
        attention_mask=tts_policy_attention_mask if tts_policy_attention_mask is not None else attention_mask,
        text_ids=text_rows,
        hidden_states=hidden_rows,
        audio_codes=code_rows,
    )
    return CausalLMOutputWithPast(logits=logits, past_key_values=None)


# The remote forward, captured at configure time (the class-level attribute
# survives the instance-level rebinding verl calls).
_TALKER_STATE: dict = {"original_forward": None}


def _get_input_embeddings(self):
    return self.tts.emb_code[0]


def _set_input_embeddings(self, value):
    self.tts.emb_code[0] = value


@OmniModelBase.register("MiniCPMO", stage="talker")
class MiniCPMTalkerAdapter(OmniModelBase):
    """MiniCPM-o 4.5 talker-stage adapter: train the TTS AR policy."""

    auto_model_class = MiniCPMOWithTTS

    # Module stripping and freezing

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Keep the TTS stage (unlike the thinker adapter, which strips it).

        Only the code2wav renderer is removed, at configure time — it is not
        needed for gradient computation and would waste GPU memory.
        """
        return []

    @classmethod
    def get_fsdp_ignored_module_names(cls, model_config) -> list[str]:
        """FSDP ignores the frozen thinker and encoder towers.

        The trainable talker (``tts.*``) is NOT ignored — it is the FSDP unit
        set that receives gradients and optimizer states.
        """
        return ["llm", "vpm", "apm", "resampler"]

    @classmethod
    def configure_model(cls, module, model_config):
        """Load the full model with TTS, freeze the thinker, wrap the forward."""
        from verl_omni.pipelines.minicpm.thinker_training_adapter import (
            _apply_remote_code_patches,
            _keep_rotary_frequencies_dtype,
        )

        if getattr(model_config, "use_remove_padding", False):
            raise ValueError("MiniCPM-o talker training requires actor_rollout_ref.model.use_remove_padding=false.")

        # Base load (no stage stripping); remote-code patches already applied by the loader.
        module = super().configure_model(module, model_config)

        # Strip the code2wav renderer (audio_tokenizer): synthesis-only.
        if hasattr(module, "tts") and getattr(module.tts, "audio_tokenizer", None) is not None:
            module.tts.audio_tokenizer = None
            eval_logger.info("Stripped tts.audio_tokenizer (code2wav) for talker training")

        # Freeze the thinker (LLM backbone + vision/audio encoders + resampler).
        trainable = 0
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name.startswith("tts."))
            trainable += int(parameter.requires_grad)
        eval_logger.info("MiniCPM-o talker adapter enabled {} trainable parameter tensors", trainable)

        # Whole-model dtype casts must not quantize the non-persistent RoPE frequencies.
        _keep_rotary_frequencies_dtype(module)
        # The thinker forward inside the talker stage hits the same remote-code
        # deviations the thinker adapter patches (batched media kernels etc.).
        _apply_remote_code_patches(module)

        # verl calls module(**kwargs); install the two-stage talker forward.
        _TALKER_STATE["original_forward"] = module.__class__.forward
        module.forward = types.MethodType(_minicpm_talker_forward, module)
        # The talker's policy vocabulary is the codec table.
        module.get_input_embeddings = types.MethodType(_get_input_embeddings, module)
        module.set_input_embeddings = types.MethodType(_set_input_embeddings, module)
        # Wrap the talker's decoder layers, so each becomes its own FSDP unit.
        module._no_split_modules = ["LlamaDecoderLayer"]
        return module

    # Processor / tokenizer: shared with the thinker stage

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Same processor configuration as the thinker stage (prompt parity)."""
        from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter

        return MiniCPMThinkerAdapter.configure_processor(model_path, model_config)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Same tokenizer configuration as the thinker stage (answer-tag parity)."""
        from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter

        return MiniCPMThinkerAdapter.configure_tokenizer(model_path, model_config)

    # Input preparation

    @classmethod
    def prepare_model_inputs(cls, model_inputs: dict[str, Any], micro_batch, model_config) -> dict[str, Any]:
        """Unpack the replay payload and rebuild the thinker input.

        The verl batch carries the codec frames as the response tokens plus
        the media tensors; the replay payload carries the thinker's full text
        sequence and the response start within it. The thinker input is rebuilt
        from the replay text (with the batch's media), so the frozen forward
        sees exactly the rollout's context.
        """
        from verl_omni.pipelines.minicpm.thinker_training_adapter import (
            _apply_media_bounds,
            split_minicpm_forward_kwargs,
        )

        model_inputs = dict(model_inputs)
        # rmpad flattening invalidates the processor's per-sample bounds coordinates;
        # the talker stage requires use_remove_padding=false anyway.
        model_inputs.pop("image_bound", None)
        model_inputs.pop("audio_bounds", None)
        model_inputs.pop("image_sizes", None)

        # Replay payload.
        sample_extra_fields = tu.get(micro_batch, "extra_fields")
        if sample_extra_fields is None:
            raise RuntimeError(
                f"MiniCPM-o talker actor inputs require the {MINICPM_TALKER_REPLAY_KEY!r} replay payload."
            )
        if not isinstance(sample_extra_fields, list):
            sample_extra_fields = list(sample_extra_fields)
        if len(sample_extra_fields) != model_inputs["input_ids"].shape[0]:
            raise RuntimeError("MiniCPM-o talker extra_fields do not match the actor batch size.")
        if any(not isinstance(item, dict) or MINICPM_TALKER_REPLAY_KEY not in item for item in sample_extra_fields):
            raise RuntimeError(
                f"MiniCPM-o talker actor inputs require the {MINICPM_TALKER_REPLAY_KEY!r} replay payload."
            )
        fields = [item[MINICPM_TALKER_REPLAY_KEY] for item in sample_extra_fields]
        if any("text_ids" not in item or "audio_codes" not in item or "response_start" not in item for item in fields):
            raise RuntimeError(
                "MiniCPM-o talker replay payloads must contain text_ids, response_start and audio_codes."
            )

        device = model_inputs["input_ids"].device
        texts = [torch.as_tensor(item["text_ids"], dtype=torch.long).reshape(-1) for item in fields]
        codes = [torch.as_tensor(item["audio_codes"], dtype=torch.long).reshape(-1) for item in fields]
        starts = [int(item["response_start"]) for item in fields]
        ends = [int(item.get("response_end", t.numel())) for item, t in zip(fields, texts, strict=True)]
        if any(item.numel() < 1 for item in texts):
            raise ValueError("MiniCPM-o talker replay text sequences must be non-empty.")
        if any(item.numel() < 1 for item in codes):
            raise ValueError("MiniCPM-o talker replay codec sequences must be non-empty.")
        if any(not 0 <= s < e <= t.numel() for s, e, t in zip(starts, ends, texts, strict=True)):
            raise ValueError("MiniCPM-o talker replay response span is outside its text sequence.")
        if any(c.min() < 0 or c.max() >= model_config.hf_config.tts_config.num_audio_tokens for c in codes):
            raise ValueError("MiniCPM-o talker replay codec ids are outside the codec vocabulary.")

        # Verify the batch response tokens are the codec frames.
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        for index, code in enumerate(codes):
            response_len = int(code.numel())
            # The codec frames are the trailing valid tokens; derive their start
            # from the mask's valid positions so any padding layout works.
            valid_positions = attention_mask[index].nonzero(as_tuple=True)[0]
            response_start = int(valid_positions[-response_len]) if len(valid_positions) >= response_len else -1
            if response_start < 1 or response_start + response_len > input_ids.shape[1]:
                raise ValueError(f"Invalid MiniCPM-o talker response alignment for sample {index}.")
            if not torch.equal(
                input_ids[index, response_start : response_start + response_len].long(), code.to(input_ids.device)
            ):
                raise RuntimeError(
                    f"MiniCPM-o talker codec trajectory is not aligned with actor labels: sample={index}."
                )

        # Keep the engine's mixed policy ids ([dummy prompt, codec frames])
        # and attention mask for the actor-side alignment check; the text
        # sequence and its mask replace what the thinker forward consumes.
        policy_ids = model_inputs.get("input_ids")
        policy_attention_mask = model_inputs.get("attention_mask")

        # Rebuild the thinker input from the replay text.
        batch_size = len(texts)
        max_text_len = max(int(t.numel()) for t in texts)
        max_code_len = max(int(c.numel()) for c in codes)
        text_buffer = torch.zeros((batch_size, max_text_len), dtype=torch.long, device=device)
        text_mask = torch.zeros((batch_size, max_text_len), dtype=torch.long, device=device)
        code_buffer = torch.zeros((batch_size, max_code_len), dtype=torch.long, device=device)
        text_lens = torch.empty(batch_size, dtype=torch.long, device=device)
        codec_lens = torch.empty(batch_size, dtype=torch.long, device=device)
        for index, (text, code) in enumerate(zip(texts, codes, strict=True)):
            text_buffer[index, : text.numel()] = text.to(device)
            text_mask[index, : text.numel()] = 1
            code_buffer[index, : code.numel()] = code.to(device)
            text_lens[index] = text.numel()
            codec_lens[index] = code.numel()

        # The thinker consumes the TEXT sequence; replace the codec-mixed ids.
        model_inputs["input_ids"] = text_buffer
        model_inputs["attention_mask"] = text_mask
        model_inputs["position_ids"] = (
            torch.arange(max_text_len, device=device).long().unsqueeze(0).expand(batch_size, -1)
        )
        data, llm_kwargs = split_minicpm_forward_kwargs(model_inputs)
        _apply_media_bounds(data, model_config)

        return {
            "data": data,
            "tts_policy_ids": policy_ids,
            "tts_policy_attention_mask": policy_attention_mask,
            "tts_text_ids": text_buffer,
            "tts_text_lens": text_lens,
            "tts_response_starts": torch.tensor(starts, dtype=torch.long, device=device),
            "tts_response_ends": torch.tensor(ends, dtype=torch.long, device=device),
            "tts_audio_codes": code_buffer,
            "tts_codec_lens": codec_lens,
            **llm_kwargs,
        }
