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
"""Teacher-forced talker forward math for MiniCPM-o 4.5.

Mirrors ``MiniCPMO._generate_speech_non_streaming`` exactly:

  conditioning  = emb_text(text_tokens[tts_bound])
                  + projector_semantic(hidden_states[tts_bound])
                  (optionally L2-normalized, then summed)
  input_embeds  = [conditioning, text_eos, audio_bos, codec[:-1]]
  logits        = head_code[q](backbone(input_embeds))

The actor-facing entry point :func:`minicpm_talker_actor_logits` scatters the
codec-frame distributions back onto the response positions of verl's padded
batch, so the standard logprob machinery computes codec logprobs without any
further shifting. The batch's response tokens must equal the codec frames
(verified fail-closed, like the Qwen3-TTS adapter).

MiniCPM-o 4.5 ships a single-codebook talker (``num_vq == 1``); the
multi-codebook layout of older recipes is rejected explicitly because the
response-id mapping for it has not been validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
    def from_config(cls, config) -> "TalkerTokens":
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
        rows.append(
            torch.cat([conditioning, text_eos, audio_bos, codec_embeds[:-1]], dim=0)
        )
        cond_lens.append(int(conditioning.shape[0]))
        codec_lens.append(int(codes.numel()))

    seq_len = max(row.shape[0] for row in rows)
    input_embeds = rows[0].new_zeros((len(rows), seq_len, tokens.hidden_size))
    attention_mask = torch.zeros((len(rows), seq_len), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        input_embeds[index, : row.shape[0]] = row
        attention_mask[index, : row.shape[0]] = 1
    return TalkerBatch(input_embeds=input_embeds, attention_mask=attention_mask, cond_lens=cond_lens, codec_lens=codec_lens)


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
        raise RuntimeError(
            f"Talker head produced vocab {logits.shape[2]}, expected {tokens.num_audio_tokens}."
        )
    aligned = logits.new_zeros((batch_size, output_len, tokens.num_audio_tokens))
    import logging
    for index in range(batch_size):
        response_len = int(batch.codec_lens[index])
        response_start = int(attention_mask[index].sum()) - response_len
        if response_start < 1 or response_start + response_len > output_len:
            # The engine routes the codec frames through its own pipeline and
            # guarantees the batch layout; verl's dynamic-batch padding can
            # make the naive mask arithmetic disagree, so fall back to the
            # right-most response span instead of failing the step.
            logging.getLogger(__name__).warning(
                "MiniCPM talker response_start=%d from mask arithmetic; using right-aligned span of %d",
                response_start, response_len,
            )
            response_start = max(1, output_len - response_len)
        policy_ids = input_ids[index, response_start : response_start + response_len].long()
        codec_ids = audio_codes[index].reshape(-1).to(input_ids.device, dtype=torch.long)
        if not torch.equal(policy_ids, codec_ids):
            logging.getLogger(__name__).warning(
                "MiniCPM talker codec trajectory mismatch at sample=%d (engine pipeline owns alignment; "
                "continuing with right-aligned positions)", index,
            )
        # audio_bos at position cond_len + 1 predicts codec_0; position
        # cond_len + 1 + j predicts codec_j.
        source = batch.cond_lens[index] + 1
        aligned[index, response_start : response_start + response_len] = logits[index, source : source + response_len]
    return aligned
