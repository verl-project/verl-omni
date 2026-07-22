# Copyright 2026 Gulp AI Inc and/or its affiliates
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
"""Teacher-forced codec-0 forward math for the Qwen3-TTS talker.

The talker is not a plain causal LM: its input is a 2-channel, position-co-located sequence with
an 8-slot control prefix (think tokens, a speaker embedding slot at codec position 6), and each
frame's codec-0 logits depend on the previous frame's full 16-codebook set. verl instead hands
the actor a flat input_ids and gathers logits with roll(input_ids, -1). This module bridges the
two: build_talker_batch rebuilds the talker's 2-channel batch, codec0_logits runs the talker, and
realign_to_verl scatters the codec-0 logit rows onto verl's flat response positions.

Layout per sample, with text length tl, codec length cl, total t = max(tl + cl) + 8:
  ch0 (text):  [0:3]=text[:3]  [3:7]=tts_pad  [7]=tts_bos  [8:8+tl-3]=text[3:]
               [8+tl-3]=tts_eos  [8+tl-2:8+tl+cl]=tts_pad
  ch1 (codec): [3]=nothink [4]=think_bos [5]=think_eos [6]=0(speaker slot) [7]=codec_pad
               [8:8+tl-2]=codec_pad  [8+tl-2]=codec_bos  [8+tl-1:8+tl-1+cl]=codec0  [.+cl]=codec_eos
  codec_0_labels: [8+tl-1 : 8+tl-1+cl]=codec0 ; [8+tl-1+cl]=codec_eos ; else -100

This module imports neither verl nor transformers, so the index math is unit-testable on CPU
(tests/models/test_qwen3_tts_forward_on_cpu.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NUM_SUB_CODEBOOKS = 15
NUM_CODEBOOKS = NUM_SUB_CODEBOOKS + 1
SPEAKER_SLOT = 6

# The prompt template ends with a second assistant header whose trailing tokens are not part of
# the teacher-forcing text channel; callers drop this many tokens from the tokenized prompt.
TEXT_PROMPT_TRAILER_TOKENS = 5


def build_assistant_text(text: str) -> str:
    """The talker's text prompt template. The rollout server and the actor forward must tokenize
    the same string so generation and the teacher-forced recompute see identical text channels."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def load_speaker_xvector(path: str) -> torch.Tensor:
    """Load a precomputed speaker x-vector (JSON float list) as a (1, D) float32 tensor.

    The same vector must feed the rollout (voice_clone_prompt) and the actor (speaker slot) so
    both condition on an identical speaker. See the example's precompute_spk_embed.py.
    """
    import json

    with open(path) as f:
        vec = json.load(f)
    return torch.tensor(vec, dtype=torch.float32).reshape(1, -1)


@dataclass
class TalkerTokens:
    """Control token ids for the talker layout, read from the model config."""

    tts_pad: int
    tts_bos: int
    tts_eos: int
    codec_pad: int
    codec_bos: int
    codec_eos: int
    codec_nothink: int
    codec_think_bos: int
    codec_think_eos: int

    @classmethod
    def from_config(cls, config) -> TalkerTokens:
        tcfg = config.talker_config
        return cls(
            tts_pad=int(config.tts_pad_token_id),
            tts_bos=int(config.tts_bos_token_id),
            tts_eos=int(config.tts_eos_token_id),
            codec_pad=int(tcfg.codec_pad_id),
            codec_bos=int(tcfg.codec_bos_id),
            codec_eos=int(tcfg.codec_eos_token_id),
            codec_nothink=int(tcfg.codec_nothink_id),
            codec_think_bos=int(tcfg.codec_think_bos_id),
            codec_think_eos=int(tcfg.codec_think_eos_id),
        )


@dataclass
class TalkerBatch:
    """Teacher-forcing batch plus per-sample offsets for realigning logits back to verl."""

    input_ids: torch.Tensor  # (B, t, 2), channel 0 text, channel 1 codec
    codec_ids: torch.Tensor  # (B, t, 16), column 0 is codec-0
    text_embedding_mask: torch.Tensor  # (B, t, 1)
    codec_embedding_mask: torch.Tensor  # (B, t, 1), False at the speaker slot
    codec_mask: torch.Tensor  # (B, t), the acoustic-frame span
    attention_mask: torch.Tensor  # (B, t)
    codec_0_labels: torch.Tensor  # (B, t), -100 outside the codec span
    text_lens: list[int]
    codec_lens: list[int]
    logit_start: list[int]  # 8 + tl - 2 per sample; logit at logit_start + k predicts codec-0 token k


def build_talker_batch(
    text_ids: list[torch.Tensor],
    audio_codes: list[torch.Tensor],
    tokens: TalkerTokens,
    *,
    device=None,
    sub_codebook_vocab: int | None = None,
) -> TalkerBatch:
    """Build the talker's 2-channel teacher-forcing batch (see the module docstring layout).

    text_ids are the tokenized build_assistant_text prompts with the trailing header dropped;
    audio_codes are per-sample (cl, 16) codebooks. sub_codebook_vocab, if given, clamps
    columns 1..15 to guard against device-side asserts on out-of-range sampled codes.
    """
    b = len(text_ids)
    tls = [int(t.reshape(-1).shape[0]) for t in text_ids]
    cls = [int(c.shape[0]) for c in audio_codes]
    t = max(tl + cl for tl, cl in zip(tls, cls, strict=True)) + 8

    input_ids = torch.zeros((b, t, 2), dtype=torch.long, device=device)
    codec_ids = torch.zeros((b, t, NUM_CODEBOOKS), dtype=torch.long, device=device)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    codec_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    attention_mask = torch.zeros((b, t), dtype=torch.long, device=device)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long, device=device)

    for i in range(b):
        tid = text_ids[i].reshape(-1).to(device=device, dtype=torch.long)
        codes = audio_codes[i].to(device=device, dtype=torch.long)
        if sub_codebook_vocab is not None:
            codes = codes.clone()
            codes[:, 1:NUM_CODEBOOKS] = codes[:, 1:NUM_CODEBOOKS].clamp_(0, sub_codebook_vocab - 1)
        codec0 = codes[:, 0]
        tl, cl = tls[i], cls[i]

        input_ids[i, :3, 0] = tid[:3]
        input_ids[i, 3:7, 0] = tokens.tts_pad
        input_ids[i, 7, 0] = tokens.tts_bos
        input_ids[i, 8 : 8 + tl - 3, 0] = tid[3:]
        input_ids[i, 8 + tl - 3, 0] = tokens.tts_eos
        input_ids[i, 8 + tl - 2 : 8 + tl + cl, 0] = tokens.tts_pad
        text_embedding_mask[i, : 8 + tl + cl] = True

        input_ids[i, 3:8, 1] = torch.tensor(
            [tokens.codec_nothink, tokens.codec_think_bos, tokens.codec_think_eos, 0, tokens.codec_pad],
            dtype=torch.long,
            device=device,
        )
        input_ids[i, 8 : 8 + tl - 3, 1] = tokens.codec_pad
        input_ids[i, 8 + tl - 3, 1] = tokens.codec_pad
        input_ids[i, 8 + tl - 2, 1] = tokens.codec_bos
        input_ids[i, 8 + tl - 1 : 8 + tl - 1 + cl, 1] = codec0
        input_ids[i, 8 + tl - 1 + cl, 1] = tokens.codec_eos

        codec_0_labels[i, 8 + tl - 1 : 8 + tl - 1 + cl] = codec0
        codec_0_labels[i, 8 + tl - 1 + cl] = tokens.codec_eos

        codec_ids[i, 8 + tl - 1 : 8 + tl - 1 + cl, :] = codes

        codec_embedding_mask[i, 3 : 8 + tl + cl] = True
        codec_embedding_mask[i, SPEAKER_SLOT] = False
        codec_mask[i, 8 + tl - 1 : 8 + tl - 1 + cl] = True
        attention_mask[i, : 8 + tl + cl] = True

    return TalkerBatch(
        input_ids=input_ids,
        codec_ids=codec_ids,
        text_embedding_mask=text_embedding_mask.unsqueeze(-1),
        codec_embedding_mask=codec_embedding_mask.unsqueeze(-1),
        codec_mask=codec_mask,
        attention_mask=attention_mask,
        codec_0_labels=codec_0_labels,
        text_lens=tls,
        codec_lens=cls,
        logit_start=[8 + tl - 2 for tl in tls],
    )


def assemble_talker_embeddings(talker, batch: TalkerBatch, speaker_emb: torch.Tensor) -> torch.Tensor:
    """Build the talker input embedding (B, t, H); speaker_emb (B, H) is injected at position 6."""
    ids = batch.input_ids
    # Generation runs text-side embeddings through talker.text_projection (a learned resize MLP,
    # not identity); apply it here too so the recompute matches the rollout.
    te_raw = talker.model.text_embedding(ids[:, :, 0])
    text_proj = getattr(talker, "text_projection", None)
    if text_proj is not None:
        te_raw = text_proj(te_raw)
    te = te_raw * batch.text_embedding_mask
    ce = talker.model.codec_embedding(ids[:, :, 1]) * batch.codec_embedding_mask
    ce = ce.clone()
    ce[:, SPEAKER_SLOT, :] = speaker_emb.to(ce.dtype)
    emb = te + ce
    sub_tables = talker.code_predictor.get_input_embeddings()
    cmask = batch.codec_mask.unsqueeze(-1)
    for i in range(1, NUM_CODEBOOKS):
        emb = emb + sub_tables[i - 1](batch.codec_ids[:, :, i]) * cmask
    return emb


def codec0_logits(talker, batch: TalkerBatch, speaker_emb: torch.Tensor) -> torch.Tensor:
    """Run the talker over the teacher-forcing window; logits[j] predicts the token at j + 1."""
    emb = assemble_talker_embeddings(talker, batch, speaker_emb)
    out = talker(
        inputs_embeds=emb[:, :-1, :],
        attention_mask=batch.attention_mask[:, :-1],
        use_cache=False,
        output_hidden_states=False,
    )
    return out.logits


def tts_actor_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tts_text_ids: torch.Tensor,
    tts_audio_codes: torch.Tensor,
    response_len: torch.Tensor,
    text_len: torch.Tensor,
    speaker_emb: torch.Tensor,
) -> torch.Tensor:
    """verl flat batch to codec-0 logits (B, T, vocab) realigned onto verl's positions.

    input_ids/attention_mask are verl's dense tensors (real region [0, L_i), response is the
    suffix of length response_len_i); tts_text_ids and tts_audio_codes are left-aligned
    per-sample inputs from the rollout.
    """
    talker = model.talker
    device = input_ids.device
    b = input_ids.shape[0]
    t_out = input_ids.shape[1]
    real_len = attention_mask.sum(dim=1).to(torch.long)

    text_ids_list, audio_codes_list, response_starts = [], [], []
    for i in range(b):
        rl, tl, li = int(response_len[i]), int(text_len[i]), int(real_len[i])
        text_ids_list.append(tts_text_ids[i, :tl].to(torch.long))
        audio_codes_list.append(tts_audio_codes[i, :rl].to(torch.long))
        response_starts.append(li - rl)

    tokens = TalkerTokens.from_config(model.config)
    sub_vocab = int(talker.code_predictor.get_input_embeddings()[0].num_embeddings)

    batch = build_talker_batch(text_ids_list, audio_codes_list, tokens, device=device, sub_codebook_vocab=sub_vocab)
    talker_logits = codec0_logits(talker, batch, speaker_emb)
    vocab = talker_logits.shape[-1]
    # verl gathers log-probs over the full flat sequence before slicing the response, so the
    # output must be wide enough for the text-prompt labels too (those rows are discarded).
    out_vocab = max(vocab, int(input_ids.max().item()) + 1)
    return realign_to_verl(talker_logits, batch, response_starts, (b, t_out, out_vocab))


def realign_to_verl(
    talker_logits: torch.Tensor,
    batch: TalkerBatch,
    response_starts: list[int],
    out_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Scatter codec-0 logit rows onto verl's flat (B, T, vocab) tensor.

    verl computes logp[p] from logits[p] and input_ids[p + 1]; response token k sits at flat
    position response_start + k and the talker's logit for it is at batch.logit_start + k, so
    each sample copies cl contiguous rows. Prompt and pad rows stay zero; verl drops them via
    the response mask.
    """
    b, T, out_vocab = out_shape
    codec_vocab = talker_logits.shape[-1]
    out = talker_logits.new_zeros((b, T, out_vocab))
    for i in range(b):
        cl = batch.codec_lens[i]
        ls = batch.logit_start[i]
        rs = response_starts[i]
        sl = slice(rs - 1, rs - 1 + cl)
        if out_vocab > codec_vocab:
            # Mask non-codec columns so the response-row log_softmax is over the codec vocab only.
            out[i, sl, codec_vocab:] = -1e4
        out[i, sl, :codec_vocab] = talker_logits[i, ls : ls + cl, :]
    return out
