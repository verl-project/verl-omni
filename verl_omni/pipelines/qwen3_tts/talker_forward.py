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
"""Teacher-forced codec-0 forward math for Qwen3-TTS."""

import json
from dataclasses import dataclass

import torch

NUM_CODEBOOKS = 16
TEXT_PROMPT_TRAILER_TOKENS = 5


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def load_speaker_xvector(path: str) -> torch.Tensor:
    with open(path) as file:
        vector = json.load(file)
    return torch.tensor(vector, dtype=torch.float32).reshape(1, -1)


@dataclass
class TalkerTokens:
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
    def from_config(cls, config):
        talker = config.talker_config
        return cls(
            int(config.tts_pad_token_id),
            int(config.tts_bos_token_id),
            int(config.tts_eos_token_id),
            int(talker.codec_pad_id),
            int(talker.codec_bos_id),
            int(talker.codec_eos_token_id),
            int(talker.codec_nothink_id),
            int(talker.codec_think_bos_id),
            int(talker.codec_think_eos_id),
        )


@dataclass
class TalkerBatch:
    input_ids: torch.Tensor
    codec_ids: torch.Tensor
    text_embedding_mask: torch.Tensor
    codec_embedding_mask: torch.Tensor
    codec_mask: torch.Tensor
    attention_mask: torch.Tensor
    codec_lens: list[int]
    logit_start: list[int]
    speaker_slots: list[int]


def build_talker_batch(
    text_ids,
    audio_codes,
    tokens,
    *,
    sub_codebook_vocab: int,
    device=None,
) -> TalkerBatch:
    if not text_ids or len(text_ids) != len(audio_codes):
        raise ValueError("Qwen3-TTS teacher forcing requires matching non-empty text and codec batches.")
    if sub_codebook_vocab <= 0:
        raise ValueError(f"sub_codebook_vocab must be positive, got {sub_codebook_vocab}.")
    if any(ids.reshape(-1).numel() < 3 for ids in text_ids):
        raise ValueError("Qwen3-TTS teacher-forcing text sequences must contain at least three prefix tokens.")
    if any(codes.ndim != 2 or codes.shape[-1] != NUM_CODEBOOKS for codes in audio_codes):
        raise ValueError(f"Qwen3-TTS codec codes must have shape (frames, {NUM_CODEBOOKS}).")
    text_lens = [int(ids.reshape(-1).shape[0]) for ids in text_ids]
    codec_lens = [int(codes.shape[0]) for codes in audio_codes]
    speaker_slot = 6
    text_start = 8
    seq_len = max(t + c for t, c in zip(text_lens, codec_lens, strict=True)) + 8
    batch_size = len(text_ids)
    input_ids = torch.zeros((batch_size, seq_len, 2), dtype=torch.long, device=device)
    codec_ids = torch.zeros((batch_size, seq_len, NUM_CODEBOOKS), dtype=torch.long, device=device)
    text_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    codec_embedding_mask = torch.zeros_like(text_mask)
    codec_mask = torch.zeros_like(text_mask)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    for index, (sample_text, sample_codes) in enumerate(zip(text_ids, audio_codes, strict=True)):
        ids = sample_text.reshape(-1).to(device=device, dtype=torch.long)
        codes = sample_codes.to(device=device, dtype=torch.long)
        if torch.any((codes[:, 1:] < 0) | (codes[:, 1:] >= sub_codebook_vocab)):
            raise ValueError("Qwen3-TTS residual codec IDs are outside the code-predictor vocabulary.")
        text_len, codec_len = text_lens[index], codec_lens[index]

        input_ids[index, :3, 0] = ids[:3]
        input_ids[index, 3:speaker_slot, 0] = tokens.tts_pad
        input_ids[index, speaker_slot, 0] = tokens.tts_pad
        input_ids[index, speaker_slot + 1, 0] = tokens.tts_bos
        input_ids[index, text_start : text_start + text_len - 3, 0] = ids[3:]
        input_ids[index, text_start + text_len - 3, 0] = tokens.tts_eos
        input_ids[index, text_start + text_len - 2 : text_start + text_len + codec_len, 0] = tokens.tts_pad
        text_mask[index, : text_start + text_len + codec_len] = True

        input_ids[index, 3:speaker_slot, 1] = torch.tensor(
            [tokens.codec_nothink, tokens.codec_think_bos, tokens.codec_think_eos], device=device
        )
        input_ids[index, speaker_slot + 1, 1] = tokens.codec_pad
        input_ids[index, text_start : text_start + text_len - 2, 1] = tokens.codec_pad
        input_ids[index, text_start + text_len - 2, 1] = tokens.codec_bos
        start = text_start + text_len - 1
        input_ids[index, start : start + codec_len, 1] = codes[:, 0]
        input_ids[index, start + codec_len, 1] = tokens.codec_eos
        codec_ids[index, start : start + codec_len] = codes
        codec_embedding_mask[index, 3 : text_start + text_len + codec_len] = True
        codec_embedding_mask[index, speaker_slot] = False
        codec_mask[index, start : start + codec_len] = True
        attention_mask[index, : text_start + text_len + codec_len] = True

    return TalkerBatch(
        input_ids,
        codec_ids,
        text_mask.unsqueeze(-1),
        codec_embedding_mask.unsqueeze(-1),
        codec_mask,
        attention_mask,
        codec_lens,
        [text_start + text_len - 2 for text_len in text_lens],
        [speaker_slot] * batch_size,
    )


def require_auto_language(language) -> str:
    """Limit RL training to the prompt layout validated by the actor forward."""
    normalized = str(language).strip()
    if normalized.lower() != "auto":
        raise ValueError(
            "Qwen3-TTS RL currently supports only tts_language=Auto; "
            "language-specific codec prefixes require a separately validated actor forward."
        )
    return "Auto"


def codec0_input_embeddings(talker, batch: TalkerBatch, speaker_embedding: torch.Tensor) -> torch.Tensor:
    ids = batch.input_ids
    text_embeddings = talker.text_projection(talker.model.text_embedding(ids[:, :, 0]))
    codec_embeddings = talker.model.codec_embedding(ids[:, :, 1]) * batch.codec_embedding_mask
    codec_embeddings = codec_embeddings.clone()
    sample_indices = torch.arange(codec_embeddings.shape[0], device=codec_embeddings.device)
    codec_embeddings[sample_indices, batch.speaker_slots] = speaker_embedding.to(codec_embeddings.dtype)
    embeddings = text_embeddings * batch.text_embedding_mask + codec_embeddings
    sub_embeddings = talker.code_predictor.get_input_embeddings()
    codec_mask = batch.codec_mask.unsqueeze(-1)
    for codebook in range(1, NUM_CODEBOOKS):
        embeddings += sub_embeddings[codebook - 1](batch.codec_ids[:, :, codebook]) * codec_mask
    return embeddings


def codec0_logits(talker, batch: TalkerBatch, speaker_embedding: torch.Tensor) -> torch.Tensor:
    embeddings = codec0_input_embeddings(talker, batch, speaker_embedding)
    output = talker(
        inputs_embeds=embeddings[:, :-1],
        attention_mask=batch.attention_mask[:, :-1],
        use_cache=False,
        output_hidden_states=False,
    )
    return output.logits


def mask_codec0_logits(logits: torch.Tensor, codebook_vocab: int, codec_eos_token_id: int) -> torch.Tensor:
    """Match the codec-token vocabulary exposed by the rollout model."""
    if not 1 < codebook_vocab <= logits.shape[-1]:
        raise ValueError(f"codebook_vocab must be in [2, {logits.shape[-1]}], got {codebook_vocab}.")
    if not 0 <= codec_eos_token_id < logits.shape[-1]:
        raise ValueError(f"codec_eos_token_id must be in [0, {logits.shape[-1]}), got {codec_eos_token_id}.")
    valid = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    valid[1:codebook_vocab] = True
    valid[codec_eos_token_id] = True
    return logits.masked_fill(~valid, -1e4)


def tts_actor_logits(
    model,
    input_ids,
    attention_mask,
    tts_text_ids,
    tts_audio_codes,
    response_len,
    text_len,
    speaker_embedding,
) -> torch.Tensor:
    batch_size, output_len = input_ids.shape
    texts, codes, response_starts = [], [], []
    for index in range(batch_size):
        response_size, text_size = int(response_len[index]), int(text_len[index])
        response_start = int(attention_mask[index].sum()) - response_size
        if response_start < 1 or response_start + response_size > output_len:
            raise ValueError("Invalid Qwen3-TTS response alignment.")
        policy_ids = input_ids[index, response_start : response_start + response_size].long()
        codec0_ids = tts_audio_codes[index, :response_size, 0].long()
        if not torch.equal(policy_ids, codec0_ids):
            mismatch = int(torch.nonzero(policy_ids != codec0_ids, as_tuple=False)[0])
            raise RuntimeError(
                f"Qwen3-TTS codec trajectory is not aligned with actor labels: sample={index}, frame={mismatch}."
            )
        texts.append(tts_text_ids[index, :text_size].long())
        codes.append(tts_audio_codes[index, :response_size].long())
        response_starts.append(response_start)

    talker = model.talker
    sub_vocab = int(talker.code_predictor.get_input_embeddings()[0].num_embeddings)
    tokens = TalkerTokens.from_config(model.config)
    batch = build_talker_batch(
        texts,
        codes,
        tokens,
        device=input_ids.device,
        sub_codebook_vocab=sub_vocab,
    )
    logits = mask_codec0_logits(
        codec0_logits(talker, batch, speaker_embedding),
        sub_vocab,
        int(model.config.talker_config.codec_eos_token_id),
    )
    if int(input_ids.min()) < 0 or int(input_ids.max()) >= logits.shape[-1]:
        raise ValueError("Qwen3-TTS actor token IDs are outside the codec-0 logit vocabulary.")
    output_vocab = logits.shape[-1]
    aligned = logits.new_zeros((batch_size, output_len, output_vocab))
    for index, response_start in enumerate(response_starts):
        codec_len = batch.codec_lens[index]
        target = slice(response_start - 1, response_start - 1 + codec_len)
        source = batch.logit_start[index]
        aligned[index, target] = logits[index, source : source + codec_len]
    return aligned
