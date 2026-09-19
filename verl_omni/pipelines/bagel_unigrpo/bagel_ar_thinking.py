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
"""Trainside AR "thinking" decode + log-prob replay on BAGEL's understanding pathway.

Ported from UniRL ``models/bagel/ar.py`` (``BagelARStep``/``BagelARStage``) onto
verl-omni's ``BagelForSFT`` text pathway. No vLLM: generation runs directly on the
live module.

Decode uses an incremental KV cache (:class:`_TextKVCache`) so sampling is O(T)
rather than the O(T^2) prefix-recompute, which makes ``max_new_tokens`` in the
thousands feasible. The cache drives token *sampling* only; the returned per-token
log-probs are recomputed with a single full-sequence teacher-forced forward
(:func:`replay_thinking_logprobs`), so the recorded rollout log-prob is bit-identical
to the training replay and the on-policy importance ratio is exactly 1 at bs=1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from verl.utils.device import get_device_id, get_device_name, is_cuda_available

from ..bagel_flow_grpo.bagel_model import _apply_rotary_emb

if TYPE_CHECKING:
    from ..bagel_flow_grpo.bagel_sft_model import BagelForSFT


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per row; returns ``(token_id [B], log_prob [B])``.

    log_prob is taken from the temperature-scaled distribution (matches UniRL
    ``BagelARStep.step``); top_k/top_p only restrict the sampling support.
    """
    if logits.dim() != 2:
        raise ValueError(f"sample_next_token: expected [B, vocab], got {tuple(logits.shape)}")
    if temperature <= 0.0:
        logp_full = F.log_softmax(logits.float(), dim=-1)
        token_id = logp_full.argmax(dim=-1)
        return token_id, logp_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)

    scaled = logits.float() / temperature
    logp_full = F.log_softmax(scaled, dim=-1)
    filtered = scaled
    if 0 < top_k < filtered.shape[-1]:
        kth = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = torch.where(filtered < kth, torch.full_like(filtered, float("-inf")), filtered)
    if top_p < 1.0:
        sorted_vals, sorted_idx = torch.sort(filtered, dim=-1, descending=True)
        cumprob = torch.softmax(sorted_vals, dim=-1).cumsum(dim=-1)
        cutoff = (cumprob > top_p).float()
        cutoff = torch.cat([torch.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], dim=-1)
        sorted_vals = sorted_vals.masked_fill(cutoff > 0, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(-1, sorted_idx, sorted_vals)
    probs = F.softmax(filtered, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return token_id, logp_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)


def _text_logits(model: BagelForSFT, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Causal text-only forward through the understanding pathway → logits ``[B, L, V]``."""
    from ..bagel_flow_grpo.bagel_sft_model import _segment_attention_mask

    batch_size, seq_len = input_ids.shape
    sequence = model.embed_tokens(input_ids)
    valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
    latent_mask = torch.zeros_like(valid_mask)
    attention_mask = _segment_attention_mask(valid_mask, [(0, seq_len, "causal")])
    hidden = model._run_sft_sequence(
        sequence,
        position_ids=position_ids,
        text_mask=valid_mask,
        latent_mask=latent_mask,
        valid_mask=valid_mask,
        attention_mask=attention_mask,
    )
    return model.lm_head(hidden)


class _TextKVCache:
    """Incremental KV cache for BAGEL's understanding (text) attention pathway.

    Reproduces the ``BagelForSFT`` text forward one token at a time: per layer it
    caches the post-QK-norm, post-RoPE keys and values (``num_kv_heads``, bf16) so a
    decode step forwards only the new token against the cached prefix — O(T) instead
    of re-running the whole prefix each step. It is a *sampling* helper only; recorded
    log-probs come from the full-sequence :func:`replay_thinking_logprobs`.

    Only the text projections (``q_proj``/``k_proj``/``v_proj``, ``q_norm``/``k_norm``,
    ``o_proj``, ``mlp``) are exercised — a thinking chain routes every token through the
    understanding experts, so the MoT latent path is never touched.
    """

    def __init__(self, model: BagelForSFT) -> None:
        self._model = model
        self._layers = model.layers
        self._key: list[torch.Tensor | None] = [None] * len(self._layers)
        self._value: list[torch.Tensor | None] = [None] * len(self._layers)
        self.length = 0

    def _layer_step(self, layer, hidden: torch.Tensor, position_ids: torch.Tensor, index: int, *, causal: bool):
        attn = layer.self_attn
        cos, sin = layer.rotary_emb(position_ids)
        normalized = layer.input_layernorm(hidden)
        batch, span, _ = normalized.shape
        query = attn.q_proj(normalized).view(batch, span, attn.num_heads, attn.head_dim).float()
        key = attn.k_proj(normalized).view(batch, span, attn.num_kv_heads, attn.head_dim).float()
        value = attn.v_proj(normalized).view(batch, span, attn.num_kv_heads, attn.head_dim)
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        query, key = _apply_rotary_emb(query, key, cos.unsqueeze(2), sin.unsqueeze(2))
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if self._key[index] is None:
            cached_key, cached_value = key, value
        else:
            cached_key = torch.cat([self._key[index], key], dim=1)
            cached_value = torch.cat([self._value[index], value], dim=1)
        self._key[index] = cached_key
        self._value[index] = cached_value

        total = cached_key.shape[1]
        if attn.num_kv_heads < attn.num_heads:
            repeats = attn.num_heads // attn.num_kv_heads
            expanded_key = cached_key.unsqueeze(3).expand(-1, -1, -1, repeats, -1)
            expanded_key = expanded_key.reshape(batch, total, attn.num_heads, attn.head_dim)
            expanded_value = cached_value.unsqueeze(3).expand(-1, -1, -1, repeats, -1)
            expanded_value = expanded_value.reshape(batch, total, attn.num_heads, attn.head_dim)
        else:
            expanded_key, expanded_value = cached_key, cached_value

        # A single decode query attends to every cached key (all at positions <= its own),
        # so causal masking is only needed while prefilling the >1-token prompt.
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            expanded_key.transpose(1, 2),
            expanded_value.transpose(1, 2),
            is_causal=causal and span > 1,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, span, -1)
        hidden = hidden + attn.o_proj(attention.to(attn.o_proj.weight.dtype))
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _forward(self, token_ids: torch.Tensor, position_ids: torch.Tensor, *, causal: bool) -> torch.Tensor:
        hidden = self._model.embed_tokens(token_ids)
        for index, layer in enumerate(self._layers):
            hidden = self._layer_step(layer, hidden, position_ids, index, causal=causal)
        last_hidden = self._model.norm(hidden[:, -1:])
        return self._model.lm_head(last_hidden)[:, -1, :]

    def prefill(self, prompt_token_ids: list[int], device: torch.device) -> torch.Tensor:
        """Consume the prompt, build the cache, and return next-token logits ``[1, V]``."""
        token_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long, device=device)
        position_ids = torch.arange(token_ids.shape[1], device=device).unsqueeze(0)
        logits = self._forward(token_ids, position_ids, causal=True)
        self.length = int(token_ids.shape[1])
        return logits

    def step(self, token_id: int, device: torch.device) -> torch.Tensor:
        """Append one token at the current position and return next-token logits ``[1, V]``."""
        token_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        position_ids = torch.full((1, 1), self.length, dtype=torch.long, device=device)
        logits = self._forward(token_ids, position_ids, causal=False)
        self.length += 1
        return logits


@torch.no_grad()
def generate_thinking(
    model: BagelForSFT,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_token_ids: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
) -> tuple[list[int], list[float]]:
    """Autoregressively sample a thinking chain from a single prompt (navit bs=1).

    Sampling runs on an incremental :class:`_TextKVCache` (O(T)). The returned
    ``per_token_log_probs`` are NOT the cache's sampling logits — they are recomputed
    with a single full-sequence teacher-forced forward (:func:`replay_thinking_logprobs`)
    so they are bit-identical to the training replay and the on-policy ratio is exactly
    1 at bs=1. Returns ``(generated_token_ids, per_token_log_probs)``, both length
    ``<= max_new_tokens`` (truncated at the first stop token, which is included).
    """
    import os
    import time

    device = next(model.parameters()).device
    stop = set(int(t) for t in stop_token_ids) if stop_token_ids else set()
    prompt = [int(t) for t in prompt_token_ids]
    generated: list[int] = []

    _verbose = os.environ.get("UNIGRPO_ROLLOUT_VERBOSE", "0") == "1" and os.environ.get("RANK", "0") == "0"
    _t0 = time.time()

    cache = _TextKVCache(model)
    logits = cache.prefill(prompt, device)
    steps = int(max_new_tokens)
    for i in range(steps):
        token, _ = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
        token_id = int(token.item())
        generated.append(token_id)
        if _verbose and len(generated) <= 3:
            print(f"[ar] token {len(generated)} ({time.time() - _t0:.2f}s cumulative)", flush=True)
        if token_id in stop:
            break
        if i + 1 < steps:
            logits = cache.step(token_id, device)

    if not generated:
        return generated, []
    # Record the on-policy log-probs from the full forward, not the cached sampling
    # logits, so old_logp == new_logp under the replay at unchanged weights (ratio == 1).
    recorded = replay_thinking_logprobs(model, prompt, generated, temperature=temperature)
    return generated, [float(v) for v in recorded.tolist()]


def replay_thinking_logprobs(
    model: BagelForSFT,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Grad-capable teacher-forced per-token log-probs of ``response`` given ``prompt``.

    Returns ``[len(response)]`` under the CURRENT weights; response token ``k`` is
    scored from the hidden state at the token preceding it (standard next-token TF),
    matching :func:`generate_thinking` when temperature=1 and weights are unchanged.
    """
    # Compute happens on the CUDA device even when FSDP has the (sharded) params offloaded to CPU,
    # so build inputs on the live CUDA device rather than the parameter device.
    device = torch.device(get_device_name(), get_device_id()) if is_cuda_available else next(model.parameters()).device
    if not response_token_ids:
        return torch.zeros(0, device=device)
    prompt_len = len(prompt_token_ids)
    response_len = len(response_token_ids)
    full = torch.tensor([list(prompt_token_ids) + list(response_token_ids)], dtype=torch.long, device=device)
    position_ids = torch.arange(full.shape[1], device=device).unsqueeze(0)
    logits = _text_logits(model, full, position_ids)[0]
    predict = logits[prompt_len - 1 : prompt_len - 1 + response_len]
    temp = float(temperature) if float(temperature) > 0.0 else 1.0
    logp_full = torch.log_softmax(predict.float() / temp, dim=-1)
    response = torch.tensor(response_token_ids, dtype=torch.long, device=device)
    return logp_full.gather(-1, response.unsqueeze(-1)).squeeze(-1)


__all__ = ["sample_next_token", "generate_thinking", "replay_thinking_logprobs"]
