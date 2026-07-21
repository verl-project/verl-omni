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
"""CPU parity tests for SD3 token-id-native prompt encoding.

Necessity: the SD3 FlowGRPO rollout adapter consumes per-text-encoder token ids
produced once by the agent loop instead of decoding ids back to text inside the
pipeline. These tests verify that ``SD3TokenIdPromptMixin.encode_prompt_from_token_ids``
reproduces the vLLM-Omni text-based ``encode_prompt`` exactly, including
per-encoder padding, truncation, and CLIP/T5 embedding concatenation.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "special_e2e"))

from build_sd3_tiny_random import get_dummy_sd3_components  # noqa: E402

from verl_omni.pipelines.sd3_flow_grpo.common import (  # noqa: E402
    SD3TokenIdPromptMixin,
    pad_token_id_batch,
)

_vllm_omni_sd3 = pytest.importorskip(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3",
    reason="vllm-omni is required for the text-path reference implementation",
)

T5_MAX_SEQUENCE_LENGTH = 16


class _ParityPipeline(SD3TokenIdPromptMixin):
    """Tiny SD3 components + the real vLLM-Omni text encode path as reference."""

    # Borrow the unbound text-path methods so the reference cannot drift from
    # the actual rollout pipeline implementation.
    encode_prompt = _vllm_omni_sd3.StableDiffusion3Pipeline.encode_prompt
    _get_clip_prompt_embeds = _vllm_omni_sd3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    _get_t5_prompt_embeds = _vllm_omni_sd3.StableDiffusion3Pipeline._get_t5_prompt_embeds

    def __init__(self):
        components = get_dummy_sd3_components()
        self.tokenizer = components["tokenizer"]
        self.tokenizer_2 = components["tokenizer_2"]
        self.tokenizer_3 = components["tokenizer_3"]
        # Mimic SD3.5, where CLIP-G pads with a different token ("!") than
        # CLIP-L ("<|endoftext|>"), to exercise per-encoder padding.
        self.tokenizer_2.pad_token = "<|startoftext|>"
        self.text_encoder = components["text_encoder"].eval()
        self.text_encoder_2 = components["text_encoder_2"].eval()
        self.text_encoder_3 = components["text_encoder_3"].eval()
        self.transformer = components["transformer"]
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.device = torch.device("cpu")
        self.od_config = SimpleNamespace(dtype=torch.float32)


def _agent_loop_tokenize(tokenizer, texts: list[str], max_length: int) -> list[list[int]]:
    """Mirror ``DiffusionSingleTurnAgentLoop._tokenize_per_encoder`` for one tokenizer."""
    return [
        tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"] for text in texts
    ]


@pytest.fixture(scope="module")
def pipeline() -> _ParityPipeline:
    return _ParityPipeline()


def _encode_both_paths(pipeline: _ParityPipeline, prompts: list[str], num_images_per_prompt: int = 1):
    clip_ids = _agent_loop_tokenize(pipeline.tokenizer, prompts, pipeline.tokenizer_max_length)
    t5_ids = _agent_loop_tokenize(pipeline.tokenizer_3, prompts, T5_MAX_SEQUENCE_LENGTH)

    with torch.no_grad():
        ref_embeds, ref_pooled = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            max_sequence_length=T5_MAX_SEQUENCE_LENGTH,
            num_images_per_prompt=num_images_per_prompt,
        )
        embeds, pooled = pipeline.encode_prompt_from_token_ids(
            clip_prompt_ids=clip_ids,
            t5_prompt_ids=t5_ids,
            max_sequence_length=T5_MAX_SEQUENCE_LENGTH,
            num_images_per_prompt=num_images_per_prompt,
        )
    return (ref_embeds, ref_pooled), (embeds, pooled)


@pytest.mark.parametrize("num_images_per_prompt", [1, 2])
def test_token_id_encode_matches_text_encode(pipeline, num_images_per_prompt):
    prompts = ["a red circle on a white background", "a blue square"]
    (ref_embeds, ref_pooled), (embeds, pooled) = _encode_both_paths(pipeline, prompts, num_images_per_prompt)

    assert embeds.shape == ref_embeds.shape
    assert pooled.shape == ref_pooled.shape
    torch.testing.assert_close(embeds, ref_embeds, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pooled, ref_pooled, rtol=0.0, atol=0.0)


def test_empty_prompt_parity(pipeline):
    (ref_embeds, ref_pooled), (embeds, pooled) = _encode_both_paths(pipeline, [""])

    torch.testing.assert_close(embeds, ref_embeds, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pooled, ref_pooled, rtol=0.0, atol=0.0)


def test_long_prompt_truncation_parity(pipeline):
    long_prompt = " ".join(["a red circle on a white background"] * 40)
    (ref_embeds, ref_pooled), (embeds, pooled) = _encode_both_paths(pipeline, [long_prompt])

    torch.testing.assert_close(embeds, ref_embeds, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pooled, ref_pooled, rtol=0.0, atol=0.0)


def test_pad_token_id_batch_pads_and_truncates():
    padded = pad_token_id_batch([[1, 2], [3, 4, 5, 6, 7]], max_length=4, pad_token_id=0, device=torch.device("cpu"))

    assert padded.shape == (2, 4)
    assert padded.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6]]
