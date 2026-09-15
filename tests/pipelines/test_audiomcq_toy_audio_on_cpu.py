# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""The shared toy checkpoint must actually encode audio, not just load for text."""

import json

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tests.special_e2e.build_qwen3_omni_multimodal_tiny_random import (
    _MM_EXTRA_SPECIAL_TOKENS,
    _checkpoint_has_required_mm_token_ids,
    build,
)


def test_toy_audio_processor_and_forward(tmp_path):
    model_path = str(tmp_path / "model")
    build(model_path, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(model_path, attn_implementation="sdpa").eval()
    assert model.config.thinker_config.text_config.tie_word_embeddings is False
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": "unused.wav"}, {"type": "text", "text": "Listen."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    waveform = np.sin(np.arange(8160, dtype=np.float32) / 10)
    inputs = processor(text=[text], audio=[waveform], sampling_rate=16000, return_tensors="pt", padding=True)
    assert inputs["input_features"].shape[1] == model.config.thinker_config.audio_config.num_mel_bins == 128
    audio_id = model.config.thinker_config.audio_token_id
    assert (inputs["input_ids"] == audio_id).sum() > 0
    with torch.no_grad():
        output = model.thinker(**inputs)
    assert output.logits.shape[:2] == inputs["input_ids"].shape
    assert torch.isfinite(output.logits).all()
    # The smoke uses top_k=1 to avoid random input-side modality markers in
    # its text response; no correctness or nonzero-reward assertion belongs here.
    with torch.no_grad():
        generated = model.thinker.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=1.0, top_k=1)
    response = generated[:, inputs["input_ids"].shape[1] :]
    assert response.numel() > 0
    for token in _MM_EXTRA_SPECIAL_TOKENS.values():
        assert not (response == processor.tokenizer.convert_tokens_to_ids(token)).any()
    assert _checkpoint_has_required_mm_token_ids(model_path)
    # An old cached toy without this field must not silently bypass rebuilding.
    config_path = tmp_path / "model" / "config.json"
    saved_config = json.loads(config_path.read_text())
    del saved_config["thinker_config"]["text_config"]["tie_word_embeddings"]
    config_path.write_text(json.dumps(saved_config))
    assert not _checkpoint_has_required_mm_token_ids(model_path)
