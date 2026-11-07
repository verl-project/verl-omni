# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real image conditioning and frozen-tower gradients for the shared toy Thinker."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tests.special_e2e.build_qwen3_omni_multimodal_tiny_random import build


def test_image_conditioning_and_language_model_backward(tmp_path):
    torch.manual_seed(42)
    model_path = str(tmp_path / "model")
    build(model_path, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(model_path, attn_implementation="sdpa").thinker.eval()
    for tower in (model.visual, model.audio_tower):
        tower.requires_grad_(False)
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the diagram."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum()
    assert image_tokens == inputs["image_grid_thw"].prod() // processor.image_processor.merge_size**2
    output = model(**inputs).logits
    assert torch.isfinite(output).all()
    changed = dict(inputs)
    changed["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
    with torch.no_grad():
        changed_output = model(**changed).logits
    assert not torch.allclose(output[:, -1], changed_output[:, -1], atol=1e-6, rtol=1e-6)
    output[:, -1].float().square().mean().backward()
    assert all(p.grad is None for p in model.visual.parameters())
    assert all(p.grad is None for p in model.audio_tower.parameters())
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_megatron_image_boundary_matches_real_thinker_logits_and_gradients(tmp_path, monkeypatch):
    """Exercise the adapter's BSHD call with a real vision tower, not a mock tensor sink."""
    from types import SimpleNamespace

    import pytest

    util = pytest.importorskip("verl.models.mcore.util")
    from verl_omni.pipelines.qwen3_omni.megatron_inputs import qwen3_omni_forward_model_engine

    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(util.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(42)
    path = str(tmp_path / "model")
    build(path, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(path)
    thinker = AutoModelForMultimodalLM.from_pretrained(path, attn_implementation="sdpa").thinker.eval()
    thinker.visual.requires_grad_(False)
    thinker.audio_tower.requires_grad_(False)
    image = Image.fromarray(np.random.default_rng(43).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the diagram."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    class ThinkerBoundary(torch.nn.Module):
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self, model):
            super().__init__()
            self.thinker = model

        def forward(self, **kwargs):
            # HF expects integer padding masks; the Megatron boundary uses bool.
            kwargs["attention_mask"] = kwargs["attention_mask"].long()
            return self.thinker(**kwargs).logits

    ids = torch.nested.nested_tensor(list(inputs["input_ids"]), layout=torch.jagged)
    mm = {key: inputs[key] for key in ("pixel_values", "image_grid_thw")}
    actual = qwen3_omni_forward_model_engine(
        ThinkerBoundary(thinker),
        ids,
        mm,
        vision_model=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    ).values()
    expected = thinker(**inputs).logits.squeeze(0)
    torch.testing.assert_close(actual, expected)
    actual[-1].square().mean().backward()
    actual_grad = thinker.lm_head.weight.grad.detach().clone()
    thinker.zero_grad(set_to_none=True)
    expected[-1].square().mean().backward()
    torch.testing.assert_close(thinker.lm_head.weight.grad, actual_grad)
    assert actual_grad.abs().sum() > 0
    assert all(p.grad is None for p in thinker.visual.parameters())
