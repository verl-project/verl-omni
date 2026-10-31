# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real video and audio inputs across the Megatron boundary; synthetic media, CPU only."""

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tests.special_e2e.build_qwen3_omni_multimodal_tiny_random import build


def test_megatron_video_audio_boundary_matches_real_thinker_logits_and_gradients(tmp_path, monkeypatch):
    """Exercise the adapter's BSHD call with real video/audio towers and the processor sampling clock."""
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
    frames = np.random.default_rng(43).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    waveform = np.sin(np.arange(8160, dtype=np.float32) / 10)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "audio", "audio": "unused.wav"},
                    {"type": "text", "text": "Describe the scene."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        videos=[frames],
        audio=[waveform],
        sampling_rate=16000,
        videos_kwargs={"fps": 2.0, "do_sample_frames": False},
        return_tensors="pt",
    )
    assert inputs["video_second_per_grid"].numel() == 1
    assert inputs["pixel_values_videos"].numel() > 0 and inputs["input_features"].numel() > 0

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
    mm = {
        key: inputs[key]
        for key in (
            "pixel_values_videos",
            "video_grid_thw",
            "video_second_per_grid",
            "input_features",
            "feature_attention_mask",
            "audio_feature_lengths",
        )
        if key in inputs
    }
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

    assert all(p.grad is None for p in thinker.audio_tower.parameters())
