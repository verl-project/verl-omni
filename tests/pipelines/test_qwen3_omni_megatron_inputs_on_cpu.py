# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Pin the actual module-call audio boundary without requiring Megatron in CPU CI."""

from types import SimpleNamespace

import pytest
import torch
from verl.utils.model import extract_multi_modal_inputs

from verl_omni.pipelines.qwen3_omni.megatron_inputs import qwen3_omni_megatron_inputs


class RecordingModel(torch.nn.Module):
    def forward(self, **kwargs):
        self.seen = kwargs
        return kwargs.get("input_features", kwargs["input_ids"].float()).sum()


def test_audio_reaches_model_and_autograd_while_vision_is_preserved():
    model = RecordingModel()
    features = torch.randn(1, 128, 5, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.long)
    lengths = mask.sum(-1)
    image = torch.randn(1, 3)
    inputs = {"input_features": features, "feature_attention_mask": mask, "audio_feature_lengths": lengths}
    with qwen3_omni_megatron_inputs(model, inputs):
        output = model(input_ids=torch.ones(1, 4, dtype=torch.long), position_ids=torch.arange(4), pixel_values=image)
    output.backward()
    assert model.seen["input_features"] is features
    assert model.seen["feature_attention_mask"] is mask
    assert model.seen["audio_feature_lengths"] is lengths
    assert model.seen["pixel_values"] is image
    assert model.seen["position_ids"] is None
    assert torch.equal(features.grad, torch.ones_like(features))
    assert not model._forward_pre_hooks


def test_audio_hook_wraps_the_pinned_mcore_bshd_forward(monkeypatch):
    model_forward = pytest.importorskip("verl.models.mcore.model_forward")
    mcore_util = pytest.importorskip("verl.models.mcore.util")
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mcore_util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)

    class MCoreRecordingModel(RecordingModel):
        pre_process = True
        post_process = False
        config = SimpleNamespace(fp8=None)

    model = MCoreRecordingModel()
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        layout=torch.jagged,
    )
    features = torch.randn(2, 128, 5, requires_grad=True)
    feature_mask = torch.ones(2, 5, dtype=torch.long)
    image = torch.randn(2, 3)
    inputs = {
        "input_features": features,
        "feature_attention_mask": feature_mask,
        "audio_feature_lengths": feature_mask.sum(-1),
        "pixel_values": image,
    }

    with qwen3_omni_megatron_inputs(model, inputs):
        output = model_forward.gptmodel_forward_model_engine(
            model=model,
            input_ids=input_ids,
            multi_modal_inputs=inputs,
            data_format="bshd",
        )
    output.backward()

    assert model.seen["input_ids"].shape == (2, 3)
    assert "packed_seq_params" not in model.seen
    assert model.seen["position_ids"] is None
    assert model.seen["input_features"] is features
    assert model.seen["feature_attention_mask"] is feature_mask
    assert model.seen["audio_feature_lengths"] is inputs["audio_feature_lengths"]
    assert model.seen["pixel_values"] is image
    assert torch.equal(features.grad, torch.ones_like(features))
    assert not model._forward_pre_hooks


def test_variable_audio_frames_are_padded_before_forward():
    rows = [
        {"input_features": torch.ones(1, 128, n), "feature_attention_mask": torch.ones(1, n, dtype=torch.long)}
        for n in (3, 5)
    ]
    model = RecordingModel()
    with qwen3_omni_megatron_inputs(model, extract_multi_modal_inputs(rows)):
        model(input_ids=torch.ones(2, 4, dtype=torch.long))
    assert model.seen["input_features"].shape == (2, 128, 5)
    assert model.seen["feature_attention_mask"].sum(-1).tolist() == [3, 5]
    assert not model.seen["input_features"][0, :, 3:].any()


def test_packed_input_fails_and_hook_is_removed():
    model = RecordingModel()
    with pytest.raises(ValueError, match="BSHD"), qwen3_omni_megatron_inputs(model, {}):
        model(input_ids=torch.ones(1, 4, dtype=torch.long), packed_seq_params=object())
    assert not model._forward_pre_hooks


def test_text_only_still_uses_model_mrope_and_later_calls_are_unmodified():
    model = RecordingModel()
    ids = torch.ones(1, 4, dtype=torch.long)
    positions = torch.arange(4)
    with qwen3_omni_megatron_inputs(model, {}):
        model(input_ids=ids, position_ids=positions)
        assert model.seen["position_ids"] is None
        assert "input_features" not in model.seen
    model(input_ids=ids, position_ids=positions)
    assert model.seen["position_ids"] is positions
