# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exercise the explicit Omni BSHD model call against pinned verl on CPU."""

from types import SimpleNamespace

import pytest
import torch
from verl.utils.model import extract_multi_modal_inputs

pytest.importorskip("verl.models.mcore.util", reason="Megatron-Core is optional in CPU CI")

from verl_omni.pipelines.qwen3_omni.megatron_inputs import qwen3_omni_forward_model_engine


class RecordingModel(torch.nn.Module):
    pre_process = True
    post_process = False
    config = SimpleNamespace(fp8=None)

    def forward(self, **kwargs):
        self.seen = kwargs
        return kwargs.get("input_features", kwargs["input_ids"].float()).sum()


@pytest.fixture(autouse=True)
def single_rank_mcore(monkeypatch):
    mcore_util = pytest.importorskip("verl.models.mcore.util")
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(mcore_util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)


def _nested_ids(*rows):
    return torch.nested.nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


def test_audio_reaches_model_and_autograd_while_vision_is_preserved():
    model = RecordingModel()
    features = torch.randn(1, 128, 5, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.long)
    lengths = mask.sum(-1)
    image = torch.randn(1, 3)
    inputs = {"input_features": features, "feature_attention_mask": mask, "audio_feature_lengths": lengths}
    inputs["pixel_values"] = image
    output = qwen3_omni_forward_model_engine(model, _nested_ids([1, 2, 3, 4]), inputs)
    output.backward()
    assert model.seen["input_features"] is features
    assert model.seen["feature_attention_mask"] is mask
    assert model.seen["audio_feature_lengths"] is lengths
    assert model.seen["pixel_values"] is image
    assert model.seen["position_ids"] is None
    assert torch.equal(features.grad, torch.ones_like(features))
    assert not model._forward_pre_hooks


def test_direct_forward_matches_pinned_bshd_text_and_gradients():
    model_forward = pytest.importorskip("verl.models.mcore.model_forward")

    class TinyModel(torch.nn.Module):
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, **kwargs):
            self.seen = kwargs
            return kwargs["input_ids"].float().unsqueeze(-1) * self.weight

    ids = _nested_ids([1, 2, 3], [4, 5])
    model = TinyModel()
    reference = TinyModel()

    def process(logits, label, temperature):
        return {"log_probs": logits.squeeze(-1) * temperature + label.float()}

    args = {"label": ids.clone(), "temperature": ids.clone() * 0 + 1.5, "loss_mask": ids.clone()}
    output = qwen3_omni_forward_model_engine(
        model, ids, {}, logits_processor=process, logits_processor_args=args, vision_model=True, pad_token_id=0
    )
    expected = model_forward.gptmodel_forward_model_engine(
        reference,
        ids,
        {},
        logits_processor=process,
        logits_processor_args=dict(args),
        vision_model=True,
        pad_token_id=0,
        data_format="bshd",
    )
    torch.testing.assert_close(output["log_probs"].values(), expected["log_probs"].values())
    output["log_probs"].values().sum().backward()
    expected["log_probs"].values().sum().backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    assert model.seen["position_ids"] is reference.seen["position_ids"] is None
    assert not model._forward_pre_hooks


def test_audio_logits_path_matches_explicit_bshd_call_and_gradients():
    from verl.models.mcore.util import postprocess_bshd_engine, preprocess_bshd_engine

    class TinyAudioModel(torch.nn.Module):
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, **kwargs):
            self.seen = kwargs
            audio = kwargs["input_features"].mean(dim=(1, 2)).view(-1, 1, 1)
            return (kwargs["input_ids"].float().unsqueeze(-1) + audio) * self.weight

    ids = _nested_ids([1, 2, 3], [4, 5])
    features = torch.randn(2, 128, 5, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    mask = torch.ones(2, 5, dtype=torch.long)
    model = TinyAudioModel()
    reference = TinyAudioModel()

    def process(logits, label, temperature):
        return {"log_probs": logits.squeeze(-1) * temperature + label.float()}

    args = {"label": ids.clone(), "temperature": ids.clone() * 0 + 1.5, "loss_mask": ids.clone()}
    inputs = {"input_features": features, "feature_attention_mask": mask, "audio_feature_lengths": mask.sum(-1)}
    output = qwen3_omni_forward_model_engine(model, ids, inputs, logits_processor=process, logits_processor_args=args)

    # This is the pinned verl BSHD preprocessing/logits/postprocessing flow,
    # with only the Omni audio tensors and model-built M-RoPE supplied directly.
    padded_ids, attention_mask, _ = preprocess_bshd_engine(ids)
    logits = reference(
        input_ids=padded_ids,
        attention_mask=attention_mask,
        position_ids=None,
        input_features=reference_features,
        feature_attention_mask=mask,
        audio_feature_lengths=mask.sum(-1),
    )
    label = preprocess_bshd_engine(args["label"], need_roll=True)[0]
    temperature = preprocess_bshd_engine(args["temperature"])[0]
    expected = postprocess_bshd_engine(process(logits, label, temperature)["log_probs"], attention_mask)
    torch.testing.assert_close(output["log_probs"].values(), expected.values())
    output["log_probs"].values().sum().backward()
    expected.values().sum().backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    torch.testing.assert_close(features.grad, reference_features.grad)
    assert model.seen["position_ids"] is None


def test_variable_audio_frames_are_padded_before_forward():
    rows = [
        {"input_features": torch.ones(1, 128, n), "feature_attention_mask": torch.ones(1, n, dtype=torch.long)}
        for n in (3, 5)
    ]
    model = RecordingModel()
    qwen3_omni_forward_model_engine(model, _nested_ids([1, 2, 3, 4], [5, 6]), extract_multi_modal_inputs(rows))
    assert model.seen["input_features"].shape == (2, 128, 5)
    assert model.seen["feature_attention_mask"].sum(-1).tolist() == [3, 5]
    assert not model.seen["input_features"][0, :, 3:].any()


def test_text_only_uses_model_mrope_and_does_not_modify_later_calls():
    model = RecordingModel()
    ids = torch.ones(1, 4, dtype=torch.long)
    positions = torch.arange(4)
    qwen3_omni_forward_model_engine(model, _nested_ids([1, 2, 3, 4]), {})
    assert model.seen["position_ids"] is None
    assert "input_features" not in model.seen
    model(input_ids=ids, position_ids=positions)
    assert model.seen["position_ids"] is positions
