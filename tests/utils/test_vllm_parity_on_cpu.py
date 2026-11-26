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
"""CPU tests for opt-in vLLM kernel parity helpers (no GPU / vLLM kernels)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import verl_omni.utils.vllm_parity as vllm_parity
import verl_omni.workers.engine.lora_export as lora_export
from verl_omni.utils.vllm_parity import (
    PARITY_SWITCH,
    install_qwen3_parity,
    merge_lora_weight,
    parity_enabled,
)


@pytest.fixture(autouse=True)
def _clear_parity_env(monkeypatch):
    monkeypatch.delenv(PARITY_SWITCH, raising=False)


def test_parity_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv(PARITY_SWITCH, raising=False)
    assert parity_enabled() is False
    monkeypatch.setenv(PARITY_SWITCH, "0")
    assert parity_enabled() is False
    monkeypatch.setenv(PARITY_SWITCH, "1")
    assert parity_enabled() is True


def test_merge_lora_weight_matches_bf16_rounded_ba(monkeypatch):
    """merge_lora_weight is bf16(W) + persistent(B,A)*s in fp32, then bf16."""

    class _BI:
        @staticmethod
        def matmul_persistent(b, a):
            return b @ a

    monkeypatch.setattr(vllm_parity, "_bi", lambda: _BI)
    monkeypatch.setattr(vllm_parity, "_no_autocast", lambda: torch.autocast(device_type="cpu", enabled=False))

    base = torch.randn(6, 4, dtype=torch.float32)
    lora_a = torch.randn(2, 4, dtype=torch.float32)
    lora_b = torch.randn(6, 2, dtype=torch.float32)
    scaling = 0.5

    got = merge_lora_weight(base, lora_a, lora_b, scaling)
    a = lora_a.detach().to(torch.bfloat16).float()
    b = lora_b.detach().to(torch.bfloat16).float()
    expected = (base.detach().to(torch.bfloat16).float() + (b @ a) * scaling).to(torch.bfloat16)

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("cfg_kwargs", "match"),
    [
        ({"rope_parameters": {"rope_type": "llama3"}}, "default RoPE"),
        ({"attention_bias": True}, "attention bias"),
        ({"hidden_act": "gelu"}, "SiLU"),
    ],
)
def test_install_qwen3_parity_rejects_unsupported_configs(cfg_kwargs, match):
    cfg = dict(
        rope_parameters={"rope_type": "default"},
        rope_theta=10000.0,
        head_dim=16,
        hidden_size=32,
        num_attention_heads=2,
        attention_bias=False,
        use_sliding_window=False,
        hidden_act="silu",
        max_position_embeddings=32,
        num_hidden_layers=1,
    )
    cfg.update(cfg_kwargs)
    causal_lm = SimpleNamespace(
        model=SimpleNamespace(
            _vllm_parity_installed=False,
            config=SimpleNamespace(**cfg),
            layers=[SimpleNamespace()],
        ),
        lm_head=SimpleNamespace(),
    )
    with pytest.raises(NotImplementedError, match=match):
        install_qwen3_parity(causal_lm)


def test_install_qwen3_parity_is_idempotent(monkeypatch):
    monkeypatch.setattr(vllm_parity, "build_rope_cache", lambda *args, **kwargs: torch.zeros(8, 16))

    layer = SimpleNamespace(forward=None)
    model = SimpleNamespace(
        _vllm_parity_installed=False,
        config=SimpleNamespace(
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            rope_theta=10000.0,
            head_dim=16,
            hidden_size=32,
            num_attention_heads=2,
            attention_bias=False,
            use_sliding_window=False,
            hidden_act="silu",
            max_position_embeddings=32,
            num_hidden_layers=1,
        ),
        layers=[layer],
        forward=None,
    )
    causal_lm = SimpleNamespace(model=model, lm_head=SimpleNamespace(forward=None, weight=torch.zeros(8, 32)))

    install_qwen3_parity(causal_lm)
    first_layer_forward = layer.forward
    first_model_forward = model.forward
    install_qwen3_parity(causal_lm)
    assert layer.forward is first_layer_forward
    assert model.forward is first_model_forward
    assert model._vllm_parity_installed is True


def test_thinker_configure_model_parity_opt_in(monkeypatch):
    from verl_omni.pipelines.minicpm import thinker_training_adapter as adapter
    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter

    called = []

    class _LLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4, 2)

        def get_input_embeddings(self):
            return self.embed

        def set_input_embeddings(self, embeddings):
            self.embed = embeddings

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(version="4.5")
            self.llm = _LLM()
            self.llm.model = nn.Module()
            self.llm.model.embed_tokens = self.llm.embed
            self.tts = nn.Linear(2, 2)
            self.apm = nn.Module()
            self.apm.conv1 = nn.Linear(2, 2)
            self.apm.layers = nn.ModuleList([])
            self.vpm = nn.Linear(2, 2)
            self.resampler = nn.Linear(2, 2)

        def forward(self, data, **kwargs):
            return data["input_ids"]

        def get_vision_embedding(self, data):
            return []

        def get_vllm_embedding(self, data):
            return self.llm.model.embed_tokens(data["input_ids"]), []

        def get_audio_embedding(self, data, **kwargs):
            return []

        def get_omni_embedding(self, data, input_embeddings, **kwargs):
            return input_embeddings

    monkeypatch.setattr(adapter, "_apply_remote_code_patches", lambda module: None)
    monkeypatch.setattr(adapter, "_keep_rotary_frequencies_dtype", lambda module: None)

    monkeypatch.delenv(PARITY_SWITCH, raising=False)
    MiniCPMThinkerAdapter.configure_model(_Module(), SimpleNamespace())
    assert called == []

    monkeypatch.setenv(PARITY_SWITCH, "1")
    monkeypatch.setattr(vllm_parity, "install_qwen3_parity", lambda llm: called.append(llm))
    MiniCPMThinkerAdapter.configure_model(_Module(), SimpleNamespace())
    assert len(called) == 1


def test_lora_merge_uses_shared_merge_when_parity_on(monkeypatch):
    monkeypatch.setattr(lora_export, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(vllm_parity, "parity_enabled", lambda: True)

    merges = []

    def _fake_merge(base, lora_a, lora_b, scaling):
        merges.append((tuple(base.shape), tuple(lora_a.shape), tuple(lora_b.shape), scaling))
        return base.to(torch.bfloat16) + 1

    monkeypatch.setattr(vllm_parity, "merge_lora_weight", _fake_merge)

    base = nn.Linear(4, 6, bias=False)
    layer = MagicMock()
    layer.lora_A = {"default": SimpleNamespace(weight=torch.randn(2, 4))}
    layer.lora_B = {"default": SimpleNamespace(weight=torch.randn(6, 2))}
    layer.scaling = {"default": 0.25}
    entry = lora_export._LoraMerge(weight=base.weight)
    entry.layers.append((layer, ["default"]))

    out = entry.merge()
    assert merges == [((6, 4), (2, 4), (6, 2), 0.25)]
    assert torch.equal(out, base.weight.data.to(torch.bfloat16) + 1)


def test_lora_merge_default_path_unchanged_when_parity_off(monkeypatch):
    monkeypatch.setattr(vllm_parity, "parity_enabled", lambda: False)
    base = nn.Linear(4, 6, bias=False)
    layer = MagicMock()
    delta = torch.ones_like(base.weight.data)
    layer.get_delta_weight.return_value = delta
    entry = lora_export._LoraMerge(weight=base.weight)
    entry.layers.append((layer, ["default"]))

    out = entry.merge()
    torch.testing.assert_close(out, base.weight.data + delta)
    layer.get_delta_weight.assert_called_once_with("default")


def test_merged_lora_stream_refuses_inplace_under_parity(monkeypatch):
    monkeypatch.setattr(vllm_parity, "parity_enabled", lambda: True)
    monkeypatch.setattr(lora_export, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(lora_export, "_vanilla_lora_linears", lambda module: None)

    with pytest.raises(RuntimeError, match="vanilla PEFT"):
        next(
            lora_export._merged_lora_stream(
                nn.Linear(2, 2, bias=False),
                key_prefix="",
                offload_fn=None,
            )
        )


def test_omni_prepare_model_outputs_routes_logprobs_under_parity(monkeypatch):
    import verl.workers.engine.fsdp.transformer_impl as transformer_impl

    from verl_omni.workers.engine.fsdp.omni_impl import OmniFSDPEngine

    seen = {}

    def _fake_parity_lp(logits, labels, temperature=None):
        seen["parity"] = (
            tuple(logits.shape),
            tuple(labels.shape),
            None if temperature is None else tuple(temperature.shape),
        )
        return torch.zeros(labels.shape, dtype=torch.float32)

    monkeypatch.setattr(vllm_parity, "logprobs_from_logits", _fake_parity_lp)
    monkeypatch.setattr(vllm_parity, "parity_enabled", lambda: True)

    engine = object.__new__(OmniFSDPEngine)
    engine.engine_config = SimpleNamespace(batch_invariant=False)

    def _super_prepare(self, output, output_args, micro_batch, logits_processor_func):
        assert self.engine_config.batch_invariant is True
        out = transformer_impl.logprobs_from_logits(
            torch.zeros(2, 8),
            torch.zeros(2, dtype=torch.long),
            batch_invariant=True,
            temperature=torch.ones(2),
        )
        return {"log_probs": out}

    monkeypatch.setattr(
        transformer_impl.FSDPEngineWithLMHead,
        "prepare_model_outputs",
        _super_prepare,
    )

    result = OmniFSDPEngine.prepare_model_outputs(engine, None, {}, {}, None)
    assert seen["parity"] == ((2, 8), (2,), (2,))
    assert engine.engine_config.batch_invariant is False
    assert result["log_probs"].shape == (2,)


def test_omni_prepare_model_outputs_unchanged_when_parity_off(monkeypatch):
    import verl.workers.engine.fsdp.transformer_impl as transformer_impl

    from verl_omni.workers.engine.fsdp.omni_impl import OmniFSDPEngine

    monkeypatch.setattr(vllm_parity, "parity_enabled", lambda: False)
    engine = object.__new__(OmniFSDPEngine)
    engine.engine_config = SimpleNamespace(batch_invariant=False)
    called = []

    def _super_prepare(self, output, output_args, micro_batch, logits_processor_func):
        called.append(True)
        return {"ok": True}

    monkeypatch.setattr(
        transformer_impl.FSDPEngineWithLMHead,
        "prepare_model_outputs",
        _super_prepare,
    )

    assert OmniFSDPEngine.prepare_model_outputs(engine, None, {}, {}, None) == {"ok": True}
    assert called == [True]
    assert engine.engine_config.batch_invariant is False
