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
"""Regression: upstream fixes that made patches unnecessary for Qwen3-Omni.

# TODO (mike): remove this comment once the patches are dropped.
Patches dropped from the adapter:
- ``_apply_tie_embeddings_fix``   → v5 config defaults ``tie_word_embeddings=False``
- ``_install_moe_unfuse_hook``    → PEFT >= 0.19.0 handles MoE natively
- ``module._no_split_modules``    → thinker class already correct
"""

import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from packaging.version import parse as parse_version


def _require_version(pkg_name: str, min_version: str):
    """Raise ``AssertionError`` if *pkg_name* is below *min_version*."""
    ver = importlib.metadata.version(pkg_name)
    assert parse_version(ver) >= parse_version(min_version), f"{pkg_name} >= {min_version} is required, got {ver}"


def _has_lora(module: nn.Module) -> bool:
    """Return True if *module* was wrapped with LoRA by PEFT."""
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


def test_configure_processor_binds_multimodal_pad_dedup(monkeypatch):
    """The V1 processor path must collapse image, video, and audio pad runs."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")

    from transformers import AutoConfig, AutoProcessor

    class _FakeOmniModelBase:
        @classmethod
        def register(cls, *args, **kwargs):
            return lambda subclass: subclass

    model_base = types.ModuleType("verl_omni.pipelines.model_base")
    model_base.OmniModelBase = _FakeOmniModelBase
    for package_name in ("verl_omni", "verl_omni.pipelines", "verl_omni.pipelines.qwen3_omni"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl_omni.pipelines.model_base", model_base)

    module_name = "verl_omni.pipelines.qwen3_omni.thinker_training_adapter"
    adapter_path = Path(__file__).parents[2] / "verl_omni" / "pipelines" / "qwen3_omni" / "thinker_training_adapter.py"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    assert spec is not None and spec.loader is not None
    adapter_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, adapter_module)
    spec.loader.exec_module(adapter_module)
    Qwen3OmniThinkerAdapter = adapter_module.Qwen3OmniThinkerAdapter

    token_ids = {
        "<|image_pad|>": 101,
        "<|video_pad|>": 102,
        "<|audio_pad|>": 103,
    }
    tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: token_ids.get(token, 0),
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        audio_token="<|audio_pad|>",
    )
    config = SimpleNamespace(
        thinker_config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        talker_config=SimpleNamespace(vision_start_token_id=104),
    )

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    class AgentLoopWorker:
        def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs=None):
            del input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs

    agent_loop_tq_module = types.ModuleType("verl.trainer.ppo.v1.agent_loop_tq")
    agent_loop_tq_module.AgentLoopWorkerTQ = SimpleNamespace(
        __ray_metadata__=SimpleNamespace(modified_class=AgentLoopWorker)
    )
    for package_name in (
        "verl",
        "verl.trainer",
        "verl.trainer.ppo",
        "verl.trainer.ppo.v1",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.agent_loop_tq", agent_loop_tq_module)

    configured = Qwen3OmniThinkerAdapter.configure_processor(
        "/fake/qwen3-omni",
        SimpleNamespace(trust_remote_code=False),
    )

    assert configured is processor
    assert hasattr(configured, "dedup_pad_tokens")
    assert configured.dedup_pad_tokens([7, 7, 101, 101, 101, 8, 102, 102, 9, 103, 103, 103, 7, 7]) == [
        7,
        7,
        101,
        8,
        102,
        9,
        103,
        7,
        7,
    ]


def test_v1_adapter_forwards_qwen3_omni_audio_lengths_to_rope(monkeypatch):
    """The V1 adapter must install the audio-aware agent-loop RoPE path."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")

    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    class _FakeOmniModelBase:
        @classmethod
        def register(cls, *args, **kwargs):
            return lambda subclass: subclass

    model_base = types.ModuleType("verl_omni.pipelines.model_base")
    model_base.OmniModelBase = _FakeOmniModelBase
    for package_name in ("verl_omni", "verl_omni.pipelines", "verl_omni.pipelines.qwen3_omni"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl_omni.pipelines.model_base", model_base)

    module_name = "verl_omni.pipelines.qwen3_omni.thinker_training_adapter"
    adapter_path = Path(__file__).parents[2] / "verl_omni" / "pipelines" / "qwen3_omni" / "thinker_training_adapter.py"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    assert spec is not None and spec.loader is not None
    adapter_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, adapter_module)
    spec.loader.exec_module(adapter_module)

    class Qwen3OmniMoeProcessor:
        def __init__(self):
            self.audio_seqlens = None
            self.tokenizer = None

    class AgentLoopWorker:
        def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs=None):
            del mm_processor_kwargs
            position_ids, _ = self.processor.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
            )
            return position_ids

    class AgentLoopWorkerTQ(AgentLoopWorker):
        # Ray copies inherited methods when @ray.remote builds the actor class.
        _compute_position_ids = AgentLoopWorker._compute_position_ids

    agent_loop_tq_module = types.ModuleType("verl.trainer.ppo.v1.agent_loop_tq")
    agent_loop_tq_module.AgentLoopWorkerTQ = SimpleNamespace(
        __ray_metadata__=SimpleNamespace(modified_class=AgentLoopWorkerTQ)
    )
    for package_name in (
        "verl",
        "verl.trainer",
        "verl.trainer.ppo",
        "verl.trainer.ppo.v1",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.agent_loop_tq", agent_loop_tq_module)

    def _get_rope_index(
        processor,
        *,
        input_ids,
        attention_mask,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_seqlens=None,
    ):
        del attention_mask, image_grid_thw, video_grid_thw
        processor.audio_seqlens = audio_seqlens
        _ = audio_seqlens[0]
        return torch.zeros((3, *input_ids.shape), dtype=torch.float32), torch.zeros((input_ids.shape[0], 1))

    processor = Qwen3OmniMoeProcessor()
    config = SimpleNamespace(
        thinker_config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        talker_config=SimpleNamespace(vision_start_token_id=104),
    )
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(Qwen3OmniMoeThinkerForConditionalGeneration, "get_rope_index", _get_rope_index)

    configured = adapter_module.Qwen3OmniThinkerAdapter.configure_processor(
        "/fake/qwen3-omni",
        SimpleNamespace(trust_remote_code=False),
    )
    worker = type("Worker", (), {"processor": configured})()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {"feature_attention_mask": torch.tensor([[1, 1, 1, 0]])}

    AgentLoopWorkerTQ._compute_position_ids(worker, input_ids, attention_mask, multi_modal_inputs)

    torch.testing.assert_close(configured.audio_seqlens, torch.tensor([3]))


class _FusedMoEExperts(nn.Module):
    """Minimal Qwen3-Omni-style fused expert group.

    ``gate_up_proj`` is a 3D ``nn.Parameter``, not ``nn.Linear``.
    Per-expert forward mirrors
    ``Qwen3OmniMoeThinkerTextExperts.forward``:
    ``F.linear(x, gate_up_proj[e]).chunk(2)`` → act → ``F.linear(…, down_proj[e])``.
    """

    def __init__(self, num_experts=4, hidden=64, intermediate=128):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
        self.act_fn = nn.functional.silu
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Token-level sparse-MoE forward matching the Qwen3-Omni pattern.

        Args:
            hidden_states: ``(total_tokens, hidden)``
            top_k_index: ``(total_tokens, num_experts_per_tok)`` — expert indices
            top_k_weights: ``(total_tokens, num_experts_per_tok)`` — routing weights
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (E, K, T)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            # fused gate+up projection
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class _MoEModel(nn.Module):
    """Minimal model with ``config.model_type`` for PEFT conversion routing.

    Contains one dense ``nn.Linear`` block and one fused MoE expert
    block with top-k routing so we can verify PEFT correctly routes
    LoRA to both.
    """

    def __init__(self, hidden=64, intermediate=128, num_experts=4, top_k=2):
        super().__init__()
        self.config = type("C", (), {"model_type": "qwen3_omni_moe"})()

        # Dense layers — standard nn.Linear, PEFT wraps these directly.
        self.dense = nn.Module()
        self.dense.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense.o_proj = nn.Linear(hidden, hidden, bias=False)

        # MoE router and experts — fused parameters, PEFT maps via target_parameters.
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.experts = _FusedMoEExperts(num_experts, hidden, intermediate)
        self.hidden = hidden

    @staticmethod
    def _unwrapped_module(module: nn.Module) -> nn.Module:
        """Walk through PEFT ``ParamWrapper`` layers to reach the raw module."""
        while hasattr(module, "base_layer"):
            module = module.base_layer
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Minimal forward exercising both dense and MoE LoRA‑wrapped paths."""
        # Scaled dot-product attention (single-head) so all four projections are exercised.
        q = self.dense.q_proj(x).unsqueeze(0)  # (1, B, H)
        k = self.dense.k_proj(x).unsqueeze(0)  # (1, B, H)
        v = self.dense.v_proj(x).unsqueeze(0)  # (1, B, H)
        scale = q.size(-1) ** -0.5
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        attn_out = self.dense.o_proj((attn_weights @ v).squeeze(0))  # (B, H)

        # MoE top-k routing, mirroring Qwen3OmniMoeThinkerTextSparseMoeBlock.forward.
        router_logits = self.router(x)  # (batch, num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        experts = self._unwrapped_module(self.experts)
        expert_out = experts(x, top_k_indices, top_k_weights)

        return attn_out + expert_out


def test_peft_lora_attaches_to_fused_moe_natively():
    """PEFT converts gate_proj+up_proj → gate_up_proj with doubled rank."""
    pytest.importorskip("peft")
    _require_version("peft", "0.19.0")
    from peft import LoraConfig, get_peft_model

    model = _MoEModel()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=["gate_up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lora_config)

    # 1. Dense nn.Linear layers get standard LoRA.
    assert _has_lora(peft_model.dense.q_proj), "dense q_proj missing LoRA"
    assert _has_lora(peft_model.dense.k_proj), "dense k_proj missing LoRA"
    assert _has_lora(peft_model.dense.v_proj), "dense v_proj missing LoRA"
    assert _has_lora(peft_model.dense.o_proj), "dense o_proj missing LoRA"

    # 2. PEFT fused gate_proj+up_proj → gate_up_proj, and down_proj, in target_parameters.
    cfg = peft_model.peft_config["default"]
    assert "gate_up_proj" in cfg.target_parameters, (
        f"PEFT should move gate_proj+up_proj to target_parameters, got {cfg.target_parameters}"
    )
    assert "down_proj" in cfg.target_parameters, (
        f"PEFT should move down_proj to target_parameters, got {cfg.target_parameters}"
    )

    # 3. PEFT multiplies rank by num_experts internally (base .r stays unchanged).
    experts_mod = peft_model.experts
    # gate_up_proj: in_features=hidden=64, effective rank = r * num_experts = 32
    assert experts_mod.base_layer.lora_A["default"].weight.shape[0] == 8 * 4, (
        f"gate_up_proj effective rank should be r*num_experts=32, "
        f"got {experts_mod.base_layer.lora_A['default'].weight.shape[0]}"
    )
    # down_proj: in_features=intermediate=128, effective rank = r * num_experts = 32
    assert experts_mod.lora_A["default"].weight.shape[0] == 8 * 4, (
        f"down_proj effective rank should be r*num_experts=32, got {experts_mod.lora_A['default'].weight.shape[0]}"
    )

    # 4. gate_up_proj parameter is preserved through PEFT wrapping.
    obj = peft_model.experts
    while hasattr(obj, "base_layer"):
        obj = obj.base_layer
    assert hasattr(obj, "gate_up_proj"), "gate_up_proj parameter should still be reachable through PEFT wrapping"


def test_tie_word_embeddings_is_false_by_default():
    """v5 config: ``tie_word_embeddings=False`` on the thinker sub-config."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig

    cfg = Qwen3OmniMoeConfig()
    # The umbrella config delegates to sub-configs; the thinker sub-config
    # is what FSDP uses after adapter strips non-thinker modules.
    assert cfg.thinker_config.tie_word_embeddings is False, (
        "thinker_config.tie_word_embeddings should default to False in transformers >= 5.0"
    )


def test_thinker_class_no_split_modules_is_correct():
    """Thinker subclass already uses the right FSDP layer class name."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers.models.qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    required = {
        "Qwen3OmniMoeAudioEncoder",
        "Qwen3OmniMoeVisionEncoder",
    }
    optional = {"Qwen3OmniMoeThinkerTextDecoderLayer"}
    actual = Qwen3OmniMoeThinkerForConditionalGeneration._no_split_modules
    actual_set = set(actual)
    assert required.issubset(actual_set), (
        f"_no_split_modules should include {sorted(required)} in transformers >= 5.0, got {actual}"
    )
    unexpected = actual_set - required - optional
    assert not unexpected, (
        f"_no_split_modules has unexpected entries {sorted(unexpected)} in transformers >= 5.0, got {actual}"
    )


def test_peft_wrapped_model_forwards():
    """PEFT-wrapped _MoEModel runs a forward pass and produces valid output."""
    pytest.importorskip("peft")
    _require_version("peft", "0.19.0")
    from peft import LoraConfig, get_peft_model

    model = _MoEModel()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=["gate_up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.train()

    batch, hidden = 2, 64
    x = torch.randn(batch, hidden)

    # 1. Forward pass succeeds and output shape is correct.
    out = peft_model(x)
    assert out.shape == (batch, hidden), f"expected ({batch}, {hidden}), got {out.shape}"
    assert not torch.isnan(out).any(), "forward output contains NaN"
    assert not torch.isinf(out).any(), "forward output contains Inf"

    # 2. Gradients flow through LoRA adapters.
    loss = out.sum()
    loss.backward()

    # Check that at least some LoRA weights received gradients.
    lora_grad_count = 0
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if param.grad is not None:
                lora_grad_count += 1
    assert lora_grad_count > 0, "no LoRA parameters received gradients"

    # 3. Output differs from the bare model (LoRA is active).
    bare_model = _MoEModel()
    with torch.no_grad():
        bare_out = bare_model(x)
    assert not torch.allclose(out.detach(), bare_out, atol=1e-6), (
        "LoRA-wrapped output should differ from bare model output"
    )
