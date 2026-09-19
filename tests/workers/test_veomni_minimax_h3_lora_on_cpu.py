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
"""H3 LoRA export/layout parity; CPU collectives stand-ins, not FSDP/TP execution."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from torch.nn import functional as F

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MiniMaxH3RolloutWeightSyncMixin
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin
from verl_omni.utils import fsdp_utils
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp import diffusers_impl as fsdp_impl
from verl_omni.workers.engine.veomni import diffusion_impl as veomni_impl

veomni_lora = pytest.importorskip("veomni.lora")

_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
_BLOCKS = ["transformer_blocks.0", "transformer_blocks.1", "token_refiner.refiner_blocks.0"]
_RANK, _ALPHA, _FFN = 4, 8, 64


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def engines(request, monkeypatch):
    """Real H3 modules and export code, with only device/collective work mocked."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        actor = MiniMaxH3Transformer3DModel(
            num_attention_heads=4,
            attention_head_dim=16,
            hidden_size=48,  # H3 attention width != residual width.
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=_FFN,
            text_dim=32,
            freq_dim=16,
            time_embed_hidden_dim=48,
            time_embed_dim=32,
            rope_freq_dim=2,
        )
        veomni_actor = veomni_lora.VeOmniLoraModel(
            deepcopy(actor), veomni_lora.VeOmniLoraConfig(r=_RANK, lora_alpha=_ALPHA, target_modules=_TARGETS)
        )
        actor.add_adapter(LoraConfig(r=_RANK, lora_alpha=_ALPHA, target_modules=_TARGETS), adapter_name="default")
        with torch.no_grad():
            for name, param in actor.named_parameters():
                if "lora_" in name:
                    param.copy_(torch.randn_like(param) / 4)  # Nonzero, distinct A/B for every projection.
        veomni_actor.get_base_model().load_state_dict(actor.state_dict(), strict=True)
        # Uniform storage dtype isolates transport; production FP32 islands are not tested here.
        actor.to(request.param)
        veomni_actor.to(request.param)

    # __post_init__ loads tokenizers; only the export fields are needed here.
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora", {"merge": False})
    object.__setattr__(model_config, "fsdp_layer_prefixes", ["transformer_blocks.", "token_refiner.refiner_blocks."])
    fsdp = object.__new__(fsdp_impl.PPODiffusersFSDPEngine)
    fsdp.module = actor
    fsdp.model_config = model_config
    fsdp._is_offload_param = False
    fsdp._uses_fsdp2_cpu_offload_policy = False
    veomni = object.__new__(veomni_impl.VeOmniDiffusionEngine)
    veomni.module = veomni_actor
    veomni._is_lora = True
    veomni._is_offload_param = False
    veomni.engine_config = SimpleNamespace(model_dtype="fp32" if request.param == torch.float32 else "bf16")

    monkeypatch.setattr(fsdp_impl, "load_fsdp_model_to_gpu", lambda *_: None)
    monkeypatch.setattr(fsdp_impl, "log_gpu_memory_usage", lambda *a, **k: None)
    monkeypatch.setattr(fsdp_impl, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(veomni_impl, "load_model_to_gpu", lambda *_: None)
    monkeypatch.setattr(veomni_impl, "get_device_id", lambda: torch.device("cpu"))
    # Select the real FSDP2 collection branch without initializing a process group.
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(fsdp_utils, "_iter_fsdp2_submodules", lambda module: iter([("", module)]))
    monkeypatch.setattr(
        torch.distributed.fsdp.FullyShardedDataParallel, "summon_full_params", lambda *a, **k: nullcontext()
    )
    return fsdp, veomni


def _export(engine, *, base_sync_done=True):
    tensors, config = engine.get_per_tensor_param(base_sync_done=base_sync_done)
    return {name: tensor.detach().clone() for name, tensor in tensors}, config


def _assert_tensors_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=lambda msg, name=name: f"{name}: {msg}")


@pytest.mark.parametrize("base_sync_done", [False, True], ids=["base", "adapter"])
def test_minimax_h3_export_matches_fsdp2(engines, base_sync_done):
    fsdp, veomni = engines
    fsdp_tensors, fsdp_config = _export(fsdp, base_sync_done=base_sync_done)
    veomni_tensors, veomni_config = _export(veomni, base_sync_done=base_sync_done)
    assert fsdp_tensors
    _assert_tensors_equal(fsdp_tensors, veomni_tensors)
    assert not any("base_model.model" in name or ".base_layer" in name for name in veomni_tensors)
    for key in ("r", "lora_alpha", "lora_dropout", "bias"):
        assert fsdp_config[key] == veomni_config[key]
    assert set(fsdp_config["target_modules"]) == set(veomni_config["target_modules"]) == set(_TARGETS)
    if base_sync_done:
        assert len(fsdp_tensors) == len(_BLOCKS) * len(_TARGETS) * 2
        assert all("lora_" in name for name in fsdp_tensors)
    else:
        assert not any("lora_" in name for name in fsdp_tensors)
        assert "transformer.proj_in.weight" in fsdp_tensors
        assert "transformer.audio_proj_in.weight" in fsdp_tensors


def test_minimax_h3_lora_projection_forward_and_gradients_match_fsdp2(engines):
    """Same-weight dense LoRA math, not full H3 attention/FSDP backward parity."""
    fsdp, veomni = engines
    left = dict(fsdp.module.named_modules())
    right = dict(veomni.module.get_base_model().named_modules())
    generator = torch.Generator().manual_seed(2026)
    tested = 0
    for name, layer in left.items():
        if not hasattr(layer, "lora_A"):
            continue
        inputs = torch.randn(3, layer.in_features, generator=generator).to(layer.weight.dtype)
        expected = layer(inputs)
        actual = right[name](inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        for adapter in ("lora_A", "lora_B"):
            expected_grad = getattr(layer, adapter)["default"].weight.grad
            actual_grad = getattr(right[name], adapter)["default"].weight.grad
            assert expected_grad is not None and torch.count_nonzero(expected_grad) > 0
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        tested += 1
    assert tested == len(_BLOCKS) * len(_TARGETS)


@pytest.fixture(params=["flow_grpo", "nft", "ref2va"])
def h3_mapper(request):
    component = "transformers_ref" if request.param == "ref2va" else "transformer"
    cls = MiniMaxH3RolloutWeightSyncMixin if request.param == "nft" else MiniMaxH3WeightSyncMixin
    mapper = object.__new__(cls)
    setattr(mapper, component, SimpleNamespace(arch=SimpleNamespace(ffn_hidden_size=_FFN)))
    if request.param == "ref2va":
        mapper.partition = "combined"
    return mapper, component


def test_minimax_h3_rollout_mapping_and_lora_deltas_match_fsdp2(engines, h3_mapper):
    """Actual vLLM lookup plus independent BA references catch shared mapping bugs."""
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    fsdp, veomni = engines
    mapper, component = h3_mapper
    source, config = _export(fsdp)
    mapped, mapped_config = mapper.map_lora_update_to_engine(source, config)
    veomni_mapped, veomni_config = mapper.map_lora_update_to_engine(*_export(veomni))
    _assert_tensors_equal(mapped, veomni_mapped)
    assert mapped_config["target_modules"] == veomni_config["target_modules"]
    assert mapped_config["r"] == veomni_config["r"] == _RANK
    assert mapped_config["lora_alpha"] == veomni_config["lora_alpha"] == _ALPHA
    assert len(mapped) == len(_BLOCKS) * 7 * 2

    loaded = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors={name: tensor.clone() for name, tensor in veomni_mapped.items()},
        peft_helper=PEFTHelper.from_dict(veomni_config),
        device="cpu",
        dtype=next(iter(source.values())).dtype,
    )
    manager = object.__new__(DiffusionLoRAManager)
    generator = torch.Generator().manual_seed(42)
    consumed = set()
    for block in _BLOCKS:
        runtime_block = block.replace("transformer_blocks.", "blocks.").replace(
            "token_refiner.refiner_blocks.", "token_refiner.blocks."
        )
        projections = [
            ("attn.to_q", "attn.to_q", None),
            ("attn.to_k", "attn.to_k", None),
            ("attn.to_v", "attn.to_v", None),
            ("attn.to_out.0", "attn.out_proj", None),
            ("ff.net.0.proj", "mlp.fc1_0", slice(_FFN, None)),  # gate, then up
            ("ff.net.0.proj", "mlp.fc1_1", slice(None, _FFN)),
            ("ff.net.2", "mlp.fc2", None),
        ]
        for actor_name, runtime_name, rows in projections:
            name = f"{component}.{runtime_block}.{runtime_name}"
            lora = manager._get_lora_weights(loaded, name)
            assert lora is not None, name
            lora.optimize()  # The manager folds alpha/r into B before binding.
            consumed.add(name)
            a = source[f"transformer.{block}.{actor_name}.lora_A.weight"]
            b = source[f"transformer.{block}.{actor_name}.lora_B.weight"]
            if rows is not None:
                b = b[rows]
            torch.testing.assert_close(lora.lora_a, a, rtol=0, atol=0)
            torch.testing.assert_close(lora.lora_b, b * (_ALPHA / _RANK), rtol=0, atol=0)
            # Evaluate in float64 to isolate mapping algebra from BF16 GEMM rounding.
            inputs = torch.randn(3, a.shape[1], generator=generator, dtype=torch.float64)
            expected = F.linear(inputs, b.double() @ a.double()) * (_ALPHA / _RANK)
            actual = F.linear(F.linear(inputs, lora.lora_a.double()), lora.lora_b.double())
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert consumed == set(loaded.loras)


def test_minimax_h3_fp32_adapter_exports_match_at_rollout_dtype(engines, h3_mapper):
    """FSDP retains FP32 adapter masters; VeOmni casts exports to model_dtype."""
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    fsdp, veomni = engines
    fsdp.module.float()
    veomni.module.float()
    veomni.engine_config.model_dtype = "bf16"
    mapper, _ = h3_mapper
    loaded = []
    for engine in (fsdp, veomni):
        tensors, config = _export(engine)
        expected_dtype = torch.float32 if engine is fsdp else torch.bfloat16
        assert all(tensor.dtype == expected_dtype for tensor in tensors.values())
        mapped, config = mapper.map_lora_update_to_engine(tensors, config)
        loaded.append(
            LoRAModel.from_lora_tensors(
                lora_model_id=1,
                tensors=mapped,
                peft_helper=PEFTHelper.from_dict(config),
                device="cpu",
                dtype=torch.bfloat16,
            )
        )
    assert loaded[0].loras.keys() == loaded[1].loras.keys()
    for name, expected in loaded[0].loras.items():
        actual = loaded[1].loras[name]
        expected.optimize()
        actual.optimize()
        torch.testing.assert_close(actual.lora_a, expected.lora_a, rtol=0, atol=0)
        torch.testing.assert_close(actual.lora_b, expected.lora_b, rtol=0, atol=0)


def test_minimax_h3_existing_mapper_also_accepts_legacy_veomni_prefix(engines, h3_mapper):
    """Unlike Qwen's old path, H3 already strips wrappers at its block anchors."""
    _, veomni = engines
    mapper, _ = h3_mapper
    tensors, config = _export(veomni)
    legacy = {
        name.replace("transformer.", "transformer.base_model.model.", 1): tensor for name, tensor in tensors.items()
    }
    expected, expected_config = mapper.map_lora_update_to_engine(tensors, config)
    actual, actual_config = mapper.map_lora_update_to_engine(legacy, config)
    _assert_tensors_equal(actual, expected)
    assert actual_config == expected_config
