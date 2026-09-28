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

"""CPU contract tests for the frozen trainable-snapshot benchmark harness."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark_module():
    scripts = Path(__file__).parents[2] / "scripts"
    metrics_spec = importlib.util.spec_from_file_location(
        "snapshot_benchmark_metrics", scripts / "snapshot_benchmark_metrics.py"
    )
    assert metrics_spec is not None and metrics_spec.loader is not None
    metrics = importlib.util.module_from_spec(metrics_spec)
    previous_metrics = sys.modules.get(metrics_spec.name)
    sys.modules[metrics_spec.name] = metrics
    metrics_spec.loader.exec_module(metrics)
    spec = importlib.util.spec_from_file_location(
        "snapshot_benchmark_contract_under_test", scripts / "benchmark_trainable_snapshot.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if previous_metrics is None:
        del sys.modules[metrics_spec.name]
    else:
        sys.modules[metrics_spec.name] = previous_metrics


class _Adapter(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))


class _LoRATarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        self.lora_A = torch.nn.ModuleDict({"default": _Adapter()})
        self.lora_B = torch.nn.ModuleDict({"default": _Adapter()})


def _qwen_image_module():
    class _Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = _LoRATarget()
            self.to_k = _LoRATarget()
            self.to_v = _LoRATarget()
            self.to_out = torch.nn.ModuleList([_LoRATarget()])
            self.add_q_proj = _LoRATarget()
            self.add_k_proj = _LoRATarget()
            self.add_v_proj = _LoRATarget()
            self.to_add_out = _LoRATarget()

    class _MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Identity(), _LoRATarget()])
            self.net[0].proj = _LoRATarget()

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attention()
            self.img_mlp = _MLP()
            self.txt_mlp = _MLP()

    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([_Block()])
    return model


def _contract_inputs(benchmark_module):
    return SimpleNamespace(module=_qwen_image_module()), SimpleNamespace(layers=1)


def test_parameter_contract_uses_qwen_image_attention_hierarchy(benchmark_module):
    engine, args = _contract_inputs(benchmark_module)

    result = benchmark_module._validate_parameter_contract(engine, args)

    assert result["effective_target_modules"] == [
        f"transformer_blocks.0.{'attn.' if target in benchmark_module.TARGETS[:8] else ''}{target}"
        for target in benchmark_module.TARGETS
    ]
    assert result["trainable_parameter_count"] == 2 * len(benchmark_module.TARGETS)


@pytest.mark.parametrize("fault", ["missing", "extra_adapter", "wrong_dtype"])
def test_parameter_contract_rejects_invalid_qwen_image_lora_expansion(benchmark_module, fault):
    engine, args = _contract_inputs(benchmark_module)
    target = engine.module.transformer_blocks[0].attn.to_q
    if fault == "missing":
        del engine.module.transformer_blocks[0].attn.to_q
    elif fault == "extra_adapter":
        target.lora_A["unexpected"] = _Adapter()
    else:
        target.lora_A["default"].weight.data = target.lora_A["default"].weight.data.bfloat16()

    with pytest.raises(RuntimeError):
        benchmark_module._validate_parameter_contract(engine, args)


def test_weight_provenance_keeps_tensor_and_file_bytes_separate(benchmark_module, monkeypatch, tmp_path):
    model = tmp_path / "model"
    (model / "transformer").mkdir(parents=True)
    (model / "scheduler").mkdir()
    metadata = {
        "model_index.json": b"{}",
        "scheduler/scheduler_config.json": b"{}",
        "transformer/config.json": json.dumps(
            {
                "_class_name": "QwenImageTransformer2DModel",
                "num_layers": 60,
                "num_attention_heads": 24,
                "attention_head_dim": 128,
            }
        ).encode(),
    }
    sizes = {}
    manifest_rows = []
    weight_map = {}
    pointers = {}
    for shard in range(1, 10):
        name = f"diffusion_pytorch_model-{shard:05d}-of-00009.safetensors"
        relative = f"transformer/{name}"
        payload = bytes([shard]) * 100
        (model / relative).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        sizes[relative] = len(payload)
        manifest_rows.append(f"{relative}|{len(payload)}|{digest}")
        weight_map[f"tensor_{shard}"] = name
        pointers[relative] = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(payload)}\n"
    metadata["transformer/diffusion_pytorch_model.safetensors.index.json"] = json.dumps(
        {"metadata": {"total_size": 90}, "weight_map": weight_map}
    ).encode()
    for relative, payload in metadata.items():
        (model / relative).write_bytes(payload)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(manifest_rows) + "\n")

    def fake_git_show(command, cwd, text=False):
        assert command[:2] == ["git", "show"]
        _, relative = command[2].split(":", 1)
        value = pointers.get(relative, (model / relative).read_bytes())
        return value if text else value.encode() if isinstance(value, str) else value

    monkeypatch.setattr(benchmark_module.subprocess, "check_output", fake_git_show)
    monkeypatch.setattr(benchmark_module, "_git", lambda *_: "a" * 40)
    args = SimpleNamespace(model=model, checkpoint_revision="a" * 40, checkpoint_manifest=manifest)

    result = benchmark_module._verify_weights(args)

    assert result["index_tensor_bytes"] == 90
    assert result["shard_file_bytes"] == sum(sizes.values()) == 900
