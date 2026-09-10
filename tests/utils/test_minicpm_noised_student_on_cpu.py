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
"""MiniCPM-o noised-student checkpoint preparation."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture
def noise_module():
    path = Path(__file__).resolve().parents[2] / "examples/opd_trainer/minicpm_o/prepare_noised_student.py"
    spec = importlib.util.spec_from_file_location("minicpm_noised_student", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    shard1 = {
        "apm.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "llm.layer.weight": torch.arange(1, 7, dtype=torch.float32).reshape(2, 3),
    }
    shard2 = {
        "llm.bfloat": torch.arange(1, 33, dtype=torch.bfloat16),
        "llm.zero": torch.zeros(3, dtype=torch.bfloat16),
        "vpm.position_ids": torch.arange(4),
    }
    save_file(shard1, path / "model-00001-of-00002.safetensors", metadata={"format": "pt"})
    save_file(shard2, path / "model-00002-of-00002.safetensors")
    tensors = [*shard1.values(), *shard2.values()]
    index = {
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors)},
        "weight_map": {
            "apm.weight": "model-00001-of-00002.safetensors",
            "llm.layer.weight": "model-00001-of-00002.safetensors",
            "llm.bfloat": "model-00002-of-00002.safetensors",
            "llm.zero": "model-00002-of-00002.safetensors",
            "vpm.position_ids": "model-00002-of-00002.safetensors",
        },
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    (path / "config.json").write_text('{"model_type":"minicpmo"}\n')


def test_create_noised_checkpoint_changes_only_nonzero_llm_float_tensors(tmp_path, noise_module):
    source = tmp_path / "source"
    output = tmp_path / "student"
    _write_checkpoint(source)

    manifest = noise_module.create_noised_checkpoint(source, output, ratio=0.2, seed=17)

    source_shard1 = load_file(source / "model-00001-of-00002.safetensors")
    output_shard1 = load_file(output / "model-00001-of-00002.safetensors")
    source_shard2 = load_file(source / "model-00002-of-00002.safetensors")
    output_shard2 = load_file(output / "model-00002-of-00002.safetensors")
    delta = output_shard1["llm.layer.weight"] - source_shard1["llm.layer.weight"]
    relative_norm = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(source_shard1["llm.layer.weight"])
    assert relative_norm == pytest.approx(0.2, rel=1e-6)
    bfloat_delta = output_shard2["llm.bfloat"].float() - source_shard2["llm.bfloat"].float()
    bfloat_ratio = torch.linalg.vector_norm(bfloat_delta) / torch.linalg.vector_norm(
        source_shard2["llm.bfloat"].float()
    )
    assert bfloat_ratio == pytest.approx(0.2, rel=5e-4)
    assert torch.equal(output_shard1["apm.weight"], source_shard1["apm.weight"])
    assert torch.equal(output_shard2["llm.zero"], source_shard2["llm.zero"])
    assert torch.equal(output_shard2["vpm.position_ids"], source_shard2["vpm.position_ids"])
    assert (output / "config.json").read_text() == (source / "config.json").read_text()
    assert manifest["tensor_prefix"] == "llm."
    assert manifest["relative_l2_ratio"] == 0.2
    assert manifest["perturbed_tensors"] == 2


def test_noise_is_deterministic_per_tensor_name(tmp_path, noise_module):
    source = tmp_path / "source"
    output1 = tmp_path / "student1"
    output2 = tmp_path / "student2"
    _write_checkpoint(source)

    noise_module.create_noised_checkpoint(source, output1, ratio=0.2, seed=42)
    noise_module.create_noised_checkpoint(source, output2, ratio=0.2, seed=42)

    for shard_name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        tensors1 = load_file(output1 / shard_name)
        tensors2 = load_file(output2 / shard_name)
        assert tensors1.keys() == tensors2.keys()
        assert all(torch.equal(tensors1[name], tensors2[name]) for name in tensors1)


def test_noised_checkpoint_rejects_invalid_destinations_and_ratios(tmp_path, noise_module):
    source = tmp_path / "source"
    _write_checkpoint(source)

    with pytest.raises(ValueError, match="between 0 and 1"):
        noise_module.create_noised_checkpoint(source, tmp_path / "bad-ratio", ratio=1.0, seed=42)
    with pytest.raises(ValueError, match="must not be inside"):
        noise_module.create_noised_checkpoint(source, source / "student", ratio=0.2, seed=42)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        noise_module.create_noised_checkpoint(source, existing, ratio=0.2, seed=42)
