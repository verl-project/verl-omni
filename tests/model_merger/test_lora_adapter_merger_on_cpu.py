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

"""Automatic LoRA checkpoint export with real Diffusers and PEFT reload."""

import importlib
import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers import QwenImageTransformer2DModel
from hydra import compose, initialize_config_dir
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file


@pytest.fixture
def api(monkeypatch):
    # Isolate training/GPU registration; checkpoint I/O and PEFT remain real.
    before = set(sys.modules)
    root = Path(__file__).parents[2]
    for name in ("verl_omni", "verl_omni.utils"):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(root.joinpath(*name.split(".")))]
            monkeypatch.setitem(sys.modules, name, package)
    if "verl.utils.fsdp_utils" not in sys.modules:
        upstream = ModuleType("verl.utils.fsdp_utils")
        for name in ("fsdp_version", "collect_lora_params", "layered_summon_lora_params"):
            setattr(upstream, name, Mock(side_effect=AssertionError("Offline export must not call live FSDP")))
        monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    yield importlib.import_module("verl_omni.utils.fsdp_utils")
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


@pytest.fixture
def case(tmp_path):
    torch.manual_seed(12)
    base = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    source, target, base_dir = tmp_path / "actor", tmp_path / "export", tmp_path / "base"
    base.save_pretrained(base_dir)
    base.save_config(source / "huggingface")
    model = deepcopy(base)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_v", "to_out.0"])
    model.add_adapter(config, adapter_name="default")
    model.add_adapter(deepcopy(config), adapter_name="old")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    (source / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 1}))
    (source / "lora_train_meta.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "task_type": "CAUSAL_LM"}))
    state = {key: value.clone() for key, value in model.state_dict().items() if "lora_" in key}
    rank_file = source / "model_world_size_1_rank_0.pt"
    torch.save(state, rank_file)
    command = [
        "merge",
        "--backend=fsdp",
        f"--local_dir={source}",
        f"--target_dir={target}",
        f"--base_model={base_dir}",
        "--output-format=transformer",
        "--trust-checkpoint",
    ]
    return SimpleNamespace(
        source=source, target=target, rank_file=rank_file, model=model, base=base, state=state, command=command
    )


def run_cli(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["model_merger", *arguments])
    runpy.run_module("verl_omni.model_merger", run_name="__main__")


@pytest.mark.parametrize("adapter_name", ["default", "old"])
@pytest.mark.parametrize("full_checkpoint", [False, True])
def test_cli_auto_export_and_reload(api, case, monkeypatch, capsys, adapter_name, full_checkpoint):
    if full_checkpoint:
        state = {
            "_fsdp_wrapped_module.base_model.model." + key.replace(".lora_", "._fsdp_wrapped_module.lora_"): value
            for key, value in case.model.state_dict().items()
        }
        (case.source / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 2}))
        case.rank_file.unlink()
        for rank in range(2):
            torch.save(state, case.source / f"model_world_size_2_rank_{rank}.pt")
    else:
        # LoRA-only export does not need the base model on disk or a new output mode.
        case.command = [arg for arg in case.command if not arg.startswith(("--base_model=", "--output-format="))]
        case.command.append("--base_model=original/base")
    if adapter_name != "default":
        case.command.append(f"--adapter-name={adapter_name}")
    run_cli(monkeypatch, case.command)
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    adapter_dir = case.target / "lora_adapter"
    assert result["output_dir"] == str(case.target if full_checkpoint else adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert config["task_type"] is None
    assert set(config["target_modules"]) == {"to_q", "to_v", "to_out.0"}
    assert (config["r"], config["lora_alpha"]) == (2, 4)
    weights = load_file(adapter_dir / "adapter_model.safetensors")
    assert set(weights) == {
        "base_model.model." + key.replace(f".{adapter_name}.weight", ".weight")
        for key in case.state
        if f".{adapter_name}.weight" in key
    }
    base = deepcopy(case.base)
    if full_checkpoint:
        base = QwenImageTransformer2DModel.from_pretrained(case.target, local_files_only=True)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, case.base.state_dict()[key], rtol=0, atol=0)
        run_cli(monkeypatch, ["test", "--backend=fsdp", f"--test_hf_dir={case.target}"])
    else:
        assert not (case.target / "config.json").exists()
        assert not (case.target / "diffusion_pytorch_model.safetensors").exists()
    restored = PeftModel.from_pretrained(base, adapter_dir)
    case.model.set_adapter(adapter_name)
    case.model.eval()
    case.model.set_attention_backend("native")
    base.set_attention_backend("native")
    inputs = dict(
        hidden_states=torch.randn(1, 4, 16),
        encoder_hidden_states=torch.randn(1, 3, 16),
        encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.bool),
        timestep=torch.tensor([0.5]),
        img_shapes=[[(1, 2, 2)]],
    )
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs).sample, case.model(**inputs).sample)
    run_cli(monkeypatch, ["test", "--backend=fsdp", f"--test_hf_dir={adapter_dir}"])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["integrity"] == "passed"


@pytest.mark.parametrize("fault", ["missing_metadata", "unknown_adapter"])
def test_invalid_adapter_export_leaves_no_output(api, case, monkeypatch, fault):
    if fault == "missing_metadata":
        (case.source / "lora_train_meta.json").unlink()
    elif fault == "unknown_adapter":
        case.command.append("--adapter-name=absent")
    with pytest.raises((ValueError, FileNotFoundError)):
        run_cli(monkeypatch, case.command)
    assert not case.target.exists()
    assert not list(case.target.parent.glob(".export.merge*"))


@pytest.mark.parametrize(
    "overrides,expected", [([], False), (["actor_rollout_ref.actor.checkpoint.save_lora_only=true"], True)]
)
def test_lora_checkpoint_config_override(overrides, expected):
    config_dir = Path(__file__).parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=overrides)
    assert config.actor_rollout_ref.actor.checkpoint.save_lora_only is expected


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("replica_delta", [0, 1])
def test_legacy_export_keeps_metadata_configuration(api, case, normalized, replica_delta):
    expected = {
        "base_model.model." + key.replace(".default.weight", ".weight"): value
        for key, value in case.state.items()
        if ".default.weight" in key
    }
    (case.source / "fsdp_config.json").write_text(json.dumps({"world_size": 2}))
    case.rank_file.unlink()
    for rank in range(2):
        state = {}
        for key, value in case.state.items():
            if normalized and ".default.weight" not in key:
                continue
            key = key.replace(".default.weight", ".weight") if normalized else key
            key = "_fsdp_wrapped_module." + key.replace(".lora_", "._fsdp_wrapped_module.lora_")
            state[key] = value + rank * replica_delta
        torch.save(state, case.source / f"model_world_size_2_rank_{rank}.pt")
    if replica_delta:
        with pytest.raises(ValueError, match="replicas disagree"):
            api.export_fsdp_lora_adapter(case.source, case.target, "original/base")
        assert not case.target.exists()
        return
    summary = api.export_fsdp_lora_adapter(case.source, case.target, "original/base")
    assert summary["world_size"] == 2
    config = json.loads((case.target / "adapter_config.json").read_text())
    assert config["task_type"] == "CAUSAL_LM"
    assert config["r"] == 2
    weights = load_file(case.target / "adapter_model.safetensors")
    assert weights.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(weights[key], expected[key], rtol=0, atol=0)
