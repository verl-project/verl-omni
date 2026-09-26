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
"""LoRA checkpoints fused into complete transformers and pipelines."""

import copy
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from model_fixtures import run_forward, tiny_transformer
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file
from test_architectures_on_cpu import _case, _h3_native_state, _h3_pipeline_case, _tensor_outputs

from verl_omni.model_merger import ModelMergerConfig, merge_model, validate_artifact
from verl_omni.model_merger.architectures import _TRANSFORMERS
from verl_omni.model_merger.base_model_merger import generate_config_from_args, parse_args
from verl_omni.model_merger.fsdp_model_merger import _pipeline_class
from verl_omni.model_merger.lora import fuse_lora, plan_lora_fusion
from verl_omni.model_merger.utils import inventory, tree_files, weight_files, write_json

RANK, ALPHA = 4, 8


def _lora_actor(architecture):
    """Base-identical actor with two differently valued LoRA adapters on every linear layer."""
    actor = tiny_transformer(architecture)
    targets = [name for name, module in actor.named_modules() if isinstance(module, torch.nn.Linear)]
    for name in ("default", "old"):
        actor.add_adapter(LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=targets), adapter_name=name)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for key, value in actor.named_parameters():
            if "lora_" in key:
                value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
    actor.set_adapter("default")
    return actor


def _write_lora_source(source: Path, actor, lora_only: bool) -> None:
    state = actor.state_dict()
    if lora_only:
        # FSDPCheckpointManager(save_lora_only=True) keeps exactly these tensors.
        state = {key: value for key, value in state.items() if "lora_" in key}
    torch.save(state, source / "model_world_size_1_rank_0.pt")
    write_json(source / "lora_train_meta.json", {"r": RANK, "lora_alpha": ALPHA, "task_type": "CAUSAL_LM"})


def _lora_case(tmp_path, architecture, *, pipeline=False, lora_only=False):
    if pipeline and architecture == "MiniMaxH3Pipeline":
        config, _ = _h3_pipeline_case(tmp_path)
    else:
        config, _ = _case(
            tmp_path, architecture, pipeline=pipeline, dual_wan=pipeline and architecture == "WanPipeline"
        )
    actor = _lora_actor(architecture)
    if not lora_only:
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                if "lora_" not in name:
                    parameter.add_(0.005)
    _write_lora_source(Path(config.local_dir), actor, lora_only)
    return config, actor


def _peft_merged_state(actor, adapter_name="default"):
    merged = copy.deepcopy(actor)
    for module in merged.modules():
        if isinstance(module, BaseTunerLayer):
            module.merge(adapter_names=[adapter_name])
    return {key.replace(".base_layer.", "."): value for key, value in merged.state_dict().items() if "lora_" not in key}


def _published_state(root: Path, weights_name=None):
    values = {}
    for path in set(weight_files(root, weights_name).values()):
        values.update(load_file(path))
    return values


@pytest.mark.parametrize("lora_only", [False, True], ids=["full_state", "lora_only"])
@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_lora_component_matches_peft_merge_and_actor_forward(tmp_path, architecture, lora_only):
    config, actor = _lora_case(tmp_path, architecture, lora_only=lora_only)
    expected = run_forward(actor, architecture)
    result = merge_model(config)
    manifest = validate_artifact(result.output_dir)
    fusion = manifest["lora_fusion"]
    assert fusion["adapter_name"] == "default" and fusion["excluded_adapters"] == ["old"]
    assert fusion["r"] == RANK and fusion["lora_alpha"] == ALPHA and fusion["scaling"] == ALPHA / RANK
    assert fusion["base_weights"] == ("base_model" if lora_only else "checkpoint")
    assert fusion["fused_modules"] == sorted(
        name for name, module in actor.named_modules() if isinstance(module, BaseTunerLayer)
    )
    assert "lora_train_meta.json" in manifest["source_files"]

    torch.testing.assert_close(_published_state(result.output_dir), _peft_merged_state(actor), rtol=0, atol=0)
    loaded = type(actor).from_pretrained(result.output_dir, local_files_only=True)
    assert not any(isinstance(module, BaseTunerLayer) for module in loaded.modules())
    torch.testing.assert_close(run_forward(loaded, architecture), expected, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def lora_dtensor_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("lora-dtensors")
    cases = {}
    for architecture in sorted(_TRANSFORMERS):
        if architecture == "BooguImagePipeline" and importlib.util.find_spec("boogu") is None:
            continue
        for lora_only in (False, True):
            config, actor = _lora_case(root / architecture / str(lora_only), architecture, lora_only=lora_only)
            source = Path(config.local_dir)
            (source / "model_world_size_1_rank_0.pt").rename(source / "full.pt")
            write_json(source / "fsdp_config.json", {"world_size": 2, "FSDP_version": 2})
            cases[architecture, lora_only] = (config, actor)
    run = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("dtensor_checkpoint.py")),
            *(config.local_dir for config, _ in cases.values()),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    for config, _ in cases.values():
        (Path(config.local_dir) / "full.pt").unlink()
    return cases


@pytest.mark.parametrize("lora_only", [False, True], ids=["full_state", "lora_only"])
@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_lora_two_rank_dtensor_checkpoints(architecture, lora_only, request):
    if architecture == "BooguImagePipeline":
        pytest.importorskip("boogu")
    config, actor = request.getfixturevalue("lora_dtensor_cases")[architecture, lora_only]
    result = merge_model(config)
    torch.testing.assert_close(_published_state(result.output_dir), _peft_merged_state(actor), rtol=0, atol=0)
    loaded = type(actor).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(
        run_forward(loaded, architecture), run_forward(actor, architecture), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("lora_only", [False, True], ids=["full_state", "lora_only"])
@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS.keys() - {"MiniMaxH3Pipeline"}))
def test_lora_pipeline_reloads_with_unchanged_frozen_components(tmp_path, architecture, lora_only):
    config, actor = _lora_case(tmp_path, architecture, pipeline=True, lora_only=lora_only)
    before = inventory(Path(config.base_model), tree_files(Path(config.base_model)))
    result = merge_model(config)
    loaded = _pipeline_class(architecture).from_pretrained(result.output_dir, local_files_only=True)
    assert not any(isinstance(module, BaseTunerLayer) for module in loaded.transformer.modules())
    torch.testing.assert_close(loaded.transformer.state_dict(), _peft_merged_state(actor), rtol=0, atol=0)
    torch.testing.assert_close(
        run_forward(loaded.transformer, architecture),
        run_forward(actor, architecture),
        rtol=1e-5,
        atol=1e-5,
    )
    after = inventory(result.output_dir, tree_files(result.output_dir))
    for name, digest in before.items():
        if not name.startswith("transformer/"):
            assert after[name] == digest, name
    assert validate_artifact(result.output_dir)["lora_fusion"]["base_weights"] == (
        "base_model" if lora_only else "checkpoint"
    )


def test_lora_minimax_h3_native_pipeline_folds_before_conversion(tmp_path):
    config, actor = _lora_case(tmp_path, "MiniMaxH3Pipeline", pipeline=True)
    merged = tiny_transformer("MiniMaxH3Pipeline")
    merged.load_state_dict(_peft_merged_state(actor))
    result = merge_model(config)
    assert validate_artifact(result.output_dir)["artifact_type"] == "minimax_h3_pipeline"
    actual = _published_state(result.output_dir / "transformer", "model.safetensors")
    expected = _h3_native_state(merged)
    assert set(actual) == set(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lora_only_native_minimax_h3_pipeline_is_rejected(tmp_path):
    config, _ = _lora_case(tmp_path, "MiniMaxH3Pipeline", pipeline=True, lora_only=True)
    with pytest.raises(ValueError, match="native MiniMax H3"):
        merge_model(config)
    assert not Path(config.target_dir).exists()


def test_adapter_name_selects_one_adapter(tmp_path):
    config, actor = _lora_case(tmp_path, "FluxPipeline")
    result = merge_model(replace(config, adapter_name="old"))
    assert validate_artifact(result.output_dir)["lora_fusion"]["excluded_adapters"] == ["default"]
    torch.testing.assert_close(_published_state(result.output_dir), _peft_merged_state(actor, "old"), rtol=0, atol=0)
    missing = replace(config, adapter_name="missing", target_dir=str(tmp_path / "missing-output"))
    with pytest.raises(ValueError, match="adapter 'missing' not found"):
        merge_model(missing)
    assert not Path(missing.target_dir).exists()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("no_metadata", "require lora_train_meta.json"),
        ("bad_metadata", "positive integer r"),
        ("unpaired", "Unpaired LoRA"),
        ("old_unpaired", "Unpaired LoRA"),
        ("old_rank", "differs from lora_train_meta.json"),
        ("rank", "differs from lora_train_meta.json"),
        ("a_not_2d", "not a linear weight"),
        ("b_not_2d", "not a linear weight"),
        ("shape", "does not match the base weight"),
        ("dora", "Unsupported adapter tensor"),
        ("lora_bias", "Unsupported adapter tensor"),
        ("embedding", "Unsupported adapter tensor"),
        ("unknown_adapter", "Unsupported adapter tensor"),
        ("no_adapter", "not found in checkpoint"),
        ("no_base_layer", "lacks a base_layer weight"),
        ("duplicate_base", "Ambiguous LoRA base tensor"),
        ("partial", "Incomplete transformer state"),
        ("mixed", "Incomplete transformer state"),
    ],
)
def test_lora_checkpoint_faults_fail_closed(tmp_path, fault, message):
    config, actor = _lora_case(tmp_path, "QwenImagePipeline")
    source = Path(config.local_dir)
    state = dict(actor.state_dict())
    lora_a = next(key for key in state if ".lora_A.default." in key)
    if fault == "no_metadata":
        (source / "lora_train_meta.json").unlink()
    elif fault == "bad_metadata":
        write_json(source / "lora_train_meta.json", {"r": 0, "lora_alpha": ALPHA})
    elif fault == "unpaired":
        state.pop(lora_a.replace("lora_A", "lora_B"))
    elif fault == "old_unpaired":
        state.pop(lora_a.replace("lora_A.default", "lora_B.old"))
    elif fault == "old_rank":
        old_a = lora_a.replace(".default.", ".old.")
        state[old_a] = state[old_a][:1]
    elif fault == "rank":
        write_json(source / "lora_train_meta.json", {"r": RANK * 2, "lora_alpha": ALPHA})
    elif fault == "a_not_2d":
        state[lora_a] = state[lora_a].unsqueeze(-1)
    elif fault == "b_not_2d":
        lora_b = lora_a.replace("lora_A", "lora_B")
        state[lora_b] = state[lora_b].unsqueeze(-1)
    elif fault == "shape":
        state[lora_a] = state[lora_a][:, :1]
    elif fault == "dora":
        state[lora_a.replace("lora_A", "lora_magnitude_vector")] = torch.ones(1)
    elif fault == "lora_bias":
        state[lora_a.replace("lora_A", "lora_B").replace(".weight", ".bias")] = torch.zeros(1)
    elif fault == "embedding":
        state[lora_a.replace("lora_A.default.weight", "lora_embedding_A.default")] = torch.ones(1)
    elif fault == "unknown_adapter":
        state[lora_a.replace("lora_A", "adapter_custom")] = torch.ones(1)
    elif fault == "no_adapter":
        state = {key: value for key, value in state.items() if "lora_" not in key}
    elif fault in {"no_base_layer", "duplicate_base"}:
        base = lora_a.replace("lora_A.default", "base_layer")
        state[base.replace(".base_layer.", ".")] = state[base]
        if fault == "no_base_layer":
            state.pop(base)
    elif fault == "partial":
        state.pop(next(key for key in state if ".base_layer." in key))
    else:
        state = {key: value for key, value in state.items() if "lora_" in key or key.endswith(".bias")}
    torch.save(state, source / "model_world_size_1_rank_0.pt")
    with pytest.raises(ValueError, match=message):
        merge_model(config)
    assert not Path(config.target_dir).exists()


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_lora_only_nonfinite_base_weights_fail_closed(tmp_path, value):
    config, _ = _lora_case(tmp_path, "QwenImagePipeline", lora_only=True)
    mapping = weight_files(Path(config.base_model))
    key = next(name for name in mapping if name.endswith(".bias"))
    weights = load_file(mapping[key])
    weights[key].fill_(value)
    save_file(weights, str(mapping[key]), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="non-finite weights"):
        merge_model(config)
    assert not Path(config.target_dir).exists()


@pytest.mark.parametrize("field", ["r", "lora_alpha"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", None, float("inf"), float("nan")])
def test_lora_metadata_requires_positive_integers(field, value):
    metadata = {"r": RANK, "lora_alpha": ALPHA, field: value}
    with pytest.raises(ValueError, match="positive integer"):
        plan_lora_fusion({}, {}, metadata, "default")


def test_lora_non_linear_target_is_rejected():
    shapes = {"conv.lora_A.default.weight": (RANK, 4), "conv.lora_B.default.weight": (8, RANK)}
    with pytest.raises(ValueError, match="not a linear weight"):
        plan_lora_fusion(shapes, {"conv.weight": (8, 4, 1, 1)}, {"r": RANK, "lora_alpha": ALPHA}, "default")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lora_fusion_accumulates_in_fp32_before_casting(dtype):
    generator = torch.Generator().manual_seed(17)
    base, a, b = [torch.randn(shape, generator=generator).to(dtype) for shape in ((8, 8), (4, 8), (8, 4))]
    expected = (base.float() + 2 * (b.float() @ a.float())).to(dtype)
    torch.testing.assert_close(fuse_lora(base, a, b, 2), expected, rtol=0, atol=0)
    assert not torch.equal(expected, base + 2 * (b @ a))


@pytest.mark.parametrize("dtype, magnitude", [(torch.float32, 1e30), (torch.float16, 1e3)])
def test_lora_fusion_overflow_fails_closed(dtype, magnitude):
    base = torch.zeros(8, 8, dtype=dtype)
    a = torch.full((4, 8), magnitude, dtype=dtype)
    b = torch.full((8, 4), magnitude, dtype=dtype)
    with pytest.raises(ValueError, match="non-finite weights"):
        fuse_lora(base, a, b, 2)


@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
@pytest.mark.parametrize("lora_only", [False, True], ids=["full_state", "lora_only"])
def test_lora_dtype_policy_and_fp32_islands(tmp_path, architecture, lora_only):
    config, actor = _lora_case(tmp_path, architecture, lora_only=lora_only)
    result = merge_model(replace(config, dtype="bfloat16"))
    expected = _peft_merged_state(actor)
    islands = tuple(getattr(actor, "_keep_in_fp32_modules", None) or ())
    for name, value in expected.items():
        if value.is_floating_point():
            dtype = torch.float32 if any(part in name.split(".") for part in islands) else torch.bfloat16
            expected[name] = value.to(dtype)
    torch.testing.assert_close(_published_state(result.output_dir), expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def fsdp2_lora_cases(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fsdp2-lora-training")
    architectures = sorted(_TRANSFORMERS)
    if importlib.util.find_spec("boogu") is None:
        architectures.remove("BooguImagePipeline")
    cases = {}
    for architecture in architectures:
        root = tmp_path / architecture
        root.mkdir()
        if architecture == "MiniMaxH3Pipeline":
            config, _ = _h3_pipeline_case(root)
        else:
            config, _ = _case(root, architecture, pipeline=True, dual_wan=architecture == "WanPipeline")
        shutil.rmtree(config.local_dir)
        cases[architecture] = config

    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(filter(None, (str(Path.cwd()), env.get("PYTHONPATH")))),
        TORCH_COMPILE_DISABLE="1",
        TORCHINDUCTOR_DISABLE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("fsdp2_tiny_train_checkpoint.py")),
            str(tmp_path),
            *architectures,
            "--lora",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stdout + run.stderr

    return cases


@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_fsdp2_lora_training_checkpoints_fuse_to_the_trained_actor(architecture, request):
    if architecture == "BooguImagePipeline":
        pytest.importorskip("boogu")
    config = request.getfixturevalue("fsdp2_lora_cases")[architecture]
    root = Path(config.local_dir).parent
    evidence = torch.load(root / "expected.pt", weights_only=True)
    assert torch.isfinite(torch.tensor(evidence["loss"]))
    expected = evidence["after"]
    assert any(not torch.equal(before, after) for before, after in zip(evidence["before"], expected, strict=True))
    # The LoRA-only checkpoint takes frozen weights from a Diffusers transformer base.
    component_base = Path(config.base_model)
    if architecture == "MiniMaxH3Pipeline":
        component_base = root / "diffusers-base"
        tiny_transformer(architecture).save_pretrained(component_base)
    component = merge_model(
        replace(
            config,
            local_dir=str(root / "actor_lora_only"),
            hf_model_config_path=str(root / "actor_lora_only/huggingface"),
            target_dir=str(root / "component-output"),
            base_model=str(component_base),
            output_format="transformer",
        )
    )
    assert validate_artifact(component.output_dir)["lora_fusion"]["excluded_adapters"] == ["old"]
    loaded = type(tiny_transformer(architecture)).from_pretrained(component.output_dir, local_files_only=True)
    assert not any(isinstance(module, BaseTunerLayer) for module in loaded.modules())
    torch.testing.assert_close(_tensor_outputs(run_forward(loaded, architecture)), expected, rtol=1e-5, atol=1e-6)

    result = merge_model(config)
    assert validate_artifact(result.output_dir)["lora_fusion"]["base_weights"] == "checkpoint"
    if architecture == "MiniMaxH3Pipeline":
        actual = _published_state(result.output_dir / "transformer", "model.safetensors")
        torch.testing.assert_close(actual, _h3_native_state(loaded), rtol=0, atol=0)
        native_load = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("minimax_h3_native_load.py")), str(result.output_dir)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert native_load.returncode == 0, native_load.stdout + native_load.stderr
    else:
        pipeline = _pipeline_class(architecture).from_pretrained(result.output_dir, local_files_only=True)
        assert not any(isinstance(module, BaseTunerLayer) for module in pipeline.transformer.modules())
        torch.testing.assert_close(
            _tensor_outputs(run_forward(pipeline.transformer, architecture)), expected, rtol=1e-5, atol=1e-6
        )


def test_adapter_name_cli_and_validation(monkeypatch):
    common = ["--backend", "fsdp", "--local_dir", "actor", "--base_model", "base", "--trust-checkpoint"]
    monkeypatch.setattr(sys, "argv", ["model_merger", "merge", *common, "--adapter-name", "old"])
    assert generate_config_from_args(parse_args()).adapter_name == "old"
    monkeypatch.setattr(sys, "argv", ["model_merger", "merge", *common])
    assert generate_config_from_args(parse_args()).adapter_name == "default"
    with pytest.raises(ValueError, match="adapter_name"):
        ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir="actor",
            base_model="base",
            trust_checkpoint=True,
            adapter_name="default.weight",
        )
