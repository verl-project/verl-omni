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
"""Every repository Diffusers architecture: real component and complete-pipeline round trips."""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import diffusers
import pytest
import torch
from model_fixtures import run_forward, tiny_pipeline, tiny_transformer

from verl_omni.model_merger import ModelMergerConfig, merge_model, validate_artifact
from verl_omni.model_merger.architectures import _PIPELINES, _TRANSFORMERS
from verl_omni.model_merger.fsdp_model_merger import (
    _h3_conversion_plan,
    _h3_native_name,
    _pipeline_class,
)
from verl_omni.model_merger.utils import inventory, read_json, tree_files, weight_files, write_json


def _tensor_outputs(output):
    return output if isinstance(output, tuple) else (output,)


def _h3_native_config(diffusers_config):
    renamed = {
        "num_refiner_layers": "token_refiner_num_layers",
        "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim",
        "audio_in_channels": "audio_latents_dim",
        "freq_dim": "timestep_input_dim",
        "time_embed_hidden_dim": "time_embed_hidden_size",
        "rope_freq_dim": "rope_inv_freq_len",
    }
    shared = (
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "patch_size",
        "text_dim",
        "time_embed_dim",
        "norm_eps",
        "qk_norm_eps",
        "final_norm_eps",
    )
    result = {name: diffusers_config[name] for name in shared}
    result.update({target: diffusers_config[source] for source, target in renamed.items()})
    result.update(
        _class_name="MiniMaxH3DiTModel",
        _diffusers_version=diffusers_config.get("_diffusers_version"),
        adaln_out_features=18 * diffusers_config["hidden_size"],
        final_adaln_out_features=2 * diffusers_config["hidden_size"],
    )
    return result


def _h3_native_state(model):
    state = model.state_dict()
    plan = _h3_conversion_plan({name: tuple(value.shape) for name, value in state.items()})
    heads = model.config.num_attention_heads
    head_dim = model.config.attention_head_dim
    ff_half = model.config.ffn_dim
    result = {}
    for target, (kind, names) in plan.items():
        values = [state[name] for name in names]
        if kind == "qkv":
            result[target] = torch.stack([value.reshape(heads, head_dim, -1) for value in values], dim=1).reshape(
                heads * 3 * head_dim, -1
            )
        elif kind == "geglu":
            up, gate = values[0].split(ff_half, dim=0)
            result[target] = torch.cat([gate, up], dim=0)
        else:
            result[target] = values[0]
    rope_len = model.config.rope_freq_dim
    result["rope.inv_freq"] = model.config.rope_theta ** (
        -(torch.arange(0, 2 * rope_len, 2, dtype=torch.float32) / (2 * rope_len))
    )
    return result


def _h3_pipeline_case(tmp_path):
    base_model = tiny_transformer("MiniMaxH3Pipeline")
    base = tmp_path / "base"
    transformer = base / "transformer"
    base.mkdir()
    native_config = _h3_native_config(dict(base_model.config))
    from verl_omni.model_merger import utils

    utils.write_weights(
        transformer,
        iter(_h3_native_state(base_model).items()),
        4096,
        weights_name="model.safetensors",
    )
    write_json(transformer / "config.json", native_config)
    components = {
        "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
        "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
        "processor": ["transformers", "Qwen3VLProcessor"],
    }
    for name in components:
        root = base / name
        root.mkdir()
        write_json(root / "config.json", {"_class_name": components[name][1]})
    write_json(base / "tokenizer/tokenizer_config.json", {"tokenizer_class": "Qwen2TokenizerFast"})
    write_json(
        base / "model_index.json",
        {
            "_class_name": "MiniMaxH3Pipeline",
            "_minimax_h3": {"schema_version": 1, "partition": "fl2va", "tasks": ["t2va", "fl2va"]},
            "scheduler": None,
            "transformer": ["diffusers", "MiniMaxH3DiTModel"],
            **components,
        },
    )
    model = tiny_transformer("MiniMaxH3Pipeline")
    model.load_state_dict(base_model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.005)
    source = tmp_path / "actor"
    source.mkdir()
    model.save_config(source / "huggingface")
    torch.save(model.state_dict(), source / "model_world_size_1_rank_0.pt")
    write_json(source / "fsdp_config.json", {"FSDP_version": 2, "world_size": 1})
    return ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(source),
        target_dir=str(tmp_path / "output"),
        base_model=str(base),
        output_format="pipeline",
        max_shard_size=4096,
        trust_checkpoint=True,
    ), model


def _case(tmp_path, architecture, pipeline=False, dual_wan=False):
    if architecture == "BooguImagePipeline":
        pytest.importorskip("boogu", reason="Install the optional boogu-image package to test its canonical class")
    model = tiny_transformer(architecture)
    base = tmp_path / "base"
    if pipeline:
        pipe = tiny_pipeline(architecture, model)
        if dual_wan:
            pipe.register_modules(transformer_2=tiny_transformer(architecture))
            pipe.register_to_config(boundary_ratio=0.875, expand_timesteps=True)
        pipe.save_pretrained(base)
    else:
        model.save_pretrained(base)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.005)
    source = tmp_path / "actor"
    source.mkdir()
    model.save_config(source / "huggingface")
    torch.save(model.state_dict(), source / "model_world_size_1_rank_0.pt")
    write_json(source / "fsdp_config.json", {"FSDP_version": 2, "world_size": 1})
    return ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(source),
        target_dir=str(tmp_path / "output"),
        base_model=str(base),
        output_format="pipeline" if pipeline else "transformer",
        max_shard_size=4096,
        trust_checkpoint=True,
    ), model


@pytest.fixture(scope="module")
def dtensor_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("architecture-dtensors")
    sources = {}
    for architecture in _TRANSFORMERS:
        if architecture == "BooguImagePipeline" and importlib.util.find_spec("boogu") is None:
            continue
        config, _ = _case(root / architecture, architecture)
        source = Path(config.local_dir)
        (source / "model_world_size_1_rank_0.pt").rename(source / "full.pt")
        write_json(source / "fsdp_config.json", {"world_size": 2, "FSDP_version": 2})
        sources[architecture] = source
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("dtensor_checkpoint.py")), *map(str, sources.values())],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return sources


@pytest.mark.parametrize("layout", ["single", "dtensor"])
@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_all_components_reload_and_forward(tmp_path, architecture, layout, request):
    config, model = _case(tmp_path, architecture)
    if layout == "dtensor":
        source = request.getfixturevalue("dtensor_sources")[architecture]
        model.load_state_dict(torch.load(source / "full.pt", weights_only=True))
        config = replace(config, local_dir=str(source))
    expected = run_forward(model, architecture)
    result = merge_model(config)
    manifest = validate_artifact(result.output_dir)
    assert manifest["artifact_type"] == "diffusers_transformer"
    assert manifest["tensor_directory"] == "."
    loaded = type(model).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(run_forward(loaded, architecture), expected, rtol=1e-6, atol=1e-6)
    assert not (result.output_dir / "model_index.json").exists()


@pytest.mark.parametrize("architecture", sorted(_PIPELINES - {"MiniMaxH3Pipeline"}))
def test_all_pipelines_reload_and_forward(tmp_path, architecture):
    config, model = _case(tmp_path, architecture, pipeline=True)
    before = inventory(Path(config.base_model), tree_files(Path(config.base_model)))
    result = merge_model(config)
    loaded = _pipeline_class(architecture).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.transformer.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        run_forward(loaded.transformer, architecture), run_forward(model, architecture), rtol=1e-6, atol=1e-6
    )
    after = inventory(result.output_dir, tree_files(result.output_dir))
    for name, digest in before.items():
        if not name.startswith("transformer/"):
            assert after[name] == digest, name
    assert validate_artifact(result.output_dir)["trained_components"] == ["transformer"]


def test_all_tiny_models_train_one_step_save_with_fsdp2_and_merge(tmp_path):
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
            config, _ = _case(
                root,
                architecture,
                pipeline=True,
                dual_wan=architecture == "WanPipeline",
            )
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
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stdout + run.stderr

    for architecture, config in cases.items():
        evidence = torch.load(tmp_path / architecture / "expected.pt", weights_only=True)
        assert torch.isfinite(torch.tensor(evidence["loss"]))
        expected = evidence["after"]
        if architecture == "MiniMaxH3Pipeline":
            diffusers_base = tmp_path / architecture / "diffusers-base"
            tiny_transformer(architecture).save_pretrained(diffusers_base)
            diffusers_result = merge_model(
                replace(
                    config,
                    target_dir=str(tmp_path / architecture / "diffusers-output"),
                    base_model=str(diffusers_base),
                    output_format="transformer",
                )
            )
            loaded = type(tiny_transformer(architecture)).from_pretrained(
                diffusers_result.output_dir,
                local_files_only=True,
            )
            torch.testing.assert_close(
                _tensor_outputs(run_forward(loaded, architecture)), expected, rtol=1e-5, atol=1e-6
            )
            native_result = merge_model(config)
            actual = {}
            for path in set(weight_files(native_result.output_dir / "transformer", "model.safetensors").values()):
                from safetensors.torch import load_file

                actual.update(load_file(path))
            native_expected = _h3_native_state(loaded)
            assert set(actual) == set(native_expected)
            for name in actual:
                torch.testing.assert_close(actual[name], native_expected[name], rtol=0, atol=0)
            native_load = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("minimax_h3_native_load.py")),
                    str(native_result.output_dir),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert native_load.returncode == 0, native_load.stdout + native_load.stderr
        else:
            result = merge_model(config)
            pipeline = _pipeline_class(architecture).from_pretrained(result.output_dir, local_files_only=True)
            torch.testing.assert_close(
                _tensor_outputs(run_forward(pipeline.transformer, architecture)), expected, rtol=1e-5, atol=1e-6
            )


def test_boogu_local_module_aliases_require_trust_and_remain_portable(tmp_path):
    config, _ = _case(tmp_path, "BooguImagePipeline", pipeline=True)
    root = Path(config.base_model)
    index = read_json(root / "model_index.json")
    index["transformer"][0] = "transformer_boogu"
    index["scheduler"][0] = "scheduling_flow_match_euler_discrete_time_shifting"
    write_json(root / "model_index.json", index)
    (root / "transformer/transformer_boogu.py").write_text(
        "from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel\n"
    )
    (root / "scheduler/scheduling_flow_match_euler_discrete_time_shifting.py").write_text(
        "from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting "
        "import FlowMatchEulerDiscreteScheduler\n"
    )
    with pytest.raises(ValueError, match="trust-remote-code"):
        merge_model(config)
    trusted = replace(config, target_dir=str(tmp_path / "trusted-output"), trust_remote_code=True)
    result = merge_model(trusted)
    assert validate_artifact(result.output_dir)["architecture"] == "BooguImagePipeline"
    assert (result.output_dir / "transformer/transformer_boogu.py").is_file()
    loaded = _pipeline_class("BooguImagePipeline").from_pretrained(
        result.output_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    assert type(loaded.transformer).__name__ == "BooguImageTransformer2DModel"


def test_wan_exports_both_transformers_without_a_component_choice(tmp_path):
    config, model = _case(tmp_path, "WanPipeline", pipeline=True, dual_wan=True)
    second = inventory(
        Path(config.base_model) / "transformer_2",
        tree_files(Path(config.base_model) / "transformer_2"),
    )
    result = merge_model(config)
    loaded = diffusers.WanPipeline.from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.transformer.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(run_forward(loaded.transformer, "WanPipeline"), run_forward(model, "WanPipeline"))
    assert (
        inventory(
            result.output_dir / "transformer_2",
            tree_files(result.output_dir / "transformer_2"),
        )
        == second
    )
    index = read_json(result.output_dir / "model_index.json")
    assert index["boundary_ratio"] == 0.875 and index["expand_timesteps"] is True
    assert validate_artifact(result.output_dir)["tensor_directory"] == "transformer"


@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_every_architecture_rejects_partial_checkpoint(tmp_path, architecture):
    config, model = _case(tmp_path, architecture)
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    torch.save(state, Path(config.local_dir) / "model_world_size_1_rank_0.pt")
    with pytest.raises(ValueError, match="Incomplete transformer state"):
        merge_model(config)
    assert not Path(config.target_dir).exists()


def test_architecture_is_inferred_and_conflicting_base_fails(tmp_path):
    config, _ = _case(tmp_path, "MiniMaxH3Pipeline")
    assert validate_artifact(merge_model(config).output_dir)["architecture"] == "MiniMaxH3Pipeline"
    data = read_json(Path(config.base_model) / "config.json")
    data["_class_name"] = "MiniMaxH3DiTModel"
    write_json(Path(config.base_model) / "config.json", data)
    with pytest.raises(ValueError, match="Unsupported publishing architecture"):
        merge_model(replace(config, target_dir=str(tmp_path / "bad-output")))


@pytest.mark.parametrize("layout", ["single", "dtensor"])
def test_minimax_h3_native_pipeline_conversion_is_complete(tmp_path, layout, request):
    config, model = _h3_pipeline_case(tmp_path)
    if layout == "dtensor":
        config = replace(config, local_dir=str(request.getfixturevalue("dtensor_sources")["MiniMaxH3Pipeline"]))
    expected = _h3_native_state(model)
    result = merge_model(config)
    manifest = validate_artifact(result.output_dir)
    assert manifest["artifact_type"] == "minimax_h3_pipeline"
    assert manifest["architecture"] == "MiniMaxH3Pipeline"
    actual = {}
    for path in set(weight_files(result.output_dir / "transformer", "model.safetensors").values()):
        from safetensors.torch import load_file

        actual.update(load_file(path))
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    source = model.state_dict()
    qkv = torch.stack(
        [
            source[f"transformer_blocks.0.attn.to_{name}.weight"].reshape(
                model.config.num_attention_heads,
                model.config.attention_head_dim,
                -1,
            )
            for name in ("q", "k", "v")
        ],
        dim=1,
    ).reshape(model.config.num_attention_heads * 3 * model.config.attention_head_dim, -1)
    torch.testing.assert_close(actual["blocks.0.attn.qkv_proj.weight"], qkv, rtol=0, atol=0)
    up, gate = source["transformer_blocks.0.ff.net.0.proj.weight"].split(model.config.ffn_dim, dim=0)
    torch.testing.assert_close(actual["blocks.0.mlp.fc1.weight"], torch.cat([gate, up]), rtol=0, atol=0)
    assert _h3_native_name("transformer_blocks.0.attn.to_out.0.weight") == "blocks.0.attn.out_proj.weight"
    for component in ("text_encoder", "tokenizer", "video_vae", "audio_vae", "processor"):
        assert (result.output_dir / component / "config.json").is_file()


def test_minimax_h3_custom_assets_require_explicit_trust(tmp_path):
    config, _ = _h3_pipeline_case(tmp_path)
    (Path(config.base_model) / "video_vae/native_module.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="trust-remote-code"):
        merge_model(config)
    result = merge_model(replace(config, target_dir=str(tmp_path / "trusted-output"), trust_remote_code=True))
    assert (result.output_dir / "video_vae/native_module.py").read_text() == "VALUE = 1\n"


def test_component_export_from_pipeline_does_not_copy_other_assets(tmp_path):
    config, model = _case(tmp_path, "FluxPipeline", pipeline=True)
    result = merge_model(replace(config, output_format="transformer"))
    loaded = type(model).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    assert not (result.output_dir / "vae").exists()
    assert not (result.output_dir / "model_index.json").exists()


@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_every_architecture_dtype_policy_and_fp32_islands(tmp_path, architecture):
    from safetensors.torch import load_file

    config, model = _case(tmp_path, architecture)
    result = merge_model(replace(config, dtype="bfloat16"))
    values = {}
    for path in set(weight_files(result.output_dir).values()):
        values.update(load_file(path))
    islands = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    for key, value in model.state_dict().items():
        expected = value
        if value.is_floating_point():
            dtype = torch.float32 if any(part in key.split(".") for part in islands) else torch.bfloat16
            expected = value.to(dtype)
        torch.testing.assert_close(values[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["option", "missing_model", "wrong_model", "wrong_tokenizer", "missing_processor"])
def test_pipeline_component_contract_is_not_just_an_architecture_allowlist(tmp_path, fault):
    arch = "QwenImageEditPlusPipeline" if fault == "missing_processor" else "WanPipeline"
    if fault == "wrong_tokenizer":
        arch = "FluxPipeline"
    config, _ = _case(tmp_path, arch, pipeline=True)
    root = Path(config.base_model)
    path = root / "model_index.json"
    index = read_json(path)
    if fault == "option":
        index["expand_timesteps"] = "false"
    elif fault == "missing_model":
        index["text_encoder"] = [None, None]
    elif fault == "wrong_model":
        index["text_encoder"] = ["transformers", "CLIPTextModel"]
    elif fault == "wrong_tokenizer":
        index["tokenizer"] = ["transformers", "T5Tokenizer"]
    else:
        index.pop("processor")
    write_json(path, index)
    with pytest.raises(ValueError):
        merge_model(config)
    assert not Path(config.target_dir).exists()


def test_component_directory_symlinks_cannot_escape_base(tmp_path):
    config, _ = _case(tmp_path, "FluxPipeline", pipeline=True)
    component = Path(config.base_model) / "transformer"
    external = tmp_path / "external-transformer"
    component.rename(external)
    component.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlinks"):
        merge_model(replace(config, output_format="transformer"))
    assert not Path(config.target_dir).exists()


def test_registry_covers_repository_diffusers_training_architectures():
    root = Path(__file__).resolve().parents[2] / "verl_omni/pipelines"
    found = set()
    for path in root.glob("*/diffusers_training_adapter.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "DiffusionModelBase":
                    found.add(ast.literal_eval(node.args[0]))
    # BAGEL and Lance build NonDiffusersModelBase and require native publishing,
    # not ModelMixin.save_pretrained, so neither appears in _TRANSFORMERS.
    assert found - {
        "OmniBagelForConditionalGeneration",
        "OmniLanceForConditionalGeneration",
    } == set(_TRANSFORMERS)
