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
"""Offline Diffusers export contracts and real tiny-model reload/forward parity."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from safetensors.torch import load_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from verl_omni.model_merger import ModelMergerConfig, merge_model, utils, validate_artifact
from verl_omni.model_merger.base_model_merger import generate_config_from_args, parse_args, run_model_merger
from verl_omni.model_merger.fsdp_model_merger import model_rank_files, reconstruct_tensor


def test_cli_common_arguments_live_in_base_model_merger(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            "actor",
            "--target_dir",
            "output",
            "--base_model",
            "base",
            "--trust-checkpoint",
        ],
    )
    config = generate_config_from_args(parse_args())
    assert config.operation == "merge" and config.backend == "fsdp"
    assert config.local_dir == "actor" and config.base_model == "base"
    assert config.hf_model_config_path == str(Path("actor/huggingface"))
    assert config.target_dir == "output" and not config.hf_upload
    assert not hasattr(config, "architecture") and not hasattr(config, "component")


@pytest.fixture(scope="module")
def base_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-qwen")
    torch.manual_seed(1)
    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=1,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    vae = AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1],
        num_res_blocks=1,
        temperal_downsample=[],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=260,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            bos_token_id=256,
            eos_token_id=256,
        ),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16),
    )
    encoder = Qwen2_5_VLForConditionalGeneration(config)
    vocabulary = dict(zip(sorted(ByteLevel.alphabet()), range(256), strict=True))
    vocabulary["<|endoftext|>"] = 256
    vocab = root / "vocab.json"
    utils.write_json(vocab, vocabulary)
    (root / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(vocab_file=str(vocab), merges_file=str(root / "merges.txt"))
    pipeline = QwenImagePipeline(
        transformer=transformer,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_encoder=encoder,
        tokenizer=tokenizer,
    )
    base = root / "base"
    pipeline.save_pretrained(base)
    (base / "LICENSE").write_text("Tiny synthetic fixture; no downloaded weights.\n")
    return base


@pytest.fixture
def case(tmp_path, base_pipeline):
    source = tmp_path / "actor"
    source.mkdir()
    model = QwenImageTransformer2DModel.from_pretrained(base_pipeline, subfolder="transformer")
    # Deliberately trained/non-base values; equality against the base would reject this export.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)
    model.save_config(source / "huggingface")
    state = model.state_dict()
    torch.save(state, source / "model_world_size_1_rank_0.pt")
    utils.write_json(source / "fsdp_config.json", {"world_size": 1, "FSDP_version": 2})
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(source),
        target_dir=str(tmp_path / "output"),
        base_model=str(base_pipeline),
        trust_checkpoint=True,
    )
    return config, state, model


def _forward(model):
    model.eval()
    model.set_attention_backend("native")
    generator = torch.Generator().manual_seed(7)
    return model(
        hidden_states=torch.randn(1, 4, 16, generator=generator),
        encoder_hidden_states=torch.randn(1, 3, 16, generator=generator),
        encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.bool),
        timestep=torch.tensor([0.5]),
        img_shapes=[[(1, 2, 2)]],
        return_dict=False,
    )[0]


@pytest.mark.parametrize("budget", [1024**2, 512])
def test_full_pipeline_round_trip_and_forward(case, budget):
    config, state, original = case
    result = merge_model(replace(config, max_shard_size=budget))
    manifest = validate_artifact(result.output_dir)
    assert manifest["verification"]["runtime"] == "not_run"
    assert manifest["verification"]["artifact_round_trip"] == "passed"
    assert config.local_dir not in result.manifest_path.read_text()
    assert config.base_model not in result.manifest_path.read_text()
    # This is a real complete pipeline loader, not a mocked transformer constructor.
    pipeline = QwenImagePipeline.from_pretrained(result.output_dir, local_files_only=True)
    loaded = pipeline.transformer
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(_forward(loaded), _forward(original), rtol=1e-6, atol=1e-6)
    for name in ("vae", "text_encoder", "scheduler", "tokenizer"):
        root = Path(config.base_model) / name
        for path in utils.tree_files(root):
            assert (result.output_dir / name / path.relative_to(root)).read_bytes() == path.read_bytes()
    assert (result.output_dir / "LICENSE").read_bytes() == (Path(config.base_model) / "LICENSE").read_bytes()
    assert not list(result.output_dir.parent.glob(".output.merge*"))


def test_verl_style_test_operation(case):
    config, _, _ = case
    result = merge_model(config)
    test_config = ModelMergerConfig(
        operation="test",
        backend="fsdp",
        test_hf_dir=str(result.output_dir),
        hf_upload_path="ignored/repository",
        private=True,
    )
    assert test_config.target_dir is None and test_config.hf_upload_path is None
    assert not test_config.private and not test_config.hf_upload
    manifest = run_model_merger(test_config)
    assert isinstance(manifest, dict) and manifest["verification"]["integrity"] == "passed"


def test_explicit_hf_model_config_path(case, tmp_path):
    config, _, _ = case
    original = Path(config.hf_model_config_path) / "config.json"
    override = tmp_path / "actor-config"
    override.mkdir()
    (override / "config.json").write_bytes(original.read_bytes())
    original.unlink()
    result = merge_model(
        replace(
            config,
            target_dir=str(tmp_path / "override-output"),
            hf_model_config_path=str(override),
        )
    )
    assert validate_artifact(result.output_dir)["verification"]["integrity"] == "passed"


def test_huggingface_upload_uses_verl_style_config(case, tmp_path, monkeypatch):
    config, _, _ = case
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    result = merge_model(
        replace(
            config,
            target_dir=str(tmp_path / "upload-output"),
            hf_upload_path="organization/model",
            private=True,
        )
    )
    assert result.output_dir.is_dir()
    assert calls == [
        ("create_repo", {"repo_id": "organization/model", "private": True, "exist_ok": True}),
        (
            "upload_folder",
            {
                "folder_path": str(result.output_dir),
                "repo_id": "organization/model",
                "repo_type": "model",
            },
        ),
    ]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_explicit_cast(case, dtype):
    config, state, _ = case
    result = merge_model(replace(config, dtype=dtype))
    mapping = utils.weight_files(result.output_dir / "transformer")
    loaded = {}
    for path in set(mapping.values()):
        loaded.update(load_file(path))
    for key in state:
        torch.testing.assert_close(loaded[key], state[key].to(getattr(torch, dtype)), rtol=0, atol=0)


def test_preserves_mixed_dtypes_and_integer_buffers(tmp_path):
    weights = [("fp32", torch.ones(3)), ("bf16", torch.ones(2).bfloat16()), ("count", torch.tensor(11))]
    root = tmp_path / "weights"
    specs = utils.write_weights(root, iter(weights), 100)
    loaded = load_file(root / utils.WEIGHTS_NAME)
    assert specs["count"]["dtype"] == "int64"
    for key, tensor in weights:
        torch.testing.assert_close(loaded[key], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "nan", "lora", "plain_shard", "replica"])
def test_invalid_checkpoint_leaves_no_output(case, fault):
    config, state, _ = case
    state = dict(state)
    first = next(iter(state))
    source = Path(config.local_dir)
    if fault == "missing":
        state.pop(first)
    elif fault == "extra":
        state["unknown"] = torch.ones(1)
    elif fault == "shape":
        state[first] = torch.ones(1)
    elif fault == "nan":
        state[first] = torch.full_like(state[first], float("nan"))
    elif fault == "lora":
        state["layer.lora_A.default.weight"] = torch.ones(1, 2)
    if fault in {"plain_shard", "replica"}:
        (source / "model_world_size_1_rank_0.pt").unlink()
        utils.write_json(source / "fsdp_config.json", {"world_size": 2, "FSDP_version": 2})
        torch.save(state, source / "model_world_size_2_rank_0.pt")
        state[first] = state[first][:1] if fault == "plain_shard" else state[first] + 1
        torch.save(state, source / "model_world_size_2_rank_1.pt")
    else:
        torch.save(state, source / "model_world_size_1_rank_0.pt")
    with pytest.raises(ValueError):
        merge_model(config)
    assert not Path(config.target_dir).exists()
    assert not list(source.parent.glob(".output.merge*"))


@pytest.mark.parametrize(
    "field,value", [("backend", "megatron"), ("dtype", "bad"), ("max_shard_size", 0), ("trust_checkpoint", False)]
)
def test_configuration_fails_closed(case, field, value):
    with pytest.raises(ValueError):
        replace(case[0], **{field: value})


def test_config_mismatch_even_with_same_weight_shapes(case):
    config, _, _ = case
    path = Path(config.local_dir) / "huggingface/config.json"
    data = utils.read_json(path)
    data["axes_dims_rope"] = [6, 4, 6]
    utils.write_json(path, data)
    with pytest.raises(ValueError, match="configuration mismatch"):
        merge_model(config)


def test_unknown_architecture_is_rejected(case, tmp_path):
    import shutil

    config, _, _ = case
    base = tmp_path / "unknown-base"
    shutil.copytree(config.base_model, base)
    path = base / "model_index.json"
    data = utils.read_json(path)
    data["_class_name"] = "UnknownPipeline"
    utils.write_json(path, data)
    with pytest.raises(ValueError, match="Unsupported publishing architecture"):
        merge_model(replace(config, base_model=str(base)))


def test_missing_and_extra_ranks(case):
    config, _, _ = case
    source = Path(config.local_dir)
    extra = source / "model_world_size_1_rank_1.pt"
    extra.touch()
    with pytest.raises(ValueError, match="rank files"):
        model_rank_files(source)
    extra.unlink()
    (source / "model_world_size_1_rank_0.pt").unlink()
    with pytest.raises(ValueError, match="rank files"):
        model_rank_files(source)


def test_fsdp1_checkpoint_is_rejected_upfront(case):
    config, _, _ = case
    source = Path(config.local_dir)
    utils.write_json(source / "fsdp_config.json", {"world_size": 1, "FSDP_version": 1})
    with pytest.raises(ValueError, match="Only FSDP2 checkpoints are supported"):
        model_rank_files(source)


def test_existing_target_and_input_overlap(case):
    config, _, _ = case
    with pytest.raises(FileExistsError):
        merge_model(replace(config, target_dir=config.local_dir))
    with pytest.raises(ValueError, match="overlap"):
        merge_model(replace(config, target_dir=str(Path(config.local_dir) / "output")))
    Path(config.target_dir).mkdir()
    (Path(config.target_dir) / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        merge_model(config)
    assert (Path(config.target_dir) / "keep").read_text() == "existing"


def test_publication_race_never_replaces_an_empty_target(tmp_path):
    target = tmp_path / "output"
    with pytest.raises(FileExistsError):
        with utils.publication_directory(target) as staging:
            (staging / "new").touch()
            target.mkdir()
    assert target.is_dir() and not list(target.iterdir())
    assert not list(tmp_path.glob(".output.merge*"))


def test_existing_lock_is_not_removed(tmp_path):
    lock = tmp_path / ".output.merge.lock"
    lock.write_text("another exporter")
    with pytest.raises(FileExistsError):
        with utils.publication_directory(tmp_path / "output"):
            pytest.fail("Must not acquire another exporter's lock")
    assert lock.read_text() == "another exporter"


def test_write_failure_and_written_tensor_corruption(case, monkeypatch):
    config, _, _ = case
    original = utils.save_file

    def corrupt(tensors, path, **kwargs):
        original({key: tensor + 1 for key, tensor in tensors.items()}, path, **kwargs)

    monkeypatch.setattr(utils, "save_file", corrupt)
    with pytest.raises(ValueError, match="round-trip"):
        merge_model(config)
    assert not Path(config.target_dir).exists()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        merge_model(config)
    assert not list(Path(config.target_dir).parent.glob(".output.merge*"))


def test_portable_validation_detects_corruption(case):
    result = merge_model(case[0])
    (result.output_dir / "LICENSE").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_artifact(result.output_dir)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a//b", "a\\b"])
def test_unsafe_index_paths(path):
    with pytest.raises(ValueError, match="path"):
        utils.relative_path(path)


def test_plain_replicas_and_unsupported_values():
    torch.testing.assert_close(reconstruct_tensor([torch.tensor(1), torch.tensor(1)], ()), torch.tensor(1))
    with pytest.raises(ValueError, match="replicas"):
        reconstruct_tensor([torch.tensor(1), torch.tensor(2)], ())
    with pytest.raises(ValueError, match="Dtype"):
        reconstruct_tensor([torch.ones(1), torch.ones(1).bfloat16()], (1,))
    with pytest.raises(ValueError):
        reconstruct_tensor([{"nested": torch.ones(1)}], (1,))


def test_real_cli_cpu_startup_and_export(case):
    config, _, _ = case
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    source = Path(config.local_dir)
    single = source / "model_world_size_1_rank_0.pt"
    single.rename(source / "full.pt")
    utils.write_json(source / "fsdp_config.json", {"world_size": 2, "FSDP_version": 2})
    generated = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("dtensor_checkpoint.py")), str(source)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert config.local_dir and config.target_dir and config.base_model
    command = [sys.executable, "-m", "verl_omni.model_merger"]
    run = subprocess.run(
        command
        + [
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            config.local_dir,
            "--target_dir",
            config.target_dir,
            "--base_model",
            config.base_model,
            "--trust-checkpoint",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert Path(config.target_dir, utils.MANIFEST_NAME).is_file()
    # Test with the actual verl-style CLI, not a test-only import bypass.
    run = subprocess.run(
        command + ["test", "--backend", "fsdp", "--test_hf_dir", config.target_dir],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert '"integrity": "passed"' in run.stdout


@pytest.fixture(scope="module")
def dtensor_ranks(tmp_path_factory):
    root = tmp_path_factory.mktemp("dtensor-shards")
    state = {
        "rows": torch.arange(15).reshape(5, 3).float(),
        "columns": torch.arange(15).reshape(3, 5).bfloat16(),
        "empty": torch.ones(1, 2),
        "scalar": torch.tensor(11),
        "boolean": torch.tensor(True),
    }
    torch.save(state, root / "full.pt")
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("dtensor_checkpoint.py")), str(root)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    ranks = [
        torch.load(root / f"model_world_size_2_rank_{rank}.pt", map_location="cpu", mmap=True, weights_only=False)
        for rank in range(2)
    ]
    assert not torch.distributed.is_initialized()
    return state, ranks


def test_real_serialized_dtensors_uneven_empty_and_replica(dtensor_ranks):
    state, ranks = dtensor_ranks
    for key, value in state.items():
        merged = reconstruct_tensor([rank[key] for rank in ranks], tuple(value.shape))
        torch.testing.assert_close(merged, value, rtol=0, atol=0)
    assert not torch.distributed.is_initialized()


@pytest.mark.parametrize("fault", ["replica", "partial", "placement", "extent", "mixed", "mesh"])
def test_dtensor_layout_errors(dtensor_ranks, fault):
    import copy

    from torch.distributed.tensor import Partial, Replicate, Shard

    _, ranks = dtensor_ranks
    key = "scalar" if fault == "replica" else "rows"
    values = [copy.deepcopy(rank[key]) for rank in ranks]
    if fault == "replica":
        values[1]._local_tensor.add_(1)
    elif fault == "partial":
        values[0]._spec.placements = (Partial(),)
    elif fault == "placement":
        values[1]._spec.placements = (Replicate(),)
    elif fault == "extent":
        values[1]._local_tensor = values[1]._local_tensor[:1]
    elif fault == "mixed":
        values[1] = values[1].to_local()
    elif fault == "mesh":
        values[0]._spec.placements = (Shard(0), Replicate())
    with pytest.raises(ValueError):
        reconstruct_tensor(values, tuple(ranks[0][key].shape))


@pytest.mark.parametrize("fault", ["component", "required_file", "architecture", "weight_index"])
def test_invalid_base(case, tmp_path, fault):
    import shutil

    config, _, _ = case
    base = tmp_path / "broken-base"
    shutil.copytree(config.base_model, base)
    if fault in {"component", "architecture"}:
        data = utils.read_json(base / "model_index.json")
        if fault == "component":
            data["vae"] = [None, None]
        else:
            data["_class_name"] = "UnsupportedPipeline"
        utils.write_json(base / "model_index.json", data)
    elif fault == "required_file":
        (base / "tokenizer/tokenizer_config.json").unlink()
    else:
        utils.write_json(base / "transformer" / utils.INDEX_NAME, {"weight_map": {"bad": "../outside.safetensors"}})
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_model(replace(config, base_model=str(base)))
    assert not Path(config.target_dir).exists()


def test_lora_metadata_rejects_even_an_apparently_complete_base(case):
    config, _, _ = case
    utils.write_json(Path(config.local_dir) / "lora_train_meta.json", {"r": 8, "lora_alpha": 16})
    with pytest.raises(ValueError, match="LoRA"):
        merge_model(config)


def test_cast_overflow_fails_instead_of_publishing_inf(case):
    config, state, _ = case
    state[next(iter(state))].fill_(1e20)
    torch.save(state, Path(config.local_dir) / "model_world_size_1_rank_0.pt")
    with pytest.raises(ValueError, match="cast produced non-finite"):
        merge_model(replace(config, dtype="float16"))
    assert not Path(config.target_dir).exists()


def test_input_mutation_during_export(case, monkeypatch):
    from verl_omni.model_merger import fsdp_model_merger

    config, _, _ = case
    original = fsdp_model_merger.write_weights

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = Path(config.local_dir) / "fsdp_config.json"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(fsdp_model_merger, "write_weights", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        merge_model(config)
    assert not Path(config.target_dir).exists()


def test_config_sanitization_and_hub_file_symlink(case, tmp_path):
    import shutil

    config, _, _ = case
    base = tmp_path / "cached-base"
    shutil.copytree(config.base_model, base)
    path = base / "transformer/config.json"
    data = utils.read_json(path)
    data["_name_or_path"] = "/private/training/base"
    utils.write_json(path, data)
    blob = tmp_path / "license-blob"
    (base / "LICENSE").rename(blob)
    (base / "LICENSE").symlink_to(blob)
    result = merge_model(replace(config, base_model=str(base)))
    assert not (result.output_dir / "LICENSE").is_symlink()
    assert (result.output_dir / "LICENSE").read_bytes() == blob.read_bytes()
    assert "_name_or_path" not in utils.read_json(result.output_dir / "transformer/config.json")
    manifest = validate_artifact(result.output_dir)
    assert "transformer/config.json" in manifest["config_transform"]["files"]
    assert "/private/" not in result.manifest_path.read_text()


def test_single_and_sharded_index_integrity(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    utils.save_file({"a": torch.ones(2)}, root / "part.safetensors")
    utils.write_json(root / utils.INDEX_NAME, {"weight_map": {"a": "part.safetensors", "b": "part.safetensors"}})
    with pytest.raises(ValueError, match="index"):
        utils.weight_files(root)
    utils.write_json(root / utils.INDEX_NAME, {"weight_map": {"a": "../part.safetensors"}})
    with pytest.raises(ValueError, match="path"):
        utils.weight_files(root)


def test_duplicate_json_keys_fail(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"world_size": 1, "world_size": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        utils.read_json(path)
