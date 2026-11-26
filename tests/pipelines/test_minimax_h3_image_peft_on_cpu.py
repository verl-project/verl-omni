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
"""Exercise native H3 Image through the unchanged shared LoRA and export helpers."""

import json
from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel
from verl.utils import device as device_utils

from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import build_image_layout
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.diffusers_training_adapter import (
    MiniMaxH3ImageDiffusionNFT,
    _actor_model_class,
)
from verl_omni.utils import fsdp_utils
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

diffusers = pytest.importorskip("diffusers", minversion="0.43.0")
NativeModel = diffusers.MiniMaxH3ImageTransformer2DModel


def _model():
    torch.manual_seed(17)
    return _actor_model_class()(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=48,
        text_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=2,
    )


def _engine(model, adapter_path=None):
    engine = LoRAAdapterMixin()
    engine.model_config = SimpleNamespace(
        lora_rank=2,
        lora_alpha=4,
        lora_init_weights=True,
        target_modules=["to_q", "to_v"],
        target_parameters=None,
        exclude_modules=None,
        policy_state_adapters=["default", "old"],
        lora_adapter_path=adapter_path,
        use_shm=False,
    )
    engine.module = engine._build_lora_module(model.requires_grad_(False))
    return engine


def _inputs(head):
    torch.manual_seed(23)
    positions, tags, image_indices, text_indices, _ = build_image_layout(4, 4, 3)
    return {
        "hidden_states": torch.randn(1, 4, 96),
        "encoder_hidden_states": torch.randn(1, 3, 16),
        "timestep": torch.tensor([0.25]),
        "timestep_indices": torch.zeros_like(tags, dtype=torch.long),
        "token_tags": tags,
        "position_ids": positions,
        "text_indices": text_indices,
        "image_indices": image_indices,
        "pdd_step_index": head,
        "return_dict": False,
    }


@pytest.mark.parametrize("head", range(6))
def test_local_mixin_keeps_native_forward_and_zero_lora_outputs(head):
    local = _model()
    native = NativeModel.from_config(local.config)
    native.load_state_dict(local.state_dict(), strict=True)
    inputs = _inputs(head)
    with torch.no_grad():
        expected = native(**inputs)[0]
    engine = _engine(local)
    assert engine.module is local and not isinstance(local, PeftModel)
    assert local.forward.__func__ is NativeModel.forward
    assert not any(name.startswith("base_model.") for name, _ in local.named_parameters())
    actual = local(**inputs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    local.enable_gradient_checkpointing()
    local.train()
    local(**inputs)[0].square().mean().backward()
    grads = [param.grad for param in local.parameters() if param.requires_grad and param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize("adapter_name", ["default", "old"])
def test_layered_export_covers_both_block_stacks(monkeypatch, adapter_name):
    model = _engine(_model()).module
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                parameter.fill_(0.125 if ".default." in name else 0.25)
    blocks = set(model.transformer_blocks) | set(model.token_refiner.refiner_blocks)
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda module: 2 if module in blocks else 0)
    monkeypatch.setattr(FullyShardedDataParallel, "summon_full_params", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(device_utils, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    actual = fsdp_utils._layered_summon_lora_params_diffusers(
        model,
        adapter_name=adapter_name,
        layer_prefixes=["transformer_blocks.", "token_refiner.refiner_blocks."],
    )
    expected = get_peft_model_state_dict(model, adapter_name=adapter_name)
    assert actual and set(actual) == set(expected)
    assert any("transformer_blocks." in key for key in actual)
    assert any("token_refiner.refiner_blocks." in key for key in actual)
    for name, tensor in actual.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_native_lora_optimizer_copy_reference_and_reload(tmp_path):
    engine = _engine(_model())
    model = engine.module
    inputs = _inputs(2)
    with torch.no_grad():
        baseline = model(**inputs)[0].clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(**inputs)[0].square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        changed = model(**inputs)[0].clone()
    assert not torch.equal(changed, baseline)
    engine.copy_adapter("default", "old")
    with engine.use_adapter("old"), torch.no_grad():
        torch.testing.assert_close(model(**inputs)[0], changed)
    with pytest.raises(RuntimeError, match="sentinel"):
        with engine.disable_adapter(), torch.no_grad():
            torch.testing.assert_close(model(**inputs)[0], baseline)
            raise RuntimeError("sentinel")
    assert model.active_adapters() == ["default"]
    model.save_lora_adapter(tmp_path)
    restored = _model()
    restored.load_lora_adapter(tmp_path, prefix=None, use_safetensors=True, adapter_name="default")
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs)[0], changed)


@pytest.mark.parametrize("config_path", [False, True])
def test_registry_loading_preserves_dtype_and_native_checkpoint(tmp_path, config_path, monkeypatch):
    native = NativeModel.from_config(_model().config)
    native.save_pretrained(tmp_path / "transformer")
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.model_config = SimpleNamespace(
        architecture="MiniMaxH3ImagePipeline",
        algorithm="diffusion_nft",
        external_lib=None,
        config_path=str(tmp_path / "transformer") if config_path else None,
        local_path=str(tmp_path),
        transformer_subfolder="transformer",
        attn_backend="native",
        enable_gradient_checkpointing=True,
        trust_remote_code=False,
    )
    engine.engine_config = SimpleNamespace(model_dtype="bf16", forward_only=False)
    engine.device_mesh = None
    monkeypatch.setattr(
        "verl_omni.workers.engine.fsdp.diffusers_impl.get_init_weight_context_manager", lambda **kwargs: nullcontext
    )
    model = engine._build_module()
    assert hasattr(model, "add_adapter") and model.is_gradient_checkpointing
    for name, param in model.named_parameters():
        keep = any(part in model._keep_in_fp32_modules for part in name.split("."))
        dtype = torch.float32 if keep else torch.bfloat16
        assert param.dtype == dtype, name
        torch.testing.assert_close(param, dict(native.named_parameters())[name].to(dtype), rtol=0, atol=0)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert model.can_generate() is False
    model.config.save_pretrained(checkpoint)
    config = NativeModel.load_config(checkpoint)
    assert config["_class_name"] == NativeModel.__name__
    assert config["num_pdd_heads"] == 6
    restored = diffusers.AutoModel.from_config(checkpoint)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert type(restored) is NativeModel


@pytest.mark.parametrize("expected_subfolder", ["", "transformer"])
def test_native_actor_config_from_hub(tmp_path, monkeypatch, expected_subfolder):
    repo_id = "example/h3-image"
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"_class_name": NativeModel.__name__}))

    def cached_download(repo, *, filename, subfolder, **kwargs):
        assert (repo, filename, subfolder) == (repo_id, "config.json", expected_subfolder)
        return str(config_file)

    monkeypatch.setattr("diffusers.configuration_utils.hf_hub_download", cached_download)
    config = SimpleNamespace(
        config_path=None if expected_subfolder else repo_id,
        local_path=repo_id if expected_subfolder else None,
        transformer_subfolder="transformer",
    )
    assert MiniMaxH3ImageDiffusionNFT.diffusers_model_class(config) is _actor_model_class()


def test_legacy_checkpoint_still_uses_default_loader(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Transformer3DModel"}))
    config = SimpleNamespace(config_path=str(tmp_path), local_path=None)
    monkeypatch.setattr(_actor_model_class(), "from_pretrained", lambda *a, **kw: pytest.fail("native loader called"))
    assert MiniMaxH3ImageDiffusionNFT.diffusers_model_class(config) is None


@pytest.mark.parametrize("safe", [True, False], ids=["safetensors-only", "bin-with-optimizer-safetensors"])
def test_engine_warm_start_loads_actual_adapter(tmp_path, safe):
    from safetensors.torch import save_file

    model = _engine(_model()).module
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_" in name and ".default." in name:
                param.uniform_(0.02, 0.12)
    expected = get_peft_model_state_dict(model)
    diffusers.StableDiffusion3Pipeline.save_lora_weights(
        tmp_path, transformer_lora_layers=expected, safe_serialization=safe
    )
    if not safe:
        save_file({"optimizer": torch.ones(1)}, tmp_path / "optimizer.safetensors")
    restored = _engine(_model(), adapter_path=str(tmp_path)).module
    actual = get_peft_model_state_dict(restored)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_default_prefix_would_omit_refiner():
    from verl_omni.workers.config import DiffusionModelConfig

    fields_by_name = {field.name: field for field in fields(DiffusionModelConfig)}
    prefixes = fields_by_name["fsdp_layer_prefixes"].default_factory()
    config = SimpleNamespace(lora_rank=2, target_modules=["to_q"], fsdp_layer_prefixes=prefixes)
    with pytest.raises(ValueError, match="token_refiner"):
        MiniMaxH3ImageDiffusionNFT.validate_lora_config(config)
    config.fsdp_layer_prefixes = ["transformer_blocks.", "token_refiner.refiner_blocks."]
    MiniMaxH3ImageDiffusionNFT.validate_lora_config(config)


def _check_sharded_actor(rank, directory, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2,))
        engine = object.__new__(PPODiffusersFSDPEngine)
        engine.device_mesh = mesh
        engine.engine_config = SimpleNamespace(model_dtype="bf16", forward_only=False)
        engine.model_config = SimpleNamespace(
            architecture="MiniMaxH3ImagePipeline",
            algorithm="diffusion_nft",
            external_lib=None,
            config_path=str(directory / str(rank)),
            attn_backend="native",
            enable_gradient_checkpointing=True,
            trust_remote_code=False,
        )
        model = _engine(engine._build_module()).module
        assert all(param.is_meta == bool(rank) for param in model.parameters())
        layout = [(name, param.dtype) for name, param in model.named_parameters()]
        layouts = [None, None]
        dist.all_gather_object(layouts, layout)
        assert layouts[0] == layouts[1]
        assert model.proj_in.weight.dtype == torch.float32
        assert model.transformer_blocks[0].attn.to_q.base_layer.weight.dtype == torch.bfloat16
        state = model.state_dict() if rank == 0 else {}
        for block in (*model.transformer_blocks, *model.token_refiner.refiner_blocks):
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        if rank:
            model.to_empty(device="cpu")
        # DCP replaces entries in its input mapping with local shards. Keep the full tensors for comparison.
        set_model_state_dict(
            model, dict(state), options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        )
        for name, param in model.named_parameters():
            gathered = param.full_tensor()
            assert torch.isfinite(gathered).all(), name
            if rank == 0:
                torch.testing.assert_close(gathered, state[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires the CPU Gloo backend")
def test_two_fsdp_ranks_load_only_one_copy_of_native_weights(tmp_path):
    from shutil import copyfile

    _model().save_pretrained(tmp_path / "0")
    (tmp_path / "1").mkdir()
    copyfile(tmp_path / "0/config.json", tmp_path / "1/config.json")
    torch.multiprocessing.spawn(_check_sharded_actor, args=(tmp_path, (tmp_path / "gloo").as_uri()), nprocs=2)
