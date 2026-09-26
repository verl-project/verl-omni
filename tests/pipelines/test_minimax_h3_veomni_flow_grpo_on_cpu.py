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

"""CPU parity checks for the VeOmni-native MiniMax H3 FlowGRPO actor."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.pipelines.minimax_h3_flow_grpo.veomni_training_adapter import (
    is_veomni_module,
    predict_veomni,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin

veomni_lora = pytest.importorskip("veomni.lora")
VeOmniDiffusionEngine = import_module("verl_omni.workers.engine.veomni.diffusion_impl").VeOmniDiffusionEngine

veomni_h3_config = pytest.importorskip(
    "veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer"
)
veomni_h3_model = pytest.importorskip(
    "veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer"
)

_HIDDEN = 48
_HEADS = 4
_HEAD_DIM = 16
_FFN = 64
_TEXT_DIM = 32
_FUSED_TARGETS = ["qkv_proj", "out_proj", "fc1", "fc2"]
_SPLIT_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


def _build_models():
    diffusers_model = MiniMaxH3Transformer3DModel(
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        hidden_size=_HIDDEN,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=_FFN,
        text_dim=_TEXT_DIM,
        freq_dim=16,
        time_embed_hidden_dim=_HIDDEN,
        time_embed_dim=32,
        rope_freq_dim=2,
        in_channels=24,
        audio_in_channels=32,
    )
    config = veomni_h3_config.MiniMaxH3DiTModelConfig(
        hidden_size=_HIDDEN,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        ffn_hidden_size=_FFN,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=_TEXT_DIM,
        timestep_input_dim=16,
        time_embed_hidden_size=_HIDDEN,
        time_embed_dim=32,
        adaln_out_features=18 * _HIDDEN,
        final_adaln_out_features=2 * _HIDDEN,
        rope_inv_freq_len=2,
    )
    native_model = veomni_h3_model.MiniMaxH3DiTModel(config)
    native_model.load_state_dict(_fuse_diffusers_state(diffusers_model.state_dict()), strict=True)
    return diffusers_model, native_model


def _native_name(name: str) -> str:
    top_level = {
        "proj_in": "video_patch_proj",
        "audio_proj_in": "audio_patch_proj",
        "context_embedder": "condition_proj",
        "time_embedder.linear_1": "time_embedder.proj_in",
        "time_embedder.linear_2": "time_embedder.proj_out",
        "norm_out.norm": "final_layer.norm",
        "norm_out.linear": "final_layer.adaln_proj.linear",
        "proj_out": "final_layer.video_out",
        "audio_proj_out": "final_layer.audio_out",
    }
    for source, target in top_level.items():
        if name == source or name.startswith(source + "."):
            return target + name[len(source) :]
    return (
        name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
        .replace("transformer_blocks.", "blocks.")
        .replace(".attn.norm_q.", ".attn.q_norm.")
        .replace(".attn.norm_k.", ".attn.k_norm.")
        .replace(".attn.to_out.0.", ".attn.out_proj.")
        .replace(".ff.net.2.", ".mlp.fc2.")
    )


def _fuse_diffusers_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fused = {}
    consumed = set()
    for name, tensor in state.items():
        if name in consumed:
            continue
        if name.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
            block = name.rsplit(".attn.to_", 1)[0]
            names = [f"{block}.attn.to_{part}.weight" for part in ("q", "k", "v")]
            projections = [state[key].reshape(_HEADS, _HEAD_DIM, -1) for key in names]
            target = f"{_native_name(block)}.attn.qkv_proj.weight"
            fused[f"dit.{target}"] = torch.stack(projections, dim=1).reshape(_HEADS * 3 * _HEAD_DIM, -1)
            consumed.update(names)
        elif name.endswith(".ff.net.0.proj.weight"):
            up, gate = tensor.chunk(2, dim=0)
            target = _native_name(name).replace(".ff.net.0.proj.weight", ".mlp.fc1.weight")
            fused[f"dit.{target}"] = torch.cat((gate, up))
            consumed.add(name)
        else:
            fused[f"dit.{_native_name(name)}"] = tensor
            consumed.add(name)
    assert consumed == set(state)
    return fused


def _logical_inputs(batch_size: int = 1) -> dict[str, torch.Tensor]:
    video_rows, audio_rows, text_len = 2, 4, 2
    seq_len = video_rows + audio_rows + text_len
    return {
        "hidden_states": torch.randn(batch_size, video_rows, 96),
        "audio_hidden_states": torch.randn(batch_size, audio_rows, 32),
        "encoder_hidden_states": torch.randn(batch_size, text_len, _TEXT_DIM),
        "timestep": torch.tensor([0.25, 0.5]),
        "timestep_indices": torch.tensor([0, 0, 1, 1, 1, 1, 0, 0]),
        "token_tags": torch.tensor([0, 0, 2, 2, 2, 2, 1, 1]),
        "position_ids": torch.randn(seq_len, 3),
        "video_indices": torch.tensor([0, 1]),
        "audio_indices": torch.tensor([2, 3, 4, 5]),
        "text_indices": torch.tensor([6, 7]),
        "return_dict": False,
        "_h3_video_update_mask": torch.ones(video_rows, dtype=torch.bool),
    }


def test_native_h3_forward_matches_diffusers(monkeypatch):
    core = pytest.importorskip("veomni.models.diffusers.minimax_h3.minimax_h3_core.core")
    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
    torch.manual_seed(7)
    diffusers_model, native_model = _build_models()
    logical = _logical_inputs()

    diffusers_model.eval()
    native_model.eval()
    with torch.no_grad():
        expected = diffusers_model(**{key: value for key, value in logical.items() if not key.startswith("_h3_")})
        actual = predict_veomni(native_model, logical, use_gradient_checkpointing=False)

    assert is_veomni_module(native_model)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_native_h3_lora_forward_and_gradients_match_diffusers(monkeypatch):
    core = pytest.importorskip("veomni.models.diffusers.minimax_h3.minimax_h3_core.core")
    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
    torch.manual_seed(11)
    diffusers_model, native_model = _build_models()
    diffusers_model.add_adapter(
        LoraConfig(r=4, lora_alpha=8, target_modules=_SPLIT_TARGETS),
        adapter_name="default",
    )
    native_lora = veomni_lora.VeOmniLoraModel(
        native_model,
        veomni_lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=_FUSED_TARGETS),
    )
    with torch.no_grad():
        for name, param in native_lora.named_parameters():
            if ".lora_" in name:
                param.normal_(std=0.05)

    native_dit = native_lora.get_base_model().dit
    block_pairs = (
        (diffusers_model.token_refiner.refiner_blocks[0], native_dit.token_refiner.blocks[0]),
        (diffusers_model.transformer_blocks[0], native_dit.blocks[0]),
    )
    with torch.no_grad():
        for diffusers_block, native_block in block_pairs:
            native_qkv = native_block.attn.qkv_proj
            qkv_b = native_qkv.lora_B["default"].weight.view(_HEADS, 3, _HEAD_DIM, -1)
            for index, projection in enumerate(
                (diffusers_block.attn.to_q, diffusers_block.attn.to_k, diffusers_block.attn.to_v)
            ):
                projection.lora_A["default"].weight.copy_(native_qkv.lora_A["default"].weight)
                projection.lora_B["default"].weight.copy_(qkv_b[:, index].reshape(_HEADS * _HEAD_DIM, -1))
            diffusers_block.attn.to_out[0].lora_A["default"].weight.copy_(
                native_block.attn.out_proj.lora_A["default"].weight
            )
            diffusers_block.attn.to_out[0].lora_B["default"].weight.copy_(
                native_block.attn.out_proj.lora_B["default"].weight
            )
            native_fc1 = native_block.mlp.fc1
            gate_b, up_b = native_fc1.lora_B["default"].weight.chunk(2)
            diffusers_block.ff.net[0].proj.lora_A["default"].weight.copy_(native_fc1.lora_A["default"].weight)
            diffusers_block.ff.net[0].proj.lora_B["default"].weight.copy_(torch.cat((up_b, gate_b)))
            diffusers_block.ff.net[2].lora_A["default"].weight.copy_(native_block.mlp.fc2.lora_A["default"].weight)
            diffusers_block.ff.net[2].lora_B["default"].weight.copy_(native_block.mlp.fc2.lora_B["default"].weight)

    logical = _logical_inputs()
    expected = diffusers_model(**{key: value for key, value in logical.items() if not key.startswith("_h3_")})
    actual = predict_veomni(native_lora, logical, use_gradient_checkpointing=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sum(tensor.float().square().mean() for tensor in expected).backward()
    sum(tensor.float().square().mean() for tensor in actual).backward()

    for diffusers_block, native_block in block_pairs:
        native_qkv = native_block.attn.qkv_proj
        split_qkv = (diffusers_block.attn.to_q, diffusers_block.attn.to_k, diffusers_block.attn.to_v)
        expected_a_grad = sum(projection.lora_A["default"].weight.grad for projection in split_qkv)
        expected_b_grad = torch.stack(
            [projection.lora_B["default"].weight.grad.reshape(_HEADS, _HEAD_DIM, -1) for projection in split_qkv],
            dim=1,
        ).reshape_as(native_qkv.lora_B["default"].weight.grad)
        torch.testing.assert_close(native_qkv.lora_A["default"].weight.grad, expected_a_grad)
        torch.testing.assert_close(native_qkv.lora_B["default"].weight.grad, expected_b_grad)

        native_fc1 = native_block.mlp.fc1
        diffusers_fc1 = diffusers_block.ff.net[0].proj
        up_grad, gate_grad = diffusers_fc1.lora_B["default"].weight.grad.chunk(2)
        torch.testing.assert_close(
            native_fc1.lora_A["default"].weight.grad, diffusers_fc1.lora_A["default"].weight.grad
        )
        torch.testing.assert_close(
            native_fc1.lora_B["default"].weight.grad,
            torch.cat((gate_grad, up_grad)),
        )
        for diffusers_layer, native_layer in (
            (diffusers_block.attn.to_out[0], native_block.attn.out_proj),
            (diffusers_block.ff.net[2], native_block.mlp.fc2),
        ):
            torch.testing.assert_close(
                native_layer.lora_A["default"].weight.grad,
                diffusers_layer.lora_A["default"].weight.grad,
            )
            torch.testing.assert_close(
                native_layer.lora_B["default"].weight.grad,
                diffusers_layer.lora_B["default"].weight.grad,
            )


def test_native_h3_forward_rejects_multi_sample_micro_batches():
    _, native_model = _build_models()
    with pytest.raises(ValueError, match="micro-batch size 1"):
        predict_veomni(native_model, _logical_inputs(batch_size=2), use_gradient_checkpointing=False)


@pytest.mark.parametrize(
    "implementation,backend",
    [
        ("eager", "native"),
        ("flash_attention_2", "flash"),
        ("flash_attention_3", "_flash_3"),
        ("flash_attention_2_hub", "flash_hub"),
        ("flash_attention_3_hub", "_flash_3_hub"),
    ],
)
def test_h3_attention_honors_config_without_changing_other_models(monkeypatch, implementation, backend):
    import diffusers.models.attention_dispatch as dispatch
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import core

    from verl_omni.workers.engine.veomni.patch import _apply_attention_backend

    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
    checked, loaded, calls = [], [], []
    monkeypatch.setattr(dispatch, "_check_attention_backend_requirements", checked.append)
    monkeypatch.setattr(dispatch, "_maybe_download_kernel_for_backend", loaded.append)
    dispatch_fn = dispatch.dispatch_attention_fn

    def record(query, key, value, **kwargs):
        calls.append(kwargs.pop("backend").value)
        return dispatch_fn(query, key, value, backend=dispatch.AttentionBackendName.NATIVE, **kwargs)

    monkeypatch.setattr(dispatch, "dispatch_attention_fn", record)
    torch.manual_seed(19)
    reference, native = _build_models()
    untouched = _build_models()[1]
    original_forward = untouched.dit.blocks[0].attn.forward.__func__
    state_keys = set(native.state_dict())
    global_backend = dispatch._AttentionBackendRegistry._active_backend
    _apply_attention_backend(native, implementation)
    logical = _logical_inputs()
    expected = reference(**{key: value for key, value in logical.items() if not key.startswith("_h3_")})
    actual = predict_veomni(native, logical, use_gradient_checkpointing=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sum(x.square().mean() for x in actual).backward()
    assert native.dit.blocks[0].attn.qkv_proj.weight.grad.isfinite().all()
    assert calls and set(calls) == {backend}
    assert [x.value for x in checked] == [backend]
    assert [x.value for x in loaded] == [backend]
    assert core.ATTENTION_IMPLEMENTATION == "torch"
    assert dispatch._AttentionBackendRegistry._active_backend == global_backend
    assert untouched.dit.blocks[0].attn.forward.__func__ is original_forward
    assert set(native.state_dict()) == state_keys


def test_h3_attention_kernel_load_failure_does_not_install_partial_patch(monkeypatch):
    import diffusers.models.attention_dispatch as dispatch

    from verl_omni.workers.engine.veomni.patch import _apply_attention_backend

    _, native = _build_models()
    original_forward = native.dit.blocks[0].attn.forward.__func__
    monkeypatch.setattr(dispatch, "_check_attention_backend_requirements", lambda _: None)

    def fail(_):
        raise RuntimeError("kernel unavailable")

    monkeypatch.setattr(dispatch, "_maybe_download_kernel_for_backend", fail)
    with pytest.raises(RuntimeError, match="kernel unavailable"):
        _apply_attention_backend(native, "flash_attention_3_hub")
    assert native.dit.blocks[0].attn.forward.__func__ is original_forward


@pytest.mark.parametrize("implementation", ["sdpa", "flex_attention", "invalid"])
def test_h3_attention_rejects_unsupported_backends(implementation):
    from verl_omni.workers.engine.veomni.patch import _apply_attention_backend

    _, native = _build_models()
    with pytest.raises(ValueError, match="Unsupported H3 VeOmni attention"):
        _apply_attention_backend(native, implementation)


@pytest.mark.parametrize("cu_seqlens,use_ulysses", [((0, 2, 8), False), ((0, 8), True)])
def test_h3_attention_bridge_rejects_packing_and_sp(cu_seqlens, use_ulysses):
    from verl_omni.workers.engine.veomni.patch import _apply_attention_backend

    _, native = _build_models()
    _apply_attention_backend(native, "eager")
    with pytest.raises(ValueError, match="one sample and Ulysses SP=1"):
        native.dit.blocks[0].attn(
            torch.randn(8, _HIDDEN), rope_cos=None, rope_sin=None, cu_seqlens=cu_seqlens, use_ulysses=use_ulysses
        )


def test_h3_attention_uses_upstream_fix_when_available():
    from verl_omni.workers.engine.veomni.patch import _apply_attention_backend

    _, native = _build_models()
    native._load_attention_kernel = lambda: None
    original_forward = native.dit.blocks[0].attn.forward.__func__
    _apply_attention_backend(native, "flash_attention_3")
    assert native.dit.blocks[0].attn.forward.__func__ is original_forward


def test_h3_lora_validation_accepts_one_naming_layout_and_rejects_mixed():
    for targets in (
        ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        _FUSED_TARGETS,
    ):
        MiniMaxH3FlowGRPO.validate_lora_config(SimpleNamespace(lora_rank=4, target_modules=targets))

    with pytest.raises(ValueError, match="one complete projection naming layout"):
        MiniMaxH3FlowGRPO.validate_lora_config(SimpleNamespace(lora_rank=4, target_modules=["to_q", "qkv_proj"]))


def test_h3_lora_layout_extends_the_native_qkv_mapping():
    mapper = object.__new__(MiniMaxH3WeightSyncMixin)
    native_qkv = [
        (".attn.qkv_proj", ".attn.to_q", "q"),
        (".attn.qkv_proj", ".attn.to_k", "k"),
        (".attn.qkv_proj", ".attn.to_v", "v"),
    ]
    mapper.transformer = SimpleNamespace(stacked_params_mapping=tuple(native_qkv))

    mapper.install_h3_lora_layout()
    mapper.install_h3_lora_layout()

    assert mapper.transformer.stacked_params_mapping == [
        *native_qkv,
        (".fc1", ".fc1_0", "0"),
        (".fc1", ".fc1_1", "1"),
    ]


def test_native_h3_lora_export_binds_to_fused_rollout_modules(monkeypatch):
    _, native_model = _build_models()
    lora_model = veomni_lora.VeOmniLoraModel(
        native_model,
        veomni_lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=_FUSED_TARGETS),
    )
    with torch.no_grad():
        for index, (name, param) in enumerate(lora_model.named_parameters()):
            if ".lora_" in name:
                values = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param)
                param.copy_((values + index + 1) / (param.numel() + index + 1))

    engine = object.__new__(VeOmniDiffusionEngine)
    engine.module = lora_model
    engine.model_config = SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm="flow_grpo")
    engine.engine_config = SimpleNamespace(model_dtype="fp32")
    engine._is_lora = True
    engine._is_offload_param = False
    engine._lora_base_synced = False
    engine._lora_adapter_synced = False
    monkeypatch.setattr("verl_omni.workers.engine.veomni.diffusion_impl.load_model_to_gpu", lambda *_: None)
    monkeypatch.setattr("verl_omni.workers.engine.veomni.diffusion_impl.get_device_id", lambda: torch.device("cpu"))

    base_generator, _ = engine.get_per_tensor_param(base_sync_done=False, adapter_name="default")
    base = dict(base_generator)
    assert "transformer.dit.condition_proj.weight" in base
    assert not any("conon_proj" in name for name in base)

    class _RolloutLoader:
        def load_weights(self, weights):
            params = dict(weights)
            self.transformer.load_state_dict(
                {name.removeprefix("transformer."): tensor for name, tensor in params.items()}, strict=True
            )
            return set(params)

    class _RolloutPipeline(MiniMaxH3WeightSyncMixin, _RolloutLoader):
        pass

    rollout = _RolloutPipeline()
    rollout.transformer = _build_models()[1].dit
    loaded_base = rollout.load_weights(base.items())
    assert loaded_base == {name.replace("transformer.dit.", "transformer.", 1) for name in base}
    torch.testing.assert_close(rollout.transformer.condition_proj.weight, base["transformer.dit.condition_proj.weight"])

    adapter_generator, config = engine.get_per_tensor_param(base_sync_done=True, adapter_name="default")
    adapter = dict(adapter_generator)
    assert all(name.startswith("transformer.dit.") for name in adapter)
    mapper = object.__new__(MiniMaxH3WeightSyncMixin)
    mapper.transformer = SimpleNamespace(
        arch=SimpleNamespace(
            ffn_hidden_size=_FFN,
            num_attention_heads=_HEADS,
            attention_head_dim=_HEAD_DIM,
        )
    )
    mapped, mapped_config = mapper.map_lora_update_to_engine(adapter, config)
    assert mapped_config["target_modules"] == ["fc1_0", "fc1_1", "fc2", "out_proj", "to_k", "to_q", "to_v"]
    assert len(mapped) == 2 * 7 * 2

    loaded = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors=mapped,
        peft_helper=PEFTHelper.from_dict(mapped_config),
        device="cpu",
        dtype=torch.float32,
    )
    manager = object.__new__(DiffusionLoRAManager)
    native_blocks = {
        "blocks.0": lora_model.get_base_model().dit.blocks[0],
        "token_refiner.blocks.0": lora_model.get_base_model().dit.token_refiner.blocks[0],
    }
    for block, native_block in native_blocks.items():
        qkv = native_block.attn.qkv_proj
        grouped_qkv_b = qkv.lora_B["default"].weight.view(_HEADS, 3, _HEAD_DIM, -1)
        fc1 = native_block.mlp.fc1
        fc1_b = fc1.lora_B["default"].weight.chunk(2, dim=0)
        expected = {
            "attn.to_q": (qkv.lora_A["default"].weight, grouped_qkv_b[:, 0].reshape(_HEADS * _HEAD_DIM, -1)),
            "attn.to_k": (qkv.lora_A["default"].weight, grouped_qkv_b[:, 1].reshape(_HEADS * _HEAD_DIM, -1)),
            "attn.to_v": (qkv.lora_A["default"].weight, grouped_qkv_b[:, 2].reshape(_HEADS * _HEAD_DIM, -1)),
            "attn.out_proj": (
                native_block.attn.out_proj.lora_A["default"].weight,
                native_block.attn.out_proj.lora_B["default"].weight,
            ),
            "mlp.fc1_0": (fc1.lora_A["default"].weight, fc1_b[0]),
            "mlp.fc1_1": (fc1.lora_A["default"].weight, fc1_b[1]),
            "mlp.fc2": (
                native_block.mlp.fc2.lora_A["default"].weight,
                native_block.mlp.fc2.lora_B["default"].weight,
            ),
        }
        for path, (expected_a, expected_b) in expected.items():
            actual = manager._get_lora_weights(loaded, f"transformer.{block}.{path}")
            assert actual is not None
            assert actual.scaling == 2.0
            torch.testing.assert_close(actual.lora_a, expected_a, rtol=0, atol=0)
            torch.testing.assert_close(actual.lora_b, expected_b, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.lora_b @ actual.lora_a * actual.scaling,
                expected_b @ expected_a * 2.0,
                rtol=0,
                atol=0,
            )

    MiniMaxH3WeightSyncMixin._validate_diffusion_lora_binding(
        lora_model=loaded,
        bound_lora_names=frozenset(loaded.loras),
    )
    with pytest.raises(ValueError, match="unbound modules"):
        MiniMaxH3WeightSyncMixin._validate_diffusion_lora_binding(
            lora_model=loaded,
            bound_lora_names=frozenset(set(loaded.loras) - {next(iter(loaded.loras))}),
        )

    class _RecordingLayer:
        def __init__(self, output_slices):
            self.n_slices = len(output_slices)
            self.output_slices = output_slices
            self.bound = None

        def set_lora(self, *, index, lora_a, lora_b):
            assert index == 0
            self.bound = (lora_a, lora_b)

        def reset_lora(self, index):
            raise AssertionError(f"LoRA layer {index} was unexpectedly reset")

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.stacked_params_mapping = (
        (".attn.qkv_proj", ".attn.to_q", "q"),
        (".attn.qkv_proj", ".attn.to_k", "k"),
        (".attn.qkv_proj", ".attn.to_v", "v"),
    )
    pipeline._validate_diffusion_lora_binding = MiniMaxH3WeightSyncMixin._validate_diffusion_lora_binding
    mapper = object.__new__(MiniMaxH3WeightSyncMixin)
    mapper.transformer = pipeline.transformer
    mapper.install_h3_lora_layout()

    manager.pipeline = pipeline
    manager._packed_modules_mapping = manager._compute_packed_modules_mapping()
    manager._lora_modules = {}
    for block in native_blocks:
        manager._lora_modules[f"transformer.{block}.attn.qkv_proj"] = _RecordingLayer([64, 64, 64])
        manager._lora_modules[f"transformer.{block}.attn.out_proj"] = _RecordingLayer([48])
        manager._lora_modules[f"transformer.{block}.mlp.fc1"] = _RecordingLayer([64, 64])
        manager._lora_modules[f"transformer.{block}.mlp.fc2"] = _RecordingLayer([48])
    for lora in loaded.loras.values():
        lora.optimize()
    manager._bind_adapter_weights(loaded, scale=1.0)
    assert all(layer.bound is not None for layer in manager._lora_modules.values())
