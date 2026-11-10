"""CPU tests for the native -> Diffusers MiniMax H3 image checkpoint converter."""

import json

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from safetensors.torch import load_file, save_file
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv

from verl_omni.model_merger.fsdp_model_merger import _h3_conversion_plan
from verl_omni.pipelines.minimax_h3_image_diffusion_nft import checkpoint as ckpt

_HEADS, _HEAD_DIM, _HIDDEN, _FFN = 2, 16, 32, 48
_RELEASE_CONFIG = {
    "_class_name": "MiniMaxH3DiTModel",
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
    "audio_enabled": False,
}


def _tiny_actor(seed: int = 0) -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        hidden_size=_HIDDEN,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=_FFN,
        text_dim=24,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    return model


def _native_config(actor: MiniMaxH3Transformer3DModel) -> dict:
    config = actor.config
    shared = (
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "text_dim",
        "time_embed_dim",
        "norm_eps",
        "qk_norm_eps",
        "final_norm_eps",
    )
    native = {name: config[name] for name in shared}
    native.update(
        _class_name="MiniMaxH3DiTModel",
        patch_size=list(config.patch_size),
        token_refiner_num_layers=config.num_refiner_layers,
        ffn_hidden_size=config.ffn_dim,
        latents_dim=config.in_channels,
        audio_latents_dim=config.audio_in_channels,
        timestep_input_dim=config.freq_dim,
        time_embed_hidden_size=config.time_embed_hidden_dim,
        rope_inv_freq_len=config.rope_freq_dim,
        adaln_out_features=18 * config.hidden_size,
        final_adaln_out_features=2 * config.hidden_size,
        audio_enabled=False,
    )
    return native


def _native_state(actor: MiniMaxH3Transformer3DModel) -> dict[str, torch.Tensor]:
    """Audio-free native state built with the model merger's independent Diffusers -> native rules."""
    state = {name: value for name, value in actor.state_dict().items() if not name.startswith("audio_proj_")}
    plan = _h3_conversion_plan({name: tuple(value.shape) for name, value in state.items()})
    native = {}
    for target, (kind, names) in plan.items():
        values = [state[name] for name in names]
        if kind == "qkv":
            native[target] = torch.stack([value.reshape(_HEADS, _HEAD_DIM, -1) for value in values], dim=1).reshape(
                _HEADS * 3 * _HEAD_DIM, -1
            )
        elif kind == "geglu":
            up, gate = values[0].split(_FFN, dim=0)
            native[target] = torch.cat([gate, up], dim=0)
        else:
            native[target] = values[0]
    native[ckpt.ROPE_BUFFER] = ckpt.expected_rope_inv_freq(actor.config)
    return native


def _write_release(root, actor, shards=2):
    native = _native_state(actor)
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    names = sorted(native)
    weight_map = {}
    for index in range(shards):
        filename = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        part = {name: native[name] for name in names[index::shards]}
        save_file(part, str(transformer / filename))
        weight_map.update({name: filename for name in part})
    (transformer / ckpt.NATIVE_INDEX).write_text(json.dumps({"weight_map": weight_map}))
    (transformer / "config.json").write_text(json.dumps(_native_config(actor)))
    save_file(
        {"video_out.weight": torch.randn(6, 96, _HIDDEN), "video_out.bias": torch.randn(6, 96)},
        str(root / "pdd_fused_heads.safetensors"),
    )
    (root / "pdd_config.json").write_text(
        json.dumps({"num_inference_steps": 6, "sigmas": [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0]})
    )
    return native


class TestTensorRules:
    def test_split_matches_the_native_loader_regrouping(self):
        weight = torch.randn(_HEADS * 3 * _HEAD_DIM, _HIDDEN)
        expected = _reorder_grouped_qkv_to_qkv(weight, num_query_groups=_HEADS, heads_per_group=1, head_dim=_HEAD_DIM)
        parts = ckpt.split_grouped_qkv(weight, _HEADS, _HEAD_DIM)
        assert torch.equal(torch.cat(parts), expected)
        assert ckpt.bits_equal(ckpt.join_grouped_qkv(*parts, _HEADS, _HEAD_DIM), weight)

    def test_fc1_swap_puts_the_native_gate_half_last(self):
        gate, up = torch.randn(_FFN, _HIDDEN), torch.randn(_FFN, _HIDDEN)
        swapped = ckpt.swap_halves(torch.cat([gate, up]))
        assert torch.equal(swapped, torch.cat([up, gate]))
        assert torch.equal(ckpt.swap_halves(swapped), torch.cat([gate, up]))

    def test_bits_equal_distinguishes_signed_zero_and_dtype(self):
        zero = torch.zeros(4)
        assert ckpt.bits_equal(zero, zero.clone())
        assert not ckpt.bits_equal(zero, -zero)
        assert not ckpt.bits_equal(zero, zero.to(torch.bfloat16))


class TestConfig:
    def test_release_config_translates_to_the_diffusers_fields(self):
        config = ckpt.diffusers_config(_RELEASE_CONFIG)
        assert config == {
            "_class_name": "MiniMaxH3Transformer3DModel",
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "hidden_size": 5376,
            "num_layers": 50,
            "num_refiner_layers": 2,
            "ffn_dim": 14336,
            "in_channels": 24,
            "audio_in_channels": 32,
            "patch_size": [1, 2, 2],
            "text_dim": 5120,
            "freq_dim": 256,
            "time_embed_hidden_dim": 5376,
            "time_embed_dim": 2688,
            "rope_freq_dim": 16,
            "rope_theta": 10000.0,
            "norm_eps": 1e-05,
            "qk_norm_eps": 1e-05,
            "final_norm_eps": 1e-05,
        }

    @pytest.mark.parametrize(
        "update",
        [{"audio_enabled": True}, {"_class_name": "MiniMaxH3Transformer3DModel"}, {"adaln_out_features": 1}],
        ids=["audio-enabled", "already-diffusers", "adaln-width"],
    )
    def test_rejects_configs_outside_the_audio_free_native_layout(self, update):
        with pytest.raises(ValueError):
            ckpt.diffusers_config({**_RELEASE_CONFIG, **update})


class TestStateDictRoundTrip:
    def test_conversion_reproduces_the_diffusers_state_bit_for_bit(self):
        actor = _tiny_actor()
        native = _native_state(actor)
        config = ckpt.diffusers_config(_native_config(actor))
        converted = ckpt.convert_state_dict(native, config)

        reference = actor.state_dict()
        assert set(converted) == set(reference)
        for name, tensor in converted.items():
            if name.startswith("audio_proj_"):
                assert tensor.dtype == torch.float32 and not tensor.count_nonzero()
            else:
                assert ckpt.bits_equal(tensor, reference[name]), name
        _tiny_actor(seed=1).load_state_dict(converted, strict=True)

        restored = ckpt.restore_state_dict(converted, list(native), config)
        assert set(restored) == set(native)
        assert all(ckpt.bits_equal(restored[name], native[name]) for name in native)

    def test_audio_tensors_are_rejected(self):
        with pytest.raises(ValueError, match="audio-free"):
            ckpt.convert_tensor("audio_patch_proj.weight", torch.zeros(2, 2), _HEADS, _HEAD_DIM)


class TestStreamingConversion:
    def test_sharded_release_converts_and_verifies(self, tmp_path):
        actor = _tiny_actor()
        native = _write_release(tmp_path / "release", actor)
        output = tmp_path / "converted" / "transformer"

        manifest = ckpt.convert(tmp_path / "release", output)
        summary = ckpt.verify(tmp_path / "release", output, report=tmp_path / "report.json")

        assert manifest["native_tensors"] == summary["native_tensors"] == len(native)
        assert manifest["dropped"] == [ckpt.ROPE_BUFFER]
        assert summary["diffusers_tensors"] == len(actor.state_dict())
        report = json.loads((tmp_path / "report.json").read_text())
        assert {row["native"] for row in report["tensors"]} == set(native)
        loaded = MiniMaxH3Transformer3DModel.from_pretrained(str(output))
        reference = actor.state_dict()
        assert all(
            ckpt.bits_equal(value, reference[name])
            for name, value in loaded.state_dict().items()
            if not name.startswith("audio_proj_")
        )
        with pytest.raises(FileExistsError):
            ckpt.convert(tmp_path / "release", output)

    def test_verify_catches_a_single_changed_value(self, tmp_path):
        actor = _tiny_actor()
        _write_release(tmp_path / "release", actor)
        output = tmp_path / "converted" / "transformer"
        ckpt.convert(tmp_path / "release", output)

        index = json.loads((output / ckpt.DIFFUSERS_INDEX).read_text())["weight_map"]
        name = next(name for name in sorted(index) if name.endswith(".attn.to_k.weight"))
        shard = output / index[name]
        tensors = load_file(str(shard))
        tensors[name][0, 0] = -tensors[name][0, 0] + 1.0
        save_file(tensors, str(shard), metadata={"format": "pt"})

        with pytest.raises(AssertionError, match="qkv_proj"):
            ckpt.verify(tmp_path / "release", output)

    def test_rope_buffer_that_diffusers_cannot_recompute_fails_conversion(self, tmp_path):
        actor = _tiny_actor()
        _write_release(tmp_path / "release", actor, shards=1)
        shard = tmp_path / "release" / "transformer" / "model-00001-of-00001.safetensors"
        tensors = load_file(str(shard))
        tensors[ckpt.ROPE_BUFFER] = tensors[ckpt.ROPE_BUFFER] * 1.5
        save_file(tensors, str(shard))

        with pytest.raises(ValueError, match="rope.inv_freq"):
            ckpt.convert(tmp_path / "release", tmp_path / "out")
