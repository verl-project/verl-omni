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

"""CPU-only unit tests for ``MiniMaxH3Flops``."""

import math

import pytest
import torch

from verl_omni.utils.mfu import DiffusionFlopsCounter, MiniMaxH3Flops, collect_diffusion_flops_meta
from verl_omni.utils.mfu import diffusion_flops_counter as dfc
from verl_omni.utils.mfu.diffusion_flops_counter import _REGISTRY, allgather_diffusion_flops_meta

H3_CONFIG: dict = {
    "_class_name": "MiniMaxH3Transformer3DModel",
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "ffn_dim": 14336,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 5120,
    "freq_dim": 256,
    "time_embed_hidden_dim": 5376,
    "time_embed_dim": 2688,
}


def _counter(config: dict | None = None) -> DiffusionFlopsCounter:
    return DiffusionFlopsCounter("MiniMaxH3Pipeline", config or H3_CONFIG)


def _reference_h3_flops(
    config: dict,
    *,
    video_seqlens: list[int],
    audio_seqlens: list[int],
    prompt_seqlens: list[int],
    delta_time: float,
    num_timesteps: int,
    num_forward_passes: int,
) -> float:
    """Independent FLOPs derivation from ``MiniMaxH3Transformer3DModel``."""
    heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    hidden = int(config["hidden_size"])
    layers = int(config["num_layers"])
    refiners = int(config.get("num_refiner_layers", config.get("token_refiner_num_layers")))
    ffn = int(config.get("ffn_dim", config.get("ffn_hidden_size")))
    in_channels = int(config.get("in_channels", config.get("latents_dim")))
    audio_channels = int(config.get("audio_in_channels", config.get("audio_latents_dim")))
    text_dim = int(config["text_dim"])
    freq_dim = int(config.get("freq_dim", config.get("timestep_input_dim")))
    time_hidden = int(config.get("time_embed_hidden_dim", config.get("time_embed_hidden_size")))
    time_dim = int(config["time_embed_dim"])
    patch_volume = math.prod(config["patch_size"])
    video_dim = in_channels * patch_volume
    inner = heads * head_dim

    flops_fwd = 0.0
    block_weights = 4 * hidden * inner + 3 * hidden * ffn
    for video_rows, audio_rows, prompt_rows in zip(video_seqlens, audio_seqlens, prompt_seqlens, strict=True):
        packed_rows = video_rows + audio_rows + prompt_rows
        flops_fwd += 2 * layers * block_weights * packed_rows
        flops_fwd += 4 * layers * inner * packed_rows**2
        flops_fwd += 2 * refiners * block_weights * prompt_rows
        flops_fwd += 4 * refiners * inner * prompt_rows**2
        flops_fwd += 2 * (video_dim * hidden * video_rows)
        flops_fwd += 2 * (audio_channels * hidden * audio_rows)
        flops_fwd += 2 * (text_dim * hidden * prompt_rows)
        flops_fwd += 2 * hidden * (video_dim + audio_channels) * packed_rows

    time_weights = freq_dim * time_hidden + time_hidden * time_dim
    adaln_weights = layers * time_dim * 18 * hidden + time_dim * 2 * hidden
    flops_fwd += 2 * (time_weights + adaln_weights) * len(prompt_seqlens)

    return 3 * flops_fwd * num_timesteps * num_forward_passes / delta_time / 1e12


class TestMiniMaxH3FlopsRegistry:
    def test_registered(self):
        assert _REGISTRY["MiniMaxH3Pipeline"] is MiniMaxH3Flops


class TestMiniMaxH3FlopsMetadata:
    def test_nft_metadata_includes_reference_rows(self):
        data = {
            "latent_meta": torch.tensor([[100, 20, 1, 1, 1, 1], [120, 30, 1, 1, 1, 1]]),
            "condition_video_row_count": torch.tensor([[4], [8]]),
            "condition_audio_row_count": torch.tensor([[2], [3]]),
            "prompt_embeds_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "train_timesteps": torch.ones(2, 3),
        }

        meta = collect_diffusion_flops_meta(_counter(), data)

        assert meta == {
            "latent_seqlens": [126, 161],
            "prompt_seqlens": [3, 2],
            "video_seqlens": [104, 128],
            "audio_seqlens": [22, 33],
            "timestep_group_seqlens": [4, 4],
            "num_timesteps": 3,
            "num_forward_passes": 1,
        }

    def test_nft_metadata_timestep_groups_drop_absent_reference_streams(self):
        data = {
            "latent_meta": torch.tensor([[100, 20, 1, 1, 1, 1], [120, 30, 1, 1, 1, 1]]),
            "condition_video_row_count": torch.tensor([[4], [0]]),
            "condition_audio_row_count": torch.tensor([[0], [0]]),
            "prompt_embeds_mask": torch.ones(2, 4, dtype=torch.long),
            "train_timesteps": torch.ones(2, 3),
        }

        meta = collect_diffusion_flops_meta(_counter(), data)

        # Sample 0 adds one frozen reference-video timestep; sample 1 is pure T2VA.
        assert meta["timestep_group_seqlens"] == [3, 2]

    def test_flowgrpo_metadata_uses_packed_layout_counts(self):
        data = {
            "h3_video_rows": torch.tensor([[144], [160]]),
            "h3_audio_rows": torch.tensor([[40], [48]]),
            "prompt_embeds_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "all_timesteps": torch.ones(2, 2),
        }

        meta = collect_diffusion_flops_meta(_counter(), data)

        assert meta["video_seqlens"] == [144, 160]
        assert meta["audio_seqlens"] == [40, 48]
        assert meta["latent_seqlens"] == [184, 208]
        assert meta["prompt_seqlens"] == [2, 3]
        assert meta["timestep_group_seqlens"] == [2, 2]
        assert meta["num_timesteps"] == 2


class TestMiniMaxH3FlopsFormula:
    def _kwargs(self, **overrides):
        defaults = {
            "latent_seqlens": [120, 155],
            "video_seqlens": [100, 130],
            "audio_seqlens": [20, 25],
            "prompt_seqlens": [32, 48],
            "delta_time": 2.0,
            "num_timesteps": 4,
            "num_forward_passes": 1,
        }
        defaults.update(overrides)
        return defaults

    def test_matches_hand_rolled_reference(self):
        kwargs = self._kwargs()
        estimate, _ = _counter().estimate_flops(**kwargs)
        reference = _reference_h3_flops(
            H3_CONFIG,
            **{key: value for key, value in kwargs.items() if key != "latent_seqlens"},
        )
        assert math.isclose(estimate, reference, rel_tol=1e-9), (estimate, reference)

    def test_matches_fused_rollout_config_aliases(self):
        fused_config = dict(H3_CONFIG)
        fused_config["token_refiner_num_layers"] = fused_config.pop("num_refiner_layers")
        fused_config["ffn_hidden_size"] = fused_config.pop("ffn_dim")
        fused_config["latents_dim"] = fused_config.pop("in_channels")
        fused_config["audio_latents_dim"] = fused_config.pop("audio_in_channels")
        fused_config["timestep_input_dim"] = fused_config.pop("freq_dim")
        fused_config["time_embed_hidden_size"] = fused_config.pop("time_embed_hidden_dim")
        kwargs = self._kwargs()
        estimate, _ = _counter(fused_config).estimate_flops(**kwargs)
        reference = _reference_h3_flops(
            fused_config,
            **{key: value for key, value in kwargs.items() if key != "latent_seqlens"},
        )
        assert math.isclose(estimate, reference, rel_tol=1e-9), (estimate, reference)

    def test_linear_in_timestep_and_forward_pass_counts(self):
        base, _ = _counter().estimate_flops(**self._kwargs())
        doubled_steps, _ = _counter().estimate_flops(**self._kwargs(num_timesteps=8))
        doubled_forwards, _ = _counter().estimate_flops(**self._kwargs(num_forward_passes=2))
        assert math.isclose(doubled_steps / base, 2.0, rel_tol=1e-9)
        assert math.isclose(doubled_forwards / base, 2.0, rel_tol=1e-9)

    def test_full_attention_is_superlinear_in_packed_rows(self):
        common = {"prompt_seqlens": [32], "delta_time": 1.0, "num_timesteps": 1, "num_forward_passes": 1}
        small, _ = _counter().estimate_flops(latent_seqlens=[512], video_seqlens=[480], audio_seqlens=[32], **common)
        large, _ = _counter().estimate_flops(latent_seqlens=[8192], video_seqlens=[7680], audio_seqlens=[512], **common)
        assert large / small > 16.0

    def test_timestep_groups_scale_only_the_adaln_term(self):
        kwargs = self._kwargs()
        base, _ = _counter().estimate_flops(**kwargs)  # fallback groups == batch size (2)
        more, _ = _counter().estimate_flops(**kwargs, timestep_group_seqlens=[3, 3])  # sum 6

        config = H3_CONFIG
        time_hidden = int(config["time_embed_hidden_dim"])
        time_dim = int(config["time_embed_dim"])
        timestep_weights = int(config["freq_dim"]) * time_hidden + time_hidden * time_dim
        adaln_weights = int(config["num_layers"]) * time_dim * 18 * int(config["hidden_size"]) + time_dim * 2 * int(
            config["hidden_size"]
        )
        per_group = 6 * (timestep_weights + adaln_weights)
        expected_delta = per_group * (6 - 2) * kwargs["num_timesteps"] / kwargs["delta_time"] / 1e12
        assert math.isclose(more - base, expected_delta, rel_tol=1e-9), (more - base, expected_delta)

    def test_timestep_group_length_must_match_batch(self):
        with pytest.raises(ValueError, match="timestep group counts must match"):
            _counter().estimate_flops(**self._kwargs(), timestep_group_seqlens=[2])


class TestMiniMaxH3FlopsDPGather:
    def test_dp_gather_flattens_h3_stream_metadata(self, monkeypatch):
        meta = {
            "latent_seqlens": [10],
            "prompt_seqlens": [3],
            "video_seqlens": [8],
            "audio_seqlens": [2],
            "timestep_group_seqlens": [4],
            "num_timesteps": 5,
            "num_forward_passes": 2,
        }

        def fake_all_gather_object(gathered, value, group):
            # Rank 0 contributes the local value; rank 1 a distinct payload.
            gathered[0] = list(value)
            gathered[1] = [item + 100 for item in value]

        monkeypatch.setattr(dfc.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(dfc.torch.distributed, "get_world_size", lambda group=None: 2)
        monkeypatch.setattr(dfc.torch.distributed, "all_gather_object", fake_all_gather_object)

        gathered = allgather_diffusion_flops_meta(meta, dp_group=object())

        # Every per-sample list field is flattened across both ranks in order.
        assert gathered["latent_seqlens"] == [10, 110]
        assert gathered["prompt_seqlens"] == [3, 103]
        assert gathered["video_seqlens"] == [8, 108]
        assert gathered["audio_seqlens"] == [2, 102]
        assert gathered["timestep_group_seqlens"] == [4, 104]
        # Scalar fields are constant across the DP group and stay unchanged.
        assert gathered["num_timesteps"] == 5
        assert gathered["num_forward_passes"] == 2


class TestMiniMaxH3FlopsParamCount:
    @pytest.fixture(scope="class")
    def tiny_h3(self):
        from diffusers import MiniMaxH3Transformer3DModel

        return MiniMaxH3Transformer3DModel(
            num_attention_heads=4,
            attention_head_dim=16,
            hidden_size=48,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=96,
            in_channels=24,
            audio_in_channels=32,
            patch_size=(1, 2, 2),
            text_dim=40,
            freq_dim=16,
            time_embed_hidden_dim=48,
            time_embed_dim=32,
            rope_freq_dim=4,
        )

    def test_token_scaling_weights_match_transformer_modules(self, tiny_h3):
        config = tiny_h3.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim
        expected_block = 4 * hidden * inner + 3 * hidden * config.ffn_dim

        def block_weights(block):
            attention_layers = (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.attn.to_out[0])
            attention = sum(layer.weight.numel() for layer in attention_layers)
            feed_forward = sum(parameter.numel() for parameter in block.ff.parameters() if parameter.ndim == 2)
            return attention + feed_forward

        assert block_weights(tiny_h3.transformer_blocks[0]) == expected_block
        assert block_weights(tiny_h3.token_refiner.refiner_blocks[0]) == expected_block

    def test_projection_and_timestep_weights_match_transformer_modules(self, tiny_h3):
        config = tiny_h3.config
        hidden = config.hidden_size
        video_dim = config.in_channels * math.prod(config.patch_size)

        assert tiny_h3.proj_in.weight.numel() == video_dim * hidden
        assert tiny_h3.audio_proj_in.weight.numel() == config.audio_in_channels * hidden
        assert tiny_h3.context_embedder.weight.numel() == config.text_dim * hidden
        assert tiny_h3.proj_out.weight.numel() == hidden * video_dim
        assert tiny_h3.audio_proj_out.weight.numel() == hidden * config.audio_in_channels
        assert tiny_h3.time_embedder.linear_1.weight.numel() == config.freq_dim * config.time_embed_hidden_dim
        assert tiny_h3.time_embedder.linear_2.weight.numel() == config.time_embed_hidden_dim * config.time_embed_dim
        assert tiny_h3.transformer_blocks[0].adaln_proj.linear.weight.numel() == config.time_embed_dim * 18 * hidden
        assert tiny_h3.norm_out.linear.weight.numel() == config.time_embed_dim * 2 * hidden
