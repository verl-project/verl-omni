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
"""Shared H3 implementation contracts and preserved NFT/FlowGRPO policy differences.

Weight tests use native unquantized loaders on CPU with simulated TP ranks, not
mock loaders or distributed collectives. Tests named ``*_policy_difference``
characterize current behavior; they do not endorse it as the unified contract.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel, MiniMaxH3Rope
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

from verl_omni.pipelines.minimax_h3_diffusion_nft import common as nft_common
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MiniMaxH3RolloutWeightSyncMixin,
    MiniMaxH3WeightSyncBase,
    pack_video_audio_rows,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.common import flatten_joint_latents, split_joint_latents
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin


class _WeightOnlyDiT(nn.Module):
    load_weights = MiniMaxH3DiTModel.load_weights

    def __init__(self):
        super().__init__()
        self.arch = SimpleNamespace(num_attention_heads=4, attention_head_dim=2, ffn_hidden_size=8, rope_inv_freq_len=2)
        self.blocks = nn.ModuleList([nn.Module()])
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([nn.Module()])
        for block in (self.blocks[0], self.token_refiner.blocks[0]):
            block.attn = nn.Module()
            block.attn.qkv_proj = linear.QKVParallelLinear(
                hidden_size=6, head_size=2, total_num_heads=4, bias=False, params_dtype=torch.float32
            )
            block.mlp = nn.Module()
            block.mlp.fc1 = linear.MergedColumnParallelLinear(6, [8, 8], bias=False, params_dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(6, 2, bias=False)
        self.rope = MiniMaxH3Rope(2)
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(-1)
            self.rope.inv_freq.fill_(-1)

    def post_load_weights(self):
        """Leave CPU parameters visible; this fixture does not install forward kernels."""


class _NativePipeline:
    load_weights = MiniMaxH3Pipeline.load_weights

    def __init__(self):
        self.transformer = _WeightOnlyDiT()
        self.transformers_ref = _WeightOnlyDiT()
        self.video_vae = self.audio_vae = None
        # Newer vLLM-Omni's load_weights reads these; None means no FastH3 adapter or checkpoint.
        self._fasth3 = self._fasth3_checkpoint = None

    def _finish_adaln_sidecar(self, component):
        """This fixture configures no AdaLN sidecar; newer vLLM-Omni's load_weights still calls this."""


class _NFTPipeline(MiniMaxH3RolloutWeightSyncMixin, _NativePipeline):
    pass


class _FlowPipeline(MiniMaxH3WeightSyncMixin, _NativePipeline):
    pass


@pytest.fixture(params=[(1, 0), (2, 0), (2, 1)], ids=["tp1", "tp2-rank0", "tp2-rank1"])
def pipelines(request, monkeypatch):
    """Construct real linear parameters without initializing a process group."""
    tp_size, tp_rank = request.param
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    with torch.device("cpu"):
        return _NFTPipeline(), _FlowPipeline(), tp_size, tp_rank


def _full_weights(offset=0):
    weights = {}
    for index, block in enumerate(("transformer_blocks.0", "token_refiner.refiner_blocks.0")):
        for shard, projection in enumerate(("to_q", "to_k", "to_v")):
            weights[f"transformer.{block}.attn.{projection}.base_layer.weight"] = (
                torch.arange(48, dtype=torch.float32).reshape(8, 6) + offset + index * 500 + shard * 100
            )
        weights[f"transformer.{block}.ff.net.0.proj.base_layer.weight"] = (
            torch.arange(96, dtype=torch.float32).reshape(16, 6) + offset + index * 500
        )
    weights["transformer.audio_proj_in.weight"] = torch.full((2, 6), float(offset))
    return weights


@pytest.mark.parametrize("bucket_size", [1, 3, 9])
def test_full_sync_matches_native_parameter_values_across_buckets(pipelines, bucket_size):
    """Both adapters must load every shard correctly, including a second weight update."""
    nft, flow, tp_size, tp_rank = pipelines
    for offset in (0, 2000):
        weights = _full_weights(offset)
        originals = {name: tensor.clone() for name, tensor in weights.items()}
        for pipeline in (nft, flow):
            items = list(weights.items())
            for start in range(0, len(items), bucket_size):
                pipeline.load_weights(iter(items[start : start + bucket_size]))
            for source, target in (
                ("transformer_blocks.0", "blocks.0"),
                ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0"),
            ):
                expected_qkv = torch.cat(
                    [
                        weights[f"transformer.{source}.attn.to_{shard}.base_layer.weight"].chunk(tp_size)[tp_rank]
                        for shard in ("q", "k", "v")
                    ]
                )
                up, gate = weights[f"transformer.{source}.ff.net.0.proj.base_layer.weight"].chunk(2)
                expected_fc1 = torch.cat([gate.chunk(tp_size)[tp_rank], up.chunk(tp_size)[tp_rank]])
                torch.testing.assert_close(
                    pipeline.transformer.get_parameter(f"{target}.attn.qkv_proj.weight"), expected_qkv
                )
                torch.testing.assert_close(pipeline.transformer.get_parameter(f"{target}.mlp.fc1.weight"), expected_fc1)
            torch.testing.assert_close(
                pipeline.transformer.audio_patch_proj.weight, weights["transformer.audio_proj_in.weight"]
            )
        for name, original in originals.items():
            torch.testing.assert_close(weights[name], original)


def test_combined_component_policy_difference(pipelines):
    """NFT writes transformer; FlowGRPO redirects actor weights to transformers_ref."""
    nft, flow, _, _ = pipelines
    weights = _full_weights()
    nft.load_weights(weights.items())
    expected = {name: param.detach().clone() for name, param in nft.transformer.named_parameters()}
    with torch.no_grad():
        for param in nft.transformer.parameters():
            param.fill_(-1)
    for pipeline, selected, untouched in (
        (nft, "transformer", "transformers_ref"),
        (flow, "transformers_ref", "transformer"),
    ):
        pipeline.partition = "combined"
        pipeline.load_weights(weights.items())
        for name, param in getattr(pipeline, selected).named_parameters():
            torch.testing.assert_close(param, expected[name])
        for param in getattr(pipeline, untouched).parameters():
            torch.testing.assert_close(param, torch.full_like(param, -1))


def test_rope_initialization_policy_difference(pipelines):
    """NFT synthesizes RoPE on first sync; FlowGRPO leaves the existing buffer alone."""
    nft, flow, _, _ = pipelines
    for pipeline in (nft, flow):
        pipeline.load_weights(_full_weights().items())
    torch.testing.assert_close(nft.transformer.rope.inv_freq, torch.tensor([1.0, 0.01]))
    torch.testing.assert_close(flow.transformer.rope.inv_freq, torch.full((2,), -1.0))
    for pipeline in (nft, flow):
        pipeline.transformer.rope.inv_freq.fill_(0.25)
        pipeline.load_weights(_full_weights(2000).items())
        torch.testing.assert_close(pipeline.transformer.rope.inv_freq, torch.full((2,), 0.25))


def _lora_tensors(block):
    tensors = {}
    for index, (module, rows, columns) in enumerate(
        (
            ("attn.to_q", 8, 6),
            ("attn.to_k", 8, 6),
            ("attn.to_v", 8, 6),
            ("attn.to_out.0", 6, 8),
            ("ff.net.0.proj", 16, 6),
            ("ff.net.2", 6, 8),
        )
    ):
        tensors[f"base_model.model.{block}.{module}.lora_A.weight"] = (
            torch.arange(columns * 2, dtype=torch.float32).reshape(2, columns) + index * 100
        )
        tensors[f"base_model.model.{block}.{module}.lora_B.weight"] = (
            torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) + index * 200
        )
    return tensors


@pytest.mark.parametrize(
    ("source", "target"),
    [("transformer_blocks.0", "blocks.0"), ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0")],
)
def test_lora_full_target_mapping_is_equivalent_and_does_not_mutate_inputs(pipelines, source, target):
    """Valid full-target adapters share names, A matrices, and gate/up B ordering."""
    nft, flow, _, _ = pipelines
    tensors = _lora_tensors(source)
    originals = {name: tensor.clone() for name, tensor in tensors.items()}
    config = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    }
    original_config = deepcopy(config)
    results = []
    for pipeline in (nft, flow):
        mapped, mapped_config = pipeline.map_lora_update_to_engine(tensors, config)
        expected = {}
        for source_module, target_module in (
            ("attn.to_q", "attn.to_q"),
            ("attn.to_k", "attn.to_k"),
            ("attn.to_v", "attn.to_v"),
            ("attn.to_out.0", "attn.out_proj"),
            ("ff.net.2", "mlp.fc2"),
        ):
            for kind in ("A", "B"):
                expected[f"transformer.{target}.{target_module}.lora_{kind}.weight"] = originals[
                    f"base_model.model.{source}.{source_module}.lora_{kind}.weight"
                ]
        for index in (0, 1):
            expected[f"transformer.{target}.mlp.fc1_{index}.lora_A.weight"] = originals[
                f"base_model.model.{source}.ff.net.0.proj.lora_A.weight"
            ]
            expected[f"transformer.{target}.mlp.fc1_{index}.lora_B.weight"] = originals[
                f"base_model.model.{source}.ff.net.0.proj.lora_B.weight"
            ].chunk(2)[1 - index]
        assert mapped.keys() == expected.keys()
        for name, tensor in expected.items():
            torch.testing.assert_close(mapped[name], tensor)
        assert set(mapped_config["target_modules"]) == {"to_q", "to_k", "to_v", "out_proj", "fc1_0", "fc1_1", "fc2"}
        assert mapped_config["r"] == 2
        assert mapped_config["lora_alpha"] == 4
        results.append(mapped)
    assert config == original_config
    for name, original in originals.items():
        torch.testing.assert_close(tensors[name], original)
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name])


def test_lora_subset_target_policy_difference(pipelines):
    """NFT expands a target subset; FlowGRPO keeps only the requested projections."""
    nft, flow, _, _ = pipelines
    name = "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"
    tensors = {name: torch.ones(2, 6)}
    config = {"r": 2, "target_modules": ["to_q"]}
    nft_tensors, nft_config = nft.map_lora_update_to_engine(tensors, config)
    flow_tensors, flow_config = flow.map_lora_update_to_engine(tensors, config)
    assert nft_tensors.keys() == flow_tensors.keys() == {"transformer.blocks.0.attn.to_q.lora_A.weight"}
    assert set(nft_config["target_modules"]) == {"to_q", "to_k", "to_v", "out_proj", "fc1_0", "fc1_1", "fc2"}
    assert flow_config["target_modules"] == ["to_q"]


def test_lora_combined_component_policy_difference(pipelines):
    """LoRA deltas follow the same component-selection difference as full weights."""
    nft, flow, _, _ = pipelines
    tensors = _lora_tensors("transformer_blocks.0")
    config = {"target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]}
    for pipeline in (nft, flow):
        pipeline.partition = "combined"
    nft_tensors, _ = nft.map_lora_update_to_engine(tensors, config)
    flow_tensors, _ = flow.map_lora_update_to_engine(tensors, config)
    assert {name.replace("transformer.", "transformers_ref.", 1) for name in nft_tensors} == flow_tensors.keys()
    for name, tensor in nft_tensors.items():
        torch.testing.assert_close(tensor, flow_tensors[name.replace("transformer.", "transformers_ref.", 1)])


@pytest.mark.parametrize("targets", [None, [], "all-linear", ["proj_in"]])
def test_both_sync_paths_reject_unsupported_lora_targets(pipelines, targets):
    """A shared sync implementation must retain explicit target validation."""
    nft, flow, _, _ = pipelines
    for pipeline in (nft, flow):
        with pytest.raises(ValueError, match="MiniMax H3 LoRA"):
            pipeline.map_lora_update_to_engine({}, {"target_modules": targets})


def test_lora_qualified_target_policy_difference(pipelines):
    """FlowGRPO accepts a qualified target suffix; NFT only accepts exact whitelist entries."""
    nft, flow, _, _ = pipelines
    name = "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"
    tensors = {name: torch.ones(2, 6)}
    config = {"target_modules": ["transformer_blocks.0.attn.to_q"]}
    with pytest.raises(ValueError, match="MiniMax H3 LoRA supports only"):
        nft.map_lora_update_to_engine(tensors, config)
    mapped, mapped_config = flow.map_lora_update_to_engine(tensors, config)
    assert mapped.keys() == {"transformer.blocks.0.attn.to_q.lora_A.weight"}
    assert mapped_config["target_modules"] == ["to_q"]


def test_lora_validation_policy_difference(pipelines):
    """Characterize the permissive NFT payload handling, not a desired validation policy."""
    nft, flow, _, _ = pipelines
    name = "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_B.weight"
    malformed = torch.zeros(15, 2)
    config = {"target_modules": ["ff.net.0.proj"]}
    nft_tensors, _ = nft.map_lora_update_to_engine({name: malformed}, config)
    assert nft_tensors["transformer.blocks.0.mlp.fc1_0.lora_B.weight"].shape == (8, 2)
    assert nft_tensors["transformer.blocks.0.mlp.fc1_1.lora_B.weight"].shape == (7, 2)
    with pytest.raises(ValueError, match="fc1 LoRA B rows must be 16"):
        flow.map_lora_update_to_engine({name: malformed}, config)

    outside_block = {"base_model.model.proj_in.lora_A.weight": torch.ones(2, 6)}
    nft_tensors, _ = nft.map_lora_update_to_engine(outside_block, {"target_modules": ["to_q"]})
    assert nft_tensors.keys() == outside_block.keys()
    with pytest.raises(ValueError, match="outside supported DiT blocks"):
        flow.map_lora_update_to_engine(outside_block, {"target_modules": ["to_q"]})


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("audio_rows", [0, 3])
def test_joint_latent_values_match_with_distinct_shape_contracts(batch_size, audio_rows):
    """NFT uses (B, D); FlowGRPO retains a singleton row dimension (B, 1, D)."""
    video = torch.arange(batch_size * 2 * 96, dtype=torch.float32).reshape(batch_size, 2, 96)
    audio = torch.arange(batch_size * audio_rows * 32, dtype=torch.float32).reshape(batch_size, audio_rows, 32)
    nft = pack_video_audio_rows(video, audio)
    flow = flatten_joint_latents(video, audio)
    assert nft.shape == (batch_size, 2 * 96 + audio_rows * 32)
    assert flow.shape == (batch_size, 1, nft.shape[1])
    torch.testing.assert_close(flow[:, 0], nft)
    for unpacked in (unpack_video_audio_rows(nft, 2, audio_rows), split_joint_latents(flow, 2, audio_rows)):
        torch.testing.assert_close(unpacked[0], video)
        torch.testing.assert_close(unpacked[1], audio)


def test_both_algorithms_use_the_same_weight_loader_and_layout_helpers():
    from verl_omni.pipelines.minimax_h3_diffusion_nft import diffusers_training_adapter as nft_actor
    from verl_omni.pipelines.minimax_h3_diffusion_nft import vllm_omni_rollout_adapter as nft_rollout
    from verl_omni.pipelines.minimax_h3_flow_grpo import diffusers_training_adapter as flow_actor
    from verl_omni.pipelines.minimax_h3_flow_grpo import vllm_omni_rollout_adapter as flow_rollout

    assert MiniMaxH3RolloutWeightSyncMixin.load_weights is MiniMaxH3WeightSyncBase.load_weights
    assert MiniMaxH3WeightSyncMixin.load_weights is MiniMaxH3WeightSyncBase.load_weights
    for actor in (nft_actor, flow_actor):
        assert actor.build_ref2va_layout_from_meta is nft_common.build_ref2va_layout_from_meta
        assert actor.h3_ulysses_forward is nft_common.h3_ulysses_forward
        assert actor.prepare_h3_processor_files is nft_common.prepare_h3_processor_files
    for rollout in (nft_rollout, flow_rollout):
        assert rollout.serialize_ref_blocks is nft_common.serialize_ref_blocks
        assert rollout.ref2va_reference_image_short_edge is nft_common.ref2va_reference_image_short_edge


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("audio_rows", [0, 3])
def test_flow_flatten_preserves_already_flattened_batch_inputs(batch_size, audio_rows):
    video = torch.arange(batch_size * 2 * 96, dtype=torch.float32).reshape(batch_size, 2 * 96)
    audio = torch.arange(batch_size * audio_rows * 32, dtype=torch.float32).reshape(batch_size, audio_rows * 32)
    expected = torch.cat([video, audio], dim=1).unsqueeze(1)
    torch.testing.assert_close(flatten_joint_latents(video, audio), expected, rtol=0, atol=0)


def test_nft_rope_keeps_native_prefix_groups_contiguous():
    class RecordingBase:
        def load_weights(self, weights):
            self.received = list(weights)
            return {name for name, _ in self.received}

    class Pipeline(MiniMaxH3RolloutWeightSyncMixin, RecordingBase):
        transformer = SimpleNamespace(arch=SimpleNamespace(rope_inv_freq_len=2))

    pipeline = Pipeline()
    pipeline.load_weights(
        [
            ("transformer.audio_proj_in.weight", torch.ones(2, 6)),
            ("text_encoder.layer.weight", torch.ones(2, 2)),
        ]
    )
    assert [name for name, _ in pipeline.received] == [
        "transformer.audio_patch_proj.weight",
        "transformer.rope.inv_freq",
        "text_encoder.layer.weight",
    ]
    assert pipeline._rope_inv_freq_loaded
