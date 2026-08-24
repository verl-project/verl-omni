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

from __future__ import annotations

import os

import torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MiniMaxH3RolloutWeightSyncMixin
from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

_TINY_H3 = {
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "hidden_size": 48,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 64,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 32,
    "freq_dim": 16,
    "time_embed_hidden_dim": 48,
    "time_embed_dim": 32,
    "rope_freq_dim": 2,
}
_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


def _vllm_transformer_config() -> TransformerConfig:
    config = {
        **_TINY_H3,
        "token_refiner_num_layers": _TINY_H3["num_refiner_layers"],
        "ffn_hidden_size": _TINY_H3["ffn_dim"],
        "latents_dim": _TINY_H3["in_channels"],
        "audio_latents_dim": _TINY_H3["audio_in_channels"],
        "timestep_input_dim": _TINY_H3["freq_dim"],
        "time_embed_hidden_size": _TINY_H3["time_embed_hidden_dim"],
        "adaln_out_features": 18 * _TINY_H3["hidden_size"],
        "final_adaln_out_features": 2 * _TINY_H3["hidden_size"],
        "rope_inv_freq_len": _TINY_H3["rope_freq_dim"],
    }
    return TransformerConfig.from_dict(config)


class _Pipeline(nn.Module, MiniMaxH3RolloutWeightSyncMixin):
    def __init__(self, transformer: MiniMaxH3DiTModel) -> None:
        super().__init__()
        self.transformer = transformer


def _old_adapter_payload() -> tuple[dict[str, torch.Tensor], dict]:
    actor = MiniMaxH3Transformer3DModel(**_TINY_H3)
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=_TARGET_MODULES,
        bias="none",
    )
    actor.add_adapter(config, adapter_name="default")
    actor.add_adapter(config, adapter_name="old")
    payload = {
        name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(actor, adapter_name="old").items()
    }
    for tensor in payload.values():
        tensor.copy_(torch.randn_like(tensor))
    return payload, config.to_dict()


def _check_text_encoder_tp(rank: int, device: torch.device) -> None:
    """Exercise the text-encoder TP group at ``text_encoder_tp_size=2``.

    ``text_encoder_tp_size`` never reaches the DiT: ``MiniMaxH3DiTModel`` does not
    read it, so putting it on this test's ``DiffusionParallelConfig`` would assert
    nothing. It is consumed only by ``MiniMaxH3Pipeline.__init__``, which builds the
    encoder process group and fans prompt/vision tensors to the encoder ranks. Both
    of those touch just ``_dit_rank`` and ``text_encoder_group``, so they run on a
    bare pipeline stand-in without any checkpoint on disk.
    """
    stand_in = object.__new__(MiniMaxH3Pipeline)
    stand_in._dit_rank = rank

    single = MiniMaxH3Pipeline._build_text_encoder_group(stand_in, 1)
    assert single.world_size == 1, f"ETP=1 must stay single-rank, got {single.world_size}"

    group = MiniMaxH3Pipeline._build_text_encoder_group(stand_in, 2)
    assert group.world_size == 2, f"ETP=2 group must span 2 ranks, got {group.world_size}"
    assert list(group.ranks) == [0, 1], f"ETP=2 group must cover DiT ranks [0, 1], got {group.ranks}"
    assert group.rank_in_group == rank, f"rank_in_group {group.rank_in_group} != rank {rank}"
    stand_in.text_encoder_group = group

    # Prompt IDs are broadcast from encoder rank 0; ranks > 0 pass None and must
    # recover rank 0's payload exactly, otherwise the sharded encode desynchronizes.
    source = torch.arange(12, device=device, dtype=torch.long).reshape(3, 4) if rank == 0 else None
    received = MiniMaxH3Pipeline._encoder_group_broadcast_tensor(stand_in, source, dtype=torch.long, device=device)
    expected = torch.arange(12, device=device, dtype=torch.long).reshape(3, 4)
    assert tuple(received.shape) == (3, 4), f"broadcast reshaped the tensor: {tuple(received.shape)}"
    assert received.dtype is torch.long, f"broadcast changed dtype: {received.dtype}"
    assert torch.equal(received, expected), f"rank={rank} received a mismatched broadcast payload"

    print(f"rank={rank}: text_encoder_tp=2 group={list(group.ranks)}, broadcast OK", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(local_rank=local_rank)
        initialize_model_parallel(tensor_model_parallel_size=2)
        try:
            device = torch.device(f"cuda:{local_rank}")
            od_config = OmniDiffusionConfig(
                model="tiny-minimax-h3-lora-regression",
                tf_model_config=_vllm_transformer_config(),
                dtype=torch.float32,
                parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
            )
            transformer = MiniMaxH3DiTModel(od_config).to(device)
            for parameter in transformer.parameters():
                parameter.data.zero_()
            transformer.stacked_params_mapping = [
                (".qkv_proj", ".to_q", "q"),
                (".qkv_proj", ".to_k", "k"),
                (".qkv_proj", ".to_v", "v"),
                (".fc1", ".fc1_0", "0"),
                (".fc1", ".fc1_1", "1"),
            ]
            payload, peft_config = _old_adapter_payload()

            VLLMOmniHijack.hijack()
            manager = DiffusionLoRAManager(_Pipeline(transformer), device=device, dtype=torch.bfloat16)
            request = OmniTensorLoRARequest(
                lora_name="h3-tp2-regression",
                lora_int_id=1,
                lora_path="/tmp/h3-tp2-regression-unused",
                peft_config=peft_config,
                lora_tensors=payload,
            )
            assert manager.add_adapter(request)
            manager.set_active_adapter(request)
            assert len(manager._lora_modules) == 12
            assert all(all(layer._diffusion_lora_active_slices) for layer in manager._lora_modules.values())

            qkv = manager._lora_modules["transformer.blocks.0.attn.qkv_proj"]
            fc1 = manager._lora_modules["transformer.blocks.0.mlp.fc1"]
            qkv_b_shapes = [tuple(value.shape) for value in qkv.lora_b_stacked]
            fc1_b_shapes = [tuple(value.shape) for value in fc1.lora_b_stacked]
            assert qkv_b_shapes == [(1, 1, 32, 64)] * 3
            assert fc1_b_shapes == [(1, 1, 32, 64)] * 2

            inputs = torch.randn(3, _TINY_H3["hidden_size"], device=device, dtype=torch.bfloat16)
            qkv_with_lora = qkv(inputs)[0].detach()
            fc1_with_lora = fc1(inputs)[0].detach()
            qkv.reset_lora(0)
            fc1.reset_lora(0)
            qkv_base = qkv(inputs)[0].detach()
            fc1_base = fc1(inputs)[0].detach()
            qkv_delta = (qkv_with_lora - qkv_base).abs().max()
            fc1_delta = (fc1_with_lora - fc1_base).abs().max()
            assert torch.isfinite(qkv_delta) and qkv_delta > 0
            assert torch.isfinite(fc1_delta) and fc1_delta > 0

            print(
                f"rank={rank}: qkv_delta={qkv_delta.item():.6g}, "
                f"fc1_delta={fc1_delta.item():.6g}, qkv_B={qkv_b_shapes}, fc1_B={fc1_b_shapes}",
                flush=True,
            )
            _check_text_encoder_tp(rank, device)

            torch.distributed.barrier()
            if rank == 0:
                print("MiniMax H3 LoRA TP=2 actor-to-rollout regression: PASS", flush=True)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
