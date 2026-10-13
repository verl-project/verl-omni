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

"""TP GPU contracts for H3 routing, RoPE, repeated full updates and combined LoRA.

Run with torchrun --standalone --nproc_per_node=2. These are tiny real Diffusers
and native DiTs with real loaders/kernels, not complete multimodal pipelines.
--output optionally saves tensors for comparison against another source tree.
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

from tests.special_e2e.minimax_h3_lora_sync_tp2 import _TINY_H3, _old_adapter_payload, _vllm_transformer_config
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MiniMaxH3RolloutWeightSyncMixin
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin
from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack


class _NativePipeline(torch.nn.Module):
    load_weights = MiniMaxH3Pipeline.load_weights
    _dit_modules = MiniMaxH3Pipeline._dit_modules

    def __init__(self, config, device, partition):
        super().__init__()
        self.partition = partition
        self.transformer = MiniMaxH3DiTModel(config).to(device)
        self.transformers_ref = MiniMaxH3DiTModel(config).to(device)
        self.video_vae = self.audio_vae = None
        # Newer vLLM-Omni's load_weights reads these; None means no FastH3 adapter or checkpoint.
        self._fasth3 = self._fasth3_checkpoint = None
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(-1)
            self.transformer.rope.inv_freq.fill_(0.25)
            self.transformers_ref.rope.inv_freq.fill_(0.25)

    def _finish_adaln_sidecar(self, component):
        """This fixture configures no AdaLN sidecar; newer vLLM-Omni's load_weights still calls this."""


class _NFTPipeline(MiniMaxH3RolloutWeightSyncMixin, _NativePipeline):
    pass


class _FlowPipeline(MiniMaxH3WeightSyncMixin, _NativePipeline):
    pass


def _check_combined_lora(pipeline, selected, device, rank, tp):
    if hasattr(pipeline, "_install_lora_layout"):
        pipeline._install_lora_layout()
    else:
        pipeline.install_h3_lora_layout()
    payload, config = _old_adapter_payload()
    manager = DiffusionLoRAManager(pipeline, device=device, dtype=torch.bfloat16)
    request = OmniTensorLoRARequest(
        lora_name="h3-combined-regression",
        lora_int_id=1,
        lora_path="/tmp/h3-combined-unused",
        peft_config=config,
        lora_tensors=payload,
    )
    assert manager.add_adapter(request)
    manager.set_active_adapter(request)
    mapped, _ = pipeline.map_lora_update_to_engine(payload, config)
    active = {name: layer for name, layer in manager._lora_modules.items() if any(layer._diffusion_lora_active_slices)}
    assert len(active) == 12, sorted(active)
    assert all(name.startswith(selected + ".") for name in active)
    qkv = active[f"{selected}.blocks.0.attn.qkv_proj"]
    for slot, projection in enumerate(("q", "k", "v")):
        prefix = f"{selected}.blocks.0.attn.to_{projection}"
        a = mapped[prefix + ".lora_A.weight"].to(device=device, dtype=torch.bfloat16)
        b = mapped[prefix + ".lora_B.weight"].to(device=device, dtype=torch.bfloat16).chunk(tp)[rank] * 2
        torch.testing.assert_close(qkv.lora_a_stacked[slot][0, 0], a, rtol=0, atol=0)
        torch.testing.assert_close(qkv.lora_b_stacked[slot][0, 0], b, rtol=0, atol=0)
    manager.set_active_adapter(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-lora", action="store_true", help="Compare legacy full weights without the known FC1 binding bug."
    )
    args = parser.parse_args()
    rank, local_rank, tp = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(local_rank=local_rank)
        initialize_model_parallel(tensor_model_parallel_size=tp)
        try:
            VLLMOmniHijack.hijack()
            torch.manual_seed(1234)
            actor = MiniMaxH3Transformer3DModel(**_TINY_H3)
            original = {name: value.detach().clone() for name, value in actor.named_parameters()}
            config = OmniDiffusionConfig(
                model="tiny-h3-shared-sync",
                tf_model_config=_vllm_transformer_config(),
                dtype=torch.bfloat16,
                parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp),
            )
            captures = {}
            for algorithm, cls in (("nft", _NFTPipeline), ("flow", _FlowPipeline)):
                for partition in ("fl2va", "combined"):
                    pipeline = cls(config, device, partition)
                    selected = "transformers_ref" if algorithm == "flow" and partition == "combined" else "transformer"
                    untouched = "transformer" if selected == "transformers_ref" else "transformers_ref"
                    target = getattr(pipeline, selected)
                    for version, offset in enumerate((0, 0.125)):
                        with torch.no_grad():
                            for parameter in target.parameters():
                                parameter.fill_(float("nan"))
                        weights = {name: value + offset for name, value in original.items()}
                        # Q/K/V arrive in separate buckets, including token-refiner projections.
                        for name, value in weights.items():
                            pipeline.load_weights(iter([(f"transformer.{name}", value)]))
                        for parameter in getattr(pipeline, untouched).parameters():
                            torch.testing.assert_close(parameter, torch.full_like(parameter, -1), rtol=0, atol=0)
                        expected_rope = (
                            torch.tensor([1.0, 0.01], device=device)
                            if algorithm == "nft"
                            else torch.full((2,), 0.25, device=device)
                        )
                        torch.testing.assert_close(pipeline.transformer.rope.inv_freq, expected_rope, rtol=0, atol=0)
                        torch.testing.assert_close(
                            pipeline.transformers_ref.rope.inv_freq,
                            torch.full((2,), 0.25, device=device),
                            rtol=0,
                            atol=0,
                        )
                        prefix = f"{algorithm}/{partition}/v{version}"
                        for name, value in target.named_parameters():
                            assert torch.isfinite(value).all(), name
                            captures[prefix + "/" + name] = value.detach().cpu().clone()
                        for block_name, module in (
                            ("transformer_blocks.0", target.blocks[0]),
                            ("token_refiner.refiner_blocks.0", target.token_refiner.blocks[0]),
                        ):
                            qkv = torch.cat(
                                [
                                    weights[f"{block_name}.attn.to_{part}.weight"].chunk(tp)[rank]
                                    for part in ("q", "k", "v")
                                ]
                            ).to(module.attn.qkv_proj.weight)
                            up, gate = weights[f"{block_name}.ff.net.0.proj.weight"].chunk(2)
                            fc1 = torch.cat([gate.chunk(tp)[rank], up.chunk(tp)[rank]]).to(module.mlp.fc1.weight)
                            for layer, expected in ((module.attn.qkv_proj, qkv), (module.mlp.fc1, fc1)):
                                torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
                                inputs = torch.ones(3, expected.shape[1], device=device, dtype=expected.dtype)
                                with torch.no_grad():
                                    output = layer(inputs)[0]
                                    torch.testing.assert_close(
                                        output, torch.nn.functional.linear(inputs, expected), rtol=0, atol=0
                                    )
                                captures[prefix + f"/{block_name}/{expected.shape[0]}/output"] = output.cpu()
                    if partition == "combined" and not args.skip_lora:
                        _check_combined_lora(pipeline, selected, device, rank, tp)
                    del pipeline
                    print(
                        f"rank={rank} {algorithm}/{partition}: "
                        "repeated full updates, native forward, routing and RoPE PASS",
                        flush=True,
                    )
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                torch.save(captures, args.output / f"rank{rank}.pt")
            torch.distributed.barrier()
            if rank == 0:
                print("MiniMax H3 shared sync TP contracts: PASS", flush=True)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
