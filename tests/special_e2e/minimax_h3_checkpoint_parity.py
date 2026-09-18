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
"""Stream-compare official and diffusers MiniMax H3 transformer checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv


def _open_checkpoint(stack: ExitStack, root: Path, index_name: str):
    weight_map = json.loads((root / index_name).read_text())["weight_map"]
    handles = {
        filename: stack.enter_context(safe_open(root / filename, framework="pt", device="cpu"))
        for filename in sorted(set(weight_map.values()))
    }
    return weight_map, handles


def _get(weight_map, handles, name: str) -> torch.Tensor:
    return handles[weight_map[name]].get_tensor(name)


def _diffusers_name(name: str) -> str:
    replacements = {
        "video_patch_proj": "proj_in",
        "audio_patch_proj": "audio_proj_in",
        "condition_proj": "context_embedder",
        "time_embedder.proj_in": "time_embedder.linear_1",
        "time_embedder.proj_out": "time_embedder.linear_2",
        "final_layer.norm": "norm_out.norm",
        "final_layer.adaln_proj.linear": "norm_out.linear",
        "final_layer.video_out": "proj_out",
        "final_layer.audio_out": "audio_proj_out",
    }
    for source, target in replacements.items():
        if name == source or name.startswith(source + "."):
            return target + name[len(source) :]
    name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
    name = name.replace(".attn.q_norm.", ".attn.norm_q.")
    name = name.replace(".attn.k_norm.", ".attn.norm_k.")
    name = name.replace(".attn.out_proj.", ".attn.to_out.0.")
    name = name.replace(".mlp.fc2.", ".ff.net.2.")
    if name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks.") :]
    return name


def _diffusers_block_base(vllm_name: str, suffix: str) -> str:
    base = vllm_name[: -len(suffix)]
    if base.startswith("blocks."):
        return "transformer_blocks." + base[len("blocks.") :]
    return base.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")


def compare(vllm_root: Path, diffusers_root: Path) -> None:
    started = time.time()
    vcfg = json.loads((vllm_root / "config.json").read_text())
    dcfg = json.loads((diffusers_root / "config.json").read_text())
    assert vcfg["hidden_size"] == dcfg["hidden_size"]
    assert vcfg["num_layers"] == dcfg["num_layers"]
    assert vcfg["num_attention_heads"] == dcfg["num_attention_heads"]
    consumed: set[str] = set()
    total_values = 0

    with ExitStack() as stack:
        vi, vh = _open_checkpoint(stack, vllm_root, "model.safetensors.index.json")
        di, dh = _open_checkpoint(stack, diffusers_root, "diffusion_pytorch_model.safetensors.index.json")
        for index, vname in enumerate(vi, 1):
            actual = _get(vi, vh, vname)
            if vname == "rope.inv_freq":
                dim = int(dcfg["rope_freq_dim"])
                theta = float(dcfg.get("rope_theta", 10000.0))
                expected = 1.0 / (theta ** (torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim)))
            elif vname.endswith(".attn.qkv_proj.weight"):
                base = _diffusers_block_base(vname, ".attn.qkv_proj.weight")
                names = [f"{base}.attn.to_{part}.weight" for part in ("q", "k", "v")]
                expected = torch.cat([_get(di, dh, name) for name in names])
                consumed.update(names)
                actual = _reorder_grouped_qkv_to_qkv(
                    actual,
                    num_query_groups=int(vcfg["num_attention_heads"]),
                    heads_per_group=1,
                    head_dim=int(vcfg["attention_head_dim"]),
                )
            elif vname.endswith(".mlp.fc1.weight"):
                base = _diffusers_block_base(vname, ".mlp.fc1.weight")
                dname = f"{base}.ff.net.0.proj.weight"
                first, second = _get(di, dh, dname).chunk(2)
                expected = torch.cat([second, first])
                consumed.add(dname)
            else:
                dname = _diffusers_name(vname)
                if dname not in di:
                    raise AssertionError(f"No diffusers mapping for {vname}: candidate={dname}")
                expected = _get(di, dh, dname)
                consumed.add(dname)

            if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
                raise AssertionError(
                    f"Checkpoint mismatch for {vname}: actual={actual.shape}/{actual.dtype}, "
                    f"expected={expected.shape}/{expected.dtype}"
                )
            total_values += actual.numel()
            del actual, expected
            if index % 50 == 0:
                gc.collect()
                print(f"checked {index}/{len(vi)} official tensors", flush=True)

    if consumed != set(di):
        raise AssertionError(f"Unconsumed diffusers keys: {sorted(set(di) - consumed)}")
    print(
        f"PASS: {len(vi)} official tensors exactly match {len(di)} diffusers tensors after QKV/GEGLU conversion; "
        f"values={total_values:,}; elapsed={time.time() - started:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-transformer", type=Path, required=True)
    parser.add_argument("--diffusers-transformer", type=Path, required=True)
    args = parser.parse_args()
    compare(args.vllm_transformer, args.diffusers_transformer)


if __name__ == "__main__":
    main()
