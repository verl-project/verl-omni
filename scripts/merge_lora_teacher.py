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
"""Merge a sharded FSDP LoRA checkpoint into a base diffusion model to build a distillation teacher.

Reads ``model_world_size_{W}_rank_{r}.pt`` shards saved by the FSDP checkpoint
manager (peft-wrapped keys, DTensor values), folds LoRA deltas into the base
weights, and writes a full model directory usable as
``actor_rollout_ref.ref.model_path``: the merged transformer is saved for real,
every other pipeline component is symlinked from the base model directory.

Usage:
    python scripts/merge_lora_teacher.py \
        --base_model /path/to/stable-diffusion-3.5-medium \
        --checkpoint checkpoints/<proj>/<exp>/global_step_100/actor \
        --output /path/to/teacher_dir
"""

import argparse
import glob
import json
import os
import re

import torch
from torch.distributed.tensor import DTensor


def load_full_state_dict(ckpt_dir: str) -> dict[str, torch.Tensor]:
    """Load all FSDP rank shards and concatenate DTensor locals along the shard dim."""
    shard_paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "model_world_size_*_rank_*.pt")),
        key=lambda p: int(re.search(r"rank_(\d+)\.pt$", p).group(1)),
    )
    assert shard_paths, f"No model_world_size_*_rank_*.pt found in {ckpt_dir}"
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in shard_paths]

    state_dict = {}
    for key in shards[0]:
        values = [shard[key] for shard in shards]
        if isinstance(values[0], DTensor):
            placement = values[0].placements[0]
            locals_ = [v._local_tensor.bfloat16() for v in values]
            state_dict[key] = locals_[0] if placement.is_replicate() else torch.cat(locals_, dim=placement.dim)
        elif len(values) > 1 and not all(torch.equal(values[0], v) for v in values[1:]):
            state_dict[key] = torch.cat([v.bfloat16() for v in values], dim=0)
        else:
            state_dict[key] = values[0].bfloat16()
    return state_dict


def normalize_key(key: str) -> str:
    """Map peft/FSDP-wrapped names back to plain module names."""
    for prefix in ("base_model.model.", "_orig_mod.", "module."):
        key = key.replace(prefix, "")
    return key.replace(".base_layer.", ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model directory (diffusers layout).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint actor dir with model_world_size_*.pt shards.")
    parser.add_argument("--output", required=True, help="Output teacher model directory.")
    parser.add_argument("--subfolder", default="transformer", help="Pipeline subfolder the LoRA applies to.")
    parser.add_argument("--lora_rank", type=int, default=None, help="Override LoRA r (default: lora_train_meta.json).")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override LoRA alpha (same default).")
    args = parser.parse_args()

    from diffusers import SD3Transformer2DModel

    base_model = os.path.abspath(os.path.expanduser(args.base_model))
    ckpt = os.path.abspath(os.path.expanduser(args.checkpoint))
    output = os.path.abspath(os.path.expanduser(args.output))

    r, alpha = args.lora_rank, args.lora_alpha
    for meta_path in (
        os.path.join(os.path.dirname(ckpt), "lora_train_meta.json"),
        os.path.join(ckpt, "lora_train_meta.json"),
    ):
        if os.path.exists(meta_path) and (r is None or alpha is None):
            with open(meta_path) as f:
                meta = json.load(f)
            r = r if r is not None else int(meta["r"])
            alpha = alpha if alpha is not None else int(meta["lora_alpha"])
            break
    assert r and alpha, "LoRA r/alpha not found; pass --lora_rank/--lora_alpha"
    scale = alpha / r

    # Split checkpoint into base-weight overrides and LoRA A/B pairs.
    base_overrides: dict[str, torch.Tensor] = {}
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    for key, value in load_full_state_dict(ckpt).items():
        norm = normalize_key(key)
        m = re.match(r"(.*)\.lora_(A|B)(?:\.default)?\.weight$", norm)
        if m:
            (lora_a if m.group(2) == "A" else lora_b)[m.group(1)] = value
        else:
            base_overrides[norm] = value
    assert set(lora_a) == set(lora_b), "Mismatched lora_A/lora_B key sets"

    # Overlay onto the base transformer, then fold in W += scale * B @ A.
    transformer = SD3Transformer2DModel.from_pretrained(
        base_model, subfolder=args.subfolder, torch_dtype=torch.bfloat16
    )
    full = transformer.state_dict()
    unknown = [k for k in base_overrides if k not in full]
    assert not unknown, f"Checkpoint keys not in base model (first 5): {unknown[:5]}"
    full.update(base_overrides)
    for module, a in lora_a.items():
        target = f"{module}.weight"
        assert target in full, f"LoRA target {target} missing from base model"
        full[target] = full[target] + scale * (lora_b[module].float() @ a.float()).bfloat16()
    transformer.load_state_dict(full, strict=True)
    print(f"Merged {len(lora_a)} LoRA layers (scale={scale}), overrode {len(base_overrides)} base tensors")

    os.makedirs(output, exist_ok=True)
    transformer.save_pretrained(os.path.join(output, args.subfolder))
    for entry in os.listdir(base_model):
        if entry == args.subfolder or entry.startswith("."):
            continue
        dst = os.path.join(output, entry)
        if not os.path.exists(dst):
            os.symlink(os.path.join(base_model, entry), dst)
    print(f"Teacher model written to {output} (merged {args.subfolder} + symlinked components)")


if __name__ == "__main__":
    main()
