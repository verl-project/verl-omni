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
"""Two-rank CPU FSDP2 one-step training and production checkpoint saving."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from model_fixtures import forward_inputs, tiny_transformer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def _tensor_outputs(output) -> tuple[torch.Tensor, ...]:
    values = output if isinstance(output, tuple | list) else (output,)
    tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
    if not tensors:
        raise ValueError("Tiny transformer forward returned no tensors")
    return tensors


def _patch_diffusers_checkpoint_config(model) -> None:
    """Mirror DiffusersFSDPEngine's config bridge used by the checkpoint manager."""
    model.register_to_config(_class_name=type(model).__name__)

    def save_config(config, save_directory):
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True))

    model.config.save_pretrained = save_config.__get__(model.config)
    model.can_generate = lambda: False


def _run_one(architecture: str, root: Path, mesh) -> None:
    rank = dist.get_rank()
    model = tiny_transformer(architecture)
    if architecture != "BooguImagePipeline":
        model.set_attention_backend("native")
    _patch_diffusers_checkpoint_config(model)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.eval()
    with torch.no_grad():
        before = tuple(value.detach().cpu().clone() for value in _tensor_outputs(model(**forward_inputs(architecture))))

    model.train()
    outputs = _tensor_outputs(model(**forward_inputs(architecture)))
    loss = sum((value.float() - 0.25).square().mean() for value in outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        after = tuple(value.detach().cpu().clone() for value in _tensor_outputs(model(**forward_inputs(architecture))))
    if not any(not torch.equal(left, right) for left, right in zip(before, after, strict=True)):
        raise AssertionError(f"One optimizer step did not change {architecture} output")

    output = root / architecture
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"before": before, "after": after, "loss": float(loss.detach())}, output / "expected.pt")
    manager = FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        checkpoint_config={"save_contents": ["model"], "load_contents": ["model"]},
    )
    manager.save_checkpoint(str(output / "actor"), global_step=1)
    dist.barrier()
    del manager, optimizer, model, before, after, outputs, loss
    gc.collect()


def main() -> None:
    """Train and checkpoint each requested architecture in one two-rank process group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("architectures", nargs="+")
    args = parser.parse_args()
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
        for architecture in args.architectures:
            _run_one(architecture, args.root, mesh)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
