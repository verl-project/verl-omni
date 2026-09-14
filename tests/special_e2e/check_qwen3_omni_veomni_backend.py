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
"""Two-GPU VeOmni backend smoke: asymmetric images, EP=2 and full-weight export.

Run with ``torchrun --standalone --nproc_per_node=2``. Builds a tiny checkpoint
without downloading weights. Requires the GPU training environment with VeOmni.
"""

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from build_qwen3_omni_tiny_random import _build_tiny_config
from tensordict import TensorDict
from transformers import AutoConfig
from veomni.arguments import OpsImplementationConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models.auto import build_foundation_model
from verl.trainer.config import CheckpointConfig
from verl.workers.config import VeOmniEngineConfig, VeOmniOptimizerConfig

from verl_omni.workers.engine.veomni.omni_impl import OmniVeOmniEngine


def _build_checkpoint(path):
    """Save a random Thinker with four experts per layer and in-vocabulary image IDs."""
    config = _build_tiny_config(256)
    thinker = config.thinker_config
    thinker.image_token_id, thinker.video_token_id, thinker.audio_token_id = 100, 101, 102
    thinker.text_config.pad_token_id = 0
    thinker.text_config.eos_token_id = 2
    config.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    model = build_foundation_model(
        config_path=config,
        weights_path=None,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        ops_implementation=OpsImplementationConfig(
            attn_implementation="sdpa",
            moe_implementation="eager",
            cross_entropy_loss_implementation="eager",
            load_balancing_loss_implementation="eager",
        ),
        init_device="cpu",
    )
    model.save_pretrained(path)


def run(path: Path, moe_implementation: str):
    """Exercise the real sharded engine with image inputs on only one rank."""
    rank = dist.get_rank()
    if rank == 0:
        torch.manual_seed(42)
        _build_checkpoint(path)
    dist.barrier()
    config = AutoConfig.from_pretrained(path)
    model_config = SimpleNamespace(
        architecture=config.architectures[0],
        hf_config=config,
        model_stage="thinker",
        lora_rank=0,
        lora={},
        use_remove_padding=True,
        local_hf_config_path=str(path),
        local_path=str(path),
        enable_gradient_checkpointing=True,
        enable_activation_offload=False,
    )
    engine = OmniVeOmniEngine(
        model_config=model_config,
        engine_config=VeOmniEngineConfig(
            expert_parallel_size=2, attn_implementation="sdpa", moe_implementation=moe_implementation
        ),
        optimizer_config=VeOmniOptimizerConfig(lr=1e-5, total_training_steps=2),
        checkpoint_config=CheckpointConfig(),
    )
    engine._build_model_optimizer()
    assert not engine.module.has_talker
    for step in range(2):
        # The response also contains image_token_id; it must remain ordinary text.
        ids = torch.tensor([[10, 100 if rank == 0 else 11, 12, 100, 13, 14]], device="cuda")
        batch = TensorDict(
            {
                "input_ids": torch.nested.as_nested_tensor([ids[0]], layout=torch.jagged),
                "response_mask": torch.nested.as_nested_tensor([torch.ones(3, device=ids.device)], layout=torch.jagged),
            },
            batch_size=1,
        )
        inputs = {
            "input_ids": ids,
            "position_ids": torch.arange(6, device=ids.device).view(1, 1, 6).expand(3, 1, 6),
            "use_cache": False,
        }
        if rank == 0:
            inputs.update(
                pixel_values=torch.randn(4, 3 * 2 * 16 * 16, device=ids.device, dtype=torch.bfloat16),
                image_grid_thw=torch.tensor([[1, 2, 2]], device=ids.device),
            )
        engine._apply_veomni_input_transforms(inputs, batch)
        output = engine.module(**inputs, labels=ids, shift_labels=ids.roll(-1, dims=-1), return_log_probs=True)
        loss = -output.log_probs[..., 3:].mean()
        loss.backward()
        assert all(p.grad is None for p in engine.module.thinker.visual.parameters())
        grad_norm = engine.optimizer_step()
        engine.optimizer.zero_grad()
        engine.lr_scheduler.step()
        assert torch.isfinite(loss) and torch.isfinite(torch.tensor(grad_norm))
        print(f"PASS rank={rank} step={step} loss={loss.item():.6f} grad_norm={grad_norm:.6f}", flush=True)

    params, _ = engine.get_per_tensor_param()
    # Export reuses broadcast buffers: consume each tensor before advancing.
    weights = {name: weight.detach().clone() for name, weight in params}
    for layer in range(2):
        for expert in range(4):
            for projection in ("gate", "up", "down"):
                assert f"thinker.model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight" in weights
    assert not any("gate_up_proj" in name for name in weights)
    for name, weight in sorted(weights.items()):
        reference = weight.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(weight, reference, rtol=0, atol=0, msg=lambda msg, name=name: f"{name}: {msg}")
    print(f"PASS rank={rank} EP=2 export: {len(weights)} tensors match across ranks", flush=True)


def main():
    """Run on exactly two local GPUs, cleaning up the temporary checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-implementation", default="fused_triton")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This backend smoke test requires exactly two GPUs on one node.")
        init_parallel_state(dp_size=2, dp_replicate_size=1, dp_shard_size=2, extra_parallel_sizes=(2,))
        with tempfile.TemporaryDirectory(prefix="qwen3-veomni-smoke-") as local_dir:
            shared = [local_dir if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(shared, src=0)
            run(Path(shared[0]) / "model", args.moe_implementation)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
