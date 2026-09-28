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
from verl.utils import tensordict_utils as tu
from verl.workers.config import VeOmniEngineConfig, VeOmniOptimizerConfig

from verl_omni.workers.engine.veomni.omni_impl import OmniVeOmniEngine


def _build_checkpoint(path):
    """Save a random Thinker with four experts per layer and in-vocabulary image IDs."""
    config = _build_tiny_config(256)
    # Exercise the Instruct config contract, rather than disabling speech in
    # the fixture and then merely asserting that it stayed disabled.
    config.enable_audio_output = True
    config.enable_talker = True
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
    assert not model.has_talker
    state = model.state_dict()
    state.update(
        {
            "talker.fixture.weight": torch.ones(2, 2),
            "talker.code_predictor.fixture.weight": torch.ones(2, 2),
            "code2wav.fixture.weight": torch.ones(2, 2),
        }
    )
    # Check VeOmni's strict state-dict hook and its actual checkpoint loader.
    model.load_state_dict(state, strict=True)
    model.save_pretrained(path, state_dict=state)


def run(path: Path, moe_implementation: str, attn_implementation: str):
    """Exercise the real sharded engine with image inputs on only one rank."""
    rank = dist.get_rank()
    if rank == 0:
        torch.manual_seed(42)
        _build_checkpoint(path)
    dist.barrier()
    config = AutoConfig.from_pretrained(path)
    assert config.enable_audio_output and config.enable_talker
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
    # Use VeOmni's native defaults rather than verl's conservative eager
    # defaults for the Qwen3 ops. The CPU checkpoint fixture above is separate.
    ops = OpsImplementationConfig(attn_implementation=attn_implementation, moe_implementation=moe_implementation)
    engine = OmniVeOmniEngine(
        model_config=model_config,
        engine_config=VeOmniEngineConfig(
            expert_parallel_size=2,
            attn_implementation=attn_implementation,
            moe_implementation=ops.moe_implementation,
            cross_entropy_loss_implementation=ops.cross_entropy_loss_implementation,
            rms_norm_implementation=ops.rms_norm_implementation,
            swiglu_mlp_implementation=ops.swiglu_mlp_implementation,
            rotary_pos_emb_implementation=ops.rotary_pos_emb_implementation,
            load_balancing_loss_implementation=ops.load_balancing_loss_implementation,
        ),
        optimizer_config=VeOmniOptimizerConfig(lr=1e-5, total_training_steps=2),
        checkpoint_config=CheckpointConfig(),
    )
    engine._build_model_optimizer()
    assert not engine.module.has_talker
    assert not any(
        name.rsplit(".", 1)[-1] in {"talker", "code2wav", "code_predictor"} for name, _ in engine.module.named_modules()
    )
    expected_params = {id(p) for name, p in engine.module.named_parameters() if p.requires_grad}
    assert all(name.startswith("thinker.") for name, p in engine.module.named_parameters() if p.requires_grad)
    optimizer_params = {
        id(p)
        for optimizer in engine.optimizer.optimizers_dict.values()
        for group in optimizer.param_groups
        for p in group["params"]
    }
    assert optimizer_params == expected_params
    print(f"PASS rank={rank} speech-enabled checkpoint: Thinker-only optimizer", flush=True)
    for step, temperature in enumerate((1.0, 0.8)):
        # The response also contains image_token_id; it must remain ordinary text.
        ids = torch.tensor([[10, 100 if rank == 0 else 11, 12, 100, 13, 14]], device="cuda")
        offsets = torch.tensor([0, ids.numel()], device=ids.device)
        batch = TensorDict(
            {
                "input_ids": torch.nested.as_nested_tensor([ids[0]], layout=torch.jagged),
                "position_ids": torch.nested.nested_tensor_from_jagged(
                    torch.arange(6, device=ids.device).expand(3, 6).contiguous(), offsets, jagged_dim=2
                ),
                "response_mask": torch.nested.as_nested_tensor([torch.ones(3, device=ids.device)], layout=torch.jagged),
            },
            batch_size=1,
        )
        if rank == 0:
            tu.assign_non_tensor(
                batch,
                multi_modal_inputs=[
                    {
                        "pixel_values": torch.randn(4, 3 * 2 * 16 * 16, device=ids.device, dtype=torch.bfloat16),
                        "image_grid_thw": torch.tensor([[1, 2, 2]], device=ids.device),
                    }
                ],
            )
        tu.assign_non_tensor(
            batch, use_fused_kernels=False, use_remove_padding=True, temperature=temperature, calculate_entropy=True
        )
        # Use the production batch flag -> inputs -> outputs path in both modes.
        # Disable dropout so the comparison isolates the LM output protocol.
        engine.module.eval()
        with torch.no_grad():
            inputs, output_args = engine.prepare_model_inputs(batch)
            output = engine.module(**inputs, use_cache=False)
            scalar_reference = (output.logits / temperature).float().log_softmax(-1)
            scalar_reference = scalar_reference.gather(-1, ids.roll(-1, dims=-1).unsqueeze(-1)).view(-1)
            reference = engine.prepare_model_outputs(output, output_args, batch, logits_processor_func=None)
        tu.assign_non_tensor(batch, use_fused_kernels=True)
        inputs, output_args = engine.prepare_model_inputs(batch)
        assert inputs["return_log_probs"] is True
        assert inputs["temperature"] == temperature
        torch.testing.assert_close(inputs["shift_labels"], ids.roll(-1, dims=-1))
        output = engine.module(**inputs, use_cache=False)
        processed = engine.prepare_model_outputs(output, output_args, batch, logits_processor_func=None)
        for key in ("log_probs", "entropy"):
            # Fused reductions return FP32; the logits path may retain BF16.
            torch.testing.assert_close(
                processed[key].values().float(), reference[key].values().float(), atol=0.03, rtol=0.01
            )
        torch.testing.assert_close(processed["log_probs"].values(), scalar_reference, atol=0.03, rtol=0.01)
        print(f"PASS rank={rank} fused log_probs/entropy vs logits: temperature={temperature}", flush=True)
        loss = -processed["log_probs"].values()[3:].mean()
        loss.backward()
        assert all(p.grad is None for p in engine.module.thinker.visual.parameters())
        grad_norm = engine.optimizer_step()
        engine.optimizer.zero_grad()
        engine.lr_scheduler.step()
        engine.module.train()
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
    parser.add_argument("--attn-implementation", default="flash_attention_2")
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
            run(Path(shared[0]) / "model", args.moe_implementation, args.attn_implementation)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
