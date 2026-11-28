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

"""Exercise the registered H3 Actor on CUDA or Ascend using a tiny fused checkpoint.

Run with torchrun --standalone --nproc-per-node=2 and a fresh shared fixture directory.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.workers.engine.base import EngineRegistry

from verl_omni.pipelines.minimax_h3_flow_grpo.common import flatten_joint_latents
from verl_omni.pipelines.minimax_h3_flow_grpo.veomni import FUSED_TARGETS, predict_row_velocities
from verl_omni.workers.config import DiffusionModelConfig, VeOmniDiffusionEngineConfig, VeOmniDiffusionOptimizerConfig
from verl_omni.workers.engine.veomni import VeOmniDiffusionEngine


def make_fixture(root):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
        MiniMaxH3DiTModelConfig,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    path = root / "transformer"
    path.mkdir(parents=True, exist_ok=False)
    params = dict(
        hidden_size=32,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=64,
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        adaln_out_features=576,
        final_adaln_out_features=64,
        rope_inv_freq_len=2,
    )
    torch.manual_seed(196)
    model = MiniMaxH3DiTModel(
        MiniMaxH3DiTModelConfig(
            **params,
            ops_implementation=OpsImplementationConfig(
                attn_implementation="eager",
                rms_norm_implementation="eager",
                rotary_pos_emb_implementation="eager",
            ),
        )
    )
    (path / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3DiTModel", **params}))
    (root / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3Pipeline"}))
    save_file({k.removeprefix("dit."): v for k, v in model.state_dict().items()}, path / "model.safetensors")


def make_transition(inputs, engine):
    video, audio = inputs["hidden_states"], inputs["audio_hidden_states"]
    shape = (1, 1)

    def scalar(n):
        return torch.full(shape, n, dtype=torch.long, device=video.device)

    video_t, audio_t = (s.timesteps[1].reshape(shape) for s in engine.scheduler)
    batch = TensorDict(
        dict(
            all_latents=flatten_joint_latents(video, audio).unsqueeze(1),
            all_next_latents=flatten_joint_latents(video + 0.05, audio - 0.03).unsqueeze(1),
            all_timesteps=video_t,
            h3_audio_timesteps=audio_t,
            h3_step_indices=scalar(1),
            h3_video_rows=scalar(2),
            h3_audio_rows=scalar(2),
            h3_seq_len=scalar(6),
            h3_position_ids=inputs["position_ids"].unsqueeze(0),
            h3_token_tags=inputs["token_tags"].unsqueeze(0),
            h3_video_indices=inputs["video_indices"].unsqueeze(0),
            h3_audio_indices=inputs["audio_indices"].unsqueeze(0),
            h3_text_indices=inputs["text_indices"].unsqueeze(0),
            h3_video_update_mask=torch.ones(1, 2, device=video.device, dtype=torch.bool),
            prompt_embeds=inputs["encoder_hidden_states"],
            prompt_embeds_mask=torch.ones(1, 2, device=video.device, dtype=torch.bool),
            advantages=torch.ones_like(video_t),
        ),
        batch_size=[1],
    )
    tu.assign_non_tensor(batch, micro_batch_size_per_gpu=1)
    return batch


def run(root, offload):
    import verl_omni.pipelines  # noqa: F401  (register H3)
    from verl_omni.workers.config.diffusion.actor import VeOmniDiffusionActorConfig
    from verl_omni.workers.utils.losses import diffusion_loss

    device = torch.device(get_device_name(), int(os.environ["LOCAL_RANK"]))
    get_torch_device().set_device(device)
    dist.init_process_group(get_nccl_backend())
    if dist.get_rank() == 0:
        make_fixture(root)
    dist.barrier()
    model = DiffusionModelConfig(
        path=str(root),
        algorithm="flow_grpo",
        training_backend="veomni",
        load_tokenizer=False,
        lora_init_weights=True,
        lora_rank=4,
        lora_alpha=8,
        target_modules=list(FUSED_TARGETS),
        attn_backend="_native_npu" if device.type == "npu" else "native",
    )
    config = VeOmniDiffusionEngineConfig(
        attn_implementation="veomni_sdpa",
        mixed_precision_cast_forward_inputs=False,
        param_offload=offload,
        optimizer_offload=offload,
    )
    engine_cls = EngineRegistry.get_engine_cls("diffusion_model", "veomni")
    assert engine_cls is VeOmniDiffusionEngine
    engine = engine_cls(
        model,
        config,
        VeOmniDiffusionOptimizerConfig(lr=0.01, total_training_steps=2, betas=(0.9, 0.999), eps=1e-8, fused=False),
        CheckpointConfig(),
    )
    engine.initialize()
    assert isinstance(engine.runtime.optimizer, torch.optim.AdamW)
    for group in engine.runtime.optimizer.param_groups:
        assert tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8 and group["fused"] is False
    torch.manual_seed(82)

    def tensor(values):
        return torch.tensor(values, device=device)

    inputs = dict(
        hidden_states=torch.randn(1, 2, 96, device=device),
        audio_hidden_states=torch.randn(1, 2, 32, device=device),
        encoder_hidden_states=torch.randn(1, 2, 32, device=device),
        position_ids=torch.zeros(6, 3, device=device),
        token_tags=tensor([0, 0, 2, 2, 1, 1]),
        video_indices=tensor([0, 1]),
        audio_indices=tensor([2, 3]),
        text_indices=tensor([4, 5]),
        timestep=tensor([0.375, 0.75]),
        timestep_indices=tensor([0, 0, 1, 1, 0, 0]),
    )
    batch = make_transition(inputs, engine)
    actor_config = VeOmniDiffusionActorConfig(rollout_n=2, ppo_micro_batch_size_per_gpu=1)
    with engine.eval_mode(), torch.no_grad():
        before = predict_row_velocities(engine.module, inputs)
        old = engine.forward_backward_batch(batch, loss_function=None, forward_only=True)
        batch["old_log_probs"] = old["model_output"]["log_probs"].detach().clone()
    with engine.train_mode():
        trained = engine.forward_backward_batch(batch, loss_function=partial(diffusion_loss, actor_config))
        log_ratio = trained["model_output"]["log_probs"].detach() - batch["old_log_probs"]
        torch.testing.assert_close(log_ratio.exp(), torch.ones_like(log_ratio), atol=1e-5, rtol=1e-5)
        loss = trained["loss"][0][0]
        norm = engine.optimizer_step()
        assert 0 < norm < float("inf"), norm
        engine.lr_scheduler_step()
    with engine.eval_mode(), torch.no_grad():
        after = predict_row_velocities(engine.module, inputs)
        assert all(v.isfinite().all() for v in after)
        assert any(not torch.equal(x, y) for x, y in zip(before, after, strict=True))
    params, _ = engine.get_per_tensor_param(base_sync_done=False)
    base = dict(params)
    assert "transformer.condition_proj.weight" in base
    for path in engine.runtime.fp32_modules:
        prefix = "transformer." + path.removeprefix("dit.")
        assert base[prefix + ".weight"].dtype == torch.float32
        assert base[prefix + ".bias"].dtype == torch.float32
    params, cfg = engine.get_per_tensor_param(base_sync_done=True)
    delta = dict(params)
    assert set(cfg["target_modules"]) == set(FUSED_TARGETS) and any(
        v.any() for k, v in delta.items() if ".lora_B." in k
    )
    if dist.get_rank() == 0:
        print(
            f"PASS ranks={dist.get_world_size()} loss={loss} "
            f"ratio={log_ratio.exp().mean().item()} grad_norm={norm} LoRA tensors={len(delta)}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--offload", action="store_true", help="Exercise the colocated recipe CPU offload policy.")
    args = parser.parse_args()
    run(args.fixture, args.offload)
