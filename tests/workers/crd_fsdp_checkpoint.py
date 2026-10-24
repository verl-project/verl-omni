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
"""Run with torchrun: real CRD FSDP2 updates, independent EMA and exact checkpoint restoration."""

import argparse
import copy
import math
import os
from functools import partial

import torch
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig

from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig, DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin
from verl_omni.workers.utils.losses import diffusion_loss


class Policy(torch.nn.Module):
    """Small named-adapter module with PEFT's trainability switching behavior."""

    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.tensor([value]))
                for name, value in [("default", 4.0), ("old", 0.0), ("moving_reference", 2.0)]
            }
        )
        self.set_adapter("default")

    def set_adapter(self, name):
        self.active_adapter = name
        for key, value in self.lora_A.items():
            value.requires_grad_(key == name)


class Engine(LoRAAdapterMixin):
    """Use the production copy/EMA helpers without FSDP or model loading."""

    def __init__(self):
        self.module = Policy()


def local_state(value):
    """Clone state without retaining references to mutable parameters or optimizer buffers."""
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: local_state(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(local_state(item) for item in value)
    return copy.deepcopy(value)


def assert_same_state(actual, expected):
    """Require exact model, optimizer and scheduler restoration, including frozen adapters."""
    torch.testing.assert_close(local_state(actual), local_state(expected), rtol=0, atol=0)


def assert_adapter_updates(before, trained, refreshed, decays):
    """Check only default changed under the optimizer and each EMA used its own decay."""
    changed = False
    for name, old_value in before.items():
        if ".default." in name:
            changed |= not torch.equal(trained[name], old_value)
        else:
            assert_same_state(trained[name], old_value)
        for adapter, decay in decays.items():
            if f".{adapter}." in name:
                source = name.replace(f".{adapter}.", ".default.")
                expected = trained[name].lerp(trained[source], 1 - decay)
                torch.testing.assert_close(refreshed[name], expected)
    if not changed:
        raise AssertionError("CRD did not change the trainable adapter.")
    for adapter in ("default", *decays):
        if not any(f".{adapter}." in name for name in before):
            raise AssertionError(f"Missing policy adapter: {adapter}")


def snapshot(engine):
    """Capture local model shards and all state needed for the next optimizer step."""
    return local_state(
        {
            "model": engine.module.state_dict(),
            "optimizer": engine.optimizer.state_dict(),
            "scheduler": engine.lr_scheduler.state_dict(),
        }
    )


def exercise_checkpoint(engine, batch, actor, checkpoint_dir):
    """Compare a restored next update with the same uninterrupted update on every rank."""
    decays = {"old": 0.2, "moving_reference": 0.7}

    def update():
        with engine.train_mode():
            engine.optimizer_zero_grad()
            # Use the production accumulation and loss-dispatch paths.
            engine.forward_backward_batch(batch.clone(), partial(diffusion_loss, config=actor))
            norm = engine.optimizer_step()
            if not math.isfinite(norm) or norm <= 0:
                raise AssertionError(f"Expected a finite positive CRD gradient norm, got {norm}")
            engine.lr_scheduler_step()

    before = snapshot(engine)["model"]
    update()
    trained = snapshot(engine)["model"]
    for adapter, decay in decays.items():
        engine.ema_update_adapter(source="default", target=adapter, decay=decay)
    saved = snapshot(engine)
    assert_adapter_updates(before, trained, saved["model"], decays)
    engine.save_checkpoint(checkpoint_dir, global_step=1)
    update()
    uninterrupted = snapshot(engine)
    engine.load_checkpoint(checkpoint_dir, del_local_after_load=False)
    assert_same_state(snapshot(engine), saved)
    update()
    assert_same_state(snapshot(engine), uninterrupted)


def make_batch(model_config, device):
    """Two complete samples for one prompt, with independent noise and explicit negative conditioning."""
    cfg = model_config.transformer_config
    generator = torch.Generator().manual_seed(7)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    batch = TensorDict(
        {
            "latents_clean": rand(2, cfg["in_channels"], 8, 8),
            "forward_noise": rand(2, 1, cfg["in_channels"], 8, 8),
            "train_timesteps": torch.full((2, 1), 500, device=device),
            "prompt_embeds": rand(2, 8, cfg["joint_attention_dim"]),
            "prompt_embeds_mask": torch.ones(2, 8, device=device),
            "negative_prompt_embeds": rand(2, 8, cfg["joint_attention_dim"]),
            "negative_prompt_embeds_mask": torch.ones(2, 8, device=device),
            "pooled_prompt_embeds": rand(2, cfg["pooled_projection_dim"]),
            "negative_pooled_prompt_embeds": rand(2, cfg["pooled_projection_dim"]),
            "advantages": torch.tensor([[-1.0], [1.0]], device=device),
            "raw_rewards": torch.tensor([0.2, 0.8], device=device),
            "group_index": torch.zeros(2, dtype=torch.long, device=device),
            "group_size": torch.full((2,), 2, device=device),
        },
        [2],
    )
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(batch, micro_batch_size_per_gpu=2, use_dynamic_bsz=False)
    return batch


def main():
    """Initialize one FSDP2 rank per GPU; fail if any checkpoint/update assertion fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    try:
        model = DiffusionModelConfig(
            path=args.model,
            algorithm="crd",
            model_type="diffusion_nft_model",
            load_tokenizer=False,
            attn_backend="native",
            lora_rank=8,
            lora_alpha=16,
            policy_state_adapters=("default", "old", "moving_reference"),
        )
        config = FSDPEngineConfig(
            strategy="fsdp2",
            model_dtype="fp32",
            mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
        )
        optimizer = FSDPOptimizerConfig(lr=1e-4, total_training_steps=4)
        engine = NFTDiffusersFSDPEngine(model, config, optimizer, CheckpointConfig())
        engine.initialize()
        for adapter in ("old", "moving_reference"):
            engine.copy_adapter(source="default", target=adapter)
        actor = DiffusionActorConfig(
            strategy="fsdp2", rollout_n=2, diffusion_loss=DiffusionLossConfig(loss_mode="crd", ref_kl_coef=0.1)
        )
        batch = make_batch(model, torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        exercise_checkpoint(engine, batch, actor, args.checkpoint_dir)
        if torch.distributed.get_rank() == 0:
            print("CRD FSDP2: weight update, frozen states, independent EMA and exact save/resume passed.")
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
