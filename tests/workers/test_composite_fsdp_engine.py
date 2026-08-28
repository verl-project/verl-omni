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

"""Unit tests for ``CompositeFSDPEngine`` (Dual-GRPO AR + DiT FSDP backends).

Mirrors the trainer flow in ``PolicyGradientRayTrainer.fit``:

1. ``infer_batch`` on the AR batch (``num_prompts * rollout.n`` rows), then ``next_stage``.
2. ``infer_batch`` on the DiT batch (``num_prompts * rollout.n * rollout.m`` rows), then ``next_stage``.
3. ``train_batch`` on the AR batch (``ppo_loss``), then ``next_stage``.
4. ``train_batch`` on the DiT batch (``diffusion_loss``), then ``next_stage``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import partial

import pytest
import ray
import torch
from tensordict import TensorDict
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.workers.config import TrainingWorkerConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

from verl_omni.pipelines.utils import build_scheduler
from verl_omni.workers.config import DiffusionModelConfig, FSDPDiffusionActorConfig
from verl_omni.workers.engine_workers import TrainingWorker
from verl_omni.workers.utils.losses import diffusion_loss
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

from ..utils.gpu_test_topology import resolve_requested_num_gpus
from .test_diffusers_fsdp_engine import _diffusers_sp_supported, _create_sp_compatible_model


def _require_composite_model(model_path: str) -> None:
    if not os.path.isdir(os.path.join(model_path, "text_encoder")):
        pytest.skip(
            f"CompositeFSDPEngine requires a text_encoder/ subfolder under {model_path}. "
            "Use a Qwen-Image checkpoint that includes text_encoder/ (see composite agent-loop tests)."
        )


def create_composite_training_config(
    strategy: str,
    device_count: int,
    model_path: str,
) -> tuple[TrainingWorkerConfig, FSDPDiffusionActorConfig]:
    if strategy not in {"fsdp", "fsdp2"}:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    if device_count == 1:
        cp = fsdp_size = 1
    else:
        cp = 2 if _diffusers_sp_supported() else 1
        fsdp_size = device_count

    path = os.path.expanduser(model_path)
    tokenizer_path = os.path.join(path, "tokenizer")

    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    model_overrides = [
        "path=" + path,
        "tokenizer_path=" + tokenizer_path,
        "algorithm=dual_grpo",
        "lora_rank=8",
        "lora_alpha=16",
        "pipeline.true_cfg_scale=4.0",
        "algo.noise_level=1.2",
        "algo.sde_type=sde",
    ]
    from verl_omni.utils.diffusion_attention import fa3_available

    if cp > 1 or not fa3_available():
        model_overrides.append("attn_backend=native")

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/model")):
        cfg = compose(config_name="diffusion_model", overrides=model_overrides)
    model_config: DiffusionModelConfig = omega_conf_to_dataclass(cfg)

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor")):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=" + strategy,
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "diffusion_loss.loss_mode=dual_grpo",
                "ppo_mini_batch_size=4",
                "ppo_micro_batch_size_per_gpu=4",
                "optim.lr=1e-4",
                "optim.weight_decay=0.0001",
                "fsdp_config.param_offload=False",
                "fsdp_config.optimizer_offload=False",
                "fsdp_config.model_dtype='bfloat16'",
                "fsdp_config.dtype='bfloat16'",
                "+fsdp_config.mixed_precision.param_dtype='bfloat16'",
                "fsdp_config.forward_only=False",
                "fsdp_config.fsdp_size=" + str(fsdp_size),
                "fsdp_config.ulysses_sequence_parallel_size=" + str(cp),
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    training_config = TrainingWorkerConfig(
        model_type="diffusion_composite_model",
        model_config=model_config,
        engine_config=actor_config.engine,
        optimizer_config=actor_config.optim,
        checkpoint_config=actor_config.checkpoint,
    )
    return training_config, actor_config


def _create_ar_left_right_batch(
    batch_size: int,
    *,
    max_seq_len: int = 64,
    max_response_len: int = 32,
) -> TensorDict:
    response_ids = torch.randint(1, 1000, (batch_size, max_response_len))
    input_ids = torch.randint(1, 1000, (batch_size, max_seq_len))
    attention_mask = torch.ones(batch_size, max_seq_len)
    response_mask = torch.zeros(batch_size, max_response_len)
    response_mask[:, : max_response_len // 2] = 1
    position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return TensorDict(
        {
            "response_ids": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )


def create_ar_infer_batch(batch_size: int) -> TensorDict:
    batch = _create_ar_left_right_batch(batch_size)
    batch = left_right_2_no_padding(batch)
    tu.assign_non_tensor(
        batch,
        compute_loss=False,
        calculate_entropy=True,
        temperature=1.0,
    )
    return batch


def create_ar_train_batch(batch_size: int, *, micro_batch_size_per_gpu: int) -> TensorDict:
    max_seq_len = 64
    max_response_len = 32
    batch = _create_ar_left_right_batch(batch_size, max_seq_len=max_seq_len, max_response_len=max_response_len)
    batch["old_log_probs"] = torch.randn(batch_size, max_seq_len)
    batch["advantages"] = torch.randn(batch_size, max_response_len)
    batch = left_right_2_no_padding(batch)
    tu.assign_non_tensor(
        batch,
        compute_loss=True,
        calculate_entropy=False,
        temperature=1.0,
        global_batch_size=batch_size,
        mini_batch_size=batch_size,
        epochs=1,
        seed=42,
        dataloader_kwargs={"shuffle": False},
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
    )
    return batch


def create_dit_infer_batch(
    batch_size: int,
    model_config: DiffusionModelConfig,
    *,
    micro_batch_size_per_gpu: int,
) -> TensorDict:
    scheduler = build_scheduler(model_config)
    seq_len = 64
    latent_dim = 64
    encoder_latent_dim = 32
    vae_scale_factor = 8
    height, width = 512, 512
    latent_height, latent_width = height // vae_scale_factor // 2, width // vae_scale_factor // 2
    num_diffusion_steps = 10
    timesteps = scheduler.timesteps[None].repeat(batch_size, 1)

    batch = TensorDict(
        {
            "old_log_probs": torch.randn((batch_size, num_diffusion_steps)),
            "advantages": torch.randn((batch_size, num_diffusion_steps)),
            "all_latents": torch.randn((batch_size, num_diffusion_steps + 1, latent_height * latent_width, latent_dim)),
            "all_timesteps": timesteps,
            "prompt_embeds": torch.randn((batch_size, seq_len, encoder_latent_dim)),
            "prompt_embeds_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            "negative_prompt_embeds": torch.randn((batch_size, seq_len, encoder_latent_dim)),
            "negative_prompt_embeds_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
        },
        batch_size=batch_size,
    )
    batch = embeds_padding_2_no_padding(batch)
    tu.assign_non_tensor(
        batch,
        compute_loss=False,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
    )
    return batch


def create_dit_train_batch(
    batch_size: int,
    model_config: DiffusionModelConfig,
    *,
    micro_batch_size_per_gpu: int,
    ppo_mini_batch_size: int,
    ppo_epochs: int,
) -> TensorDict:
    batch = create_dit_infer_batch(
        batch_size,
        model_config,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
    )
    tu.assign_non_tensor(
        batch,
        compute_loss=True,
        global_batch_size=ppo_mini_batch_size,
        mini_batch_size=ppo_mini_batch_size,
        epochs=ppo_epochs,
        seed=42,
        dataloader_kwargs={"shuffle": False},
    )
    return batch


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_composite_fsdp_engine_infer_and_train(strategy: str) -> None:
    # if not torch.cuda.is_available():
    #     pytest.skip("CUDA is required for CompositeFSDPEngine GPU tests")

    rollout_n = 2 # dit
    rollout_m = 2 # ar

    ray.init()
    tmp_dir = tempfile.mkdtemp(prefix="composite_fsdp_engine_")
    try:
        device_count = resolve_requested_num_gpus(default_num_gpus=max(1, torch.cuda.device_count()))
        if device_count > 1 and device_count % 2 != 0:
            pytest.skip(f"Need even GPU count for cp=2/fsdp_size=device_count test, got {device_count}")

        sp_enabled = device_count > 1 and _diffusers_sp_supported()
        base_model_path = os.path.expanduser("~/models/tiny-random/Qwen-Image")
        if not os.path.isdir(base_model_path):
            pytest.skip(f"Missing tiny model at {base_model_path}")

        if sp_enabled:
            # SP requires num_attention_heads divisible by sp_size (cp=2).
            model_path = _create_sp_compatible_model(tmp_dir, base_model_path, num_attention_heads=2)
        else:
            model_path = base_model_path
        _require_composite_model(model_path)

        training_config, actor_config = create_composite_training_config(
            strategy=strategy,
            device_count=device_count,
            model_path=model_path,
        )

        micro_batch_size = actor_config.ppo_micro_batch_size_per_gpu
        ar_batch_size = rollout_m * device_count
        dit_batch_size = ar_batch_size * rollout_n

        # init training worker and engines
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=training_config)
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.reset()

        # Two-stage forward: AR then DiT
        # --- infer w/o loss: AR first (CompositeFSDPEngine starts on AR), then DiT via next_stage ---
        # stage 1: _compute_ar_old_log_prob
        ar_infer_td = create_ar_infer_batch(ar_batch_size)
        ar_infer_out = wg.infer_batch(ar_infer_td).get()
        for key in ["log_probs", "metrics", "entropy"]:
            assert key in ar_infer_out

        # stage 2: _compute_old_log_prob
        dit_infer_td = create_dit_infer_batch(
            dit_batch_size,
            training_config.model_config,
            micro_batch_size_per_gpu=micro_batch_size,
        )
        dit_infer_out = wg.infer_batch(dit_infer_td).get()
        assert dit_infer_out is not None
        assert "log_probs" in dit_infer_out
        assert "metrics" in dit_infer_out

        ar_log_probs = ar_infer_out["log_probs"]
        dit_log_probs = dit_infer_out["log_probs"]
        if isinstance(ar_log_probs, torch.Tensor):
            if ar_log_probs.is_nested:
                assert len(ar_log_probs.unbind()) == ar_batch_size
            else:
                assert ar_log_probs.shape[0] == ar_batch_size
        if isinstance(dit_log_probs, torch.Tensor):
            assert dit_log_probs.shape[0] == dit_batch_size
        assert ar_batch_size != dit_batch_size

        # --- train w/ loss: AR (token-level ppo_loss), then DiT (diffusion_loss / dual_grpo) ---
        # stage 1
        wg.set_loss_fn(partial(ppo_loss, config=actor_config))
        ar_train_td = create_ar_train_batch(ar_batch_size, micro_batch_size_per_gpu=micro_batch_size)
        ar_train_out = wg.train_batch(ar_train_td).get()
        assert ar_train_out is not None
        assert "metrics" in ar_train_out

        # stage 2
        wg.set_loss_fn(partial(diffusion_loss, config=actor_config))
        dit_train_td = create_dit_train_batch(
            dit_batch_size,
            training_config.model_config,
            micro_batch_size_per_gpu=micro_batch_size,
            ppo_mini_batch_size=actor_config.ppo_mini_batch_size * device_count,
            ppo_epochs=actor_config.ppo_epochs,
        )
        dit_train_out = wg.train_batch(dit_train_td).get()
        assert dit_train_out is not None
        assert "metrics" in dit_train_out
    finally:
        ray.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_composite_fsdp_engine_next_stage_toggles_engine() -> None:
    """Unit-check stage switching without loading weights (single-process mock)."""
    from unittest.mock import MagicMock

    from verl_omni.workers.engine.fsdp.diffusers_impl import CompositeFSDPEngine

    engine = CompositeFSDPEngine.__new__(CompositeFSDPEngine)
    engine.ar_engine = MagicMock(name="ar_engine")
    engine.dit_engine = MagicMock(name="dit_engine")
    engine.ar_stage = True
    engine.current_engine = engine.ar_engine

    engine.next_stage()
    assert engine.ar_stage is False
    assert engine.current_engine is engine.dit_engine

    engine.next_stage()
    assert engine.ar_stage is True
    assert engine.current_engine is engine.ar_engine
