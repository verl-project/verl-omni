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

"""End-to-end tests for the VeOmni-backed diffusion FSDP2 actor engine.

Mirrors :mod:`tests.workers.test_diffusers_fsdp_engine` but exercises the
``veomni`` strategy in isolation so that VeOmni-specific imports and config
overrides do not bleed into the diffusers FSDP/FSDP2 test surface.
"""

import os
import shutil
import tempfile
from functools import partial

import numpy as np
import pytest
import ray
import torch
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.workers.config import TrainingWorkerConfig

from verl_omni.pipelines.utils import build_scheduler
from verl_omni.workers.config import DiffusionModelConfig, VeOmniDiffusionActorConfig
from verl_omni.workers.engine_workers import TrainingWorker
from verl_omni.workers.utils.losses import diffusion_loss
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

from ..utils.gpu_test_topology import resolve_requested_num_gpus


def create_training_config(model_type, device_count, model):
    cp = 1
    fsdp_size = device_count
    path = os.path.expanduser(model)
    tokenizer_path = os.path.join(path, "tokenizer")

    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/model")):
        cfg = compose(
            config_name="diffusion_model",
            overrides=[
                "path=" + path,
                "tokenizer_path=" + tokenizer_path,
                "lora_rank=0",
                "pipeline.true_cfg_scale=4.0",
                "algo.noise_level=1.2",
                "algo.sde_type=sde",
                "config_path=" + os.path.join(path, "transformer"),
            ],
        )
    model_config: DiffusionModelConfig = omega_conf_to_dataclass(cfg)

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor")):
        cfg = compose(
            config_name="veomni_diffusion_actor",
            overrides=[
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_mini_batch_size=4",
                "ppo_micro_batch_size_per_gpu=4",
                "optim.lr=1e-4",
                "optim.weight_decay=0.0001",
                "veomni_config.param_offload=False",
                "veomni_config.optimizer_offload=False",
                "veomni_config.mixed_precision=True",
                "veomni_config.forward_only=False",
                "veomni_config.fsdp_size=" + str(fsdp_size),
                "veomni_config.ulysses_parallel_size=" + str(cp),
                "diffusion_loss.loss_mode='flow_grpo'",
            ],
        )
    actor_config: VeOmniDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    training_config = TrainingWorkerConfig(
        model_type=model_type,
        model_config=model_config,
        engine_config=actor_config.engine,
        optimizer_config=actor_config.optim,
        checkpoint_config=actor_config.checkpoint,
    )
    return training_config, actor_config


def create_data_samples(num_device: int, model_config: DiffusionModelConfig) -> DataProto:
    from tensordict import TensorDict

    scheduler = build_scheduler(model_config)

    batch_size = 8 * num_device
    seq_len = 64
    latent_dim = 64
    encoder_latent_dim = 32
    vae_scale_factor = 8
    height, width = 512, 512
    latent_height, latent_width = height // vae_scale_factor // 2, width // vae_scale_factor // 2
    num_diffusion_steps = 10
    timesteps = scheduler.timesteps[None].repeat(batch_size, 1)

    torch.manual_seed(1)
    np.random.seed(1)

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
    data = DataProto(batch=batch)
    data.meta_info["micro_batch_size_per_gpu"] = 4
    data.meta_info["height"] = height
    data.meta_info["width"] = width
    data.meta_info["vae_scale_factor"] = vae_scale_factor

    return data


def test_veomni_fsdp_engine():
    ray.init()
    tmp_dir = tempfile.mkdtemp(prefix="qwen_image_veomni_")
    try:
        visible_gpus = torch.cuda.device_count()
        device_count = resolve_requested_num_gpus(default_num_gpus=max(1, visible_gpus))
        if device_count > 1 and device_count % 2 != 0:
            pytest.skip(f"Need even GPU count for fsdp_size=device_count test, got {device_count}")

        model_path = os.path.expanduser("~/models/tiny-random/Qwen-Image")
        training_config, actor_config = create_training_config(
            model_type="diffusion_model",
            device_count=device_count,
            model=model_path,
        )

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=training_config)
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.reset()

        data_td = create_data_samples(device_count, training_config.model_config).to_tensordict()
        data_td = embeds_padding_2_no_padding(data_td)
        tu.assign_non_tensor(
            data_td,
            compute_loss=False,
            height=training_config.model_config.get("height", 512),
            width=training_config.model_config.get("width", 512),
            vae_scale_factor=training_config.model_config.get("vae_scale_factor", 8),
        )
        output = wg.infer_batch(data_td)
        output_dict = output.get()

        for key in ["log_probs", "metrics"]:
            assert key in output_dict

        loss_fn = partial(diffusion_loss, config=actor_config)
        wg.set_loss_fn(loss_fn)

        data_td = create_data_samples(device_count, training_config.model_config).to_tensordict()
        data_td = embeds_padding_2_no_padding(data_td)
        ppo_mini_batch_size = 4
        ppo_epochs = actor_config.ppo_epochs
        seed = 42
        shuffle = actor_config.shuffle
        tu.assign_non_tensor(
            data_td,
            global_batch_size=ppo_mini_batch_size * device_count,
            mini_batch_size=ppo_mini_batch_size * device_count,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )
        output = wg.train_mini_batch(data_td)
        output_dict = output.get()

        assert "metrics" in output_dict.keys()
    finally:
        ray.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)
