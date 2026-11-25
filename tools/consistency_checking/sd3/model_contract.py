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

"""Fixed public SD3.5 functional fixture and local checkpoint identity.

The path is supplied by the caller. This module never downloads a model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
WEIGHT_NAME = "transformer/diffusion_pytorch_model.safetensors"
WEIGHT_BYTES = 4939433672
WEIGHT_SHA256 = "f81eb93a348be218af757dd095cbda45090380d9bbda91049442e0e40bb4cc95"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weight(model_path: Path) -> str:
    weight = model_path / WEIGHT_NAME
    if not weight.is_file() or weight.stat().st_size != WEIGHT_BYTES or sha(weight) != WEIGHT_SHA256:
        raise ValueError("SD3.5 Medium transformer checkpoint differs from the fixed fixture")
    return WEIGHT_SHA256


def model_identity(model_path: Path) -> dict:
    # Check the complete local pipeline inventory before allocating a GPU.
    components = ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3")
    configs = {name: sha(model_path / name) for name in ("model_index.json", "scheduler/scheduler_config.json")}
    weights = []
    for component in components:
        config = model_path / component / "config.json"
        if not config.is_file():
            raise ValueError(f"missing {component} config")
        configs[str(config.relative_to(model_path))] = sha(config)
        files = sorted(
            path
            for path in (model_path / component).rglob("*")
            if path.is_file() and path.suffix in (".safetensors", ".bin")
        )
        if not files:
            raise ValueError(f"missing {component} weights")
        weights.extend([[str(path.relative_to(model_path)), path.stat().st_size] for path in files])
    for component in ("tokenizer", "tokenizer_2", "tokenizer_3"):
        directory = model_path / component
        if not directory.is_dir() or not any(path.is_file() for path in directory.iterdir()):
            raise ValueError(f"missing {component} files")
    if [row for row in weights if row[0] == WEIGHT_NAME] != [[WEIGHT_NAME, WEIGHT_BYTES]]:
        raise ValueError("transformer checkpoint layout differs from fixed fixture")
    return {
        "path": str(model_path.resolve(strict=True)),
        "configs": configs,
        "weights": weights,
        "tokenizer_dirs": ["tokenizer", "tokenizer_2", "tokenizer_3"],
        "revision_declaration": MODEL_REVISION,
        "integrity_scope": (
            "transformer weight SHA256 verified separately; other weight files inventoried by path and size"
        ),
    }


def configure(model_path: Path):
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig

    from verl_omni.workers.config import DiffusionModelConfig, FSDPDiffusionActorConfig
    from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig
    from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig, DiffusionRolloutAlgoConfig

    model = DiffusionModelConfig(
        path=str(model_path),
        algorithm="flow_grpo",
        load_tokenizer=False,
        attn_backend="native",
        enable_gradient_checkpointing=False,
        lora_rank=0,
        pipeline=DiffusionPipelineConfig(height=256, width=256, num_inference_steps=4, guidance_scale=0.0),
        algo=DiffusionRolloutAlgoConfig(noise_level=0.7, sde_type="sde"),
    )
    engine = FSDPEngineConfig(
        strategy="fsdp2",
        model_dtype="fp32",
        dtype="fp32",
        fsdp_size=1,
        ulysses_sequence_parallel_size=1,
        forward_only=True,
        param_offload=False,
        optimizer_offload=False,
        use_torch_compile=False,
        mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
        seed=1234,
        full_determinism=True,
    )
    optim = FSDPOptimizerConfig(
        optimizer="AdamW",
        optimizer_impl="torch.optim",
        lr=1e-5,
        weight_decay=0.0,
        clip_grad=1.0,
        total_training_steps=2,
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
    )
    actor = FSDPDiffusionActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        fsdp_config=engine,
        optim=optim,
        checkpoint=CheckpointConfig(),
        model_config=model,
        diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2, adv_clip_max=5.0),
    )
    return model, actor


def plain(tensor):
    import torch
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("DTensor materialization requires world size one")
        tensor = tensor.to_local()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(type(tensor))
    return tensor.detach().cpu().contiguous().clone()
