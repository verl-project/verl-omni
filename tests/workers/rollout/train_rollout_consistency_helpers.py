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
"""Shared helpers for train–rollout consistency GPU tests."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import TrainingWorkerConfig
from verl.workers.rollout.replica import RolloutMode

from verl_omni.workers.config import DiffusionModelConfig, FSDPDiffusionActorConfig
from verl_omni.workers.engine_workers import TrainingWorker
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

TINY_QWEN_IMAGE_PATH = Path(os.path.expanduser("~/models/tiny-random/Qwen-Image"))
FULL_QWEN_IMAGE_LOCAL_PATH = Path(os.path.expanduser("~/models/Qwen/Qwen-Image"))
FULL_QWEN_IMAGE_HUB_ID = "Qwen/Qwen-Image"
_MIN_PROMPT_TOKENS = 35
QwenImageModelVariant = Literal["tiny", "full"]
QwenImageRecipe = Literal["default", "ocr"]
OCR_TARGET_MODULES = (
    "['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out',"
    "'img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']"
)


@dataclass(frozen=True)
class QwenImageConsistencyProfile:
    variant: QwenImageModelVariant
    recipe: QwenImageRecipe
    model_path: str
    gpu_memory_utilization: float
    generate_timeout_s: int
    fsdp_param_offload: bool
    load_format: str
    lora_rank: int
    lora_alpha: int
    target_modules: str
    layered_summon: bool
    sde_window_size: int
    sde_window_range: tuple[int, int]
    max_sequence_length: int
    num_inference_steps: int = 10
    attn_backend: str = "native"

    @property
    def uses_hybrid_lora_sync(self) -> bool:
        return self.lora_rank > 0 and self.layered_summon


def resolve_qwen_image_profile(variant: str, recipe: str = "default") -> QwenImageConsistencyProfile:
    if recipe not in ("default", "ocr"):
        raise ValueError(f"Unknown Qwen-Image recipe {recipe!r}; expected 'default' or 'ocr'")

    if variant == "tiny":
        base = dict(
            variant="tiny",
            model_path=str(TINY_QWEN_IMAGE_PATH),
            gpu_memory_utilization=0.55,
            generate_timeout_s=600,
            fsdp_param_offload=False,
            load_format="auto",
        )
    elif variant == "full":
        model_path = os.environ.get("QWEN_IMAGE_MODEL_PATH")
        if model_path is None:
            model_path = str(
                FULL_QWEN_IMAGE_LOCAL_PATH if FULL_QWEN_IMAGE_LOCAL_PATH.exists() else FULL_QWEN_IMAGE_HUB_ID
            )
        base = dict(
            variant="full",
            model_path=model_path,
            gpu_memory_utilization=0.90,
            generate_timeout_s=1800,
            fsdp_param_offload=True,
            load_format="safetensors",
        )
    else:
        raise ValueError(f"Unknown Qwen-Image variant {variant!r}; expected 'tiny' or 'full'")

    if recipe == "default":
        return QwenImageConsistencyProfile(
            recipe="default",
            lora_rank=0,
            lora_alpha=64,
            target_modules="all-linear",
            layered_summon=False,
            sde_window_size=2,
            sde_window_range=(0, 2),
            max_sequence_length=1058,
            **base,
        )

    # OCR FlowGRPO recipe (run_qwen_image_ocr_lora.sh); tiny uses all-linear like e2e smoke.
    target_modules = OCR_TARGET_MODULES if variant == "full" else "all-linear"
    return QwenImageConsistencyProfile(
        recipe="ocr",
        lora_rank=64,
        lora_alpha=128,
        target_modules=target_modules,
        layered_summon=True,
        sde_window_size=2,
        sde_window_range=(0, 5),
        max_sequence_length=256,
        load_format="safetensors",
        **{k: v for k, v in base.items() if k != "load_format"},
    )


def resolve_tokenizer_path(model_path: str) -> str:
    expanded = Path(os.path.expanduser(model_path))
    if expanded.is_dir():
        tokenizer_dir = expanded / "tokenizer"
        if tokenizer_dir.is_dir():
            return str(tokenizer_dir)
    return model_path


def require_qwen_image_model(profile: QwenImageConsistencyProfile) -> str:
    import pytest

    if profile.variant == "tiny":
        if not Path(profile.model_path).expanduser().exists():
            pytest.skip(f"Tiny Qwen-Image fixture not found at {profile.model_path}")
        return profile.model_path

    expanded = Path(os.path.expanduser(profile.model_path))
    if expanded.exists() or profile.model_path == FULL_QWEN_IMAGE_HUB_ID:
        return profile.model_path
    pytest.skip(
        f"Full Qwen-Image not found at {profile.model_path}. "
        f"Download to {FULL_QWEN_IMAGE_LOCAL_PATH}, set QWEN_IMAGE_MODEL_PATH, "
        f"or use the Hugging Face id {FULL_QWEN_IMAGE_HUB_ID!r}."
    )


def require_tiny_qwen_image_model() -> Path:
    """Backward-compatible helper for tests that only support the tiny fixture."""
    import pytest

    profile = resolve_qwen_image_profile("tiny", recipe="default")
    if not Path(profile.model_path).expanduser().exists():
        pytest.skip(f"Tiny Qwen-Image fixture not found at {profile.model_path}")
    return Path(profile.model_path)


def compose_trainer_config(profile: QwenImageConsistencyProfile) -> DictConfig:
    """Build a minimal diffusion_trainer config for hybrid LoRA weight-sync tests."""
    from hydra import compose, initialize_config_dir

    path = profile.model_path
    tokenizer_path = resolve_tokenizer_path(path)
    sde_window_range = list(profile.sde_window_range)
    overrides = [
        "actor_rollout_ref.model.algorithm=flow_grpo",
        f"actor_rollout_ref.model.path={path}",
        f"actor_rollout_ref.model.tokenizer_path={tokenizer_path}",
        f"actor_rollout_ref.model.lora_rank={profile.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={profile.lora_alpha}",
        f"actor_rollout_ref.model.target_modules={profile.target_modules}",
        f'actor_rollout_ref.model.attn_backend="{profile.attn_backend}"',
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={profile.fsdp_param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={profile.fsdp_param_offload}",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={profile.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.load_format={profile.load_format}",
        f"actor_rollout_ref.rollout.layered_summon={profile.layered_summon}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        f"actor_rollout_ref.rollout.max_model_len={profile.max_sequence_length + 34}",
        f"actor_rollout_ref.rollout.pipeline.max_sequence_length={profile.max_sequence_length}",
        f"actor_rollout_ref.rollout.pipeline.num_inference_steps={profile.num_inference_steps}",
        "actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0",
        "actor_rollout_ref.rollout.algo.noise_level=1.2",
        "actor_rollout_ref.rollout.algo.sde_type=sde",
        f"actor_rollout_ref.rollout.algo.sde_window_size={profile.sde_window_size}",
        f"actor_rollout_ref.rollout.algo.sde_window_range={sde_window_range}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
    ]
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config")):
        return compose(config_name="diffusion_trainer", overrides=overrides)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("vllm", "vllm_omni", "verl", "verl_omni"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return versions


def tokenize_chat_prompt(text: str, model_path: str | Path) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(str(model_path)), trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    if len(token_ids) <= _MIN_PROMPT_TOKENS:
        raise ValueError(
            f"Prompt too short ({len(token_ids)} tokens). Need > {_MIN_PROMPT_TOKENS} after chat-template drop."
        )
    return token_ids


def build_rollout_server(profile: QwenImageConsistencyProfile):
    model_path = profile.model_path
    tokenizer_path = resolve_tokenizer_path(model_path)
    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 16,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": profile.load_format,
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": False,
            "disable_log_stats": True,
            "n": 1,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
            },
            "algo": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionRolloutAlgoConfig",
                "noise_level": 1.2,
                "sde_type": "sde",
                "sde_window_size": profile.sde_window_size,
                "sde_window_range": list(profile.sde_window_range),
            },
        }
    )
    model_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": model_path,
            "tokenizer_path": tokenizer_path,
            "trust_remote_code": True,
            "load_tokenizer": True,
            "algorithm": "flow_grpo",
            "lora_rank": profile.lora_rank,
            "lora_alpha": profile.lora_alpha,
            "target_modules": profile.target_modules,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.model.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
            },
            "algo": {
                "_target_": "verl_omni.workers.config.diffusion.model.DiffusionRolloutAlgoConfig",
                "noise_level": 1.2,
                "sde_type": "sde",
            },
        }
    )

    rollout_env_vars = {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "NCCL_CUMEM_ENABLE": "0",
    }
    if profile.attn_backend == "_flash_3_varlen_hub":
        rollout_env_vars["DIFFUSION_ATTENTION_BACKEND"] = "FLASH_ATTN"

    ServerCls = ray.remote(vLLMOmniHttpServer)
    server = ServerCls.options(
        runtime_env={
            "env_vars": rollout_env_vars,
        },
        max_concurrency=4,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )
    ray.get(server.launch_server.remote())
    return server


def create_training_worker(profile: QwenImageConsistencyProfile) -> tuple[RayWorkerGroup, FSDPDiffusionActorConfig]:
    path = profile.model_path
    tokenizer_path = resolve_tokenizer_path(path)
    param_offload = profile.fsdp_param_offload

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/model")):
        model_cfg = compose(
            config_name="diffusion_model",
            overrides=[
                f"path={path}",
                f"tokenizer_path={tokenizer_path}",
                f"lora_rank={profile.lora_rank}",
                f"lora_alpha={profile.lora_alpha}",
                f"target_modules={profile.target_modules}",
                f'attn_backend="{profile.attn_backend}"',
                "pipeline.height=512",
                "pipeline.width=512",
                f"pipeline.num_inference_steps={profile.num_inference_steps}",
                "pipeline.true_cfg_scale=4.0",
                "algo.noise_level=1.2",
                "algo.sde_type=sde",
            ],
        )
    model_config: DiffusionModelConfig = omega_conf_to_dataclass(model_cfg)

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor")):
        actor_cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_mini_batch_size=1",
                "ppo_micro_batch_size_per_gpu=1",
                "optim.lr=3e-4",
                f"fsdp_config.param_offload={param_offload}",
                f"fsdp_config.optimizer_offload={param_offload}",
                "fsdp_config.model_dtype=bfloat16",
                "fsdp_config.dtype=bfloat16",
                "+fsdp_config.mixed_precision.param_dtype=bfloat16",
                "fsdp_config.forward_only=True",
                "fsdp_config.fsdp_size=1",
                "fsdp_config.ulysses_sequence_parallel_size=1",
                "diffusion_loss.loss_mode=flow_grpo",
                "+fsdp_config.infer_micro_batch_size_per_gpu=1",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(actor_cfg)
    actor_config.engine.infer_micro_batch_size_per_gpu = 1

    training_config = TrainingWorkerConfig(
        model_type="diffusion_model",
        model_config=model_config,
        engine_config=actor_config.engine,
        optimizer_config=actor_config.optim,
        checkpoint_config=actor_config.checkpoint,
    )

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=training_config)
    resource_pool = RayResourcePool(process_on_nodes=[1])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.reset()
    return wg, actor_config


def _ensure_rollout_batch_dim(key: str, tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if key == "all_latents" and tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if key == "all_timesteps" and tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if key in ("prompt_embeds", "negative_prompt_embeds") and tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if key in ("prompt_embeds_mask", "negative_prompt_embeds_mask") and tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == 1 and key.endswith("_mask"):
        return tensor.unsqueeze(0)
    return tensor


def build_train_batch_from_rollout(extra_fields: dict[str, Any], *, height: int = 512, width: int = 512) -> TensorDict:
    all_latents = _ensure_rollout_batch_dim("all_latents", extra_fields["all_latents"])
    all_timesteps = _ensure_rollout_batch_dim("all_timesteps", extra_fields["all_timesteps"])
    prompt_embeds = _ensure_rollout_batch_dim("prompt_embeds", extra_fields["prompt_embeds"])
    prompt_embeds_mask = _ensure_rollout_batch_dim("prompt_embeds_mask", extra_fields["prompt_embeds_mask"])

    negative_prompt_embeds = extra_fields.get("negative_prompt_embeds")
    negative_prompt_embeds_mask = extra_fields.get("negative_prompt_embeds_mask")
    if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_prompt_embeds_mask = torch.zeros_like(prompt_embeds_mask)
    negative_prompt_embeds = _ensure_rollout_batch_dim("negative_prompt_embeds", negative_prompt_embeds)
    negative_prompt_embeds_mask = _ensure_rollout_batch_dim("negative_prompt_embeds_mask", negative_prompt_embeds_mask)

    batch = TensorDict(
        {
            "all_latents": all_latents,
            "all_timesteps": all_timesteps,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        },
        batch_size=all_latents.shape[0],
    )
    tu.assign_non_tensor(
        batch,
        compute_loss=False,
        micro_batch_size_per_gpu=1,
        height=height,
        width=width,
        vae_scale_factor=8,
    )
    return embeds_padding_2_no_padding(batch)


def rollout_sampling_params(*, profile: QwenImageConsistencyProfile | None = None, seed: int = 42) -> dict[str, Any]:
    profile = profile or resolve_qwen_image_profile("tiny", recipe="default")
    return {
        "num_inference_steps": profile.num_inference_steps,
        "true_cfg_scale": 4.0,
        "height": 512,
        "width": 512,
        "logprobs": True,
        "seed": seed,
        "noise_level": 1.2,
        "sde_type": "sde",
        "sde_window_size": profile.sde_window_size,
        "sde_window_range": list(profile.sde_window_range),
    }


def default_prompts(model_path: str) -> tuple[list[int], list[int]]:
    prompt = (
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline"
    )
    negative = (
        "blurry, low quality, unreadable text, distorted letters, smudged ink, "
        "overexposed background, washed out colors, random noise, broken typography, "
        "illegible handwriting, cropped edges, compression artifacts, and visual clutter"
    )
    return tokenize_chat_prompt(prompt, model_path), tokenize_chat_prompt(negative, model_path)
