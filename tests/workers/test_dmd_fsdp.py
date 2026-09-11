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
"""Run with torchrun to validate real multi-rank DMD2 engine updates and resume."""

import gc
import os
import shutil
import tempfile
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig

from verl_omni.workers.config import (
    DiffusionActorConfig,
    DiffusionDMDConfig,
    DiffusionLossConfig,
    DiffusionModelConfig,
    DiffusionPipelineConfig,
)
from verl_omni.workers.engine.fsdp.dmd_impl import DMDDiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import load_diffusers_lora_adapter
from verl_omni.workers.utils.losses import diffusion_loss


@pytest.fixture(scope="module")
def process_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")
    if dist.is_initialized():
        pytest.skip("This fixture owns its process group.")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    with tempfile.TemporaryDirectory(prefix="dmd2_pg_") as directory:
        dist.init_process_group(
            "nccl",
            init_method="env://" if world > 1 else f"file://{directory}/rdzv",
            rank=int(os.environ.get("RANK", "0")),
            world_size=world,
            timeout=timedelta(seconds=180),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def build_engine(strategy, model_path):
    model = DiffusionModelConfig(
        path=model_path,
        algorithm="dmd2",
        model_type="diffusion_dmd_model",
        load_tokenizer=False,
        lora_rank=2,
        lora_alpha=2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        attn_backend="native",
        enable_gradient_checkpointing=True,
        pipeline=DiffusionPipelineConfig(height=64, width=64, num_inference_steps=4, max_sequence_length=64),
    )
    config = FSDPEngineConfig(strategy=strategy, use_orig_params=True, model_dtype="bfloat16", seed=7)
    optimizer = FSDPOptimizerConfig(lr=1e-4, total_training_steps=3)
    engine = DMDDiffusersFSDPEngine(
        model, config, optimizer, CheckpointConfig(), dmd_config=DiffusionDMDConfig(ema_decay=0.5)
    )
    engine.initialize()
    return engine


def adapter_values(engine, role):
    tensors, _ = engine.get_per_tensor_param(base_sync_done=True, adapter_name=engine.adapter_names[role])
    return {key: value.detach().cpu().clone() for key, value in tensors}


def train_stage(engine, stage, batch):
    engine.select_stage(stage)
    tu.assign_non_tensor(batch, dmd_stage=stage, micro_batch_size_per_gpu=2)
    actor = DiffusionActorConfig(
        strategy=engine.engine_config.strategy, rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2")
    )
    with engine.train_mode():
        output = engine.train_batch(batch, partial(diffusion_loss, config=actor))
    assert engine.last_step_succeeded
    exits = torch.tensor(output["metrics"]["dmd/rollout_exit"].aggregate(), device="cuda")
    gathered = [torch.zeros_like(exits) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, exits)
    assert all(torch.equal(value, exits) for value in gathered)
    return output


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_qwen_dmd2_engine_update_resume_export(strategy, process_group):
    model_path = os.environ.get("QWEN_IMAGE_MODEL_PATH", os.path.expanduser("~/models/tiny-random/Qwen-Image"))
    if not Path(model_path, "model_index.json").is_file():
        pytest.skip(f"Tiny Qwen checkpoint not found: {model_path}")
    paths = [tempfile.mkdtemp(prefix="dmd2_fsdp_") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    directory = Path(paths[0])
    engine = build_engine(strategy, model_path)
    batch = TensorDict({"dummy_tensor": torch.zeros(3, 1)}, batch_size=[3])
    tu.assign_non_tensor_stack(
        batch,
        "raw_prompt",
        [
            [{"role": "user", "content": "cat" if dist.get_rank() % 2 else "a red apple on a table"}],
            [{"role": "user", "content": "a blue bird"}],
            [{"role": "user", "content": "a green triangle"}],
        ],
    )
    try:
        initial = {role: adapter_values(engine, role) for role in ("student", "fake_score")}
        for _ in range(3):
            for stage in ("student", "fake_score", "fake_score"):
                train_stage(engine, stage, batch)
        assert engine.optimizer_steps == {"student": 3, "fake_score": 6}
        saved = {role: adapter_values(engine, role) for role in ("student", "fake_score", "student_ema")}
        for role, values in initial.items():
            assert any(torch.count_nonzero(saved[role][key] - value) > 0 for key, value in values.items())
        engine.save_checkpoint(str(directory / "actor"), global_step=3)
        replay = train_stage(engine, "student", batch)
        expected = adapter_values(engine, "student")
        restored = engine.load_checkpoint(str(directory / "actor"), del_local_after_load=False)
        assert restored == {"student": 3, "fake_score": 6}
        for role, parameters in saved.items():
            for key, value in adapter_values(engine, role).items():
                torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
        repeated = train_stage(engine, "student", batch)
        assert repeated["loss"] == pytest.approx(replay["loss"], rel=1e-6, abs=1e-8)
        for key, value in adapter_values(engine, "student").items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        engine.export_student(str(directory / "inference"), role="student")
        assert (directory / "inference" / "adapter_model.safetensors").is_file()
        from diffusers import QwenImageTransformer2DModel
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file

        reloaded = QwenImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        load_diffusers_lora_adapter(reloaded, directory / "inference", "reloaded")
        state = get_peft_model_state_dict(reloaded, adapter_name="reloaded")
        exported = load_file(directory / "inference" / "adapter_model.safetensors")
        assert state.keys() == exported.keys()
        for key, value in state.items():
            torch.testing.assert_close(value, exported[key], rtol=0, atol=0, check_dtype=False)
        before = dict(engine.optimizer_steps)
        scheduler_step = engine.lr_scheduler.last_epoch
        engine.forward_finite = dist.get_rank() != 0
        engine.optimizer_step()
        assert not engine.last_step_succeeded
        assert engine.optimizer_steps == before and engine.lr_scheduler.last_epoch == scheduler_step
    finally:
        dist.barrier()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            shutil.rmtree(directory)
        dist.barrier()
