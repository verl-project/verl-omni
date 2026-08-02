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
"""GPU tests for the frozen diffusion teacher worker.

Uses the tiny random SD3 from ``build_sd3_tiny_random.py``. That checkpoint
cannot serve rollout (vllm-omni's SD3 pipeline builds the slow T5Tokenizer,
which the tiny builder does not write), but the teacher worker loads only the
transformer and the scheduler, so it is fine here -- and the e2e smoke covers
the rollout-backed path on a real checkpoint.

Run:  pytest tests/workers/test_diffusion_teacher_gpu.py
"""

import os
import subprocess
import sys

import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ray.util.placement_group import remove_placement_group
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu

import verl_omni
from verl_omni.workers.teacher_workers import DiffusionTeacherWorker

STUDENT_DIR = os.path.expanduser("~/models/tiny-random/sd3-tiny-student")
TEACHER_DIR = os.path.expanduser("~/models/tiny-random/sd3-tiny-teacher")

BATCH, STEPS = 2, 4
HEIGHT = WIDTH = 64
VAE_SCALE_FACTOR = 8
# tiny SD3 transformer: in_channels=16, joint_attention_dim=8, pooled_projection_dim=16
LATENT_CHANNELS, JOINT_DIM, POOLED_DIM = 16, 8, 16
SEQ_LEN = 16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def build_tiny_checkpoints():
    for path, seed in ((STUDENT_DIR, 0), (TEACHER_DIR, 1)):
        if os.path.exists(os.path.join(path, "model_index.json")):
            continue
        subprocess.run(
            [
                sys.executable,
                "tests/special_e2e/build_sd3_tiny_random.py",
                "--output-dir",
                path,
                "--seed",
                str(seed),
            ],
            check=True,
        )


def teacher_config(teacher_path):
    """The actor_rollout_ref subtree, composed from the shipped config.

    Deliberately not hand-rolled: nested nodes carry their own ``_target_`` and
    mandatory fields, so a hand-written stub silently degrades them to bare dicts
    (``pipeline``) or trips mandatory values (``profiler.save_path``) in ways the
    real trainer never would.
    """
    config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={STUDENT_DIR}",
                "actor_rollout_ref.model.algorithm=flow_grpo",
                "actor_rollout_ref.model.attn_backend=native",
                "actor_rollout_ref.model.lora_rank=0",
                f"actor_rollout_ref.model.pipeline.height={HEIGHT}",
                f"actor_rollout_ref.model.pipeline.width={WIDTH}",
                f"actor_rollout_ref.model.pipeline.num_inference_steps={STEPS}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={BATCH}",
                f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={BATCH}",
                "actor_rollout_ref.teacher.enabled=true",
                f"+actor_rollout_ref.teacher.models.default.model.path={teacher_path}",
                "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg.actor_rollout_ref


def make_request(scheduler):
    # SD3 latents stay image-shaped [B, T, C, H, W]; the token-packed [B, T, tokens, C]
    # layout belongs to Qwen-Image and trips SD3's patch-embed conv.
    latent_side = HEIGHT // VAE_SCALE_FACTOR
    torch.manual_seed(0)
    batch_td = tu.get_tensordict(
        {
            "all_latents": torch.randn(BATCH, STEPS + 1, LATENT_CHANNELS, latent_side, latent_side),
            "all_timesteps": scheduler.timesteps[:STEPS][None].repeat(BATCH, 1),
            "prompt_embeds": torch.randn(BATCH, SEQ_LEN, JOINT_DIM),
            "prompt_embeds_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.int32),
            "pooled_prompt_embeds": torch.randn(BATCH, POOLED_DIM),
        }
    )
    tu.assign_non_tensor(
        batch_td,
        compute_loss=False,
        height=HEIGHT,
        width=WIDTH,
        vae_scale_factor=VAE_SCALE_FACTOR,
    )
    return batch_td


_SPAWNED = []


def spawn_teacher(teacher_path):
    ray_cls = RayClassWithInitArgs(cls=ray.remote(DiffusionTeacherWorker), config=teacher_config(teacher_path))
    resource_pool = RayResourcePool(process_on_nodes=[1])
    worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)
    worker_group.init_model()
    _SPAWNED.append((worker_group, resource_pool))
    return worker_group


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    build_tiny_checkpoints()
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(autouse=True)
def release_worker_groups():
    """One GPU, several groups: leaking them across tests exhausts it.

    Killing the actors is not enough -- each pool reserves its own placement
    group, and an unreleased one keeps the GPU booked, so the next spawn blocks
    forever rather than failing.
    """
    yield
    while _SPAWNED:
        worker_group, resource_pool = _SPAWNED.pop()
        for handle in worker_group._workers:
            ray.kill(handle)
        for placement_group in resource_pool.pgs or ():
            remove_placement_group(placement_group)


def test_teacher_scoring_is_frozen_and_well_formed():
    """The teacher's weights do not move, and its targets satisfy the contract."""
    from verl_omni.pipelines.model_base import DiffusionModelBase
    from verl_omni.trainer.diffusion.teacher_scheduler_checks import build_cpu_scheduler
    from verl_omni.workers.config.diffusion import DiffusionModelConfig
    from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig

    # same pipeline settings as the worker, or the request's timesteps land off its grid
    model_config = DiffusionModelConfig(
        path=TEACHER_DIR,
        algorithm="flow_grpo",
        attn_backend="native",
        load_tokenizer=False,
        pipeline=DiffusionPipelineConfig(height=HEIGHT, width=WIDTH, num_inference_steps=STEPS),
    )
    scheduler = build_cpu_scheduler(model_config, DiffusionModelBase.get_class(model_config))

    worker_group = spawn_teacher(TEACHER_DIR)
    before = worker_group.teacher_param_checksum()

    output = worker_group.compute_teacher_outputs(make_request(scheduler))

    after = worker_group.teacher_param_checksum()
    assert after == before, "teacher parameters moved across a scoring pass"

    target = tu.get(output, "teacher_prev_sample_mean")
    assert target.dtype is torch.float32
    assert target.device.type == "cpu"
    assert target.shape[:2] == (BATCH, STEPS)
    assert torch.isfinite(target).all()


def test_distinct_checkpoints_give_distinct_checksums():
    """Proves the teacher is a separate model rather than a copy of the actor."""
    teacher = spawn_teacher(TEACHER_DIR)
    student_as_teacher = spawn_teacher(STUDENT_DIR)

    assert teacher.teacher_param_checksum() != student_as_teacher.teacher_param_checksum()


# The teacher = student scenario is not here on purpose. It needs a further pair of
# worker groups on the same GPU, and tearing that many down mid-session crashes the
# driver. Its substantive half -- that the teacher's targets match the student's own
# replay -- belongs with the standalone-placement work, where a same-hardware
# comparison exists; checksum equality on its own is near-tautological for a
# deterministic hash over identical weights.
