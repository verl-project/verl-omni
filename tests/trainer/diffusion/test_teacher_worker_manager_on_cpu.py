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
"""CPU tests for the frozen diffusion teacher worker."""

import pytest
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.teacher_manager import DiffusionTeacherModelManager
from verl_omni.workers.config.diffusion import (
    DiffusionTeacherConfig,
    TeacherCheckpointConfig,
    TeacherModelEntry,
)
from verl_omni.workers.teacher_workers import (
    build_teacher_response,
    build_teacher_training_config,
    parameter_checksum,
)

BATCH, STEPS, CHANNELS = 2, 4, 3


@pytest.fixture
def adapter(diffusion_model_config, fake_sd3_checkpoint):
    return DiffusionModelBase.get_class(diffusion_model_config(fake_sd3_checkpoint("probe")))


@pytest.fixture
def actor_rollout_ref_config(fake_sd3_checkpoint):
    """The actor_rollout_ref subtree as the teacher worker receives it."""
    return OmegaConf.create(
        {
            "model": {
                "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
                "path": fake_sd3_checkpoint("student"),
                "algorithm": "flow_grpo",
                "attn_backend": "native",
                "load_tokenizer": False,
                "model_type": "diffusion_model",
            },
            "actor": {
                "_target_": "verl_omni.workers.config.diffusion.FSDPDiffusionActorConfig",
                "strategy": "fsdp",
                "rollout_n": 1,
                "ppo_micro_batch_size_per_gpu": 2,
                "fsdp_config": {"_target_": "verl.workers.config.FSDPEngineConfig", "strategy": "fsdp"},
            },
            "rollout": {"log_prob_micro_batch_size_per_gpu": None},
            "teacher": {
                "enabled": True,
                "models": {"default": {"model": {"path": fake_sd3_checkpoint("teacher")}}},
                "placement": {"mode": "colocated"},
            },
        }
    )


def make_engine_output(mean=None):
    """The engine emits several keys; only prev_sample_mean crosses into the contract."""
    if mean is None:
        mean = torch.randn(BATCH, STEPS, CHANNELS, dtype=torch.bfloat16)
    return tu.get_tensordict({"prev_sample_mean": mean, "log_probs": torch.randn(*mean.shape[:2])})


class TestBuildTeacherTrainingConfig:
    def test_forward_only_and_frozen_surface(self, actor_rollout_ref_config):
        key, training_config = build_teacher_training_config(actor_rollout_ref_config, world_size=2)

        assert key == "default"
        assert training_config.engine_config.forward_only is True
        assert training_config.engine_config.fsdp_size == 2
        assert training_config.engine_config.infer_micro_batch_size_per_gpu == 2
        # no optimizer by construction; forward_only skips _build_optimizer outright
        assert training_config.optimizer_config is None
        assert training_config.model_config.path.endswith("teacher")
        assert training_config.model_config.lora_rank == 0
        assert training_config.model_config.policy_state_adapters == ()

    def test_forward_only_asserted_not_trusted(self, actor_rollout_ref_config, monkeypatch):
        """The invariant is re-checked on the built config, not assumed from the resolver."""
        from verl_omni.workers import teacher_workers

        real_resolver = teacher_workers.resolve_teacher_engine_config

        def trainable_engine(*args, **kwargs):
            engine = real_resolver(*args, **kwargs)
            object.__setattr__(engine, "forward_only", False)
            return engine

        monkeypatch.setattr(teacher_workers, "resolve_teacher_engine_config", trainable_engine)

        with pytest.raises(ValueError, match="forward_only must be True"):
            build_teacher_training_config(actor_rollout_ref_config, world_size=2)


class TestBuildTeacherResponse:
    def test_output_renamed_and_fp32_cpu(self):
        response = build_teacher_response(make_engine_output(), torch.zeros(BATCH, STEPS), "default")

        assert list(response.keys()) == ["teacher_prev_sample_mean"]
        target = response["teacher_prev_sample_mean"]
        assert target.dtype is torch.float32
        assert target.device.type == "cpu"
        assert target.shape == (BATCH, STEPS, CHANNELS)

    def test_missing_key_rejected(self):
        output = tu.get_tensordict({"log_probs": torch.randn(BATCH, STEPS)})

        with pytest.raises(ValueError, match="no 'prev_sample_mean'"):
            build_teacher_response(output, torch.zeros(BATCH, STEPS), "ocr_expert")

    def test_nonfinite_output_rejected(self):
        mean = torch.randn(BATCH, STEPS, CHANNELS)
        mean[1, 2, 0] = float("nan")

        with pytest.raises(ValueError, match="non-finite") as excinfo:
            build_teacher_response(make_engine_output(mean), torch.zeros(BATCH, STEPS), "ocr_expert")

        assert "ocr_expert" in str(excinfo.value)

    @pytest.mark.parametrize("bad_shape", [(BATCH + 1, STEPS, CHANNELS), (BATCH, STEPS - 1, CHANNELS)])
    def test_shape_mismatch_rejected(self, bad_shape):
        output = make_engine_output(torch.randn(*bad_shape))

        with pytest.raises(ValueError, match=r"\[batch, steps\] prefix"):
            build_teacher_response(output, torch.zeros(BATCH, STEPS), "default")


class TestParameterChecksum:
    def test_stable_and_sensitive(self):
        module = torch.nn.Linear(4, 4)
        before = parameter_checksum(module)

        assert parameter_checksum(module) == before

        with torch.no_grad():
            module.weight[0, 0] += 1e-3
        assert parameter_checksum(module) != before

    def test_distinct_modules_differ(self):
        torch.manual_seed(0)
        first = torch.nn.Linear(4, 4)
        torch.manual_seed(1)
        second = torch.nn.Linear(4, 4)

        assert parameter_checksum(first) != parameter_checksum(second)


class FakeTeacherWorkerGroup:
    """Stands in for the Ray worker group; the manager must never hand it out."""

    def __init__(self, response=None, raises=None, checksums=("rank0", "rank1")):
        self.response = response
        self.raises = raises
        self.checksums = list(checksums)
        self.requests = []
        self.profile_calls = []

    def compute_teacher_outputs(self, batch_td):
        self.requests.append(batch_td)
        if self.raises is not None:
            raise self.raises
        return self.response

    def teacher_param_checksum(self):
        return self.checksums

    def start_profile(self, profile_step):
        self.profile_calls.append(("start", profile_step))

    def stop_profile(self):
        self.profile_calls.append(("stop", None))


def make_teacher_config(path, key="default"):
    return DiffusionTeacherConfig(
        enabled=True, models={key: TeacherModelEntry(model=TeacherCheckpointConfig(path=path))}
    )


def make_response(rows, steps):
    """Row i carries the value i, so any reordering by the manager is visible."""
    mean = torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1).expand(rows, steps, CHANNELS).contiguous()
    return tu.get_tensordict({"teacher_prev_sample_mean": mean})


def make_batch(all_timesteps):
    rows, steps = all_timesteps.shape
    return DataProto.from_tensordict(
        tu.get_tensordict({"all_latents": torch.randn(rows, steps + 1, CHANNELS), "all_timesteps": all_timesteps})
    )


class TestTeacherManagerConstruction:
    def test_non_mvp_architecture_rejected(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        """Both sides Qwen: compatible with each other, still outside the supported matrix."""
        actor = diffusion_model_config(fake_sd3_checkpoint("student", class_name="QwenImagePipeline"))
        teacher_path = fake_sd3_checkpoint("teacher", class_name="QwenImagePipeline")

        with pytest.raises(ValueError, match="outside the supported matrix"):
            DiffusionTeacherModelManager(make_teacher_config(teacher_path), FakeTeacherWorkerGroup(), actor, adapter)

    def test_architecture_mismatch_rejected(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher_path = fake_sd3_checkpoint("teacher", class_name="QwenImagePipeline")

        with pytest.raises(ValueError, match="differs from the actor's"):
            DiffusionTeacherModelManager(make_teacher_config(teacher_path), FakeTeacherWorkerGroup(), actor, adapter)

    def test_scheduler_grid_validated_at_construction(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student", shift=1.0))
        teacher_path = fake_sd3_checkpoint("teacher", shift=3.0)

        with pytest.raises(ValueError, match="resolved timesteps differ"):
            DiffusionTeacherModelManager(make_teacher_config(teacher_path), FakeTeacherWorkerGroup(), actor, adapter)

    def test_two_teachers_rejected(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        """The manager must not guess which teacher to use.

        `enabled=False` is what lets a two-teacher config construct at all --
        the enabled path is already guarded in DiffusionTeacherConfig -- so this
        pins the manager's own contract rather than re-testing the config's.
        """
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        config = DiffusionTeacherConfig(
            enabled=False,
            models={
                "default": TeacherModelEntry(model=TeacherCheckpointConfig(path=fake_sd3_checkpoint("teacher"))),
                "expert": TeacherModelEntry(model=TeacherCheckpointConfig(path=fake_sd3_checkpoint("expert"))),
            },
        )

        with pytest.raises(NotImplementedError, match="exactly one teacher"):
            DiffusionTeacherModelManager(config, FakeTeacherWorkerGroup(), actor, adapter)

    def test_worker_group_not_publicly_exposed(self, manager_and_wg):
        manager, worker_group = manager_and_wg

        public = {name: value for name, value in vars(manager).items() if not name.startswith("_")}
        assert worker_group not in public.values()


@pytest.fixture
def manager_and_wg(adapter, fake_sd3_checkpoint, diffusion_model_config):
    actor = diffusion_model_config(fake_sd3_checkpoint("student"))
    worker_group = FakeTeacherWorkerGroup()
    manager = DiffusionTeacherModelManager(
        make_teacher_config(fake_sd3_checkpoint("teacher")), worker_group, actor, adapter
    )
    return manager, worker_group


@pytest.fixture
def on_grid_timesteps(manager_and_wg):
    manager, _ = manager_and_wg
    return manager.actor_scheduler.timesteps[:STEPS].expand(BATCH, STEPS).contiguous()


class TestTeacherManagerScoring:
    def test_request_missing_trajectory_rejected(self, manager_and_wg):
        manager, _ = manager_and_wg
        batch = DataProto.from_tensordict(tu.get_tensordict({"all_latents": torch.randn(BATCH, STEPS, CHANNELS)}))

        with pytest.raises(ValueError, match="all_timesteps"):
            manager.compute_teacher_outputs(batch)

    def test_off_grid_request_rejected(self, manager_and_wg, on_grid_timesteps):
        manager, _ = manager_and_wg
        off_grid = on_grid_timesteps.clone()
        off_grid[0, 0] += 0.5

        with pytest.raises(ValueError, match="not on the shared scheduler grid"):
            manager.compute_teacher_outputs(make_batch(off_grid))

    def test_response_validated_and_unioned(self, manager_and_wg, on_grid_timesteps):
        manager, worker_group = manager_and_wg
        worker_group.response = make_response(BATCH, STEPS)
        batch = make_batch(on_grid_timesteps)

        output = manager.compute_teacher_outputs(batch)

        assert list(output.batch.keys()) == ["teacher_prev_sample_mean"]
        assert torch.equal(output.batch["teacher_prev_sample_mean"], worker_group.response["teacher_prev_sample_mean"])
        unioned = batch.union(output)
        assert "teacher_prev_sample_mean" in unioned.batch
        # row order preserved: row i still carries the value i
        assert torch.equal(unioned.batch["teacher_prev_sample_mean"][:, 0, 0], torch.arange(BATCH, dtype=torch.float32))

    def test_row_count_mismatch_rejected(self, manager_and_wg, on_grid_timesteps):
        manager, worker_group = manager_and_wg
        worker_group.response = make_response(BATCH + 1, STEPS)

        with pytest.raises(ValueError, match="rows but the request carried"):
            manager.compute_teacher_outputs(make_batch(on_grid_timesteps))

    def test_missing_response_key_rejected(self, manager_and_wg, on_grid_timesteps):
        manager, worker_group = manager_and_wg
        worker_group.response = tu.get_tensordict({"prev_sample_mean": torch.randn(BATCH, STEPS, CHANNELS)})

        with pytest.raises(ValueError, match="no 'teacher_prev_sample_mean'"):
            manager.compute_teacher_outputs(make_batch(on_grid_timesteps))

    def test_empty_response_rejected(self, manager_and_wg, on_grid_timesteps):
        """Collect ranks assemble the response before it reaches the manager, so a
        None here is a broken teacher group, not a non-collect rank."""
        manager, worker_group = manager_and_wg
        worker_group.response = None

        with pytest.raises(ValueError, match="returned no output"):
            manager.compute_teacher_outputs(make_batch(on_grid_timesteps))

    def test_worker_failure_propagates_with_key(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher_path = fake_sd3_checkpoint("teacher")
        worker_group = FakeTeacherWorkerGroup(raises=RuntimeError("CUDA OOM in teacher forward"))
        manager = DiffusionTeacherModelManager(
            make_teacher_config(teacher_path, key="ocr_expert"), worker_group, actor, adapter
        )
        timesteps = manager.actor_scheduler.timesteps[:STEPS].expand(BATCH, STEPS).contiguous()

        with pytest.raises(RuntimeError) as excinfo:
            manager.compute_teacher_outputs(make_batch(timesteps))

        message = str(excinfo.value)
        assert "ocr_expert" in message
        assert f"{BATCH} rows" in message
        assert teacher_path in message
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "CUDA OOM" in str(excinfo.value.__cause__)

    def test_request_carries_pipeline_metadata(self, manager_and_wg, on_grid_timesteps):
        manager, worker_group = manager_and_wg
        worker_group.response = make_response(BATCH, STEPS)

        manager.compute_teacher_outputs(make_batch(on_grid_timesteps))

        request = worker_group.requests[0]
        assert tu.get(request, "compute_loss") is False
        assert tu.get(request, "height") == manager.actor_model_config.pipeline.height
        assert tu.get(request, "width") == manager.actor_model_config.pipeline.width


class TestTeacherManagerPassthroughs:
    def test_checksums_are_per_rank_list(self, manager_and_wg):
        manager, worker_group = manager_and_wg
        worker_group.checksums = ["a" * 8, "b" * 8]

        checksums = manager.teacher_param_checksums()

        assert isinstance(checksums, list)
        assert checksums == ["a" * 8, "b" * 8]

    def test_profiling_forwarded(self, manager_and_wg):
        manager, worker_group = manager_and_wg

        manager.start_profile(step=7)
        manager.stop_profile()

        assert worker_group.profile_calls == [("start", 7), ("stop", None)]
