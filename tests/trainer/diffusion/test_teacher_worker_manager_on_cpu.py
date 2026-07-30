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
"""CPU tests for the frozen diffusion teacher worker (RFC #293)."""

import pytest
import torch
from omegaconf import OmegaConf
from verl.utils import tensordict_utils as tu

from verl_omni.workers.teacher_workers import (
    build_teacher_response,
    build_teacher_training_config,
    parameter_checksum,
)

BATCH, STEPS, CHANNELS = 2, 4, 3


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
