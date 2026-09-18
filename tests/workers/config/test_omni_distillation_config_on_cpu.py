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
"""CPU tests for omni distillation configs.

Covers:
- OmniDistillationTeacherModelConfig._validate_topk_logprobs branch for vllm_omni
- hydra materialization of the distillation subtree composed by omni_trainer.yaml:
  teachers materialize as OmniDistillationTeacherModelConfig (via the yaml
  _target_ override) and are resolved by the base DistillationConfig
"""

import pytest
from hydra import compose, initialize_config_module
from hydra.errors import InstantiationException
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig

from verl_omni.workers.config.omni.distillation import OmniDistillationTeacherModelConfig

OPD_OVERRIDES = [
    "distillation.enabled=true",
    "distillation.nnodes=1",
    "distillation.n_gpus_per_node=8",
    "distillation.teacher_models.teacher_model.model_path=/tmp/teacher",
    "distillation.teacher_models.teacher_model.inference.name=vllm_omni",
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2",
    "distillation.teacher_models.teacher_model.inference.data_parallel_size=1",
    "+distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=1",
    "+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.max_logprobs=8",
    "distillation.distillation_loss.loss_mode=kl",
    "distillation.distillation_loss.use_policy_gradient=false",
]


def _materialize_distillation(overrides):
    with initialize_config_module("verl_omni.trainer.config", version_base=None):
        cfg = compose(config_name="omni_trainer", overrides=overrides)
    return omega_conf_to_dataclass(cfg.distillation)


class TestOmniDistillationTeacherConfig:
    def test_vllm_omni_topk_seeds_and_validates(self):
        # Seed branch: no max_logprobs in engine_kwargs, expect to set to topk
        teacher_cfg = {
            "model_path": "/tmp/teacher",
            "inference": {
                "name": "vllm_omni",
                "engine_kwargs": {},
            },
        }
        teacher: OmniDistillationTeacherModelConfig = omega_conf_to_dataclass(
            teacher_cfg, dataclass_type=OmniDistillationTeacherModelConfig
        )

        teacher._validate_topk_logprobs(use_topk=True, topk=5)
        assert teacher.inference.engine_kwargs["vllm_omni"]["max_logprobs"] == 5

        # Reject branch: existing max_logprobs smaller than requested topk
        bad_teacher_cfg = {
            "model_path": "/tmp/teacher",
            "inference": {
                "name": "vllm_omni",
                "engine_kwargs": {"vllm_omni": {"max_logprobs": 3}},
            },
        }
        bad_teacher: OmniDistillationTeacherModelConfig = omega_conf_to_dataclass(
            bad_teacher_cfg, dataclass_type=OmniDistillationTeacherModelConfig
        )
        with pytest.raises(ValueError):
            bad_teacher._validate_topk_logprobs(use_topk=True, topk=5)


class TestOmniDistillationComposition:
    def test_yaml_override_materializes_omni_teacher(self):
        # The _target_ override in omni_trainer.yaml routes teacher materialization
        # through the omni subclass, so the vllm_omni engine is accepted.
        # __post_init__ already resolved the single-teacher pool math.
        obj: DistillationConfig = _materialize_distillation(OPD_OVERRIDES)
        assert type(obj) is DistillationConfig

        assert set(obj.teacher_models.keys()) == {"default"}
        tm = obj.teacher_models["default"]
        assert isinstance(tm, OmniDistillationTeacherModelConfig)
        # Pool: 1 node * 8 gpus. Per-replica: 2*1*1 = 2. Expect 4 replicas.
        assert tm.num_replicas == 4

    def test_omni_teacher_passes_vllm_omni_topk(self):
        # max_logprobs=8 >= topk=5 must be accepted by the omni subclass.
        obj: DistillationConfig = _materialize_distillation(OPD_OVERRIDES)
        tm = obj.teacher_models["default"]
        tm._validate_topk_logprobs(use_topk=True, topk=5)
        assert tm.inference.engine_kwargs["vllm_omni"]["max_logprobs"] == 8

    def test_divisibility_check(self):
        # Pool: 3 gpus not divisible by per-replica 2 must raise; hydra wraps
        # the dataclass ValueError in InstantiationException.
        with pytest.raises(InstantiationException, match="must divide the distillation resource pool size"):
            _materialize_distillation(OPD_OVERRIDES + ["distillation.n_gpus_per_node=3"])
