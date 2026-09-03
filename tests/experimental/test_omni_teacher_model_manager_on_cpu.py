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
"""CPU tests for Omni teacher managers.

Covers:
- OmniTeacherModelManager uses OmniModelConfig for vllm_omni teachers
- LLM server wiring without launching real engines
- Single-teacher pool math via the base DistillationConfig with the omni
  teacher _target_ (as composed by omni_trainer.yaml)
"""

import verl.experimental.teacher_loop.teacher_model as base_teacher_mod
import verl.workers.rollout.replica as replica_mod
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig

from verl_omni.experimental.teacher_loop import teacher_model as omni_teacher_mod
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.config.omni.distillation import OmniDistillationTeacherModelConfig


class _Pool:
    def __init__(self, world_size, start_bundle_index=0):
        self.world_size = world_size
        self.start_bundle_index = start_bundle_index


def _make_distillation_cfg(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    return {
        "_target_": "verl.workers.config.DistillationConfig",
        "n_gpus_per_node": 2,
        "nnodes": 1,
        "teacher_models": {
            "teacher_model": {
                "_target_": "verl_omni.workers.config.omni.distillation.OmniDistillationTeacherModelConfig",
                "model_path": str(teacher_dir),
                "inference": {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm_omni",
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "engine_kwargs": {},
                },
            }
        },
    }


def test_omni_teacher_manager_uses_omni_model_config(monkeypatch, tmp_path):
    created = []

    class FakeReplica:
        def __init__(self, *, replica_rank, config, model_config, gpus_per_node, is_teacher_model, name_suffix):
            self.replica_rank = replica_rank
            self.config = config
            self.model_config = model_config
            self.gpus_per_node = gpus_per_node
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self._server_handle = None
            self._server_address = None
            created.append(self)

        def init_colocated(self, resource_pool):
            self._server_handle = f"handle-{self.replica_rank}"
            self._server_address = f"addr-{self.replica_rank}"

    class FakeOmniModelConfig(OmniModelConfig):
        def __post_init__(self):
            pass

    # get_rollout_replica_class and _run_all are imported lazily inside
    # OmniTeacherModelManager._initialize_llm_servers, so patch their home modules.
    monkeypatch.setattr(replica_mod, "get_rollout_replica_class", lambda name: FakeReplica)
    monkeypatch.setattr(base_teacher_mod, "_run_all", lambda tasks: None)
    monkeypatch.setattr(omni_teacher_mod.TeacherModelManager, "_initialize_load_balancer_handle", lambda self: None)
    monkeypatch.setattr(omni_teacher_mod, "OmniModelConfig", FakeOmniModelConfig)

    def _split(pool, split_size):
        count = pool.world_size // split_size
        return [_Pool(world_size=split_size, start_bundle_index=i * split_size) for i in range(count)]

    monkeypatch.setattr(omni_teacher_mod, "split_resource_pool", _split)

    distill_cfg_dict = _make_distillation_cfg(tmp_path)
    distill_cfg: DistillationConfig = omega_conf_to_dataclass(distill_cfg_dict)
    teacher_models = distill_cfg._resolve_teacher_models()
    assert set(teacher_models.keys()) == {"default"}
    teacher_cfg = teacher_models["default"]
    assert isinstance(teacher_cfg, OmniDistillationTeacherModelConfig)
    assert teacher_cfg.num_replicas == 2

    resource_pool = _Pool(world_size=distill_cfg_dict["n_gpus_per_node"] * distill_cfg_dict["nnodes"])  # 2
    manager = omni_teacher_mod.OmniTeacherModelManager(
        distillation_config=distill_cfg,
        teacher_model_config=teacher_cfg,
        resource_pool=resource_pool,
    )

    assert len(manager.rollout_replicas) == 2
    assert [rep.is_teacher_model for rep in manager.rollout_replicas] == [True, True]
    assert all(isinstance(rep.model_config, OmniModelConfig) for rep in manager.rollout_replicas)
    assert manager.server_handles == ["handle-0", "handle-1"]
    assert manager.server_addresses == ["addr-0", "addr-1"]
