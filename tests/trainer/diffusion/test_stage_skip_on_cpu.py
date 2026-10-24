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

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.skip import SkipManager

from verl_omni.utils.skip import STAGE_SKIP_ROLES, init_skip_manager, skip_stage_metrics
from verl_omni.utils.skip.stage_skip import DiffusionRolloutSkip

DUMP_DIR = "dump"


def make_config(tmp_path, *, algorithm=None, omni_skip_overrides=None, trainer_type=None) -> OmegaConf:
    omni_skip = {
        role: {
            "enable": False,
            "dump_dir": str(tmp_path / DUMP_DIR),
            "steps": [],
            "action": "empty" if role in ("actor_update", "weight_sync") else "cache",
        }
        for role in STAGE_SKIP_ROLES
    }
    for role, overrides in (omni_skip_overrides or {}).items():
        omni_skip[role].update(overrides)
    algorithm = dict(algorithm or {})
    if trainer_type is not None:
        algorithm["trainer_type"] = trainer_type
    return OmegaConf.create(
        {
            "trainer": {"experiment_name": "exp", "project_name": "proj"},
            "skip": {},
            "algorithm": algorithm,
            "omni_skip": omni_skip,
        }
    )


def enable_stage(config, role: str, steps: list[int], action: str = "cache") -> None:
    config.omni_skip[role].update({"enable": True, "steps": steps, "action": action})


class DummyTrainer:
    """Mirrors the annotated stage-method signatures of the diffusion trainers."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def _bump(self, role: str) -> None:
        self.calls[role] = self.calls.get(role, 0) + 1

    @SkipManager.annotate(role="old_log_prob")
    def compute_old_log_prob(self, data: DataProto) -> DataProto:
        self._bump("old_log_prob")
        return DataProto.from_single_dict(data={"old_log_probs": torch.full((len(data.batch["x"]), 2), 7.0)})

    @SkipManager.annotate(role="actor_update")
    def update_actor(self, data: DataProto) -> DataProto:
        self._bump("actor_update")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/loss": 1.0}})

    @SkipManager.annotate(role="weight_sync")
    def sync_weights(self) -> None:
        self._bump("weight_sync")


@pytest.fixture
def reset_skip_manager():
    SkipManager.config = None
    SkipManager.step = -1
    SkipManager.skip_instances = {}
    yield
    SkipManager.config = None
    SkipManager.step = -1
    SkipManager.skip_instances = {}


@pytest.fixture
def dummy():
    return DummyTrainer()


def _batch(n: int = 2) -> DataProto:
    return DataProto.from_single_dict(data={"x": torch.zeros(n, 3)})


def test_stage_cache_miss_then_hit_round_trip(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    SkipManager.set_step(1)

    first = dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}
    assert skip_stage_metrics() == {}  # running the stage does not count as a skip

    second = dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}  # replayed from cache
    torch.testing.assert_close(second.batch["old_log_probs"], first.batch["old_log_probs"])
    assert skip_stage_metrics() == {"skip/old_log_prob/calls": 1, "skip/old_log_prob/steps": 1}


def test_stage_not_listed_step_runs_normally(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    SkipManager.set_step(2)

    dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}
    assert skip_stage_metrics() == {}


def test_repeat_action_replays_closest_dumped_step(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[5], action="repeat")
    init_skip_manager(config)
    SkipManager.set_step(1)
    dummy.compute_old_log_prob(_batch())  # cache-miss run dumps step 1

    SkipManager.set_step(5)
    result = dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}
    torch.testing.assert_close(result.batch["old_log_probs"], torch.full((2, 2), 7.0))
    assert skip_stage_metrics() == {"skip/old_log_prob/calls": 1, "skip/old_log_prob/steps": 1}


def test_validation_batch_bypasses_stage_skip(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    SkipManager.set_step(1)
    dummy.compute_old_log_prob(_batch())  # populate the cache

    validation_batch = _batch()
    validation_batch.meta_info["validate"] = True
    dummy.compute_old_log_prob(validation_batch)
    assert dummy.calls == {"old_log_prob": 2}


def test_actor_update_empty_action_returns_empty_metrics(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "actor_update", steps=[1], action="empty")
    init_skip_manager(config)
    SkipManager.set_step(1)

    result = dummy.update_actor(_batch())
    assert dummy.calls == {}  # stage never ran
    assert result.meta_info["metrics"] == {}
    assert skip_stage_metrics() == {"skip/actor_update/calls": 1, "skip/actor_update/steps": 1}


def test_weight_sync_empty_action(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    config.omni_skip.weight_sync.update({"enable": True, "steps": [1], "action": "empty", "ack_stale_weights": True})
    init_skip_manager(config)
    SkipManager.set_step(1)

    assert dummy.sync_weights() is None
    assert dummy.calls == {}
    assert skip_stage_metrics() == {"skip/weight_sync/calls": 1, "skip/weight_sync/steps": 1}


def test_weight_sync_requires_stale_weights_ack(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    enable_stage(config, "weight_sync", steps=[1], action="empty")
    with pytest.raises(ValueError, match="ack_stale_weights"):
        init_skip_manager(config)


def test_old_log_prob_conflicts_with_rollout_correction_bypass(tmp_path, reset_skip_manager):
    config = make_config(tmp_path, algorithm={"rollout_correction": {"bypass_mode": True}})
    enable_stage(config, "old_log_prob", steps=[1])
    with pytest.raises(ValueError, match="bypass_mode"):
        init_skip_manager(config)


def test_old_log_prob_inert_for_direct_preference(tmp_path, reset_skip_manager):
    config = make_config(tmp_path, trainer_type="direct_preference")
    enable_stage(config, "old_log_prob", steps=[1])
    with pytest.raises(ValueError, match="direct_preference"):
        init_skip_manager(config)


def test_advantage_inert_for_direct_preference(tmp_path, reset_skip_manager):
    config = make_config(tmp_path, trainer_type="direct_preference")
    enable_stage(config, "advantage", steps=[1])
    with pytest.raises(ValueError, match="direct_preference"):
        init_skip_manager(config)


def test_old_log_prob_inert_when_loss_reads_no_log_probs(tmp_path, reset_skip_manager, monkeypatch):
    from verl_omni.trainer.diffusion import diffusion_trainer_utils

    monkeypatch.setattr(diffusion_trainer_utils, "skip_old_log_prob", lambda config: True)
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    with pytest.raises(ValueError, match="inert"):
        init_skip_manager(config)


def test_disabled_stage_not_instantiated(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    assert "old_log_prob" in SkipManager.skip_instances
    for role in STAGE_SKIP_ROLES:
        if role != "old_log_prob":
            assert role not in SkipManager.skip_instances


def test_diffusion_rollout_skip_restamps_input_metadata(tmp_path, reset_skip_manager):
    config = OmegaConf.create(
        {
            "enable": True,
            "dump_dir": str(tmp_path / DUMP_DIR),
            "steps": [3],
            "action": "cache",
        }
    )
    global_config = OmegaConf.create({"trainer": {"experiment_name": "exp", "project_name": "proj"}})
    skip = DiffusionRolloutSkip(config, global_config)

    cached_output = DataProto.from_single_dict(
        data={"responses": torch.ones(1, 4)},
        meta_info={"timing": {"gen": 1.0}},
    )
    cached_output.non_tensor_batch["uid"] = ["old-uid"]
    skip.prepare_data(step=3, result=cached_output)

    fresh_gen_batch = DataProto.from_single_dict(data={"responses": torch.zeros(1, 4)})
    fresh_gen_batch.non_tensor_batch["uid"] = ["new-uid"]
    fresh_gen_batch.non_tensor_batch["data_source"] = ["ocr"]

    result = skip.warp_function(3, None, None, fresh_gen_batch)

    assert result.non_tensor_batch["uid"] == ["new-uid"]
    assert result.non_tensor_batch["data_source"] == ["ocr"]
    torch.testing.assert_close(result.batch["responses"], torch.ones(1, 4))
    assert result.meta_info["timing"] == {"gen": 1.0}
