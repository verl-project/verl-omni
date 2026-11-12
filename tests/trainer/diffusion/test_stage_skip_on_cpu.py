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

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.skip import SkipManager
from verl.utils.skip.rollout_skip import RolloutTqSkip

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    BaseRayDiffusionTrainer,
    DirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
)
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync
from verl_omni.utils.skip import STAGE_SKIP_ROLES, init_skip_manager, log_skip_summary, skip_stage_metrics
from verl_omni.utils.skip.stage_skip import DiffusionRolloutSkip

DUMP_DIR = "dump"
COMPARE_SCRIPT = Path(__file__).parents[3] / "examples" / "stage_skip" / "compare_replay_metrics.py"


def make_config(tmp_path, *, algorithm=None, omni_skip_overrides=None, trainer_type=None, v1=None, skip=None):
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
    trainer = {"experiment_name": "exp", "project_name": "proj", "use_v1": True}
    # The real trainer config always ships trainer.v1 (and verl's RolloutTqSkip
    # reads trainer.v1.trainer_mode at init even when rollout_tq is disabled).
    trainer["v1"] = {"trainer_mode": "separate_async", "separate_async": {"parameter_sync_step": 1}}
    if v1 is not None:
        trainer["v1"].update(v1)
    return OmegaConf.create(
        {
            "trainer": trainer,
            "skip": skip or {},
            "algorithm": algorithm,
            "reward": {"reward_model": {"enable": True, "enable_resource_pool": False}},
            # The loss reads log probs, so the real skip_old_log_prob conflict
            # check passes and the annotated stage is created.
            "actor_rollout_ref": {"actor": {"diffusion_loss": {"loss_mode": "flow_grpo"}, "use_kl_loss": False}},
            "omni_skip": omni_skip,
        }
    )


def enable_stage(config, role: str, steps: list[int], action: str = "cache") -> None:
    config.omni_skip[role].update({"enable": True, "steps": steps, "action": action})


class DummyTrainer:
    """Mirrors the annotated stage-method signatures of the diffusion trainers."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.woken = 0
        self.checkpoint_manager = SimpleNamespace(wake_up_replicas=self._wake_up)

    def _wake_up(self) -> None:
        self.woken += 1

    def _bump(self, role: str) -> None:
        self.calls[role] = self.calls.get(role, 0) + 1

    @SkipManager.annotate(role="old_log_prob")
    def compute_old_log_prob(self, data: DataProto) -> DataProto:
        value = 7.0 + self.calls.get("old_log_prob", 0)  # per-call distinct so replay mismatches are visible
        self._bump("old_log_prob")
        return DataProto.from_single_dict(data={"old_log_probs": torch.full((len(data.batch["x"]), 2), value)})

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
    assert skip_stage_metrics() == {"skip/old_log_prob/calls": 0, "skip/old_log_prob/steps": 0}

    # Re-running init models a fresh run: instances and per-step call counters
    # start over, so the dumped step replays instead of running the stage again.
    init_skip_manager(config)
    SkipManager.set_step(1)
    second = dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}  # replayed from cache
    torch.testing.assert_close(second.batch["old_log_probs"], first.batch["old_log_probs"])
    assert skip_stage_metrics() == {"skip/old_log_prob/calls": 1, "skip/old_log_prob/steps": 1}


def test_stage_cache_replays_per_call_within_split_step(tmp_path, reset_skip_manager, dummy):
    """parameter_sync_step > 1: each mini-batch call of a step owns its own cache slot."""
    config = make_config(tmp_path, v1={"separate_async": {"parameter_sync_step": 2}})
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    SkipManager.set_step(1)

    first = dummy.compute_old_log_prob(_batch())
    second = dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 2}  # mini-batch 1 must not replay mini-batch 0

    init_skip_manager(config)
    SkipManager.set_step(1)
    replayed = [dummy.compute_old_log_prob(_batch()) for _ in range(2)]
    assert dummy.calls == {"old_log_prob": 2}
    torch.testing.assert_close(replayed[0].batch["old_log_probs"], first.batch["old_log_probs"])
    torch.testing.assert_close(replayed[1].batch["old_log_probs"], second.batch["old_log_probs"])
    assert skip_stage_metrics() == {"skip/old_log_prob/calls": 2, "skip/old_log_prob/steps": 1}


def test_repeat_ignores_incomplete_split_step(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path, v1={"separate_async": {"parameter_sync_step": 2}})
    enable_stage(config, "old_log_prob", steps=[5], action="repeat")
    init_skip_manager(config)
    SkipManager.set_step(5)
    dummy.compute_old_log_prob(_batch())  # mini-batch 0 dumped; the step cache is incomplete

    dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 2}  # an incomplete cache never replays


def test_stage_not_listed_step_runs_normally(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    SkipManager.set_step(2)

    dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}
    assert skip_stage_metrics() == {"skip/old_log_prob/calls": 0, "skip/old_log_prob/steps": 0}


def test_repeat_action_replays_closest_dumped_step(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1, 5], action="repeat")
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
    assert dummy.woken == 1  # the next step's sleep must not hit a sleeping replica
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


def test_advantage_inert_for_v0_policy_gradient(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    config.trainer.use_v1 = False  # the V0 trainer computes advantages inline
    enable_stage(config, "advantage", steps=[1])
    with pytest.raises(ValueError, match="V0"):
        init_skip_manager(config)


def test_advantage_supported_for_v1_policy_gradient(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    enable_stage(config, "advantage", steps=[1])
    init_skip_manager(config)
    assert "advantage" in SkipManager.skip_instances


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


@pytest.mark.parametrize(
    "use_v1, trainer_mode",
    [(False, "separate_async"), (True, "sync")],
    ids=["v0", "v1_sync"],
)
def test_weight_sync_inert_without_mid_cycle_sync(tmp_path, reset_skip_manager, use_v1, trainer_mode):
    config = make_config(tmp_path, v1={"trainer_mode": trainer_mode})
    config.trainer.use_v1 = use_v1
    config.omni_skip.weight_sync.update({"enable": True, "steps": [1], "ack_stale_weights": True})
    with pytest.raises(ValueError, match="mid-cycle"):
        init_skip_manager(config)


def test_reward_inert_with_streaming_reward(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    config.reward.reward_model.enable_resource_pool = True
    enable_stage(config, "reward", steps=[1])
    with pytest.raises(ValueError, match="colocated reward"):
        init_skip_manager(config)


def test_ref_log_prob_inert_without_reference_policy(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    enable_stage(config, "ref_log_prob", steps=[1])
    with pytest.raises(ValueError, match="no reference policy"):
        init_skip_manager(config)


def test_enabled_stage_without_steps_rejected(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[])
    with pytest.raises(ValueError, match="non-empty steps"):
        init_skip_manager(config)


def test_misspelled_stage_rejected(tmp_path, reset_skip_manager):
    config = make_config(tmp_path)
    OmegaConf.update(config, "omni_skip.old_logprob", {"enable": True, "steps": [1]})
    with pytest.raises(ConfigKeyError):
        init_skip_manager(config)


@pytest.mark.parametrize("role", STAGE_SKIP_ROLES)
@pytest.mark.parametrize("enabled", [False, True])
def test_misspelled_action_rejected_even_when_disabled(tmp_path, reset_skip_manager, role, enabled):
    config = make_config(tmp_path, omni_skip_overrides={role: {"action": "cahce", "enable": enabled, "steps": [1]}})
    config.omni_skip.weight_sync.ack_stale_weights = True
    with pytest.raises(ValueError, match="[Aa]ction"):
        init_skip_manager(config)


@pytest.mark.parametrize(
    "trainer_cls,method_name,role",
    [
        (PolicyGradientDiffusionTrainerV1, "_compute_reward_colocate", "reward"),
        (PolicyGradientDiffusionTrainerV1, "_compute_old_log_prob", "old_log_prob"),
        (PolicyGradientDiffusionTrainerV1, "_compute_ref_log_prob", "ref_log_prob"),
        (PolicyGradientDiffusionTrainerV1, "_compute_ref_noise_pred", "ref_log_prob"),
        (PolicyGradientDiffusionTrainerV1, "_compute_advantage", "advantage"),
        (PolicyGradientDiffusionTrainerV1, "_update_actor", "actor_update"),
        (PolicyGradientDiffusionTrainerV1, "_update_rollout_weights_mid_cycle", "weight_sync"),
        (PolicyGradientDiffusionTrainerV1SeparateAsync, "_compute_old_log_prob", "old_log_prob"),
        (BaseRayDiffusionTrainer, "_compute_reward_colocate", "reward"),
        (BaseRayDiffusionTrainer, "_update_actor", "actor_update"),
        (PolicyGradientRayTrainer, "_compute_old_log_prob", "old_log_prob"),
        (PolicyGradientRayTrainer, "_compute_ref_log_prob", "ref_log_prob"),
        (DirectPreferenceRayTrainer, "_update_actor", "actor_update"),
        (DirectPreferenceRayTrainer, "_compute_ref_noise_pred", "ref_log_prob"),
    ],
)
def test_real_trainer_stage_uses_expected_skip_role(reset_skip_manager, trainer_cls, method_name, role):
    intercepted = []
    sentinel = object()

    def replay(step, func, *args, **kwargs):
        intercepted.append((step, func.__name__))
        return sentinel

    SkipManager.skip_instances = {
        role: SimpleNamespace(
            is_enabled=lambda: True,
            support_online_step=False,
            steps=[1],
            meet_precondition=lambda *args, **kwargs: True,
            warp_function=replay,
        )
    }
    SkipManager.set_step(1)
    trainer = SimpleNamespace(parameter_sync_step=1)
    if trainer_cls is PolicyGradientDiffusionTrainerV1SeparateAsync:
        trainer = object.__new__(trainer_cls)
        trainer.parameter_sync_step = 1
    args = () if role == "weight_sync" else (_batch(),)
    assert getattr(trainer_cls, method_name)(trainer, *args) is sentinel
    assert intercepted == [(1, method_name)]


def test_v0_trainer_ignores_leftover_parameter_sync_step(tmp_path, reset_skip_manager, dummy):
    config = make_config(tmp_path, v1={"separate_async": {"parameter_sync_step": 4}})
    config.trainer.use_v1 = False
    enable_stage(config, "old_log_prob", steps=[1, 2], action="repeat")
    init_skip_manager(config)
    SkipManager.set_step(1)
    dummy.compute_old_log_prob(_batch())

    SkipManager.set_step(2)
    dummy.compute_old_log_prob(_batch())
    assert dummy.calls == {"old_log_prob": 1}  # step 1 counts as a complete cache


def test_failed_dump_raises(tmp_path, reset_skip_manager, dummy, monkeypatch):
    config = make_config(tmp_path)
    enable_stage(config, "old_log_prob", steps=[1])
    init_skip_manager(config)
    SkipManager.set_step(1)

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        dummy.compute_old_log_prob(_batch())


def test_log_skip_summary_logs_counters_only_when_enabled(tmp_path, reset_skip_manager, dummy):
    logged = []
    tracking = SimpleNamespace(log=lambda data, step: logged.append((step, data)))
    init_skip_manager(make_config(tmp_path))
    log_skip_summary(tracking, step=3)
    assert logged == []

    config = make_config(tmp_path)
    enable_stage(config, "actor_update", steps=[1], action="empty")
    init_skip_manager(config)
    SkipManager.set_step(1)
    dummy.update_actor(_batch())
    log_skip_summary(tracking, step=3)
    assert logged == [(3, {"skip/actor_update/calls": 1, "skip/actor_update/steps": 1})]


def _tq_trainer(submitted: list) -> SimpleNamespace:
    return SimpleNamespace(
        _next_train_batch=lambda: {"uid": ["new-uid"]},
        _submit_batch_to_rollout=submitted.append,
    )


def test_v1_submit_replays_rollout_tq_cache_instead_of_generating(tmp_path, reset_skip_manager, monkeypatch):
    skip = {"rollout_tq": {"enable": True, "dump_dir": str(tmp_path / DUMP_DIR), "steps": [1], "action": "cache"}}
    config = make_config(tmp_path, skip=skip)
    init_skip_manager(config)
    SkipManager.set_step(1)
    rollout_tq = SkipManager.skip_instances["rollout_tq"]
    injected = []
    monkeypatch.setattr(RolloutTqSkip, "load_dump_data", lambda self, **kwargs: injected.append(kwargs))

    submitted = []
    PolicyGradientDiffusionTrainerV1._add_batch_to_generate(_tq_trainer(submitted))
    assert len(submitted) == 1 and injected == []  # cache miss: the prompts go to rollout

    slot = rollout_tq._get_v1_inner_dir(1, 0)
    slot.mkdir(parents=True)
    (slot / rollout_tq.tq_batch_name).touch()
    (slot / rollout_tq.meta_name).touch()
    PolicyGradientDiffusionTrainerV1._add_batch_to_generate(_tq_trainer(submitted))
    assert len(submitted) == 1  # cache hit: no generation
    assert injected[0]["new_prompt_uids"] == ["new-uid"]


def test_rollout_tq_replay_keeps_record_pass_row_order(tmp_path, reset_skip_manager, monkeypatch):
    """Stage caches are row-aligned, so a replayed session must map back to its own trajectory."""
    skip = {"rollout_tq": {"enable": True, "dump_dir": str(tmp_path / DUMP_DIR), "steps": [1], "action": "cache"}}
    init_skip_manager(make_config(tmp_path, skip=skip))
    rollout_tq = SkipManager.skip_instances["rollout_tq"]

    row_ids = {"a_0_0": 0, "a_1_0": 1, "b_0_0": 2, "b_1_0": 3}
    prompt_index = {"a": 0, "b": 1}
    injected = {}

    def kv_batch_get(keys, partition_id):
        return TensorDict({"row": torch.tensor([row_ids[key] for key in keys])}, batch_size=[len(keys)])

    def kv_batch_put(keys, partition_id, fields=None, tags=None):
        if fields is not None:
            injected.update(zip(keys, fields["row"].tolist(), strict=True))

    fake_tq = SimpleNamespace(kv_batch_get=kv_batch_get, kv_batch_put=kv_batch_put)
    monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq)

    storage_order = ["b_1_0", "a_0_0", "b_0_0", "a_1_0"]
    tags = [{"gen_batch_seq": 0, "prompt_index": prompt_index[key[0]]} for key in storage_order]
    sampled = KVBatchMeta(partition_id="train", keys=storage_order, tags=tags)
    rollout_tq.prepare_data(step=1, batch=sampled, global_steps=1)
    rollout_tq.load_dump_data(step=1, new_prompt_uids=["x", "y"], n=2, global_steps=1)

    assert injected == {"x_0_0": 0, "x_1_0": 1, "y_0_0": 2, "y_1_0": 3}


def test_v0_rollout_replays_through_verl_agent_loop_hook(tmp_path, reset_skip_manager):
    """The V0 trainer calls verl's annotated AgentLoopManager.generate_sequences directly."""
    skip = {"rollout": {"enable": True, "dump_dir": str(tmp_path / DUMP_DIR), "steps": [3], "action": "cache"}}
    config = make_config(tmp_path, skip=skip)
    init_skip_manager(config)
    SkipManager.set_step(3)
    rollout = SkipManager.skip_instances["rollout"]
    assert isinstance(rollout, DiffusionRolloutSkip)

    rollout.prepare_data(step=3, result=DataProto.from_single_dict(data={"responses": torch.ones(1, 4)}))
    prompts = DataProto.from_single_dict(data={"x": torch.zeros(1, 4)})
    prompts.non_tensor_batch["uid"] = ["fresh-uid"]

    replayed = AgentLoopManager.generate_sequences(None, prompts)
    torch.testing.assert_close(replayed.batch["responses"], torch.ones(1, 4))
    assert replayed.non_tensor_batch["uid"] == ["fresh-uid"]


def _write_metrics(path: Path, rows: list[tuple[int, dict]]) -> Path:
    path.write_text("".join(json.dumps({"step": step, "data": data}) + "\n" for step, data in rows))
    return path


def _compare(record: Path, replay: Path, *stages: str) -> subprocess.CompletedProcess:
    args = [sys.executable, str(COMPARE_SCRIPT), str(record), str(replay), "--num-steps", "1"]
    if stages:
        args += ["--replayed-stages", *stages]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def test_compare_script_passes_identical_runs_and_ignores_timing(tmp_path):
    record = _write_metrics(tmp_path / "a.jsonl", [(1, {"actor/loss": 0.5, "timing_s/step": 1.0})])
    replay = _write_metrics(
        tmp_path / "b.jsonl",
        [
            (1, {"actor/loss": 0.5, "timing_s/step": 9.0}),
            (2, {"skip/old_log_prob/calls": 1, "skip/old_log_prob/steps": 1}),
        ],
    )
    result = _compare(record, replay, "old_log_prob")
    assert result.returncode == 0, result.stdout


def test_compare_script_fails_on_metric_mismatch(tmp_path):
    record = _write_metrics(tmp_path / "a.jsonl", [(1, {"actor/loss": 0.5})])
    replay = _write_metrics(tmp_path / "b.jsonl", [(1, {"actor/loss": 0.5000001})])
    result = _compare(record, replay)
    assert result.returncode == 1
    assert "actor/loss" in result.stdout


@pytest.mark.parametrize(
    "recorded,replayed,exit_code", [(float("nan"), float("nan"), 0), (float("nan"), 0.5, 1), (0.5, float("nan"), 1)]
)
def test_compare_script_handles_nan_metrics(tmp_path, recorded, replayed, exit_code):
    record = _write_metrics(tmp_path / "a.jsonl", [(1, {"actor/loss": recorded})])
    replay = _write_metrics(tmp_path / "b.jsonl", [(1, {"actor/loss": replayed})])
    assert _compare(record, replay).returncode == exit_code


@pytest.mark.parametrize("missing_from", ["record", "replay"])
def test_compare_script_rejects_missing_nan_metric(tmp_path, missing_from):
    complete = [(1, {"actor/loss": 0.5, "actor/optional": float("nan")})]
    missing = [(1, {"actor/loss": 0.5})]
    record = _write_metrics(tmp_path / "a.jsonl", missing if missing_from == "record" else complete)
    replay = _write_metrics(tmp_path / "b.jsonl", missing if missing_from == "replay" else complete)
    result = _compare(record, replay)
    assert result.returncode == 1
    assert "actor/optional" in result.stdout


def test_compare_script_fails_when_a_stage_was_not_replayed(tmp_path):
    rows = [(1, {"actor/loss": 0.5}), (2, {"skip/old_log_prob/calls": 0, "skip/old_log_prob/steps": 0})]
    record, replay = _write_metrics(tmp_path / "a.jsonl", rows), _write_metrics(tmp_path / "b.jsonl", rows)
    result = _compare(record, replay, "old_log_prob")
    assert result.returncode == 1
    assert "skip/old_log_prob/steps" in result.stdout


def test_compare_script_fails_when_nothing_is_comparable(tmp_path):
    rows = [(1, {"timing_s/step": 1.0})]
    result = _compare(_write_metrics(tmp_path / "a.jsonl", rows), _write_metrics(tmp_path / "b.jsonl", rows))
    assert result.returncode == 1
    assert "no comparable" in result.stdout


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


@pytest.mark.parametrize("keyword_prompts", [False, True])
@pytest.mark.parametrize("malformed", ["tuple", "row_count"])
def test_diffusion_rollout_rejects_invalid_cache_before_restamping(tmp_path, monkeypatch, keyword_prompts, malformed):
    from verl.utils.skip.rollout_skip import RolloutSkip

    skip = DiffusionRolloutSkip(
        OmegaConf.create({"enable": True, "dump_dir": str(tmp_path), "steps": [1], "action": "cache"}),
        make_config(tmp_path),
    )
    cached = _batch(1)
    cached.non_tensor_batch["uid"] = ["cached-uid"]
    result = (cached, {}) if malformed == "tuple" else cached
    monkeypatch.setattr(RolloutSkip, "warp_function", lambda *args, **kwargs: result)
    prompts = _batch(2)
    prompts.non_tensor_batch["uid"] = ["fresh-0", "fresh-1"]
    args, kwargs = ((None,), {"prompts": prompts}) if keyword_prompts else ((None, prompts), {})
    error, message = (TypeError, "DataProto") if malformed == "tuple" else (ValueError, "rows")
    with pytest.raises(error, match=message):
        skip.warp_function(1, None, *args, **kwargs)
    assert cached.non_tensor_batch["uid"] == ["cached-uid"]
