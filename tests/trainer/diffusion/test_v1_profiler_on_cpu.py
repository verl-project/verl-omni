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
"""Profiling configuration and V1 trainer RPC contracts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("present", [False, True])
def test_rollout_managers_ignore_missing_or_none_manager(present):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    if present:
        trainer.llm_server_manager = None
    assert trainer._rollout_server_managers() == []


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("standalone_present", [False, True])
def test_async_rollout_manager_selection(switch, standalone_present):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=switch)
    hybrid = object()
    standalone = object()
    trainer.llm_server_manager = hybrid
    if standalone_present:
        trainer.standalone_server_manager = standalone
    expected = ([hybrid] if switch else []) + ([standalone] if standalone_present else [])
    assert trainer._rollout_server_managers() == expected


@pytest.mark.parametrize(
    ("ref_worker_key", "teacher_worker_keys", "expected_key_roles"),
    [
        pytest.param(None, None, [("actor", "train")], id="actor-only"),
        pytest.param("actor", ["actor", "actor"], [("actor", "train")], id="all-share-actor-group"),
        pytest.param(
            None,
            ["teacher1", "teacher2"],
            [("actor", "train"), ("teacher1", "teacher"), ("teacher2", "teacher")],
            id="standalone-teachers",
        ),
        pytest.param(
            "ref",
            ["ref", "teacher1", "teacher1", "teacher2"],
            [("actor", "train"), ("ref", "ref"), ("teacher1", "teacher"), ("teacher2", "teacher")],
            id="shared-and-distinct-groups",
        ),
    ],
)
def test_profile_worker_groups(ref_worker_key, teacher_worker_keys, expected_key_roles):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    # Keys identify worker group objects; repeated keys represent aliases to the same object.
    workers = {key: object() for key in ("actor", "ref", "teacher1", "teacher2")}
    trainer.actor_rollout_wg = workers["actor"]
    trainer.use_reference_policy = ref_worker_key is not None
    trainer.ref_policy_wg = workers[ref_worker_key] if ref_worker_key is not None else None
    trainer.use_teacher_policy = teacher_worker_keys is not None
    if teacher_worker_keys is not None:
        trainer.teacher_model_manager = SimpleNamespace(
            teacher_wg={str(index): workers[key] for index, key in enumerate(teacher_worker_keys)}
        )
    trainer.global_steps = 7

    assert trainer._profile_worker_groups() == [
        (workers[key], {"role": role, "profile_step": 7}) for key, role in expected_key_roles
    ]


@pytest.mark.parametrize("continuous,expected", [(False, [1, 2, 4]), (True, [1, 4])])
def test_training_windows_and_shared_reference_group(continuous, expected):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    trainer.config = OmegaConf.create(
        {
            "global_profiler": {
                "steps": [1, 2, 4],
                "profile_continuous_steps": continuous,
            }
        }
    )
    trainer.actor_rollout_wg = Mock()
    trainer.ref_policy_wg = trainer.actor_rollout_wg
    trainer.use_reference_policy = True
    trainer.use_teacher_policy = False
    trainer.reward_loop_manager = None
    trainer.gpu_metrics_profiler = None
    trainer._init_profile_state()
    trainer.total_training_steps = 4
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    for step in range(1, 5):
        trainer.global_steps = step
        trainer._on_profiling_step_start()
        trainer._on_profiling_step_end()
    assert [call.kwargs["profile_step"] for call in trainer.actor_rollout_wg.start_profile.call_args_list] == expected
    assert trainer.actor_rollout_wg.stop_profile.call_count == len(expected)
    assert [call.kwargs["run_command"] for call in trainer.actor_rollout_wg.stop_profile.call_args_list] == (
        [False] * (len(expected) - 1) + [True]
    )


@pytest.mark.parametrize("engine", ["dp_diffusion", "veomni_diffusion"])
@pytest.mark.parametrize("mode,hybrid", [("sync", False), ("separate_async", False), ("separate_async", True)])
@pytest.mark.parametrize("actor,rollout", [(True, False), (False, True), (True, True)])
def test_compose_npu_discrete_manual_ranks(engine, mode, hybrid, actor, rollout):
    from hydra import compose, initialize_config_dir

    rollout_rank = 0 if mode == "sync" else 8
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"diffusion/model_engine={engine}",
                "trainer.use_v1=true",
                f"trainer.v1.trainer_mode={mode}",
                f"trainer.v1.separate_async.hybrid_rollout.enable_switch={str(hybrid).lower()}",
                "global_profiler.tool=npu",
                "global_profiler.steps=[1,3]",
                "global_profiler.profile_continuous_steps=false",
                "actor_rollout_ref.actor.profiler.tool=npu",
                f"actor_rollout_ref.actor.profiler.enable={str(actor).lower()}",
                "actor_rollout_ref.actor.profiler.all_ranks=false",
                "actor_rollout_ref.actor.profiler.ranks=[0]",
                "actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true",
                "actor_rollout_ref.rollout.profiler.tool=npu",
                f"actor_rollout_ref.rollout.profiler.enable={str(rollout).lower()}",
                "actor_rollout_ref.rollout.profiler.all_ranks=false",
                f"actor_rollout_ref.rollout.profiler.ranks=[{rollout_rank}]",
                "actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=true",
            ],
        )
    for role, enabled, rank in [("actor", actor, 0), ("rollout", rollout, rollout_rank)]:
        role_cfg = OmegaConf.to_container(cfg.actor_rollout_ref[role].profiler, resolve=True)
        assert role_cfg["tool"] == "npu"
        assert role_cfg["enable"] is enabled
        assert role_cfg["all_ranks"] is False
        assert role_cfg["ranks"] == [rank]
        assert role_cfg["tool_config"]["npu"]["discrete"] is True


@pytest.mark.parametrize("engine", ["dp_diffusion", "veomni_diffusion"])
@pytest.mark.parametrize("tool", ["torch", "npu"])
def test_compose_actor_and_rollout_profiling(engine, tool):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"diffusion/model_engine={engine}",
                "trainer.use_v1=true",
                f"global_profiler.tool={tool}",
                "global_profiler.steps=[1,2,4]",
                "global_profiler.save_path=outputs/profile_v1",
                "global_profiler.relocate_results=true",
                "global_profiler.finish_hook_cmd=echo done",
                "global_profiler.finish_hook_ranks=[0]",
                f"actor_rollout_ref.actor.profiler.tool={tool}",
                "actor_rollout_ref.actor.profiler.enable=true",
                "actor_rollout_ref.actor.profiler.all_ranks=true",
                f"actor_rollout_ref.rollout.profiler.tool={tool}",
                "actor_rollout_ref.rollout.profiler.enable=true",
                "actor_rollout_ref.rollout.profiler.all_ranks=true",
            ],
        )
    for role in ("actor", "rollout"):
        role_cfg = OmegaConf.to_container(cfg.actor_rollout_ref[role].profiler, resolve=True)
        assert role_cfg["enable"] is True
        assert role_cfg["tool"] == tool
        assert role_cfg["save_path"] == "outputs/profile_v1"
        assert role_cfg["relocate_results"] is True
        assert role_cfg["finish_hook_cmd"] == "echo done"
        assert role_cfg["finish_hook_ranks"] == [0]
        assert role_cfg["tool_config"][tool]["discrete"] is False
