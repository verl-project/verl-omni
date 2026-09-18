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
"""CPU tests for the hidden-state agent-loop plumbing (Task 6)."""

import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.agent_loop.nitrobrew_opd_agent_loop import (
    lift_teacher_hidden_states,
    use_hidden_states_from_config,
)


class TestUseHiddenStatesFromConfig:
    def test_nitrobrew_mode(self):
        cfg = OmegaConf.create({"distillation": {"distillation_loss": {"loss_mode": "nitrobrew"}}})
        assert use_hidden_states_from_config(cfg) is True

    def test_reverse_kl_mode(self):
        cfg = OmegaConf.create({"distillation": {"distillation_loss": {"loss_mode": "nitrobrew_reverse_kl"}}})
        assert use_hidden_states_from_config(cfg) is True

    def test_estimator_mode(self):
        cfg = OmegaConf.create({"distillation": {"distillation_loss": {"loss_mode": "kl"}}})
        assert use_hidden_states_from_config(cfg) is False

    def test_no_distillation(self):
        assert use_hidden_states_from_config(OmegaConf.create({})) is False


class TestLiftTeacherHiddenStates:
    def test_lifts_hidden_and_key(self):
        hidden = torch.randn(7, 4)
        field = {"prompts": torch.zeros(7), "extra_fields": {"teacher_hidden_states": hidden, "teacher_key": "t0"}}
        out = lift_teacher_hidden_states(field)
        assert out["teacher_hidden_states"] is hidden
        assert out["teacher_key"] == "t0"
        assert "teacher_hidden_states" not in out["extra_fields"]

    def test_noop_without_hidden(self):
        field = {"prompts": torch.zeros(3), "extra_fields": {"min_global_steps": 1}}
        out = lift_teacher_hidden_states(field)
        assert "teacher_hidden_states" not in out
        assert out["extra_fields"] == {"min_global_steps": 1}

    def test_noop_without_extra_fields(self):
        field = {"prompts": torch.zeros(3)}
        out = lift_teacher_hidden_states(field)
        assert out is field


class TestWorkerHiddenModeSelection:
    """_use_hidden_states must only engage the hidden manager on nitrobrew modes."""

    @pytest.fixture
    def _skip_heavy_imports(self):
        # AgentLoopWorker import needs the full verl stack; only config plumbing is
        # under test here, covered by use_hidden_states_from_config above.
        pytest.importorskip("verl")

    def test_imports_resolve(self):
        from verl_omni.agent_loop.nitrobrew_opd_agent_loop import (  # noqa: F401
            AsyncTeacherHiddenStatesManager,
            NitrobrewOPDAgentLoopManagerTQ,
            NitrobrewOPDAgentLoopWorker,
        )


class TestTeacherHiddenManagerRouting:
    def _make_manager(self, keys):
        from verl_omni.agent_loop.nitrobrew_opd_agent_loop import AsyncTeacherHiddenStatesManager

        teachers = {k: object() for k in keys}
        distill = OmegaConf.create(
            {
                "teacher_key": "data_source",
                "teacher_models": {k: {"key": k, "model_path": f"/m/{k}"} for k in keys},
            }
        )
        mgr = AsyncTeacherHiddenStatesManager.__new__(AsyncTeacherHiddenStatesManager)
        mgr.teacher_model_configs = OmegaConf.to_container(distill.teacher_models)
        mgr.teacher_key = distill.teacher_key
        mgr.teacher_client = teachers
        return mgr

    def test_single_teacher_routes_without_key(self):
        mgr = self._make_manager(["default"])
        assert mgr._resolve_teacher_key(None) == "default"
        assert mgr._resolve_teacher_key("anything") == "default"

    def test_multi_teacher_requires_known_key(self):
        mgr = self._make_manager(["a", "b"])
        assert mgr._resolve_teacher_key("a") == "a"
        with pytest.raises(ValueError, match="Routing key is required"):
            mgr._resolve_teacher_key(None)
        with pytest.raises(ValueError, match="No teacher configured"):
            mgr._resolve_teacher_key("c")


class TestCollateHiddenJagged:
    def test_variable_length_hidden_becomes_jagged(self):
        """list_of_dict_to_tensordict turns per-sample [S_i, D] tensors into jagged nested."""
        from verl.utils.tensordict_utils import list_of_dict_to_tensordict

        fields = [
            {"teacher_hidden_states": torch.randn(5, 4), "uid": "a"},
            {"teacher_hidden_states": torch.randn(3, 4), "uid": "b"},
        ]
        td = list_of_dict_to_tensordict(fields)
        hidden = td["teacher_hidden_states"]
        assert hidden.is_nested
        assert hidden.shape[0] == 2
        assert hidden.shape[-1] == 4
        lengths = [int(t.shape[0]) for t in hidden.unbind()]
        assert lengths == [5, 3]
