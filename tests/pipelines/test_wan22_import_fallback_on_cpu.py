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

import importlib
import logging
import sys
import types

import pytest

import verl_omni.pipelines.wan22_dance_grpo as wan22

ADAPTER = "verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter"


@pytest.fixture
def reload_wan22(monkeypatch):
    """Reload the package with a replaced rollout adapter, then restore the real one."""
    importlib.import_module(ADAPTER)

    def _reload(adapter_module):
        monkeypatch.setitem(sys.modules, ADAPTER, adapter_module)
        return importlib.reload(wan22)

    yield _reload
    monkeypatch.undo()
    importlib.reload(wan22)


def test_missing_rollout_dependency_is_reported_with_its_cause(reload_wan22, caplog):
    with caplog.at_level(logging.WARNING, logger=wan22.__name__):
        module = reload_wan22(None)  # None in sys.modules makes the import raise ImportError

    assert any("unavailable" in record.getMessage() for record in caplog.records)
    with pytest.raises(RuntimeError, match="unavailable") as excinfo:
        module.Wan22DanceGRPOPipelineWithLogProb()
    assert isinstance(excinfo.value.__cause__, ImportError)
    assert "GPU" not in str(excinfo.value)


def test_non_dependency_import_failure_is_raised(reload_wan22):
    broken = types.ModuleType(ADAPTER)

    def _fail(name):
        raise RuntimeError(f"broken adapter while resolving {name}")

    broken.__getattr__ = _fail
    with pytest.raises(RuntimeError, match="broken adapter"):
        reload_wan22(broken)
