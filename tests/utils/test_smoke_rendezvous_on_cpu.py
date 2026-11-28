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
import os

import pytest

from tests.smoke_rendezvous import configure_gloo_file_store


@pytest.mark.parametrize("group", ["rollout_standalone_0", "rollout_standalone_1", "rollout_reward_standalone_0"])
def test_standalone_group_uses_its_own_store(monkeypatch, tmp_path, group):
    monkeypatch.delenv("DIST_INIT_METHOD", raising=False)
    monkeypatch.setenv("VERL_SMOKE_FILESTORE_DIR", str(tmp_path))
    monkeypatch.setenv("WG_PREFIX", group)

    configure_gloo_file_store()

    assert os.environ["DIST_INIT_METHOD"] == f"file://{tmp_path}/{group}"


@pytest.mark.parametrize("group", ["", "actor", "rollout_reward_colocate_0"])
def test_other_workers_keep_default_rendezvous(monkeypatch, group):
    monkeypatch.delenv("DIST_INIT_METHOD", raising=False)
    monkeypatch.delenv("VERL_SMOKE_FILESTORE_DIR", raising=False)
    monkeypatch.setenv("WG_PREFIX", group)

    configure_gloo_file_store()

    assert "DIST_INIT_METHOD" not in os.environ


def test_explicit_rendezvous_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_SMOKE_FILESTORE_DIR", str(tmp_path))
    monkeypatch.setenv("WG_PREFIX", "rollout_standalone_0")
    monkeypatch.setenv("DIST_INIT_METHOD", "env://")

    configure_gloo_file_store()

    assert os.environ["DIST_INIT_METHOD"] == "env://"
