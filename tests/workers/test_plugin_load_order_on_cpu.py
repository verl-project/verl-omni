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
"""CPU test: verl plugins load before verl_omni's engines read the device.

The omni recipes export ``VERL_USE_EXTERNAL_MODULES=verl_omni``, and verl imports
those modules before its ``verl.plugins`` entry points. A fake plugin installed
into a temporary directory records whether the engine modules were already
imported when it loaded.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PLUGIN = """\
import sys

ENGINES_IMPORTED_FIRST = "verl_omni.workers.engine" in sys.modules
"""

_PROBE = "import verl, verl_omni, fake_verl_plugin; print(fake_verl_plugin.ENGINES_IMPORTED_FIRST)"


def _install_fake_plugin(site_dir) -> None:
    (site_dir / "fake_verl_plugin.py").write_text(_PLUGIN)
    dist_info = site_dir / "fake_verl_plugin-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-verl-plugin\nVersion: 0.0.0\n")
    (dist_info / "entry_points.txt").write_text("[verl.plugins]\nfake = fake_verl_plugin\n")


def _run_probe(site_dir, **env_overrides) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(site_dir), env.get("PYTHONPATH")]))
    env.pop("VERL_USE_EXTERNAL_PLUGINS", None)
    env.update(env_overrides)
    return subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.parametrize("external_modules", ["verl_omni", ""])
def test_plugins_load_before_engine_imports(tmp_path, external_modules):
    _install_fake_plugin(tmp_path)
    result = _run_probe(tmp_path, VERL_USE_EXTERNAL_MODULES=external_modules)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_plugin_policy_none_is_respected(tmp_path):
    _install_fake_plugin(tmp_path)
    result = _run_probe(tmp_path, VERL_USE_EXTERNAL_MODULES="verl_omni", VERL_USE_EXTERNAL_PLUGINS="none")
    assert result.returncode == 0, result.stderr
    # Nothing loaded the plugin, so the probe's own import is the first one.
    assert result.stdout.strip().splitlines()[-1] == "True"
