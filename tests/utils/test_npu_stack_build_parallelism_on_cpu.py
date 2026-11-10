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
"""Keep native dependency compilation independent of model thread limits."""

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "cpus,override,expected",
    [(1, None, "1"), (2, None, "2"), (64, None, "8"), (64, "", "8"), (64, "3", "3")],
)
def test_native_build_parallelism_preserves_runtime_limits(tmp_path, cpus, override, expected):
    """Bound automatic concurrency and honor an explicit build-only limit."""
    result = _configure(tmp_path, cpus, override)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{expected}|1|1"


@pytest.mark.parametrize("override", ["0", "-1", "00", "bad"])
def test_invalid_native_build_parallelism_fails(tmp_path, override):
    """Do not treat malformed concurrency as an unbounded or serial fallback."""
    result = _configure(tmp_path, 64, override)
    assert result.returncode != 0
    assert "must be a positive integer" in result.stderr


def _configure(tmp_path, cpus, override):
    source = (Path(__file__).resolve().parents[2] / "tests/npu_smoke/sync_npu_stack.sh").read_text()
    helper = re.search(r"^configure_native_build_parallelism\(\) \{\n.*?^\}", source, re.M | re.S)
    assert helper is not None
    probe = tmp_path / "nproc"
    probe.write_text(
        '#!/bin/sh\n[ -z "${OMP_NUM_THREADS+x}" ] && [ -z "${OMP_THREAD_LIMIT+x}" ] || exit 9\n'
        f"printf '%s\\n' {cpus}\n"
    )
    probe.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1")
    env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
    if override is not None:
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = override
    command = (
        helper.group()
        + "\nconfigure_native_build_parallelism\n"
        + 'bash -c \'printf "%s|%s|%s\\n" "$CMAKE_BUILD_PARALLEL_LEVEL" "$OMP_NUM_THREADS" "$OMP_THREAD_LIMIT"\''
    )
    return subprocess.run(["bash", "-euc", command], env=env, capture_output=True, text=True, timeout=10)
