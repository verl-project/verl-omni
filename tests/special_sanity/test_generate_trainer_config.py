# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Guards for scripts/generate_trainer_config.sh.

The script's verl probe decides whether the omni config checks run; a failed
probe must abort the script in CI instead of silently skipping them.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_trainer_config.sh"


def test_verl_probe_failure_fails_closed_in_ci():
    """With CI set and a python3 that cannot import verl, the script must exit 1.

    Uses the OS python (PATH=/usr/bin:/bin), which never has verl installed,
    so the probe fails and the fail-closed branch is exercised.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"CI": "1", "PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"})
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}\n{output}"
    assert "failed in CI" in output, f"missing fail-closed message:\n{output}"
