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
"""Exercise attention defaults at the shared smoke launcher command boundary."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "special_e2e"
_LAUNCHERS = ("run_flowgrpo_bagel_pickscore.sh", "run_flowgrpo_qwen_image_separate.sh")


@pytest.fixture
def validate_attention_pair():
    # Load the stdlib-only validator without importing optional trainer dependencies.
    spec = importlib.util.spec_from_file_location(
        "smoke_attention_validation", _SCRIPTS.parents[1] / "verl_omni/utils/diffusion_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_attention_consistency


def _launch(tmp_path, launcher, actor, rollout, fa_available, trainer_exit=0):
    env = {key: value for key, value in os.environ.items() if key not in {"ATTN_BACKEND", "ROLLOUT_ATTN_BACKEND"}}
    if actor is not None:
        env["ATTN_BACKEND"] = actor
    if rollout is not None:
        env["ROLLOUT_ATTN_BACKEND"] = rollout
    capture = tmp_path / "trainer.json"
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(
        r"""python3() {
    if [[ "$1" == -c ]]; then
        echo probe >> "$PROBE_LOG"
        return "$FA_EXIT"
    elif [[ "$1" == - ]]; then
        cat >/dev/null
    elif [[ "$1" == -m && "$2" == verl_omni.trainer.main_diffusion ]]; then
        "$TEST_PYTHON" - "$@" <<'PY_CAPTURE'
import json
import os
import sys
from pathlib import Path

Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
PY_CAPTURE
        return "$TRAINER_EXIT"
    fi
}
nvidia-smi() { return 0; }
"""
    )
    env.update(
        BASH_ENV=str(bash_env),
        TEST_PYTHON=sys.executable,
        CAPTURE=str(capture),
        PROBE_LOG=str(tmp_path / "probe.log"),
        FA_EXIT="0" if fa_available else "1",
        TRAINER_EXIT=str(trainer_exit),
        NUM_GPUS="4",
        MODEL_PATH=str(tmp_path / "model with spaces"),
        DATA_DIR=str(tmp_path / "data"),
    )
    result = subprocess.run(
        ["bash", str(_SCRIPTS / launcher)],
        cwd=_SCRIPTS.parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    argv = json.loads(capture.read_text()) if capture.exists() else None
    return result, argv, (tmp_path / "probe.log").exists()


@pytest.mark.parametrize("launcher", _LAUNCHERS)
@pytest.mark.parametrize("fa_available", [False, True])
@pytest.mark.parametrize(
    "actor,rollout,expected",
    [
        (None, None, None),
        ("native", None, ("native", "TORCH_SDPA")),
        ("_native_npu", "", ("_native_npu", "TORCH_SDPA")),
        ("flash_varlen_hub", None, ("flash_varlen_hub", "FLASH_ATTN")),
        ("_flash_3_varlen_hub", None, ("_flash_3_varlen_hub", "FLASH_ATTN")),
        (None, "TORCH_SDPA", ("native", "TORCH_SDPA")),
        ("", "FLASH_ATTN", ("_flash_3_varlen_hub", "FLASH_ATTN")),
        (None, "FLASH_ATTN_HUB", ("_flash_3_varlen_hub", "FLASH_ATTN_HUB")),
        (None, "FLASH_ATTN_3_HUB", ("_flash_3_varlen_hub", "FLASH_ATTN_3_HUB")),
        ("_native_npu", "TORCH_SDPA", ("_native_npu", "TORCH_SDPA")),
        ("flash_varlen_hub", "FLASH_ATTN_HUB", ("flash_varlen_hub", "FLASH_ATTN_HUB")),
    ],
)
def test_launchers_select_compatible_attention_pair(
    tmp_path, launcher, fa_available, actor, rollout, expected, validate_attention_pair
):
    """One explicit backend determines its counterpart independently of FA availability."""
    result, argv, probed = _launch(tmp_path, launcher, actor, rollout, fa_available)
    assert result.returncode == 0, result.stdout + result.stderr
    args = dict(arg.split("=", 1) for arg in argv[2:])
    actual = (args["actor_rollout_ref.model.attn_backend"], args["actor_rollout_ref.rollout.rollout_attn_backend"])
    if expected is None:
        expected = ("_flash_3_varlen_hub", "FLASH_ATTN") if fa_available else ("native", "TORCH_SDPA")
    assert actual == expected
    assert probed is (not actor and not rollout)
    validate_attention_pair(
        SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor={"strategy": "fsdp"},
                model={"attn_backend": actual[0]},
                rollout={"rollout_attn_backend": actual[1]},
            )
        )
    )


@pytest.mark.parametrize("launcher", _LAUNCHERS)
@pytest.mark.parametrize("trainer_exit", [23, 134])
def test_attention_defaults_preserve_trainer_failure(tmp_path, launcher, trainer_exit):
    """A compatible inferred pair must not swallow the trainer's nonzero status."""
    result, argv, _ = _launch(tmp_path, launcher, "native", None, True, trainer_exit)
    assert argv is not None
    assert result.returncode == trainer_exit
    assert "test passed" not in result.stdout


@pytest.mark.parametrize("launcher", _LAUNCHERS)
def test_explicit_attention_mismatch_is_not_rewritten(tmp_path, launcher, validate_attention_pair):
    """Keep both explicit values so the trainer can reject a conflicting pair."""
    result, argv, probed = _launch(tmp_path, launcher, "native", "FLASH_ATTN", True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "actor_rollout_ref.model.attn_backend=native" in argv
    assert "actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN" in argv
    assert not probed
    with pytest.raises(ValueError, match="mismatch"):
        validate_attention_pair(
            SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(
                    actor={"strategy": "fsdp"},
                    model={"attn_backend": "native"},
                    rollout={"rollout_attn_backend": "FLASH_ATTN"},
                )
            )
        )


@pytest.mark.parametrize("launcher", _LAUNCHERS)
@pytest.mark.parametrize("actor,rollout", [("unknown", None), (None, "UNKNOWN")])
def test_unknown_partial_attention_fails_before_trainer(tmp_path, launcher, actor, rollout):
    """Report an unsupported single backend without guessing its counterpart."""
    result, argv, probed = _launch(tmp_path, launcher, actor, rollout, True)
    assert result.returncode == 2
    assert "Cannot infer" in result.stderr
    assert argv is None
    assert not probed
