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
"""Keep shell and Python NPU smoke launchers on the same runtime settings."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.special_e2e.npu_runtime import npu_precompile_threads, npu_runtime_overrides

_SCRIPTS = Path(__file__).resolve().parents[1] / "special_e2e"


@pytest.fixture
def h3_runner(monkeypatch):
    # Fixture construction is outside the launcher-configuration boundary.
    builder = ModuleType("tests.special_e2e.build_minimax_h3_tiny_random")
    builder.DEFAULT_OUTPUT_DIR = "/unused"
    builder._checkpoint_is_complete = lambda path: True
    builder.ensure_tiny_minimax_h3_checkpoint = lambda *args, **kwargs: None
    data = ModuleType("tests.special_e2e.create_dummy_h3_data")
    for task in ("t2va", "fl2va", "ref2va"):
        setattr(data, f"build_dummy_h3_{task}_data", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    monkeypatch.setitem(sys.modules, data.__name__, data)
    monkeypatch.syspath_prepend(str(_SCRIPTS.parents[1]))
    spec = importlib.util.spec_from_file_location("h3_runtime_test", _SCRIPTS / "run_flowgrpo_minimax_h3_tiny.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _h3_overrides(runner, device):
    return runner._hydra_overrides(
        tiny_model_dir="/model",
        train_parquet="/train",
        val_parquet="/val",
        reward_stub_path="/reward",
        output_dir="/output",
        task="t2va",
        num_gpus=4,
        actor_sp=2,
        rollout_tp=2,
        text_encoder_tp=1,
        total_training_steps=1,
        ray_num_cpus=40,
        height=160,
        width=288,
        num_frames=97,
        num_inference_steps=4,
        device=device,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_shell_and_h3_forward_shared_runtime(monkeypatch, h3_runner, parallel):
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("NUM_NPUS", "4")
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "" if parallel else "1")
    monkeypatch.setenv("NPU_SMOKE_SKIP_GLOBAL_CLEANUP", "1" if parallel else "0")
    monkeypatch.setenv("RAY_MASTER_PORT_RANGE", "21000,22000" if parallel else "")
    monkeypatch.setenv("TORCHNPU_PRECOMPILE_THREADS", "3" if parallel else "2")
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""set -euo pipefail
source() {
    case "$1" in */set_env.sh) return 0 ;; *) builtin source "$@" ;; esac
}
python3() {
    if [[ "$1" == - ]]; then cat >/dev/null; else "$TEST_PYTHON" "$@"; fi
}
source "$TEST_LIB"
"$TEST_PYTHON" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${NPU_RUNTIME_ARGS[@]}"
""",
        ],
        env={**os.environ, "TEST_PYTHON": sys.executable, "TEST_LIB": str(_SCRIPTS / "lib_npu_e2e.sh")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    shared = json.loads(result.stdout)
    assert shared == npu_runtime_overrides()
    h3 = _h3_overrides(h3_runner, "npu")
    assert h3[-len(shared) - 1 : -1] == shared
    assert h3[-1] == "actor_rollout_ref.rollout.enforce_eager=True"
    assert "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0" in shared
    assert "++reward.reward_model.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0" in shared
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ="0"' in shared
    threads = "3" if parallel else "2"
    assert f'++ray_kwargs.ray_init.runtime_env.env_vars.TORCHNPU_PRECOMPILE_THREADS="{threads}"' in shared
    assert "data.dataloader_num_workers=0" in shared
    assert "transfer_queue.backend.SimpleStorage.num_data_storage_units=2" in shared
    assert ("+ray_kwargs.ray_init.address=local" in shared) is parallel
    assert ("trainer.ray_master_port_range=[21000,22000]" in shared) is parallel


def test_gpu_h3_does_not_read_npu_environment(monkeypatch, h3_runner):
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(h3_runner, "npu_runtime_overrides", lambda: pytest.fail("GPU requested NPU runtime"))
    overrides = _h3_overrides(h3_runner, "cuda")
    assert not any("weight_nz_mode" in item or "ASCEND" in item for item in overrides)
    assert "actor_rollout_ref.actor.strategy=fsdp2" in overrides


def test_shell_propagates_runtime_configuration_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHNPU_PRECOMPILE_THREADS", raising=False)
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""set -euo pipefail
source() {
    case "$1" in */set_env.sh) return 0 ;; *) builtin source "$@" ;; esac
}
python3() {
    if [[ "$1" == - ]]; then cat >/dev/null; else return 23; fi
}
source "$TEST_LIB"
touch "$AFTER_SETUP"
""",
        ],
        env={**os.environ, "TEST_LIB": str(_SCRIPTS / "lib_npu_e2e.sh"), "AFTER_SETUP": str(tmp_path / "started")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 23, result.stderr
    assert not (tmp_path / "started").exists()


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "two"])
def test_compile_thread_setting_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        npu_precompile_threads({"TORCHNPU_PRECOMPILE_THREADS": value})


def test_compile_thread_setting_defaults_and_explicit_override():
    assert npu_precompile_threads({}) == "2"
    assert npu_precompile_threads({"TORCHNPU_PRECOMPILE_THREADS": "4"}) == "4"


def test_shared_smoke_config_accepts_caller_overrides(monkeypatch):
    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "0,1")
    overrides = npu_runtime_overrides()
    with initialize_config_dir(config_dir=str(_SCRIPTS.parents[1] / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=overrides
            + [
                "data.dataloader_num_workers=3",
                "transfer_queue.backend.SimpleStorage.num_data_storage_units=4",
            ],
        )
    assert cfg.data.dataloader_num_workers == 3
    assert cfg.transfer_queue.backend.SimpleStorage.num_data_storage_units == 4
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode == 0


def test_shell_rejects_compile_threads_before_device_preflight(tmp_path):
    marker = tmp_path / "device-preflight"
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""set -euo pipefail
source() {
    case "$1" in */set_env.sh) return 0 ;; *) builtin source "$@" ;; esac
}
python3() { touch "$PREFLIGHT_MARKER"; }
source "$TEST_LIB"
""",
        ],
        env={
            **os.environ,
            "TEST_LIB": str(_SCRIPTS / "lib_npu_e2e.sh"),
            "TORCHNPU_PRECOMPILE_THREADS": "0",
            "PREFLIGHT_MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not marker.exists()


def test_direct_h3_npu_configures_compiler_before_preflight(monkeypatch, h3_runner):
    monkeypatch.delenv("TORCHNPU_PRECOMPILE_THREADS", raising=False)
    monkeypatch.setattr(h3_runner, "_parse_args", lambda: SimpleNamespace(device="npu"))

    def preflight():
        assert os.environ["TORCHNPU_PRECOMPILE_THREADS"] == "2"
        raise RuntimeError("preflight reached")

    monkeypatch.setattr(h3_runner, "_require_minimax_h3_diffusers", preflight)
    with pytest.raises(RuntimeError, match="preflight reached"):
        h3_runner.main()
