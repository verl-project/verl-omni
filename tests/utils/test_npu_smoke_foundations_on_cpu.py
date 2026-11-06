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
"""Check smoke ownership and exit boundaries without devices or process signals."""

import importlib.util
import os
import signal
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def smoke_modules(monkeypatch, tmp_path):
    """Load the real controller and supervisor with isolated module bindings."""
    scripts = Path(__file__).resolve().parents[2] / "tests/npu_smoke"
    loaded = []
    for name in ("run_npu_smoke_processes", "run_npu_smoke_parallel"):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    processes, controller = loaded
    registry = tmp_path / "repo/tests/npu_smoke"
    registry.mkdir(parents=True)
    for group, layout in controller._LAYOUTS.items():
        ids = [i for _, _, items in layout for i in items]
        (registry / f"run_npu_smoke_{group}.sh").write_text("".join(f'run_test {i} "fixture" true\n' for i in ids))
    monkeypatch.setattr(controller, "_REPO_ROOT", tmp_path / "repo")
    for name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_SMOKE_TEST_IDS",
        "NPU_SMOKE_FULL_TOPOLOGY",
        "NPU_SMOKE_PARALLEL_CHILD",
    ):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(controller=controller, processes=processes, registry=registry)


@pytest.mark.parametrize("value", [None, ""])
def test_default_runtime_device_list(smoke_modules, monkeypatch, value):
    """An absent or empty runtime mask uses the supplied default."""
    if value is not None:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", value)
    assert smoke_modules.controller._device_list("ASCEND_RT_VISIBLE_DEVICES", "0,1", expected=2) == ["0", "1"]


@pytest.mark.parametrize("value", ["0", "0,0", "0,00", "1,0", "0,1,2", "0, 1", "0,,1", "-1,0", "0,1,"])
def test_runtime_device_list_rejects_invalid_masks(smoke_modules, monkeypatch, value):
    """Runtime lanes require the exact count, unique IDs and ascending order."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", value)
    with pytest.raises(ValueError):
        smoke_modules.controller._device_list("ASCEND_RT_VISIBLE_DEVICES", "0,1", expected=2)


@pytest.mark.parametrize("value,valid", [("4,5,0,1", True), ("1,0", True), ("0", False), ("0,00", False)])
def test_physical_allocation_may_include_other_lanes(smoke_modules, monkeypatch, value, valid):
    """Physical allocations permit rotation and supersets without duplicates."""
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", value)
    if valid:
        assert smoke_modules.controller._device_list("ASCEND_VISIBLE_DEVICES", "0,1", expected=2) == value.split(",")
    else:
        with pytest.raises(ValueError):
            smoke_modules.controller._device_list("ASCEND_VISIBLE_DEVICES", "0,1", expected=2)


@pytest.mark.parametrize("group", ["core", "e2e"])
def test_default_selection_matches_registry(smoke_modules, group):
    """Default selection includes every registered case exactly once."""
    expected = {i for _, _, ids in smoke_modules.controller._LAYOUTS[group] for i in ids}
    assert smoke_modules.controller._selected_ids(group) == expected


@pytest.mark.parametrize("group", ["core", "e2e"])
def test_shipped_layout_matches_group_registry(smoke_modules, monkeypatch, group):
    """The checked-in runners and controller plan must agree before activation."""
    monkeypatch.setattr(smoke_modules.controller, "_REPO_ROOT", Path(__file__).resolve().parents[2])
    assert smoke_modules.controller._selected_ids(group)


@pytest.mark.parametrize("value", ["00", "0,", "0 3", "0,0", "2", "99", "-1", ",0"])
def test_selection_rejects_invalid_duplicate_or_foreign_ids(smoke_modules, monkeypatch, value):
    """Malformed and wrong-group selectors cannot turn into a partial pass."""
    monkeypatch.setenv("NPU_SMOKE_TEST_IDS", value)
    with pytest.raises(ValueError):
        smoke_modules.controller._selected_ids("core")


def test_selection_accepts_subset(smoke_modules, monkeypatch):
    """Explicit selection preserves the requested registered subset."""
    monkeypatch.setenv("NPU_SMOKE_TEST_IDS", "0,3")
    assert smoke_modules.controller._selected_ids("core") == {0, 3}


@pytest.mark.parametrize("replacement", ['run_test 0 "only" true\n', 'run_test 99 "foreign" true\n', None])
def test_selection_rejects_registry_drift(smoke_modules, replacement):
    """Missing, foreign and repeated registrations fail before launching work."""
    path = smoke_modules.registry / "run_npu_smoke_core.sh"
    path.write_text(replacement if replacement is not None else path.read_text() + 'run_test 0 "duplicate" true\n')
    with pytest.raises(ValueError, match="registry"):
        smoke_modules.controller._selected_ids("core")


@pytest.fixture
def fake_proc(smoke_modules, monkeypatch, tmp_path):
    """Redirect only the supervisor's /proc reads to synthetic stat files."""
    root = tmp_path / "proc"

    def write(pid, parent, start):
        directory = root / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        fields = ["S", str(parent), *(["0"] * 17), str(start)]
        (directory / "stat").write_text(f"{pid} (worker with ) parentheses) {' '.join(fields)}\n")

    for args in ((100, 1, 10), (101, 100, 11), (102, 101, 12), (200, 1, 20)):
        write(*args)
    monkeypatch.setattr(smoke_modules.processes, "Path", lambda path: root / Path(path).relative_to("/proc"))
    return SimpleNamespace(write=write, root=root, identity=smoke_modules.processes._Process(100, 10, 1, "S"))


def test_process_identity_and_ancestry_read_stat_fields(smoke_modules, fake_proc):
    """Stat parsing handles parenthesized comm names and follows parent links."""
    module = smoke_modules.processes
    assert module.process_identity(102) == module._Process(102, 12, 101, "S")
    assert [p.pid for p in module._ancestry(102, fake_proc.identity)] == [102, 101, 100]


@pytest.mark.parametrize("case", ["missing", "reused-root", "cycle", "foreign"])
def test_ancestry_refuses_unowned_or_stale_chain(smoke_modules, fake_proc, case):
    """A stale PID, missing process or disconnected chain cannot establish ownership."""
    if case == "missing":
        (fake_proc.root / "102/stat").unlink()
    elif case == "reused-root":
        fake_proc.write(100, 1, 99)
    elif case == "cycle":
        fake_proc.write(101, 102, 11)
    else:
        fake_proc.write(101, 200, 11)
    assert smoke_modules.processes._ancestry(102, fake_proc.identity) is None


def test_ancestry_rechecks_parent_identity(smoke_modules, fake_proc, monkeypatch):
    """Parent PID reuse after the first read invalidates the entire chain."""
    module = smoke_modules.processes
    original = module.process_identity

    def changing_identity(pid):
        result = original(pid)
        if pid == 101:
            fake_proc.write(101, 100, result.start + 1)
        return result

    monkeypatch.setattr(module, "process_identity", changing_identity)
    assert module._ancestry(102, fake_proc.identity) is None


@pytest.mark.parametrize("mode", [0o600, 0o644, 0o640, 0o604, 0o660])
def test_state_requires_private_permissions(smoke_modules, tmp_path, mode):
    """Only private supervisor metadata can be trusted."""
    path = tmp_path / "state.json"
    path.write_text('{"token": "fixture"}')
    path.chmod(mode)
    if mode == 0o600:
        assert smoke_modules.processes._read_state(path) == {"token": "fixture"}
    else:
        with pytest.raises(RuntimeError, match="owned private regular"):
            smoke_modules.processes._read_state(path)


@pytest.mark.parametrize("kind", ["foreign-owner", "symlink", "nonregular"])
def test_state_rejects_untrusted_metadata(smoke_modules, tmp_path, monkeypatch, kind):
    """Ownership, file type and no-follow policy protect supervisor state."""
    path = tmp_path / "state.json"
    path.write_text("{}")
    path.chmod(0o600)
    module = smoke_modules.processes
    if kind == "foreign-owner":
        monkeypatch.setattr(module.os, "getuid", lambda: path.stat().st_uid + 1)
    elif kind == "nonregular":
        monkeypatch.setattr(
            module.os, "fstat", lambda fd: SimpleNamespace(st_uid=os.getuid(), st_mode=stat.S_IFIFO | 0o600)
        )
    else:
        link = tmp_path / "state-link"
        link.symlink_to(path)
        path = link
    with pytest.raises((RuntimeError, OSError)):
        module._read_state(path)


@pytest.mark.parametrize("kind", ["failed", "success", "foreign-owner", "permissions", "symlink", "file"])
def test_runtime_removal_requires_success_and_owned_private_directory(smoke_modules, tmp_path, monkeypatch, kind):
    """Failed evidence and changed runtime paths must survive cleanup."""
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    marker = runtime / "evidence"
    marker.write_text("keep")
    module = smoke_modules.controller
    if kind == "foreign-owner":
        monkeypatch.setattr(module.os, "getuid", lambda: runtime.stat().st_uid + 1)
    elif kind == "permissions":
        runtime.chmod(0o755)
    elif kind == "symlink":
        link = tmp_path / "runtime-link"
        link.symlink_to(runtime, target_is_directory=True)
        runtime = link
    elif kind == "file":
        runtime = tmp_path / "regular-file"
        runtime.write_text("keep")
    child = {"runtime": runtime, "successful": kind != "failed"}
    if kind in ("failed", "success"):
        module._discard_successful_runtime(child)
        assert runtime.exists() is (kind == "failed")
    else:
        with pytest.raises(RuntimeError, match="Refusing"):
            module._discard_successful_runtime(child)
        assert runtime.exists() and marker.read_text() == "keep"


@pytest.mark.parametrize("cleanup_error", [None, OSError, RuntimeError])
@pytest.mark.parametrize(
    "status,error,cancel_signal",
    [
        (0, None, 0),
        (1, None, 0),
        (134, None, 0),
        (2, RuntimeError, 0),
        (2, OSError, 0),
        (130, InterruptedError, signal.SIGINT),
        (143, InterruptedError, signal.SIGTERM),
        (129, InterruptedError, signal.SIGHUP),
    ],
)
def test_main_preserves_failure_and_signal_status(
    smoke_modules, monkeypatch, tmp_path, capsys, status, error, cancel_signal, cleanup_error
):
    """Cleanup failure cannot make a workload failure, abort or signal successful."""
    module = smoke_modules.controller
    instance = Mock(cancel_signal=cancel_signal)
    instance.run.return_value = status
    if error:
        instance.run.side_effect = error("injected workload failure")
    if cleanup_error:
        instance.close.side_effect = cleanup_error("injected cleanup failure")
    monkeypatch.setattr(module, "enable_subreaper", Mock())
    monkeypatch.setattr(module, "_Controller", Mock(return_value=instance))
    monkeypatch.setattr(sys, "argv", ["smoke", "core", "--num-npus", "8", "--log-dir", str(tmp_path / "logs")])
    assert module.main() == (2 if status == 0 and cleanup_error is not None else status)
    instance.close.assert_called_once_with()
    assert ("[FAIL] cleanup:" in capsys.readouterr().err) is (cleanup_error is not None)


@pytest.mark.parametrize("error", [OSError, RuntimeError, InterruptedError])
def test_main_reports_incomplete_construction(smoke_modules, monkeypatch, tmp_path, error):
    """An absent controller must not hide the original construction error."""
    module = smoke_modules.controller
    monkeypatch.setattr(module, "enable_subreaper", Mock())
    monkeypatch.setattr(module, "_Controller", Mock(side_effect=error("injected construction failure")))
    monkeypatch.setattr(sys, "argv", ["smoke", "core", "--num-npus", "8", "--log-dir", str(tmp_path / "logs")])
    assert module.main() == 2
