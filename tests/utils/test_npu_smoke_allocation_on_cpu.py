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
"""Check physical allocation metadata and runtime masks without starting workers."""

import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "tests" / "npu_smoke"
_RUNTIME_IDS = tuple(str(device) for device in range(8))


@pytest.fixture
def parallel_smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(_SCRIPT_ROOT))
    return importlib.import_module("run_npu_smoke_parallel")


@pytest.mark.parametrize("offset", range(8), ids=["ascending", "core-ci-order", "e2e-ci-order", *range(3, 8)])
def test_physical_allocation_preserves_every_rotation(monkeypatch, parallel_smoke, offset):
    devices = _RUNTIME_IDS[offset:] + _RUNTIME_IDS[:offset]
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", ",".join(devices))

    assert parallel_smoke._device_list("ASCEND_VISIBLE_DEVICES", ",".join(_RUNTIME_IDS)) == list(devices)


@pytest.mark.parametrize(
    "devices",
    [
        tuple(str(device) for device in range(8, 16)),
        tuple(str(device) for device in range(101, 110)),
        tuple(str(device) for device in range(15, -1, -1)),
    ],
)
def test_physical_allocation_keeps_offset_or_larger_sets(monkeypatch, parallel_smoke, devices):
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", ",".join(devices))

    assert parallel_smoke._device_list("ASCEND_VISIBLE_DEVICES", ",".join(_RUNTIME_IDS)) == list(devices)


@pytest.mark.parametrize("override", [None, ""])
def test_absent_physical_allocation_uses_supplied_default(monkeypatch, parallel_smoke, override):
    if override is None:
        monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", override)
    default = "101,102,103,104,105,106,107,108"

    assert parallel_smoke._device_list("ASCEND_VISIBLE_DEVICES", default) == default.split(",")


@pytest.mark.parametrize(
    "value",
    [
        "0,1,2,3,4,5,6",
        "0,1,2,3,4,5,6,6",
        "0,1,2,3,4,5,6,00",
        "0,1,2,3,4,5,6,a",
        "-1,0,1,2,3,4,5,6",
        "0,1,2,3,4,5,6, 7",
        "0,1,2,3,4,5,6,7,",
    ],
)
def test_invalid_physical_allocations_still_fail(monkeypatch, parallel_smoke, value):
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", value)

    with pytest.raises(ValueError, match="ASCEND_VISIBLE_DEVICES"):
        parallel_smoke._device_list("ASCEND_VISIBLE_DEVICES", ",".join(_RUNTIME_IDS))


@pytest.mark.parametrize("offset", range(1, 8))
def test_runtime_masks_still_reject_nonascending_order(monkeypatch, parallel_smoke, offset):
    devices = _RUNTIME_IDS[offset:] + _RUNTIME_IDS[:offset]
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(devices))

    with pytest.raises(ValueError, match="ASCEND_RT_VISIBLE_DEVICES.*ascending"):
        parallel_smoke._device_list("ASCEND_RT_VISIBLE_DEVICES", ",".join(_RUNTIME_IDS))


@pytest.mark.parametrize("count", [7, 9])
def test_runtime_masks_require_exactly_eight_devices(monkeypatch, parallel_smoke, count):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(str(device) for device in range(count)))

    with pytest.raises(ValueError, match="ASCEND_RT_VISIBLE_DEVICES"):
        parallel_smoke._device_list("ASCEND_RT_VISIBLE_DEVICES", ",".join(_RUNTIME_IDS))


@pytest.mark.parametrize("first", [0, 8])
def test_ascending_runtime_masks_are_preserved(monkeypatch, parallel_smoke, first):
    devices = [str(device) for device in range(first, first + 8)]
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(devices))

    assert parallel_smoke._device_list("ASCEND_RT_VISIBLE_DEVICES", ",".join(_RUNTIME_IDS)) == devices


@pytest.mark.parametrize(
    "physical",
    ["1,2,3,4,5,6,7,0", "2,3,4,5,6,7,0,1", "101,102,103,104,105,106,107,108,109"],
)
@pytest.mark.parametrize(
    "group,name,count,offset,ids",
    [
        ("core", "exclusive", 8, 0, (0, 3, 13, 14, 28)),
        ("core", "lane0", 4, 0, (7, 11, 12)),
        ("core", "lane1", 4, 4, (4, 5, 6, 15, 17, 30)),
        ("e2e", "exclusive", 8, 0, (1,)),
        ("e2e", "lane0", 2, 0, (18, 20, 23)),
        ("e2e", "lane1", 2, 2, (19, 21, 25, 34)),
        ("e2e", "lane2", 4, 4, (2, 22, 24, 26)),
    ],
)
def test_lane_launch_slices_only_runtime_mask(
    monkeypatch, parallel_smoke, tmp_path, physical, group, name, count, offset, ids
):
    class LaunchCaptured(Exception):
        pass

    captured = {}

    def capture_launch(command, *, env, **kwargs):
        captured.update(command=command, env=env)
        raise LaunchCaptured

    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", physical)
    monkeypatch.setenv("NPU_VISIBLE_DEVICES", physical)
    monkeypatch.setattr(parallel_smoke.subprocess, "Popen", capture_launch)
    controller = object.__new__(parallel_smoke._Controller)
    controller.group = group
    controller.log_dir = tmp_path
    controller.devices = (list(_RUNTIME_IDS), physical.split(","))
    controller.cancel_signal = 0
    controller.identity = SimpleNamespace(pid=4242, start=12345)
    (tmp_path / ".processes").mkdir()
    try:
        with pytest.raises(LaunchCaptured):
            controller._start(name, count, ids, offset, 20000)
    finally:
        runtime_file = tmp_path / ".processes" / name / "runtime.json"
        if runtime_file.exists():
            shutil.rmtree(json.loads(runtime_file.read_text())["runtime"])

    env = captured["env"]
    assert env["NUM_NPUS"] == str(count)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(_RUNTIME_IDS[offset : offset + count])
    assert env["ASCEND_VISIBLE_DEVICES"] == physical
    assert env["NPU_VISIBLE_DEVICES"] == physical
    assert env["NPU_SMOKE_TEST_IDS"] == ",".join(str(test_id) for test_id in ids)
