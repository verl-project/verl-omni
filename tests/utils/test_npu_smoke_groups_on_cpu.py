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
"""Exercise group overlap and failure propagation with command-boundary stubs."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def shell_runtime(tmp_path):
    """Copy the launchers and replace only device/process commands."""
    root = tmp_path / "repo"
    scripts = root / "tests/npu_smoke"
    shutil.copytree(REPO_ROOT / "tests/npu_smoke", scripts)
    bins = tmp_path / "bin"
    bins.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    def executable(name, text):
        path = bins / name
        path.write_text(text)
        path.chmod(0o755)

    executable(
        "setsid", f"#!{sys.executable}\nimport os,sys\nos.setsid()\nos.execvpe(sys.argv[1],sys.argv[1:],os.environ)\n"
    )
    executable("sleep", "#!/bin/sh\nexit 0\n")
    for name in ("ray", "pkill", "npu-smi"):
        executable(
            name,
            f"#!{sys.executable}\n"
            "import os,pathlib,sys\n"
            "p=pathlib.Path(os.environ['EVIDENCE'])\n"
            "active=list(p.glob('*.active'))\n"
            f"with (p/'cleanup.jsonl').open('a') as f: f.write(str(({name!r},[x.name for x in active]))+'\\n')\n",
        )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NPU_SMOKE_", "ASCEND_", "RAY_", "MODEL_", "TOKENIZER_", "DATA_", "TRAIN_", "VAL_"))
    }
    env.update(
        PATH=f"{bins}:{os.environ['PATH']}",
        LD_PRELOAD="/fixture/libjemalloc.so",
        LOG_DIR=str(tmp_path / "logs"),
        EVIDENCE=str(evidence),
    )
    return root, scripts, evidence, env


@pytest.mark.parametrize("core_exit,e2e_exit", [(0, 0), (134, 0), (0, 134), (7, 9)])
@pytest.mark.parametrize("shard", [None, "0", "1"])
def test_parallel_groups_overlap_and_preserve_failures(shell_runtime, core_exit, e2e_exit, shard):
    """Both groups start together and a failed child cannot become a suite pass."""
    _, scripts, evidence, env = shell_runtime
    for group, rc in (("core", core_exit), ("e2e", e2e_exit)):
        (scripts / f"run_npu_smoke_{group}.sh").write_text(
            f"exec {sys.executable} - <<'PY'\n"
            "import json,os,pathlib,time\n"
            "p=pathlib.Path(os.environ['EVIDENCE'])\n"
            f"group={group!r}\n"
            "(p/(group+'.active')).touch()\n"
            "(p/(group+'.started')).touch()\n"
            "deadline=time.monotonic()+5\n"
            "while len(list(p.glob('*.started'))) != 2:\n"
            "    assert time.monotonic()<deadline, 'groups did not overlap'\n"
            "    time.sleep(.02)\n"
            "(p/(group+'.json')).write_text(json.dumps(dict(os.environ)))\n"
            "(p/(group+'.active')).unlink()\n"
            f"raise SystemExit({rc})\nPY\n"
        )
    env["NPU_SMOKE_PARALLEL"] = "1"
    if shard is not None:
        env["NPU_SMOKE_SHARD_INDEX"] = shard
    env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    env["RAY_ADDRESS"] = "ray://unrelated-cluster:10001"
    result = subprocess.run(
        ["bash", str(scripts / "run_npu_smoke_tests.sh"), "--num-npus", "8"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == int(bool(core_exit or e2e_exit)), result.stdout + result.stderr
    values = {name: json.loads((evidence / f"{name}.json").read_text()) for name in ("core", "e2e")}
    assert values["core"]["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
    assert values["e2e"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
    assert values["core"]["NUM_NPUS"] == "4" and values["e2e"]["NUM_NPUS"] == "4"
    assert values["core"]["RAY_TMPDIR"] != values["e2e"]["RAY_TMPDIR"]
    assert values["core"]["RAY_MASTER_PORT_RANGE"] != values["e2e"]["RAY_MASTER_PORT_RANGE"]
    for value in values.values():
        assert value.get("NPU_SMOKE_SHARD_INDEX") == shard
        assert value["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] == ""
        assert value["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == ""
        assert value["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"
        assert value["NPU_SMOKE_SKIP_GLOBAL_CLEANUP"] == "1"
        assert "VERL_ZMQ_DIR" not in value
        assert "RAY_ADDRESS" not in value
    assert all(line.endswith(", [])") for line in (evidence / "cleanup.jsonl").read_text().splitlines())
    assert (Path(env["LOG_DIR"]) / "group-exits.tsv").read_text() == f"core\t{core_exit}\ne2e\t{e2e_exit}\n"


@pytest.mark.parametrize(
    "overrides",
    [
        {"NUM_NPUS": "4"},
        {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"},
        {"MODEL_PATH": "/model"},
        {"TOKENIZER_PATH": "/tokenizer"},
        {"DATA_DIR": "/data"},
        {"TRAIN_FILES": "/train.parquet"},
        {"VAL_FILES": "/test.parquet"},
        {"NPU_SMOKE_TEST_IDS": "0"},
        {"NPU_SMOKE_SKIP_GLOBAL_CLEANUP": "1"},
    ],
)
def test_invalid_parallel_plan_fails_without_cleanup(shell_runtime, overrides):
    """Invalid allocation or shared model overrides cannot stop existing jobs."""
    _, scripts, evidence, env = shell_runtime
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(scripts / "run_npu_smoke_parallel.sh")], env=env, capture_output=True, timeout=10
    )
    assert result.returncode == 2
    assert not (evidence / "cleanup.jsonl").exists()


def test_default_log_directories_do_not_reuse_previous_results(shell_runtime):
    """Repeated launches keep separate logs even within the same timestamp."""
    root, scripts, _, env = shell_runtime
    env.pop("LOG_DIR")
    env.update(NPU_SMOKE_GROUP="fixture", NPU_SMOKE_SKIP_GLOBAL_CLEANUP="1")
    date = Path(env["PATH"].split(os.pathsep)[0]) / "date"
    date.write_text("#!/bin/sh\nprintf '20261109_120000\\n'\n")
    date.chmod(0o755)
    paths = []
    for _ in range(2):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'lib=$1; shift; source "$lib"; printf "log_dir=%s\\n" "$LOG_DIR"',
                "fixture",
                str(scripts / "lib_npu_smoke.sh"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        paths.append(Path(result.stdout.splitlines()[-1].removeprefix("log_dir=")))
    assert paths[0] != paths[1]
    assert all(path.is_dir() and path.parent == root / "logs/npu_smoke/fixture" for path in paths)


@pytest.mark.parametrize("rc,tee_rc", [(0, 0), (134, 0), (0, 3), (134, 3)])
def test_workload_and_log_failures_remain_failures(shell_runtime, rc, tee_rc):
    """A completed pytest banner never masks aborts or a failed log writer."""
    _, scripts, evidence, env = shell_runtime
    bins = Path(env["PATH"].split(os.pathsep)[0])
    if tee_rc:
        tee = bins / "tee"
        tee.write_text(f"#!/bin/sh\ncat >/dev/null\nexit {tee_rc}\n")
        tee.chmod(0o755)
    env.update(NPU_SMOKE_GROUP="fixture", NPU_SMOKE_SKIP_GLOBAL_CLEANUP="1")
    command = (
        'lib=$1; child_rc=$2; set --; source "$lib"; '
        'run_test 3 fixture bash -c \'echo "1 passed"; exit "$1"\' bash "$child_rc"; '
        'printf "%s" "${TEST_RESULTS[0]}"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "fixture", str(scripts / "lib_npu_smoke.sh"), str(rc)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("FAIL" if rc or tee_rc else "PASS")
    assert not (evidence / "cleanup.jsonl").exists()


@pytest.mark.parametrize("value", [None, "", "1"])
def test_shared_launcher_preserves_explicit_empty_noset(shell_runtime, value):
    """Ray-managed child workers keep the empty value; serial defaults stay 1."""
    _, scripts, _, env = shell_runtime
    env.update(NPU_SMOKE_GROUP="fixture", NPU_SMOKE_SKIP_GLOBAL_CLEANUP="1")
    if value is not None:
        env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = value
    result = subprocess.run(
        [
            "bash",
            "-c",
            'lib=$1; shift; source "$lib"; printf "value=%s\\n" "$RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"',
            "fixture",
            str(scripts / "lib_npu_smoke.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith(f"value={'1' if value is None else value}")


@pytest.mark.parametrize("script", ["run_flowgrpo_qwen_image_npu.sh", "run_dancegrpo_wan22_npu.sh"])
@pytest.mark.parametrize("value,raw_exit", [(None, 0), ("", 0), ("", 134)])
def test_trainer_launchers_preserve_device_policy_paths_and_exit(shell_runtime, script, value, raw_exit):
    """Real shell launchers pass empty NOSET values, quoted paths and trainer status."""
    root, _, evidence, env = shell_runtime
    bins = Path(env["PATH"].split(os.pathsep)[0])
    python = bins / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "if sys.argv[1:3]==['-m','verl_omni.trainer.main_diffusion']:\n"
        "    p=pathlib.Path(os.environ['EVIDENCE'])/'trainer.json'\n"
        "    noset=os.environ.get('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES')\n"
        "    p.write_text(json.dumps(dict(argv=sys.argv[3:],noset=noset)))\n"
        f"    raise SystemExit({raw_exit})\n"
    )
    python.chmod(0o755)
    data = root / "data with spaces"
    data.mkdir()
    (data / "train.parquet").touch()
    (data / "test.parquet").touch()
    env.update(NUM_NPUS="6", DATA_DIR=str(data), MODEL_PATH=str(root / "model with spaces"))
    if value is not None:
        env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = value
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source() { return 0; }; export -f source; exec bash "$1"',
            "fixture",
            str(REPO_ROOT / "tests/special_e2e" / script),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == raw_exit, result.stdout + result.stderr
    captured = json.loads((evidence / "trainer.json").read_text())
    assert captured["noset"] == ("1" if value is None else value)
    assert f"actor_rollout_ref.model.path={env['MODEL_PATH']}" in captured["argv"]
    assert f"data.train_files={data}/train.parquet" in captured["argv"]


@pytest.mark.skipif(sys.platform != "linux", reason="Real group runners require Linux Bash 4+")
@pytest.mark.parametrize("group", ["core", "e2e"])
def test_two_shards_execute_every_registered_case_exactly_once(shell_runtime, group):
    """Real group dispatch partitions IDs without losing or duplicating a workload."""
    _, scripts, _, env = shell_runtime
    bins = Path(env["PATH"].split(os.pathsep)[0])
    shell = shutil.which("bash")
    for name in ("bash", "pytest", "python3"):
        stub = bins / name
        stub.write_text("#!/bin/sh\necho 'stub workload completed'\n")
        stub.chmod(0o755)
    script = scripts / f"run_npu_smoke_{group}.sh"
    registered = {int(value) for value in re.findall(r"^run_test (\d+) ", script.read_text(), re.M)}
    selections = []
    for index in (0, 1):
        log_dir = scripts.parent / f"shard-{index}"
        env.update(NPU_SMOKE_SHARD_INDEX=str(index), LOG_DIR=str(log_dir))
        result = subprocess.run(
            [shell, str(script), "--num-npus", "8"], env=env, capture_output=True, text=True, timeout=20
        )
        assert result.returncode == 0, result.stdout + result.stderr
        rows = re.findall(r"^(\d+)\s+(PASS|FAIL)\s+", (log_dir / "summary.log").read_text(), re.M)
        assert rows and all(status == "PASS" for _, status in rows)
        selections.append({int(case_id) for case_id, _ in rows})
    assert not selections[0] & selections[1]
    assert selections[0] | selections[1] == registered


@pytest.mark.parametrize("shard", ["-1", "2", "bad", "00"])
def test_invalid_shard_never_cleans_or_launches_devices(shell_runtime, shard):
    """Reject malformed selection before any process cleanup can run."""
    _, scripts, evidence, env = shell_runtime
    env["NPU_SMOKE_SHARD_INDEX"] = shard
    result = subprocess.run(["bash", str(scripts / "run_npu_smoke_core.sh")], env=env, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert not (evidence / "cleanup.jsonl").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Real group runners require Linux Bash 4+")
def test_empty_shard_is_not_reported_as_a_pass(shell_runtime):
    """A selection with no executed tests must fail closed."""
    _, scripts, _, env = shell_runtime
    env.update(NPU_SMOKE_GROUP="fixture", NPU_SMOKE_SHARD_INDEX="0", NPU_SMOKE_SKIP_GLOBAL_CLEANUP="1")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'lib=$1; set --; source "$lib"; run_test 1 unselected false; print_summary',
            "fixture",
            str(scripts / "lib_npu_smoke.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Passed: 0" in result.stdout


@pytest.mark.parametrize(
    "event,labels,groups,jobs",
    [
        ("push", [], ["core", "e2e"], ["shard-0", "shard-1"]),
        ("pull_request", ["ci-npu"], ["core", "e2e"], ["shard-0", "shard-1"]),
        ("pull_request", ["ready-for-ci"], ["core", "e2e"], ["shard-0", "shard-1"]),
        ("pull_request", ["ci-npu-core"], ["core"], ["core"]),
        ("pull_request", ["ci-npu-e2e"], ["e2e"], ["e2e"]),
        ("pull_request", ["ci-npu-core", "ci-npu-e2e"], ["core", "e2e"], ["shard-0", "shard-1"]),
        ("pull_request", ["unrelated"], [], []),
    ],
)
def test_workflow_label_resolution_retains_single_group_runs(tmp_path, event, labels, groups, jobs):
    """Execute the workflow's actual resolver, including label unions and rejected inputs."""
    import yaml

    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/npu_smoke.yml").read_text())
    # Companion model-preparation steps use the stable matrix.group interface.
    assert workflow["jobs"]["npu_smoke"]["strategy"]["matrix"] == {
        "group": "${{ fromJSON(needs.resolve_groups.outputs.jobs) }}"
    }
    step = workflow["jobs"]["resolve_groups"]["steps"][0]
    output = tmp_path / "github-output"
    env = dict(os.environ, EVENT_NAME=event, PR_LABELS=json.dumps(labels), GITHUB_OUTPUT=str(output))
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == (0 if groups else 1), result.stderr
    if groups:
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert json.loads(values["groups"]) == groups
        assert json.loads(values["jobs"]) == jobs
