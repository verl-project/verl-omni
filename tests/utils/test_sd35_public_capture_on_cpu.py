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

"""Portable contract tests for the public SD3.5 capture recipe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/flowgrpo_trainer/sd35/consistency"


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, RECIPE / f"{name}.py")
    assert spec and spec.loader
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_cli_help_without_training_imports():
    for name in ("capture", "source_manifest", "pin_producer"):
        script = RECIPE / f"{name}.py"
        # -S avoids site packages; --help must terminate before training imports.
        done = subprocess.run(
            [sys.executable, "-S", str(script), "--help"], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr
        assert "usage:" in done.stdout


def test_debugger_classmethod_admission_is_signature_only(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))
    capture = module("capture")

    class Compatible:
        def __init__(self, *, config_path):
            raise AssertionError("constructor must not run")

        @classmethod
        def start(cls, model=None):
            raise AssertionError("start must not run")

        @classmethod
        def stop(cls):
            raise AssertionError("stop must not run")

        @classmethod
        def step(cls):
            raise AssertionError("step must not run")

    capture._validate_debugger_api(Compatible)

    class Incompatible(Compatible):
        @classmethod
        def start(cls, required_input):
            raise AssertionError("start must not run")

    with pytest.raises(TypeError):
        capture._validate_debugger_api(Incompatible)


def test_source_manifest_bytes_and_no_overwrite(tmp_path, monkeypatch):
    inventory = module("source_manifest")
    source = tmp_path / "src"
    source.mkdir()
    monkeypatch.setattr(inventory, "OWNERS", ("owner.py",))
    (source / "owner.py").write_bytes(b"old\n")
    target = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["source_manifest", "--source", str(source), "--head", "a" * 40, "--output", str(target)]
    )
    inventory.main()
    manifest = json.loads(target.read_text())
    assert manifest == {
        "head": "a" * 40,
        "files": [{"path": "owner.py", "sha256": hashlib.sha256(b"old\n").hexdigest(), "size": 4}],
    }
    with pytest.raises(ValueError, match="fresh"):
        inventory.main()
    (source / "owner.py").write_bytes(b"new\n")
    # The capture side rejects changed owner bytes before training imports.
    sys.path.insert(0, str(RECIPE))
    try:
        import importlib

        capture = importlib.import_module("capture")
        monkeypatch.setattr(capture, "OWNERS", ("owner.py",))
        with pytest.raises(ValueError, match="byte mismatch"):
            capture.validate_source_manifest(source, target, hashlib.sha256(target.read_bytes()).hexdigest(), "a" * 40)
    finally:
        sys.path.remove(str(RECIPE))
        sys.modules.pop("capture", None)


def test_source_manifest_rejects_escaped_owner(tmp_path, monkeypatch):
    inventory = module("source_manifest")
    source = tmp_path / "src"
    source.mkdir()
    external = tmp_path / "external.py"
    external.write_text("x")
    (source / "alias.py").symlink_to(external)
    monkeypatch.setattr(inventory, "OWNERS", ("alias.py",))
    monkeypatch.setattr(
        sys,
        "argv",
        ["source_manifest", "--source", str(source), "--head", "a" * 40, "--output", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="escaped"):
        inventory.main()


def test_pins_exact_eight_and_no_overwrite(tmp_path, monkeypatch, capsys):
    pins = module("pin_producer")
    root = tmp_path / "run"
    names = ["rollout/manifest.json", "rollout/actor_batch.pt", "actor/comparison.json", "actor/comparison.pt"]
    for role in ("rollout", "actor"):
        prefix = f"{role}/captures/{role}/id"
        names += [f"{prefix}/record.json", f"{prefix}/dump/step0/rank0/dump.json"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    out = tmp_path / "pins.sha256"
    monkeypatch.setattr(sys, "argv", ["pin_producer", "--root", str(root), "--output", str(out)])
    pins.main()
    data = out.read_bytes()
    assert len(data.splitlines()) == 8
    assert data.decode().splitlines() == [f"{hashlib.sha256(name.encode()).hexdigest()}  {name}" for name in names]
    assert hashlib.sha256(data).hexdigest() in capsys.readouterr().out
    with pytest.raises(ValueError, match="fresh"):
        pins.main()
    monkeypatch.setattr(sys, "argv", ["pin_producer", "--root", str(root), "--output", str(root / "internal-pin")])
    with pytest.raises(ValueError, match="outside"):
        pins.main()


def test_admission_token_missing_and_stale(tmp_path, monkeypatch):
    sys.path.insert(0, str(RECIPE))
    try:
        import importlib

        capture = importlib.import_module("capture")
        args = type("Args", (), {"role": "rollout", "rollout_offload": "none"})()
        identity = {"owner": "byte-1"}
        monkeypatch.setattr(
            capture,
            "provenance",
            lambda args, *_: {
                "source": "byte-1",
                "transformer_weight_hash_verified": True,
                "settings": {"rollout_offload": args.rollout_offload},
            },
        )
        with pytest.raises(FileNotFoundError):
            capture.validate_admission(args, identity, {}, tmp_path)
        path = tmp_path / "admit/admission.json"
        path.parent.mkdir()
        payload = {
            "role": "admit",
            "identity": identity,
            "provenance": {
                "source": "byte-1",
                "transformer_weight_hash_verified": False,
                "settings": {"rollout_offload": "none"},
            },
            "cuda_initialized": False,
            "scheduler_timesteps": [1, 2, 3, 4],
            "rollout_transformer_config": {"num_layers": 1},
        }
        path.write_text(json.dumps(payload))
        assert capture.validate_admission(args, identity, {}, tmp_path) == hashlib.sha256(path.read_bytes()).hexdigest()
        args.rollout_offload = "dit-layerwise"
        with pytest.raises(ValueError, match="token"):
            capture.validate_admission(args, identity, {}, tmp_path)
        args.rollout_offload = "none"
        payload["identity"] = {"owner": "stale"}
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="token"):
            capture.validate_admission(args, identity, {}, tmp_path)
    finally:
        sys.path.remove(str(RECIPE))
        sys.modules.pop("capture", None)


def test_admit_shell_masks_cuda_visibility():
    source = (RECIPE / "run.sh").read_text()
    assert 'CUDA_VISIBLE_DEVICES=\'\' "$PYTHON" "$recipe/capture.py" --role admit' in source


def test_rollout_offload_config_and_invalid_mode():
    capture = module("capture")
    assert capture.rollout_offload_kwargs("none") == {}
    assert capture.rollout_offload_kwargs("dit-layerwise") == {
        "diffusion_offload_config": {"mode": "layer", "components": ["dit"]}
    }
    with pytest.raises(ValueError, match="unsupported rollout offload"):
        capture.rollout_offload_kwargs("other")
    done = subprocess.run(
        [sys.executable, "-S", str(RECIPE / "capture.py"), "--rollout-offload", "other"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert "invalid choice" in done.stderr


@pytest.mark.parametrize("setting", [None, "dit-layerwise"])
def test_launcher_forwards_rollout_offload(tmp_path, setting):
    launcher = RECIPE / "run.sh"
    fake_python = tmp_path / "python"
    log = tmp_path / "calls.jsonl"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    env = {
        "SOURCE_ROOT": str(ROOT),
        "PYTHON": str(fake_python),
        "MODEL_DIR": str(tmp_path / "model"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "RUN_ID": "test-run",
        "SOURCE_MANIFEST": str(tmp_path / "manifest"),
        "SOURCE_MANIFEST_SHA256": "a" * 64,
        "SOURCE_HEAD": "b" * 40,
        "PIN_FILE": str(tmp_path / "pin"),
        "CALL_LOG": str(log),
    }
    if setting is not None:
        env["SD35_ROLLOUT_OFFLOAD"] = setting
    import os

    launch_env = {**os.environ, **env}
    if setting is None:
        launch_env.pop("SD35_ROLLOUT_OFFLOAD", None)
    for mode in ("admit", "capture"):
        done = subprocess.run(["bash", str(launcher), mode], env=launch_env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[1:3] for call in calls[:3]] == [["--role", role] for role in ("admit", "rollout", "actor")]
    assert all(call[call.index("--rollout-offload") + 1] == (setting or "none") for call in calls[:3])


@pytest.mark.parametrize("script", ["source_manifest", "pin_producer"])
def test_exclusive_output_survives_creation_race(tmp_path, monkeypatch, script):
    from unittest.mock import patch

    owner = module(script)
    output = tmp_path / "output"
    original_open = Path.open

    def raced_open(path, mode="r", *args, **kwargs):
        if path == output and mode == "xb":
            output.write_bytes(b"other writer")
        return original_open(path, mode, *args, **kwargs)

    if script == "source_manifest":
        source = tmp_path / "src"
        source.mkdir()
        (source / "owner.py").write_text("x")
        monkeypatch.setattr(owner, "OWNERS", ("owner.py",))
        argv = [script, "--source", str(source), "--head", "a" * 40, "--output", str(output)]
    else:
        root = tmp_path / "run"
        names = ["rollout/manifest.json", "rollout/actor_batch.pt", "actor/comparison.json", "actor/comparison.pt"]
        for role in ("rollout", "actor"):
            prefix = f"{role}/captures/{role}/id"
            names += [f"{prefix}/record.json", f"{prefix}/dump/step0/rank0/dump.json"]
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        argv = [script, "--root", str(root), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(Path, "open", raced_open), pytest.raises(FileExistsError):
        owner.main()
    assert output.read_bytes() == b"other writer"


@pytest.mark.parametrize("actor_status,pinned", [(0, True), (10, True), (1, False)])
def test_capture_shell_preserves_mismatch_and_skips_incomplete(tmp_path, actor_status, pinned):
    source = tmp_path / "source"
    source.mkdir()
    fake = tmp_path / "fake-python"
    fake.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
name = Path(sys.argv[1]).name
if name == 'capture.py':
    role = sys.argv[sys.argv.index('--role') + 1]
    if role == 'actor':
        sys.exit(int(os.environ['FAKE_ACTOR_STATUS']))
elif name == 'pin_producer.py':
    Path(sys.argv[sys.argv.index('--output') + 1]).write_text('eight pins')
""")
    fake.chmod(0o755)
    pin = tmp_path / "pins.sha256"
    env = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "SOURCE_ROOT": str(source),
        "PYTHON": str(fake),
        "MODEL_DIR": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path),
        "RUN_ID": "run",
        "SOURCE_MANIFEST": str(tmp_path / "manifest"),
        "SOURCE_MANIFEST_SHA256": "a" * 64,
        "SOURCE_HEAD": "b" * 40,
        "PIN_FILE": str(pin),
        "FAKE_ACTOR_STATUS": str(actor_status),
    }
    result = subprocess.run(
        ["bash", str(RECIPE / "run.sh"), "capture"], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == actor_status
    assert pin.exists() is pinned
