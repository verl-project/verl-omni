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

"""Exercise recipe hygiene and prove shell inspection cannot run recipe code."""

from pathlib import Path

import pytest
from check_example_recipes import findings

VALID_HEADER = "#!/usr/bin/env bash\nset -euo pipefail\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('model = "/home/alice/weights"', ["VO014"]),
        ('model = "/Users/alice/weights"', ["VO014"]),
        ('model = "/nfs/alice/weights"', ["VO014"]),
        ('model = "/mnt/weights"', ["VO014"]),
        ('model = "/root/weights"', ["VO014"]),
        ('model = f"/home/{user}/weights"', ["VO014"]),
        ('model = f"{workspace}/home/weights"', []),
        ('model = Path("/home/alice/weights")', ["VO014"]),
        ('model = """/home/alice/weights"""', ["VO014"]),
        ('model = "/run/root/weights"', []),
        ('model = "/homegrown/weights"', []),
        ('model = "org/model"', []),
        ("model = config.model_path", []),
        ('# model = "/home/alice/weights"', []),
        ('"""Example: /home/alice/weights"""\nmodel = config.model_path', []),
        ('"""/home/alice/weights"""\nmodel = config.model_path', []),
        ('def documented():\n    """/home/alice/weights"""\n    return None', []),
        ("fixture = 'model = \"/home/alice/weights\"'", []),
    ],
)
def test_python_path_literals(source: str, expected: list[str]) -> None:
    result = findings("tests/workers/test_model_on_cpu.py", source.encode())
    assert [code for _line, code, _message in result] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('MODEL="/home/alice/weights"\npython trainer.py "$@"', ["VO014"]),
        ('MODEL=/mnt/weights\npython trainer.py "$@"', ["VO014"]),
        ('MODEL=${MODEL:-/nfs/alice/weights}\npython trainer.py "$@"', ["VO014"]),
        ('python trainer.py "model.path=/root/weights" "$@"', ["VO014"]),
        ('# MODEL=/home/alice/weights\npython trainer.py "$@"', []),
        ('MODEL="org/model"\npython trainer.py "$@"', []),
        ('python trainer.py "$@"', []),
        ('python trainer.py "${@}"', []),
        ('python trainer.py \\\n    "$@"', []),
        ('python trainer.py "$@" 2>&1 | tee "$LOG_FILE"', []),
        ('bash "$BASE_SCRIPT" "$@"', []),
        ('exec python trainer.py "$@"', []),
        ('args=(\n    "key=value"\n    "$@"\n)\npython trainer.py "${args[@]}"', []),
        ('args=("$@")\npython trainer.py', ["EX002"]),
        ('args=("$@")\necho "${args[@]}"\npython trainer.py', ["EX002"]),
        ('# python trainer.py "$@"\npython trainer.py', ["EX002"]),
        ("echo '\"$@\"'\npython trainer.py", ["EX002"]),
        ('echo "$@"\npython trainer.py', ["EX002"]),
        ('if true; then echo "$@"; fi\npython trainer.py', ["EX002"]),
        ('if true; then printf "%s\\n" "$@"; fi\npython trainer.py', ["EX002"]),
        ('env echo "$@"\npython trainer.py', ["EX002"]),
        ('if true; then python trainer.py "$@"; fi', []),
        ('printf "%s" "$@"\npython trainer.py', ["EX002"]),
        ("python trainer.py '\"$@\"'", ["EX002"]),
        ("python trainer.py $@", ["EX002"]),
        ('python trainer.py "$*"', ["EX002"]),
        ('python trainer.py "\\$@"', ["EX002"]),
        ("cat <<'EOF'\npython trainer.py \"$@\"\nEOF\npython trainer.py", ["EX002"]),
        ('cat <<-"EOF"\n\tpython trainer.py "$@"\n\tEOF\npython trainer.py', ["EX002"]),
        ('cat <<EOF\n/home/alice/weights\nEOF\npython trainer.py "$@"', []),
        ('read -r -a ARGS <<< "$INPUT"\npython trainer.py "$@"', []),
        ('if true; then\npython trainer.py "$@"', ["EX001"]),
    ],
)
def test_shell_recipe_checks(source: str, expected: list[str]) -> None:
    assert [
        code for _line, code, _message in findings("examples/model/run_train.sh", (VALID_HEADER + source).encode())
    ] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('model_path: "/home/alice/weights"', ["VO014"]),
        ("model_path: /nfs/alice/weights", ["VO014"]),
        ("paths:\n  - /root/weights", ["VO014"]),
        ("model_path: /homegrown/weights", []),
        ("model_path: org/model # /home/alice/weights", []),
        ("# model_path: /home/alice/weights", []),
        ('model_path: "org/model"', []),
        ('description: "Store weights at /home/alice/weights"', []),
    ],
)
def test_yaml_path_literals(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings("examples/model/config.yaml", source.encode())] == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("examples/model/reward.PNG", ["EX003"]),
        ("examples/model/video.mp4", ["EX003"]),
        ("examples/model/audio.wav", ["EX003"]),
        ("docs/model/reward.png", []),
        ("examples/model/README_training.md", ["EX004"]),
        ("examples/model/README.md", []),
        ("docs/model/README_training.md", []),
    ],
)
def test_example_path_checks_do_not_decode_media(path: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, b"\xff\x00")] == expected


def test_python_path_checks_respect_scope() -> None:
    assert not list(findings("docs/example.py", b'model = "/home/alice/weights"'))


@pytest.mark.parametrize("path", ["examples/model/run_train.sh", "examples/model/config.yaml"])
def test_invalid_example_text_has_a_diagnostic(path: str) -> None:
    assert [code for _line, code, _message in findings(path, b"\xff")] == ["EX001"]


def test_shell_helpers_need_valid_syntax_but_not_argument_forwarding() -> None:
    assert not list(findings("examples/model/download.sh", b'printf "%s" "org/model"'))
    assert [code for _line, code, _message in findings("examples/model/download.sh", b"if true; then")] == ["EX001"]


def test_shell_syntax_inspection_never_executes_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = tmp_path / "executed"
    startup = tmp_path / "startup.sh"
    startup.write_text(f'touch "{sentinel}"\n', encoding="utf-8")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("ENV", str(startup))
    source = f'{VALID_HEADER}touch "{sentinel}"\nRESULT=$(touch "{sentinel}")\npython trainer.py "$@"\n'
    assert not list(findings("examples/model/run_train.sh", source.encode()))
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("examples/model/run_train.sh", 'set -euo pipefail\npython trainer.py "$@"\n', ["EX005"]),
        ("examples/model/run_train.sh", '#!/bin/bash\nset -euo pipefail\npython trainer.py "$@"\n', ["EX005"]),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -x\npython trainer.py "$@"\n', ["EX005"]),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -euo pipefail\npython trainer.py "$@"\n', []),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -xeuo pipefail\npython trainer.py "$@"\n', []),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -ex\npython trainer.py "$@"\n', []),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -o errexit\npython trainer.py "$@"\n', []),
        ("examples/model/setup.sh", "echo ok\n", []),
    ],
)
def test_run_shell_recipes_require_bash_shebang_and_errexit(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, source.encode())] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("examples/dancegrpo_trainer/flux1/dataprocess/prepare_prompts.py", "x = 1\n", ["EX006"]),
        (
            "examples/dpo_trainer/qwen3_omni/qwen3_omni/run_demo.sh",
            f'{VALID_HEADER}python trainer.py "$@"\n',
            ["EX006"],
        ),
        ("examples/dpo_trainer/qwen3_omni/eval.sh", ".venv/bin/python eval.py\n", ["EX006"]),
        ("examples/dancegrpo_trainer/data_process/prepare_prompts.py", "x = 1\n", []),
        ("examples/dpo_trainer/qwen3_omni/run_demo.sh", f'{VALID_HEADER}python trainer.py "$@"\n', []),
        ("examples/dpo_trainer/qwen3_omni/eval.sh", "# .venv/bin/python\npython3 eval.py\n", []),
        ("examples/dpo_trainer/qwen3_omni/eval.sh", "cat <<EOF\n.venv/bin/python\nEOF\npython3 eval.py\n", []),
    ],
)
def test_example_layout_and_portable_python_hygiene(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, source.encode())] == expected
