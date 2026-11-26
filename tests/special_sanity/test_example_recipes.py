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
from check_example_recipes import delegates_to_base_script, findings

VALID_HEADER = "#!/usr/bin/env bash\nset -euo pipefail\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('model = "/home/alice/weights"', ["VO014"]),
        ('model = "/Users/alice/weights"', ["VO014"]),
        ('model = "/user/alice/weights"', ["VO014"]),
        ('model = "/users/alice/weights"', ["VO014"]),
        ('model = "/user"', []),
        ('model = "/user_data/weights"', []),
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
        ('MODEL="/user/alice/weights"\npython trainer.py "$@"', ["VO014"]),
        ('MODEL="/users/alice/weights"\npython trainer.py "$@"', ["VO014"]),
        ('MODEL="/user_data/weights"\npython trainer.py "$@"', []),
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
        ('python trainer.py \\ \n    "$@"', ["EX001"]),
        ('python trainer.py "$@" trainer.total_epochs=1', ["EX002"]),
        ('if true; then\npython trainer.py "$@"', ["EX001"]),
        ('args=(\n    data.dir=$(pwd)/out\n    "$@"\n)\npython trainer.py "${args[@]}"', []),
        ('args+=("$@")\npython trainer.py "${args[@]}"', []),
        ('for arg in "$@"; do echo "$arg"; done\npython trainer.py', ["EX002"]),
        ('X=$((1 << 2))\nWANDB_API_KEY="secret123456"\npython trainer.py "$@"', ["EX008"]),
        ('cat <<\\EOF\nhello\nEOF\nWANDB_API_KEY="secret123456"\npython trainer.py "$@"', ["EX008"]),
        ('cat <<- EOF\n\thello\n\tEOF\nWANDB_API_KEY="secret123456"\npython trainer.py "$@"', ["EX008"]),
        ('# comment ends with \\\nexec bash "$SCRIPT_DIR/run_base.sh" "$@"', []),
        ('echo \\\\\npkill -9 ray\npython trainer.py "$@"', ["EX008"]),
        ("awk '\nBEGIN { a.b=1 }\n' input.txt\npython trainer.py \"$@\"", []),
        ('echo "Example: MODEL=/home/alice/weights"\npython trainer.py "$@"', []),
        ('X=${X-/home/alice/weights}\npython trainer.py "$@"', ["VO014"]),
        ('X=${X:+/home/alice/weights}\npython trainer.py "$@"', ["VO014"]),
        ('PYTHONPATH="/opt:/home/alice/lib"\npython trainer.py "$@"', ["VO014"]),
        ('python trainer.py data.train_files=\'["/home/alice/a.parquet"]\' "$@"', ["VO014"]),
        ('python trainer.py data.train_files=\'["a","/user/bob/b.parquet"]\' "$@"', ["VO014"]),
        (
            'ARGS=(\n    MODEL="/user/alice/weights"\n    trainer.epochs=1\n    trainer.epochs=2\n)\n'
            'python trainer.py "$@"',
            ["VO014", "EX007"],
        ),
        ('ARGS=(\n    data.dir=$(pwd)/out\n    "$@"\n)\npython trainer.py "${ARGS[@]}"', []),
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
        ('model_path: "/user/alice/weights"', ["VO014"]),
        ('model_path: "/users/alice/weights"', ["VO014"]),
        ('model_path: "/user_data/weights"', []),
        ("model_path: /nfs/alice/weights", ["VO014"]),
        ("paths:\n  - /root/weights", ["VO014"]),
        ("model_path: /homegrown/weights", []),
        ("model_path: org/model # /home/alice/weights", []),
        ("# model_path: /home/alice/weights", []),
        ('model_path: "org/model"', []),
        ('description: "Store weights at /home/alice/weights"', []),
        ('description: "Example: /home/alice/weights"', []),
    ],
)
def test_yaml_path_literals(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings("examples/model/config.yaml", source.encode())] == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("examples/model/reward.PNG", ["EX003"]),
        ("examples/model/video.mp4", ["EX003"]),
        ("examples/model/clip.mov", ["EX003"]),
        ("examples/model/audio.wav", ["EX003"]),
        ("examples/model/audio.flac", ["EX003"]),
        ("docs/model/reward.png", []),
        ("examples/model/README_training.md", ["EX004"]),
        ("examples/model/README-en.md", ["EX004"]),
        ("examples/model/readme_en.MD", ["EX004"]),
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
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -e\npython trainer.py "$@"\n', ["EX005"]),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -ex\npython trainer.py "$@"\n', ["EX005"]),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -euo pipefail\npython trainer.py "$@"\n', []),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -xeuo pipefail\npython trainer.py "$@"\n', []),
        ("examples/model/run_train.sh", '#!/usr/bin/env bash\nset -e\nset -o pipefail\npython trainer.py "$@"\n', []),
        (
            "examples/model/run_train.sh",
            '#!/usr/bin/env bash\nset -e -o pipefail\npython trainer.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            '#!/usr/bin/env bash\nset -o errexit -o pipefail\npython trainer.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            '#!/usr/bin/env bash\nset -o pipefail -e\npython trainer.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            '#!/usr/bin/env bash\ncat <<EOF\nset -euo pipefail\nEOF\npython trainer.py "$@"\n',
            ["EX005"],
        ),
        (
            "examples/model/run_train.sh",
            '#!/usr/bin/env bash\r\nset -euo pipefail\r\npython trainer.py "$@"\r\n',
            ["EX001", "EX005"],
        ),
        ("examples/model/RUN_UPPER.SH", 'set -euo pipefail\npython trainer.py "$@"\n', ["EX005"]),
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
        ("examples/dpo_trainer/qwen3_omni/eval.sh", "PYTHON_BIN=.venv/bin/python\n$PYTHON_BIN eval.py\n", ["EX006"]),
        ("examples/dancegrpo_trainer/data_process/prepare_prompts.py", "x = 1\n", []),
        ("examples/dpo_trainer/qwen3_omni/run_demo.sh", f'{VALID_HEADER}python trainer.py "$@"\n', []),
        ("examples/dpo_trainer/qwen3_omni/eval.sh", "# .venv/bin/python\npython3 eval.py\n", []),
        ("examples/dpo_trainer/qwen3_omni/eval.sh", "cat <<EOF\n.venv/bin/python\nEOF\npython3 eval.py\n", []),
    ],
)
def test_example_layout_and_portable_python_hygiene(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, source.encode())] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'python3 -m verl_omni.trainer.main_omni \\\n    data.train_batch_size=8\n    trainer.total_epochs=1 "$@"\n',
            ["EX007"],
        ),
        (
            "python3 -m verl_omni.trainer.main_omni \\\n    trainer.total_epochs=1 \\\n"
            '    trainer.total_epochs=2 "$@"\n',
            ["EX007"],
        ),
        (
            'if [ "$MODE" = "a" ]; then\n    python3 t.py trainer.total_epochs=1 "$@"\n'
            'else\n    python3 t.py trainer.total_epochs=2 "$@"\nfi\n',
            [],
        ),
        ('EXTRA_ARGS=(\n    +reward.x=1\n    +reward.y=2\n)\npython3 t.py "${EXTRA_ARGS[@]}" "$@"\n', []),
        ('case "$arg" in\n    data.max_prompt_length=*)\n        shift\n        ;;\nesac\npython3 t.py "$@"\n', []),
        (
            'case "$arg" in\n    trainer.device=npu|trainer.device=mlu)\n        shift\n        ;;\nesac\n'
            'python3 t.py "$@"\n',
            [],
        ),
        (
            "python3 -m verl_omni.trainer.main_omni \\\n    data.train_batch_size=8\n"
            '    ++trainer.total_epochs=1 "$@"\n',
            ["EX007"],
        ),
        (
            "python3 -m verl_omni.trainer.main_omni \\\n    ++trainer.total_epochs=1 \\\n"
            '    trainer.total_epochs=2 "$@"\n',
            ["EX007"],
        ),
        ('python3 t.py \\\n    data.x=1\n    ~data.y "$@"\n', ["EX007"]),
        ('local -a args=(\n    +reward.x=1\n    data.dir=$(pwd)/out\n)\npython3 t.py "${args[@]}" "$@"\n', []),
    ],
)
def test_shell_hydra_launcher_integrity(source: str, expected: list[str]) -> None:
    assert [
        code for _line, code, _message in findings("examples/model/run_train.sh", (VALID_HEADER + source).encode())
    ] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("examples/model/run_train.sh", f'{VALID_HEADER}WANDB_API_KEY="secret123456"\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}export HF_TOKEN=hf_abcdefghij\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}CUDA_VISIBLE_DEVICES=0,1\npython3 t.py "$@"\n', ["EX008"]),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}export ASCEND_RT_VISIBLE_DEVICES="0,1"\npython3 t.py "$@"\n',
            ["EX008"],
        ),
        ("examples/model/run_train.sh", f'{VALID_HEADER}MLU_VISIBLE_DEVICES=0,1\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}HIP_VISIBLE_DEVICES=0,1\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}ROCR_VISIBLE_DEVICES="0,1"\npython3 t.py "$@"\n', ["EX008"]),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}source /usr/local/Ascend/ascend-toolkit/set_env.sh\npython3 t.py "$@"\n',
            ["EX008"],
        ),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}. /usr/local/Ascend/cann-9.0.0/set_env.sh\npython3 t.py "$@"\n',
            ["EX008"],
        ),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}MLU_VISIBLE_DEVICES=${{MLU_VISIBLE_DEVICES:-0,1}}\npython3 t.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}HIP_VISIBLE_DEVICES=${{HIP_VISIBLE_DEVICES:-0,1}}\npython3 t.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            f"{VALID_HEADER}source "
            '"${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"\n'
            'python3 t.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}ENDPOINT="http://10.12.34.56:8000"\npython3 t.py "$@"\n',
            ["EX008"],
        ),
        ("examples/model/run_train.sh", f'{VALID_HEADER}cd /workspace/verl-omni\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}pkill -9 ray\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}ray stop --force\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}pip install vllm\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}uv pip install -e .\npython3 t.py "$@"\n', ["EX008"]),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0,1}}\npython3 t.py "$@"\n',
            [],
        ),
        ("examples/model/run_train.sh", f'{VALID_HEADER}WANDB_API_KEY="${{WANDB_API_KEY:-}}"\npython3 t.py "$@"\n', []),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"\npython3 t.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}# pkill -9 ray\ncat <<EOF\nray stop --force\nEOF\npython3 t.py "$@"\n',
            [],
        ),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}export "WANDB_API_KEY=secret123456"\npython3 t.py "$@"\n',
            ["EX008"],
        ),
        ("examples/model/run_train.sh", f'{VALID_HEADER}cd -- /workspace/verl-omni\npython3 t.py "$@"\n', ["EX008"]),
        ("examples/model/run_train.sh", f'{VALID_HEADER}python3 -m pip install vllm\npython3 t.py "$@"\n', ["EX008"]),
        (
            "examples/model/run_train.sh",
            f'{VALID_HEADER}IMAGE="v10.1.2.3"\nX="10.300.400.500"\npython3 t.py "$@"\n',
            [],
        ),
        ("examples/model/run_train.sh", f'{VALID_HEADER}pkill -P $$\npython3 t.py "$@"\n', []),
    ],
)
def test_shell_safety_and_portability_guards(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, source.encode())] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('exec bash "$SCRIPT_DIR/run_foo.sh" trainer.device=npu "$@"\n', True),
        ('exec "$SCRIPT_DIR/run_foo.sh" trainer.device=npu "$@"\n', True),
        ('source "$SCRIPT_DIR/run_foo.sh" "$@"\n', True),
        ('/bin/bash "$SCRIPT_DIR/run_foo.sh" "$@"\n', True),
        ('base_recipe="$SCRIPT_DIR/run_foo.sh"\nbash "$base_recipe" "$@"\n', True),
        ('BASE=${BASE:-"$SCRIPT_DIR/run_foo.sh"}\nexec bash "$BASE" "$@"\n', True),
        ('bash -n "$SCRIPT_DIR/run_foo.sh"\n', False),
        ('bash "$SCRIPT_DIR/other.sh" --ref "$SCRIPT_DIR/run_foo.sh"\n', False),
        ('bash "$SCRIPT_DIR/../other/run_foo.sh" "$@"\n', False),
        ('python3 -m verl_omni.trainer.main_diffusion trainer.device=npu "$@"\n', False),
        ('# exec bash "$SCRIPT_DIR/run_foo.sh" "$@"\npython3 -m verl_omni.trainer.main_diffusion "$@"\n', False),
        ('cat <<EOF\nbash "$SCRIPT_DIR/run_foo.sh"\nEOF\npython3 -m verl_omni.trainer.main_diffusion "$@"\n', False),
    ],
)
def test_delegates_to_base_script_ignores_comments_and_heredocs(source: str, expected: bool) -> None:
    assert delegates_to_base_script(source.encode(), "run_foo.sh") is expected


def test_multiline_shell_findings_report_exact_line_numbers() -> None:
    source = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'MODEL="/home/alice/weights"\n'
        "python3 -m verl_omni.trainer.main_omni \\\n"
        "    data.train_batch_size=8 \\\n"
        "    actor.use_kl_loss=false \\\n"
        '    data.val_file="/user/alice/val.parquet" \\\n'
        "    actor.use_kl_loss=true \\\n"
        '    "$@"\n'
    )
    hits = [(line, code) for line, code, _msg in findings("examples/model/run_train.sh", source.encode())]
    assert hits == [(3, "VO014"), (7, "VO014"), (8, "EX007")]
