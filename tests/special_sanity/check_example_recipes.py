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

"""Check example recipes, bash safety, and local paths without executing code."""

import ast
import os
import re
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import PurePosixPath

MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".wav"}
LOCAL_PATH = re.compile(r"^/(?:Users|home|nfs|mnt|root)(?:/|$)")
SHELL_LOCAL_PATH = re.compile(r"(?:^|=|:-|:=)[\"']?(/(?:Users|home|nfs|mnt|root)(?:/|$))")
YAML_LOCAL_PATH = re.compile(r"(?:^|:\s+|-\s+|[,\[{]\s*)[\"']?(/(?:Users|home|nfs|mnt|root)(?:/|$))")
SET_ERREXIT = re.compile(r"^\s*set\s+(?:-[a-zA-Z]*e[a-zA-Z]*\b|-o\s+errexit\b)", re.MULTILINE)
SET_PIPEFAIL = re.compile(r"^\s*set\s+(?:-[a-zA-Z]*o\s+pipefail|-o\s+pipefail)\b", re.MULTILINE)
HARDCODED_VENV_PYTHON = re.compile(r"(?:^|/)\.venv/bin/python")
TRAILING_BACKSLASH_SPACE = re.compile(r"\\[ \t]+$")
HYDRA_DOTTED_KEY = re.compile(r"^\+?([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)=")
SECRET_ENV_NAMES = {
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
}
DEVICE_ENV_NAMES = {"CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"}
SECRET_VALUE = re.compile(r"^[A-Za-z0-9_\-\.]{8,}$")
DEVICE_LIST_VALUE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
PRIVATE_IPV4 = re.compile(
    r"(?<![0-9.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![0-9.])"
)
COMMAND_PREFIXES = {"if", "then", "elif", "else", "do", "!", "env", "exec", "command", "builtin"}


def _python_paths(source: bytes) -> Iterator[int]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return
    docstrings = set()
    fstring_parts = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
        elif isinstance(node, ast.JoinedStr):
            fstring_parts.update(id(part) for part in node.values)
            if node.values and isinstance(node.values[0], ast.Constant) and LOCAL_PATH.match(node.values[0].value):
                yield node.lineno
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and id(node) not in fstring_parts
            and LOCAL_PATH.match(node.value)
        ):
            yield node.lineno


def _shell_tokens(source: str) -> list[tuple[int, str]]:
    tokens = []
    heredocs: list[tuple[str, bool]] = []
    continued = ""
    start_line = 1
    for line_number, line in enumerate(source.splitlines(), 1):
        if heredocs:
            marker, strip_tabs = heredocs[0]
            if (line.lstrip("\t") if strip_tabs else line) == marker:
                heredocs.pop(0)
            continue
        if not continued:
            start_line = line_number
        if line.endswith("\\"):
            continued += line[:-1] + " "
            continue
        line = continued + line
        continued = ""
        lexer = shlex.shlex(line, posix=False, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.wordchars += "${}:"
        try:
            line_tokens = list(lexer)
        except ValueError:
            continue
        for index, token in enumerate(line_tokens):
            if token == "<<" and index + 1 < len(line_tokens):
                marker = line_tokens[index + 1]
                strip_tabs = marker.startswith("-")
                marker = marker.removeprefix("-") if strip_tabs else marker
                heredocs.append((marker.strip("\"'"), strip_tabs))
        tokens.extend((start_line, token) for token in line_tokens)
        tokens.append((line_number, "\n"))
    return tokens


def _iter_commands(tokens: list[tuple[int, str]]) -> Iterator[tuple[list[tuple[int, str]], str]]:
    command: list[tuple[int, str]] = []
    in_array = False
    for line, token in [*tokens, (0, "\n")]:
        if token == "(" and len(command) == 1 and re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*\+?-?=", command[0][1]):
            in_array = True
            command = []
            continue
        if in_array:
            if token == ")":
                in_array = False
            continue
        if token in {"\n", ";", ";;", "&&", "||", "|", "&", "(", ")"}:
            if command:
                yield command, token
                command = []
        else:
            command.append((line, token))


def _strip_command_prefixes(command: list[tuple[int, str]]) -> list[tuple[int, str]]:
    work = list(command)
    while work and (work[0][1] in COMMAND_PREFIXES or re.match(r"^[a-zA-Z_][a-zA-Z_0-9]*=", work[0][1])):
        work.pop(0)
    return work


def _forwards_args(tokens: list[tuple[int, str]]) -> bool:
    command = []
    argument_arrays = set()
    array_name = None
    for _line, token in [*tokens, (0, "\n")]:
        if token == "(" and len(command) == 1 and re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*=", command[0]):
            array_name = command[0][:-1]
            command = []
            continue
        if array_name is not None:
            if token == ")":
                array_name = None
            elif token in {'"$@"', '"${@}"'}:
                argument_arrays.add(array_name)
            continue
        if token in {"\n", ";", ";;", "&&", "||", "|", "&", "(", ")"}:
            if command:
                while command and (command[0] in COMMAND_PREFIXES or re.match(r"^[a-zA-Z_][a-zA-Z_0-9]*=", command[0])):
                    command.pop(0)
            if command:
                executable = command[0]
                if executable not in {"echo", "printf", ":", "test", "[", "[[", "local", "declare", "export"}:
                    allowed_arguments = {'"$@"', '"${@}"'} | {f'"${{{name}[@]}}"' for name in argument_arrays}
                    if any(argument in allowed_arguments for argument in command[1:]):
                        return True
            command = []
        else:
            command.append(token)
    return False


def _shell_findings(source_path: PurePosixPath, source: bytes, text: str) -> Iterator[tuple[int, str, str]]:
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        if raw_line.lstrip().startswith("#"):
            continue
        if TRAILING_BACKSLASH_SPACE.search(raw_line):
            yield line_no, "EX001", "Remove trailing spaces after backslash line continuation."
            return

    environment = {
        key: value for key, value in os.environ.items() if key not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}
    }
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-n"],
        input=source,
        capture_output=True,
        env=environment,
        timeout=5,
        check=False,
    )
    if result.returncode:
        yield 1, "EX001", "Fix the shell syntax reported by bash -n."
        return

    tokens = _shell_tokens(text)
    is_run = source_path.name.startswith("run_")
    for line, token in tokens:
        stripped = token.strip("\"'")
        if SHELL_LOCAL_PATH.search(stripped):
            yield line, "VO014", "Pass local model and data paths through configuration."
        if HARDCODED_VENV_PYTHON.search(stripped):
            yield line, "EX006", "Use python3 or ${PYTHON_BIN:-python3} instead of a hardcoded .venv/bin/python path."
        if PRIVATE_IPV4.search(stripped):
            yield (
                line,
                "EX008",
                "Do not hardcode private cluster IP addresses; pass endpoints via environment variables.",
            )
        if "=" in token:
            key, raw_value = token.split("=", 1)
            value = raw_value.strip("\"'")
            if key in SECRET_ENV_NAMES and not value.startswith("$") and SECRET_VALUE.fullmatch(value):
                yield line, "EX008", f"Do not hardcode {key} in example scripts; read it from the environment."
            if key in DEVICE_ENV_NAMES and DEVICE_LIST_VALUE.fullmatch(value):
                yield line, "EX008", f"Use overridable ${{{key}:-...}} instead of hardcoding {key}={value}."

    for command, delimiter in _iter_commands(tokens):
        work = _strip_command_prefixes(command)
        if not work:
            continue
        head_line, head_tok = work[0]
        if (
            delimiter != ")"
            and HYDRA_DOTTED_KEY.match(head_tok.strip("\"'"))
            and not head_tok.strip("\"'").endswith("=*")
        ):
            yield (
                head_line,
                "EX007",
                f"Orphaned Hydra override '{head_tok.split('=')[0]}' (missing trailing backslash on previous line).",
            )
            continue
        if head_tok == "cd" and len(work) >= 2 and work[1][1].strip("\"'").startswith("/"):
            yield head_line, "EX008", "Avoid hardcoded absolute 'cd /...' paths; resolve relative to SCRIPT_DIR."
        if is_run:
            if head_tok in {"pkill", "killall"} or (head_tok == "ray" and len(work) >= 2 and work[1][1] == "stop"):
                yield head_line, "EX008", "Do not kill host-wide processes (pkill/killall/ray stop) inside run_*.sh."
            if (head_tok in {"pip", "pip3"} and len(work) >= 2 and work[1][1] == "install") or (
                head_tok == "uv" and len(work) >= 3 and work[1][1] == "pip" and work[2][1] == "install"
            ):
                yield head_line, "EX008", "Do not run pip install inside run_*.sh; document dependencies in README.md."

        hydra_positions: list[tuple[int, int, str]] = []
        at_positions: list[tuple[int, int]] = []
        for idx, (cln, ctok) in enumerate(work[1:], 1):
            if ctok in {'"$@"', '"${@}"'}:
                at_positions.append((idx, cln))
            match = HYDRA_DOTTED_KEY.match(ctok.strip("\"'"))
            if match:
                hydra_positions.append((idx, cln, match.group(1)))
        seen_keys: dict[str, int] = {}
        for _idx, cln, key in hydra_positions:
            if key in seen_keys:
                yield (
                    cln,
                    "EX007",
                    f"Duplicate Hydra override '{key}' in the same command (previously set on line {seen_keys[key]}).",
                )
            seen_keys[key] = cln
        if is_run and at_positions and hydra_positions:
            last_hydra_idx = hydra_positions[-1][0]
            for at_idx, at_line in at_positions:
                if at_idx < last_hydra_idx:
                    yield at_line, "EX002", 'Place "$@" after Hydra overrides so caller arguments take precedence.'

    if is_run:
        lines = text.splitlines()
        has_shebang = bool(lines) and lines[0].strip() == "#!/usr/bin/env bash"
        if not (has_shebang and SET_ERREXIT.search(text) and SET_PIPEFAIL.search(text)):
            yield (
                1,
                "EX005",
                "Start run_*.sh with #!/usr/bin/env bash and enable set -e + set -o pipefail (e.g. set -euo pipefail).",
            )
        if not _forwards_args(tokens):
            yield 1, "EX002", 'Forward caller overrides as a double-quoted "$@" command argument.'


def findings(path: str, source: bytes) -> Iterator[tuple[int, str, str]]:
    """Check local path literals, example names, layout, and shell recipe hygiene."""
    source_path = PurePosixPath(path)
    in_examples = path.startswith("examples/")
    if in_examples:
        dirs = source_path.parts[1:-1]
        if "dataprocess" in dirs:
            yield 1, "EX006", "Use data_process/ instead of dataprocess/ under examples/."
        if any(parent == child for parent, child in zip(dirs, dirs[1:], strict=False)):
            yield 1, "EX006", "Remove duplicate nested directory segment under examples/."
    if in_examples and source_path.suffix.lower() in MEDIA_SUFFIXES:
        yield 1, "EX003", "Keep generated example media in PR evidence instead of committing it."
        return
    if in_examples and source_path.match("README_*.md"):
        yield 1, "EX004", "Use README.md in an example directory so the docs checker discovers it."
        return
    if source_path.suffix == ".py" and path.startswith(("verl_omni/", "tests/", "examples/")):
        for line in _python_paths(source):
            yield line, "VO014", "Pass local model and data paths through configuration or temporary test fixtures."
        return
    if not in_examples or source_path.suffix not in {".sh", ".yaml", ".yml"}:
        return
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        yield 1, "EX001", "Save example shell scripts and YAML as UTF-8 text."
        return
    if source_path.suffix in {".yaml", ".yml"}:
        for line_number, line in enumerate(text.splitlines(), 1):
            lexer = shlex.shlex(line, posix=False)
            lexer.whitespace_split = True
            try:
                value = " ".join(lexer)
            except ValueError:
                continue
            if YAML_LOCAL_PATH.search(value):
                yield line_number, "VO014", "Pass local model and data paths through configuration."
        return
    yield from _shell_findings(source_path, source, text)
