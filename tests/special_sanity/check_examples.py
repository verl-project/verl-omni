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

"""Check example recipes and local paths without executing candidate code."""

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
                while command and (
                    command[0] in {"if", "then", "elif", "else", "do", "!", "env", "exec", "command", "builtin"}
                    or re.match(r"^[a-zA-Z_][a-zA-Z_0-9]*=", command[0])
                ):
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


def findings(path: str, source: bytes) -> Iterator[tuple[int, str, str]]:
    """Check local path literals, example names, and basic shell recipe hygiene."""
    source_path = PurePosixPath(path)
    in_examples = path.startswith("examples/")
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
    for line, token in tokens:
        if SHELL_LOCAL_PATH.search(token.strip("\"'")):
            yield line, "VO014", "Pass local model and data paths through configuration."
    if source_path.name.startswith("run_") and not _forwards_args(tokens):
        yield 1, "EX002", 'Forward caller overrides as a double-quoted "$@" command argument.'
