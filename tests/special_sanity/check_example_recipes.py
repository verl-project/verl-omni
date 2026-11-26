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

MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".wav", ".mov", ".webm", ".mp3", ".flac"}
_LOCAL_PATH_BODY = r"/(?:(?:Users|home|nfs|mnt|root)(?:/|$)|users?/)"
LOCAL_PATH = re.compile(rf"^{_LOCAL_PATH_BODY}")
SHELL_LOCAL_PATH = re.compile(rf"(?:^|=|:-|:=|-|:\+|:(?=/))[\"']?({_LOCAL_PATH_BODY})")
YAML_LOCAL_PATH = re.compile(rf"^(?:\s*-\s*|[A-Za-z0-9_.-]+\s*:\s*(?:\[[^\]]*,\s*)?)[\"']?({_LOCAL_PATH_BODY})")
HARDCODED_VENV_PYTHON = re.compile(r"(?:^|[=:/\-])\.venv/bin/python")
TRAILING_BACKSLASH_SPACE = re.compile(r"\\[ \t]+$")
_HYDRA_SET_KEY = re.compile(r"^(?:\+\+?)?([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)=")
_HYDRA_REMOVE_KEY = re.compile(r"^~([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)$")
_ARRAY_ASSIGN_RE = re.compile(r"^[a-zA-Z_][a-zA-Z_0-9]*\+?-?=$")
_ARRAY_DECL_PREFIXES = {"local", "declare", "typeset", "export", "readonly"}
SECRET_ENV_NAMES = {
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
}
DEVICE_ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "MLU_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}
ACCEL_VARIANT_SCRIPT = re.compile(r"^(run_.+?)(?:_(?:npu|mlu|amd|rocm))(_v1)?\.sh$")
SECRET_VALUE = re.compile(r"^[A-Za-z0-9_\-\./+=]{8,}$")
DEVICE_LIST_VALUE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
PRIVATE_IPV4 = re.compile(
    rf"(?<![A-Za-z0-9.])(?:10\.{_OCTET}\.{_OCTET}\.{_OCTET}|192\.168\.{_OCTET}\.{_OCTET}|"
    rf"172\.(?:1[6-9]|2\d|3[01])\.{_OCTET}\.{_OCTET})(?![0-9.])"
)
COMMAND_PREFIXES = {"if", "then", "elif", "else", "do", "!", "env", "exec", "command", "builtin"}


def _hydra_key(raw_token: str) -> str | None:
    stripped = raw_token.strip("\"'")
    if stripped.endswith("=*"):
        return None
    if match := _HYDRA_SET_KEY.match(stripped):
        return match.group(1)
    if match := _HYDRA_REMOVE_KEY.match(stripped):
        return match.group(1)
    return None


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


def _scan_shell_line(line: str, quote: str | None, ansi_single: bool) -> tuple[str | None, bool, bool]:
    i = 0
    length = len(line)
    while i < length:
        ch = line[i]
        if quote == "'":
            if ansi_single and ch == "\\":
                i += 2
                continue
            if ch == "'":
                quote = None
                ansi_single = False
            i += 1
            continue
        if quote == '"':
            if ch == "\\":
                if i == length - 1:
                    return quote, ansi_single, True
                i += 2
                continue
            if ch == '"':
                quote = None
            i += 1
            continue
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            return None, False, False
        if ch == "\\":
            if i == length - 1:
                return None, False, True
            i += 2
            continue
        if ch == "$" and i + 1 < length and line[i + 1] == "'":
            quote = "'"
            ansi_single = True
            i += 2
            continue
        if ch in {"'", '"'}:
            quote = ch
        i += 1
    return quote, ansi_single, False


def _consume_heredoc_tokens(line_tokens: list[str], heredocs: list[tuple[str, bool]]) -> None:
    arith_depth = 0
    index = 0
    while index < len(line_tokens):
        token = line_tokens[index]
        if token == "((":
            arith_depth += 2
        elif token == "))" and arith_depth:
            arith_depth = max(0, arith_depth - 2)
        elif token == "(" and arith_depth:
            arith_depth += 1
        elif token == ")" and arith_depth:
            arith_depth -= 1
        elif token == "<<" and arith_depth == 0 and index + 1 < len(line_tokens):
            raw_next = line_tokens[index + 1]
            if raw_next == "-" and index + 2 < len(line_tokens):
                strip_tabs = True
                raw_marker = line_tokens[index + 2]
            elif raw_next.startswith("-") and len(raw_next) > 1:
                strip_tabs = True
                raw_marker = raw_next[1:]
            else:
                strip_tabs = False
                raw_marker = raw_next
            marker = raw_marker.lstrip("\\").strip("\"'")
            if marker:
                heredocs.append((marker, strip_tabs))
        index += 1


def _shell_tokens(source: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    heredocs: list[tuple[str, bool]] = []
    logical_line_tokens: list[str] = []
    continued = ""
    start_line = 1
    quote: str | None = None
    ansi_single = False
    for line_number, line in enumerate(source.splitlines(), 1):
        if heredocs:
            marker, strip_tabs = heredocs[0]
            if (line.lstrip("\t") if strip_tabs else line) == marker:
                heredocs.pop(0)
            continue
        if not continued and quote is None:
            start_line = line_number
        quote, ansi_single, has_cont = _scan_shell_line(line, quote, ansi_single)
        if quote is not None:
            continued += line[:-1] if has_cont else line + "\n"
            continue
        segment = line[:-1] if has_cont else line
        full_line = continued + segment
        continued = ""
        lexer = shlex.shlex(full_line, posix=False, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.wordchars += "${}:"
        try:
            line_tokens = list(lexer)
        except ValueError:
            if not has_cont:
                _consume_heredoc_tokens(logical_line_tokens, heredocs)
                logical_line_tokens.clear()
                tokens.append((line_number, "\n"))
            continue
        tokens.extend((start_line, token) for token in line_tokens)
        logical_line_tokens.extend(line_tokens)
        if not has_cont:
            _consume_heredoc_tokens(logical_line_tokens, heredocs)
            logical_line_tokens.clear()
            tokens.append((line_number, "\n"))
    return tokens


def _is_array_start(command: list[tuple[int, str]]) -> str | None:
    if not command or not _ARRAY_ASSIGN_RE.fullmatch(command[-1][1]):
        return None
    for _ln, prefix_tok in command[:-1]:
        if prefix_tok not in _ARRAY_DECL_PREFIXES and not prefix_tok.startswith("-"):
            return None
    return command[-1][1].rstrip("+=")


def _iter_commands(tokens: list[tuple[int, str]]) -> Iterator[tuple[list[tuple[int, str]], str]]:
    command: list[tuple[int, str]] = []
    array_depth = 0
    case_depth = 0
    in_case_pattern = False
    for line, token in [*tokens, (0, "\n")]:
        if array_depth > 0:
            if token in {"(", "(("}:
                array_depth += 1 if token == "(" else 2
            elif token in {")", "))"}:
                array_depth = max(0, array_depth - (1 if token == ")" else 2))
            continue
        if token == "(" and _is_array_start(command) is not None:
            array_depth = 1
            command = []
            continue
        if case_depth > 0 and in_case_pattern:
            if token == ")":
                in_case_pattern = False
                command = []
            elif token == "esac":
                case_depth = max(0, case_depth - 1)
                in_case_pattern = False
                command = []
            continue
        if token in {"\n", ";", ";;", "&&", "||", "|", "&", "(", ")"}:
            if command:
                if command[0][1] == "case":
                    case_depth += 1
                    in_case_pattern = True
                elif command[0][1] == "esac":
                    case_depth = max(0, case_depth - 1)
                    in_case_pattern = False
                else:
                    yield command, token
                command = []
            if case_depth > 0 and token == ";;":
                in_case_pattern = True
        else:
            command.append((line, token))


def _strip_command_prefixes(command: list[tuple[int, str]]) -> list[tuple[int, str]]:
    work = list(command)
    while work and (work[0][1] in COMMAND_PREFIXES or re.match(r"^[a-zA-Z_][a-zA-Z_0-9]*=", work[0][1])):
        work.pop(0)
    return work


def _forwards_args(tokens: list[tuple[int, str]]) -> bool:
    command: list[tuple[int, str]] = []
    argument_arrays: set[str] = set()
    array_name: str | None = None
    array_depth = 0
    excluded = {"echo", "printf", ":", "test", "[", "[[", "local", "declare", "export", "for"}
    for line, token in [*tokens, (0, "\n")]:
        if array_depth > 0:
            if token in {"(", "(("}:
                array_depth += 1 if token == "(" else 2
            elif token in {")", "))"}:
                array_depth = max(0, array_depth - (1 if token == ")" else 2))
                if array_depth == 0:
                    array_name = None
            elif array_name is not None and token in {'"$@"', '"${@}"'}:
                argument_arrays.add(array_name)
            continue
        if token == "(" and (started := _is_array_start(command)) is not None:
            array_name = started
            array_depth = 1
            command = []
            continue
        if token in {"\n", ";", ";;", "&&", "||", "|", "&", "(", ")"}:
            work = [tok for _ln, tok in _strip_command_prefixes(command)]
            if work and work[0] not in excluded:
                allowed = {'"$@"', '"${@}"'} | {f'"${{{name}[@]}}"' for name in argument_arrays}
                if any(argument in allowed for argument in work[1:]):
                    return True
            command = []
        else:
            command.append((line, token))
    return False


def _has_errexit_and_pipefail(tokens: list[tuple[int, str]]) -> bool:
    has_errexit = False
    has_pipefail = False
    for command, _delimiter in _iter_commands(tokens):
        work = [tok for _ln, tok in _strip_command_prefixes(command)]
        if not work or work[0] != "set":
            continue
        args = work[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--":
                break
            if arg == "-o" and i + 1 < len(args):
                opt = args[i + 1]
                if opt == "errexit":
                    has_errexit = True
                elif opt == "pipefail":
                    has_pipefail = True
                i += 2
                continue
            if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
                flags = arg[1:]
                if "e" in flags:
                    has_errexit = True
                if "o" in flags and i + 1 < len(args):
                    if args[i + 1] == "errexit":
                        has_errexit = True
                    elif args[i + 1] == "pipefail":
                        has_pipefail = True
                    i += 2
                    continue
            i += 1
    return has_errexit and has_pipefail


def _token_findings(tokens: list[tuple[int, str]]) -> Iterator[tuple[int, str, str]]:
    for line, token in tokens:
        stripped = token.strip("\"'")
        if HARDCODED_VENV_PYTHON.search(stripped):
            yield line, "EX006", "Use python3 or ${PYTHON_BIN:-python3} instead of a hardcoded .venv/bin/python path."
        if PRIVATE_IPV4.search(stripped):
            yield (
                line,
                "EX008",
                "Do not hardcode private cluster IP addresses; pass endpoints via environment variables.",
            )
        if "=" in stripped:
            key, raw_value = stripped.split("=", 1)
            value = raw_value.strip("\"'")
            if key in SECRET_ENV_NAMES and not value.startswith("$") and SECRET_VALUE.fullmatch(value):
                yield line, "EX008", f"Do not hardcode {key} in example scripts; read it from the environment."
            if key in DEVICE_ENV_NAMES and DEVICE_LIST_VALUE.fullmatch(value):
                yield line, "EX008", f"Use overridable ${{{key}:-...}} instead of hardcoding {key}={value}."


def _command_safety_findings(work: list[tuple[int, str]], is_run: bool) -> Iterator[tuple[int, str, str]]:
    head_line, head_tok = work[0]
    if head_tok == "cd":
        targets = [tok.strip("\"'") for _ln, tok in work[1:] if tok not in {"--", "-P", "-L"}]
        if targets and targets[0].startswith("/"):
            yield head_line, "EX008", "Avoid hardcoded absolute 'cd /...' paths; resolve relative to SCRIPT_DIR."
    if head_tok in {"source", "."} and len(work) >= 2 and work[1][1].strip("\"'").startswith("/usr/local/Ascend/"):
        yield (
            head_line,
            "EX008",
            "Use ${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit} instead of hardcoding /usr/local/Ascend/.",
        )
    if not is_run:
        return
    if head_tok == "killall" or (head_tok == "ray" and len(work) >= 2 and work[1][1] == "stop"):
        yield head_line, "EX008", "Do not kill host-wide processes (pkill/killall/ray stop) inside run_*.sh."
    elif head_tok == "pkill":
        args_list = [tok for _ln, tok in work[1:]]
        own_children = len(args_list) >= 2 and args_list[:2] == ["-P", "$$"]
        if not own_children:
            yield head_line, "EX008", "Do not kill host-wide processes (pkill/killall/ray stop) inside run_*.sh."
    is_pip = head_tok in {"pip", "pip3"} and len(work) >= 2 and work[1][1] == "install"
    is_uv_pip = head_tok == "uv" and len(work) >= 3 and work[1][1] == "pip" and work[2][1] == "install"
    is_py_pip = (
        PurePosixPath(head_tok.strip("\"'")).name in {"python", "python3"}
        and len(work) >= 4
        and work[1][1] == "-m"
        and work[2][1] == "pip"
        and work[3][1] == "install"
    )
    if is_pip or is_uv_pip or is_py_pip:
        yield head_line, "EX008", "Do not run pip install inside run_*.sh; document dependencies in README.md."


def _shell_findings(source_path: PurePosixPath, source: bytes, text: str) -> Iterator[tuple[int, str, str]]:
    raw_lines = text.split("\n")
    has_crlf = "\r\n" in text
    if has_crlf:
        yield 1, "EX001", "Use LF (Unix) line endings in example shell scripts."
    text_lf = text.replace("\r\n", "\n")

    cleaned_lines: list[str] = []
    for line_no, raw_line in enumerate(text_lf.splitlines(), 1):
        if not raw_line.lstrip().startswith("#") and TRAILING_BACKSLASH_SPACE.search(raw_line):
            yield line_no, "EX001", "Remove trailing spaces after backslash line continuation."
            cleaned_lines.append(TRAILING_BACKSLASH_SPACE.sub("\\\\", raw_line))
        else:
            cleaned_lines.append(raw_line)

    environment = {
        key: value for key, value in os.environ.items() if key not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}
    }
    try:
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
    except subprocess.TimeoutExpired:
        yield 1, "EX001", "Shell syntax check timed out."

    parse_text = "\n".join(cleaned_lines)
    tokens = _shell_tokens(parse_text)
    is_run = source_path.name.lower().startswith("run_")
    yield from _token_findings(tokens)

    for command, delimiter in _iter_commands(tokens):
        work = _strip_command_prefixes(command)
        if not work or work[0][1] not in {"echo", "printf", ":"}:
            for ln, tok in command:
                if SHELL_LOCAL_PATH.search(tok.strip("\"'")):
                    yield ln, "VO014", "Pass local model and data paths through configuration."
        if not work:
            continue
        head_line, head_tok = work[0]
        if delimiter != ")" and _hydra_key(head_tok) is not None:
            yield (
                head_line,
                "EX007",
                f"Orphaned Hydra override '{head_tok.split('=')[0]}' (missing trailing backslash on previous line).",
            )
            continue
        yield from _command_safety_findings(work, is_run)

        hydra_positions: list[tuple[int, int, str]] = []
        at_positions: list[tuple[int, int]] = []
        for idx, (cln, ctok) in enumerate(work[1:], 1):
            if ctok in {'"$@"', '"${@}"'}:
                at_positions.append((idx, cln))
            if (key := _hydra_key(ctok)) is not None:
                hydra_positions.append((idx, cln, key))
        seen_keys: set[str] = set()
        for _idx, cln, key in hydra_positions:
            if key in seen_keys:
                yield (
                    cln,
                    "EX007",
                    f"Duplicate Hydra override '{key}' in the same command.",
                )
            seen_keys.add(key)
        if is_run and at_positions and hydra_positions:
            last_hydra_idx = hydra_positions[-1][0]
            for at_idx, at_line in at_positions:
                if at_idx < last_hydra_idx:
                    yield at_line, "EX002", 'Place "$@" after Hydra overrides so caller arguments take precedence.'

    if is_run:
        has_shebang = bool(raw_lines) and raw_lines[0] == "#!/usr/bin/env bash"
        if not (has_shebang and _has_errexit_and_pipefail(tokens)):
            yield (
                1,
                "EX005",
                "Start run_*.sh with #!/usr/bin/env bash and enable set -e + set -o pipefail (e.g. set -euo pipefail).",
            )
        if not _forwards_args(tokens):
            yield 1, "EX002", 'Forward caller overrides as a double-quoted "$@" command argument.'


def _matches_base_ref(raw_arg: str, base_name: str, base_vars: set[str]) -> bool:
    stripped = raw_arg.strip("\"'")
    if ".." in PurePosixPath(stripped).parts:
        return False
    if stripped == base_name or stripped.endswith(f"/{base_name}"):
        return True
    return any(stripped in {f"${var}", f"${{{var}}}"} for var in base_vars)


def delegates_to_base_script(source: bytes, base_name: str) -> bool:
    """Return whether a shell wrapper executes the sibling base recipe via bash/sh/exec/source."""
    try:
        text = source.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return True
    tokens = _shell_tokens(text)
    base_vars: set[str] = set()
    base_pattern = re.compile(rf'(?:^|/|["\']){re.escape(base_name)}["\']?\}}?$')
    for _line, token in tokens:
        if "=" in token and _hydra_key(token) is None:
            var, raw_val = token.split("=", 1)
            val = raw_val.strip("\"'")
            if re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", var) and ".." not in val and base_pattern.search(val):
                base_vars.add(var)
    for command, _delimiter in _iter_commands(tokens):
        work = _strip_command_prefixes(command)
        if not work:
            continue
        has_exec = any(tok == "exec" for _ln, tok in command[: len(command) - len(work)])
        if has_exec and _matches_base_ref(work[0][1], base_name, base_vars):
            return True
        head = PurePosixPath(work[0][1].strip("\"'")).name
        if head not in {"bash", "sh", "source", "."}:
            continue
        positional: str | None = None
        i = 1
        disallowed_flag = False
        while i < len(work):
            tok = work[i][1].strip("\"'")
            if tok == "--":
                if i + 1 < len(work):
                    positional = work[i + 1][1]
                break
            if tok == "-o":
                i += 2
                continue
            if tok.startswith("-") and len(tok) > 1:
                if any(flag in tok[1:] for flag in ("n", "c")):
                    disallowed_flag = True
                    break
                i += 1
                continue
            positional = work[i][1]
            break
        if not disallowed_flag and positional is not None and _matches_base_ref(positional, base_name, base_vars):
            return True
    return False


def findings(path: str, source: bytes) -> Iterator[tuple[int, str, str]]:
    """Check local path literals, example names, layout, and shell recipe hygiene."""
    source_path = PurePosixPath(path)
    in_examples = path.startswith("examples/")
    suffix = source_path.suffix.lower()
    if in_examples:
        dirs = source_path.parts[1:-1]
        if "dataprocess" in dirs:
            yield 1, "EX006", "Use data_process/ instead of dataprocess/ under examples/."
        if any(parent == child for parent, child in zip(dirs, dirs[1:], strict=False)):
            yield 1, "EX006", "Remove duplicate nested directory segment under examples/."
    if in_examples and suffix in MEDIA_SUFFIXES:
        yield 1, "EX003", "Keep generated example media in PR evidence instead of committing it."
        return
    if in_examples and re.fullmatch(r"readme[-_].+\.md", source_path.name, re.IGNORECASE):
        yield 1, "EX004", "Use README.md in an example directory so the docs checker discovers it."
        return
    if suffix == ".py" and path.startswith(("verl_omni/", "tests/", "examples/")):
        for line in _python_paths(source):
            yield line, "VO014", "Pass local model and data paths through configuration or temporary test fixtures."
        return
    if not in_examples or suffix not in {".sh", ".yaml", ".yml"}:
        return
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        yield 1, "EX001", "Save example shell scripts and YAML as UTF-8 text."
        return
    if suffix in {".yaml", ".yml"}:
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
