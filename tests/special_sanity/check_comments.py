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

"""Detect recognizable comment noise and docstring boilerplate in Python files."""

import ast
import inspect
import io
import re
import tokenize
from collections.abc import Iterator

NARRATION = re.compile(
    r"^(?:(?:added|updated|changed|modified|removed|renamed|refactored|introduced|implemented)\s+"
    r"(?:(?:handling|support)\s+for\b|(?:a|an|the|this|these|to|for)\b)|"
    r"(?:we|i)\s+(?:added|updated|changed|modified|removed|renamed|refactored|introduced|implemented)\b|"
    r"(?:here|now|first|next|finally|then)[,:]?\s+we\b|"
    r"let's\s+|"
    r"this\s+(?:function|method|helper|class|block|loop|step)\s+"
    r"(?:is|will|does|handles|computes|returns|checks|converts)\b|"
    r"fixed\s+(?:the|this|a|an|bug|issue|error|crash|race|leak|mismatch)\b|"
    r"bugfix\s*:|fix\s+for\s+#\d+\b|"
    r"previously[,:\s]+(?:we|this|it|the)\b|before\s+this\s+change\b|old\s+code\b|"
    r"(?:changed|moved|extracted)\s+from\b|replaced\s+(?:the|this|old)\b|"
    r"(?:helper|utility)\s+(?:function|method|class)\s+(?:to|for)\b|"
    r"new\s+(?:helper|function|method|class|code|implementation)\s*(?::|for\b|to\b)|"
    r"step\s+\d+(?:\.\d+)*\s*[:.)-]|"
    r"(?:新增|修改|更新|添加|移除|重命名|重构|修复|修正|解决|替换|调整|优化)"
    r"(?:了|：|:|(?:代码|处理|支持|函数|方法|类|逻辑|校验|检查)(?:以|来|用于|：|:))|"
    r"步骤\s*\d+\s*[:：.)、-])",
    re.IGNORECASE,
)
BANNER = re.compile(
    r"^(?:[#=\-_*~]{3,}(?:\s+.*?\s+[#=\-_*~]{3,})?|"
    r"(?:imports?|constants?|helpers?|utilities|end\s+of\s+\S.*))$",
    re.IGNORECASE,
)
PR_REF = re.compile(r"\bPR\s*#\d+\b", re.IGNORECASE)
DOCSTRING_TEMPLATE = re.compile(r"^\s*(?:Necessity|Motivation)\s*:", re.MULTILINE | re.IGNORECASE)
AI_DOC_FILLER = re.compile(
    r"^(?:(?:a\s+|an\s+)?(?:helper|utility|convenience|internal|private)\s+"
    r"(?:function|method|helper|routine|wrapper)\s+(?:to|for|that|which)\b|"
    r"this\s+(?:function|method|helper|class)\s+(?:is\s+used\s+to|is\s+a\s+helper|simply|just)\b)",
    re.IGNORECASE,
)
ENCODING = re.compile(r"coding[:=]\s*[-\w.]+")
DIRECTIVE_PREFIXES = (
    "noqa",
    "type:",
    "pyright:",
    "mypy:",
    "pylint:",
    "flake8:",
    "ruff:",
    "fmt:",
    "isort:",
    "pragma:",
    "todo",
    "fixme",
    "note:",
    "xxx",
    "warning:",
    "hack:",
    "copyright",
    "licensed",
    "spdx-",
    "http://",
    "https://",
)
ECHO_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "each",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "new",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
DOC_ECHO_STOPWORDS = ECHO_STOPWORDS | {"compute", "computes", "get", "gets", "return", "returns"}


def _norm_word(word: str) -> str:
    if len(word) >= 5 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _identifier_words(text: str) -> set[str]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return {_norm_word(word) for word in re.findall(r"[a-z0-9]+", snake.lower()) if len(word) > 1 or word.isdigit()}


def _is_echo_comment(text: str, code_line: str) -> bool:
    if not text or len(text) >= 60 or any(marker in text for marker in (":", "?", "!", "->", "[")):
        return False
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not 2 <= len(words) <= 6:
        return False
    content = {_norm_word(word) for word in words if word not in ECHO_STOPWORDS}
    if not content or len(content) > 5:
        return False
    code_words = _identifier_words(code_line)
    return bool(code_words) and content <= code_words


def _license_lines(tokens: list[tokenize.TokenInfo]) -> set[int]:
    lines: set[int] = set()
    started = False
    for token in tokens:
        if token.type == tokenize.ENCODING:
            continue
        if token.type != tokenize.COMMENT:
            break
        text = token.string.removeprefix("#").strip().lower()
        if not started and not any(key in text for key in ("copyright", "spdx-", "licensed under")):
            break
        started = True
        lines.add(token.start[0])
    return lines


def _is_trivial_private_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef, doc: str) -> bool:
    stripped = doc.strip()
    if "\n" in stripped or any(marker in stripped for marker in (":", "`", "(", "[", "->")):
        return False
    words = {_norm_word(word) for word in re.findall(r"[a-z0-9]+", stripped.lower()) if word not in DOC_ECHO_STOPWORDS}
    if not words or len(words) > 5:
        return False
    sig_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    sig_words = _identifier_words(node.name)
    for arg in sig_args:
        sig_words.update(_identifier_words(arg.arg))
    return words <= sig_words


def _docstring_findings(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    doc_nodes = [tree] + [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    for node in doc_nodes:
        first = node.body[0] if node.body else None
        if not (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        ):
            continue
        doc = inspect.cleandoc(first.value.value)
        if DOCSTRING_TEMPLATE.search(doc):
            yield first.lineno, "CM004", "Remove PR-template headers (Necessity/Motivation) from docstrings."
        elif AI_DOC_FILLER.match(doc.strip()):
            yield first.lineno, "CM004", "Use imperative contract docstrings instead of helper/filler phrasing."
        elif (
            path.startswith("verl_omni/")
            and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("_")
            and not (node.name.startswith("__") and node.name.endswith("__"))
            and _is_trivial_private_docstring(node, doc)
        ):
            yield first.lineno, "CM004", "Omit trivial private docstring that only repeats the function signature."


def findings(path: str, source: bytes, tree: ast.AST | None = None) -> Iterator[tuple[int, str, str]]:
    """Yield comment-shape and docstring-boilerplate violations without inferring authorship."""
    if not path.startswith(("verl_omni/", "tests/", "examples/")) or not path.endswith(".py"):
        return
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source).readline))
    except (tokenize.TokenError, SyntaxError, UnicodeError, LookupError):
        return

    license_lines = _license_lines(tokens)
    standalone_lines: dict[int, str] = {}
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line_no = token.start[0]
        text = token.string.removeprefix("#").strip()
        if line_no <= 2 and ENCODING.search(text):
            continue
        if NARRATION.match(text):
            yield line_no, "CM001", "Remove edit/step narration; keep only necessary contract or rationale."
            continue
        standalone = not token.line[: token.start[1]].strip()
        if standalone and BANNER.fullmatch(text):
            yield line_no, "CM002", "Remove decorative comment banners; use concise API documentation."
            continue
        lower = text.lower()
        is_directive = lower.startswith(DIRECTIVE_PREFIXES)
        if line_no not in license_lines and not is_directive and PR_REF.search(text):
            yield line_no, "CM005", "Remove PR references from code comments; cite issues or explain the rationale."
        if standalone and line_no not in license_lines and not is_directive:
            standalone_lines[line_no] = text

    if path.startswith("verl_omni/") and "/config/" not in path and standalone_lines:
        raw_lines = source.decode("utf-8", errors="replace").splitlines()
        for line_no, text in standalone_lines.items():
            if (line_no - 1) in standalone_lines or (line_no + 1) in standalone_lines:
                continue
            next_line = raw_lines[line_no].strip() if line_no < len(raw_lines) else ""
            if next_line and not next_line.startswith("#") and _is_echo_comment(text, next_line):
                yield (
                    line_no,
                    "CM003",
                    "Remove redundant comment repeating the next line; only keep non-obvious rationale.",
                )

    if path.startswith(("verl_omni/", "tests/")):
        if tree is None:
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                return
        yield from _docstring_findings(path, tree)
