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

"""Detect explicit edit narration and decorative banners in Python comments."""

import io
import re
import tokenize
from collections.abc import Iterator

NARRATION = re.compile(
    r"^(?:(?:added|updated|changed|modified|removed|renamed|refactored|introduced|implemented)\s+"
    r"(?:(?:handling|support)\s+for\b|(?:a|an|the|this|these|to|for)\b)|"
    r"(?:we|i)\s+(?:added|updated|changed|modified|removed|renamed|refactored|introduced|implemented)\b|"
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
ENCODING = re.compile(r"coding[:=]\s*[-\w.]+")


def findings(path: str, source: bytes) -> Iterator[tuple[int, str, str]]:
    """Yield comment-shape violations without inferring authorship or necessity."""
    if not path.startswith(("verl_omni/", "tests/", "examples/")) or not path.endswith(".py"):
        return
    try:
        for token in tokenize.tokenize(io.BytesIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            text = token.string.removeprefix("#").strip()
            if token.start[0] <= 2 and ENCODING.search(text):
                continue
            if NARRATION.match(text):
                yield token.start[0], "CM001", "Remove edit/step narration; keep only necessary contract or rationale."
            elif not token.line[: token.start[1]].strip() and BANNER.fullmatch(text):
                yield token.start[0], "CM002", "Remove decorative comment banners; use concise API documentation."
    except (tokenize.TokenError, SyntaxError, UnicodeError, LookupError):
        return
