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

"""Check recognizable comment noise while preserving technical documentation."""

from pathlib import Path

import pytest
from check_comments import findings

COMMENT_PATH = "verl_omni/trainer/diffusion/example.py"


@pytest.mark.parametrize(
    "comment",
    [
        "# Added handling for the new configuration field",
        "# Updated the reward path",
        "# Changed to use the batched path",
        "# Modified this function to support videos",
        "# Removed the unused branch",
        "# Renamed the helper",
        "# Refactored this function",
        "# Introduced a new helper",
        "# Implemented support for images",
        "# We added the missing check",
        "# I updated the implementation",
        "# New helper for parsing input",
        "# New function: normalize",
        "# New class for parsing input",
        "# New method to normalize input",
        "# Step 1: load model",
        "# STEP 2. Encode prompt",
        "# Step 2.1: encode image",
        "# Fixed the mismatch in padding",
        "# Fixed bug when prompt is empty",
        "# Bugfix: handle empty mask",
        "# Fix for #942",
        "# Previously, we dropped the audio tensor",
        "# Before this change the buffer overflowed",
        "# Old code kept bf16 here",
        "# Changed from list to tuple",
        "# Moved from trainer_base.py",
        "# Extracted from rollout loop",
        "# Replaced the legacy scheduler",
        "# Helper function to format latents",
        "# Utility method for slicing masks",
        "# 步骤 1：加载模型",
        "# 新增了图像处理",
        "# 修改代码以支持视频",
        "# 更新：采用新实现",
        "# 修复了长度截断问题",
        "# 修正：对齐上游接口",
        "# 替换了旧的调度器",
    ],
)
def test_comment_narration_is_rejected(comment: str) -> None:
    source = f"def compute():\n    {comment}\n    return 1\n"
    assert list(findings(COMMENT_PATH, source.encode())) == [
        (2, "CM001", "Remove edit/step narration; keep only necessary contract or rationale.")
    ]


@pytest.mark.parametrize(
    "comment",
    [
        "# ======",
        "# -----",
        "# #####",
        "# ______",
        "# ~~~~~",
        "# ***",
        "# ==== Loading ====",
        "# --- 初始化 ---",
        "# Imports",
        "# Constants",
        "# Helpers",
        "# Utilities",
        "# End of rollout helpers",
    ],
)
def test_decorative_comment_banners_are_rejected(comment: str) -> None:
    result = list(findings(COMMENT_PATH, f"{comment}\nvalue = 1\n".encode()))
    assert [code for _line, code, _message in result] == ["CM002"]


@pytest.mark.parametrize(
    "source",
    [
        "#!/usr/bin/env python3\n",
        "# -*- coding: utf-8 -*-\n",
        "# Copyright 2026 Bytedance Ltd. and/or its affiliates\n",
        "# SPDX-License-Identifier: Apache-2.0\n",
        "# Licensed under the Apache License, Version 2.0\n",
        "# http://www.apache.org/licenses/LICENSE-2.0\n",
        "# fmt: off\nvalue = 1\n# fmt: on\n",
        "value = 1  # noqa: F841\n",
        "# mypy: ignore-errors\n",
        "value = 1  # type: ignore[assignment]\n",
        "value = 1  # [B, C, H, W]\n",
        "# The lock must cover initialization and all readers.\n",
        "# Added tokens are discarded by the tokenizer.\n",
        "# Updated weights are broadcast before inference.\n",
        "# New tokens have no cached embeddings.\n",
        "# New class tokens must remain outside the positional grid.\n",
        "# New class IDs must not reuse the old embedding rows.\n",
        "# Added support vectors must use the training feature space.\n",
        "# 新增类 token 不参与 patch 位置编码。\n",
        "# New function values must stay within the scheduler domain.\n",
        "# 新增方法的输出仍遵循旧有 shape 契约。\n",
        "# Keep fp32 here to match the rollout scheduler contract.\n",
        "# Fixed-size ring buffer avoids reallocations during async rollout.\n",
        "# Fixed seed keeps the SDE window aligned across DP ranks.\n",
        "# Fix the seed so rollout and actor sample identical noise.\n",
        "# Previously computed latents stay in fp32 across microbatches.\n",
        "# Stage 1: AR token generation.\n",
        "# Helpers for building ragged prompt embeddings.\n",
        "value = 1  # helpers\n",
        "# 更新后的权重必须先广播才能推理。\n",
        "# 修复上游问题前先保留此分支。\n",
        "# 用同一把锁保护初始化和读取，避免线程读到半初始化模型。\n",
        "# TODO(owner): remove after upstream issue #123 is fixed.\n",
        "# See https://example.invalid/Step-1/Added-handling\n",
        '"""Return (B, T) log probabilities; missing required inputs raise ValueError."""\n',
        'def compute():\n    """Added handling describes an API change in historical documentation."""\n    return 1\n',
        'message = "# Step 1: load model"\n',
        'source = """# ======\n# Added handling for input\n"""\n',
        'message = f"# Step 1: {model}"\n',
        "value = 1  # ---\n",
    ],
)
def test_necessary_comments_docstrings_and_literals_are_preserved(source: str) -> None:
    assert not list(findings(COMMENT_PATH, source.encode()))


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("verl_omni/pipelines/example/common.py", ["CM001"]),
        ("tests/workers/test_example_on_cpu.py", ["CM001"]),
        ("examples/model/prepare.py", ["CM001"]),
        ("docs/example.py", []),
        ("scripts/local_probe.py", []),
        ("examples/model/run_train.sh", []),
    ],
)
def test_comment_rules_follow_python_source_scope(path: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, b"# Added handling for input\n")] == expected


@pytest.mark.parametrize("source", [b'"""unterminated', b"\xff", b"# coding: unknown-encoding\n", b"def run(\n"])
def test_invalid_source_is_left_to_the_python_syntax_checker(source: bytes) -> None:
    assert not list(findings(COMMENT_PATH, source))


def test_comment_rule_checks_inline_narration_and_real_line_numbers() -> None:
    source = b"\nvalue = 1  # Added handling for input\n# ======\n"
    assert [(line, code) for line, code, _message in findings(COMMENT_PATH, source)] == [(2, "CM001"), (3, "CM002")]


def test_checker_preserves_its_own_license_and_source_literals() -> None:
    checker = Path(__file__).with_name("check_comments.py")
    assert not list(findings("tests/special_sanity/check_comments.py", checker.read_bytes()))
