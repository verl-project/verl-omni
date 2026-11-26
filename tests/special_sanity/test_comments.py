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
        "# Here we normalize the prompt embeddings",
        "# Now we compute the step weights",
        "# Let's validate the shape before decoding",
        "# This function converts latents to pixels",
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
        "# Here the scheduler lets the stage advance by one step.\n",
        "# Avoid PR numbers while keeping rationale comments concise.\n",
        "# Normalize latents before scaling\nlatents = latents / scale\n",
        "# Keep fp32 here\nlatents = latents.float()\n",
        "# Copyright 2026 Bytedance\n# Encode prompts\n",
        (
            "#!/usr/bin/env python3\n# Copyright 2026 Bytedance Ltd.\n#\n"
            "# Licensed under Apache 2.0\n# Adapted from PR #123\nx = 1\n"
        ),
        "# Added to prevent NaN when sigma == 1.0\n",
        "# Changed to fp32 for rollout parity\n",
        "# Removed to avoid OOM during rollout\n",
        "# This loop is intentionally unbatched because shapes vary\n",
        '# Disable gradient checkpointing\nlogger.warning("Disable gradient checkpointing when using LoRA")\n',
        "def compute():\n    # Line 1 why\n    # Line 2 why\n    # Line 3 why\n    pass\n",
        (
            "def step():\n    # DanceGRPO SDE step from https://github.com/XueZeyue/DanceGRPO\n"
            "    # Numerically stable even when sigma is close to 1.\n"
            "    # score_estimate = -(x_t - x_0 * (1 - sigma)) / sigma^2\n"
            "    # prev_sample_mean += log_term * dsigma\n    pass\n"
        ),
        (
            "def collect():\n    # DIFF vs upstream: this wrapper exports LoRA params directly\n"
            "    # without backing up base weights to CPU, keeping peak memory\n"
            "    # flat while synchronizing multi-GPU training ranks.\n"
            "    # Remove once upstream supports streaming exports.\n    pass\n"
        ),
        (
            'def _pad_prompt_tokens(micro_batch, device):\n    """Pad BAGEL prompt_token_ids.\n\n'
            "    Args:\n        micro_batch: Batch containing prompt_token_ids.\n"
            "        device: Target device for padded tensors.\n\n"
            '    Returns:\n        (text_token_ids, mask) with shape (B, max_len).\n    """\n    pass\n'
        ),
        ('def _resolve_sleep_level():\n    """Use sleep level 1 so CUDA buffers free cleanly."""\n    return 1\n'),
        'def _helper():\n    """Short contract.\n\n    Line 2.\n    """\n    pass\n',
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


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (COMMENT_PATH, "# Encode prompts\nprompt_embeds = encode_prompt(prompt)\n", [(1, "CM003")]),
        (COMMENT_PATH, '# model outputs\nmodel_output = output.pop("model_output", {})\n', [(1, "CM003")]),
        (COMMENT_PATH, "# Return log probs\nreturn log_probs\n", [(1, "CM003")]),
        ("verl_omni/trainer/config/example.py", "# Encode prompts\nprompt_embeds = encode_prompt(prompt)\n", []),
        (
            "tests/utils/test_example.py",
            'def test_case():\n    """Necessity: keep this test."""\n    pass\n',
            [(2, "CM004")],
        ),
        (
            COMMENT_PATH,
            'def helper():\n    """Motivation: we needed this."""\n    pass\n',
            [(2, "CM004")],
        ),
        (
            COMMENT_PATH,
            'def normalize():\n    """Helper function to normalize latents."""\n    pass\n',
            [(2, "CM004")],
        ),
        (
            COMMENT_PATH,
            'def parse():\n    """This function is used to parse configs."""\n    pass\n',
            [(2, "CM004")],
        ),
        (
            COMMENT_PATH,
            'def _encode_prompts(prompts):\n    """Encode the prompts."""\n    return prompts\n',
            [(2, "CM004")],
        ),
        (
            COMMENT_PATH,
            'def _run(**kwargs):\n    """Run kwargs."""\n    return kwargs\n',
            [(2, "CM004")],
        ),
        (COMMENT_PATH, "# Keep workaround from PR #942 until upstream fix\nvalue = 1\n", [(1, "CM005")]),
        (COMMENT_PATH, "# TODO: remove workaround from PR #942\nvalue = 1\n", [(1, "CM005")]),
        (COMMENT_PATH, "# NOTE: added in PR #942\nvalue = 1\n", [(1, "CM005")]),
        (COMMENT_PATH, "# TODO: remove after upstream issue #942 is fixed\nvalue = 1\n", []),
        (
            "verl_omni/pipelines/demo.py",
            'def fn():\n    """Implemented in PR #942."""\n    return 1\n',
            [(2, "CM005")],
        ),
        (
            "tests/special_e2e/demo.py",
            '"""Added for PR #942 verification."""\nx = 1\n',
            [],
        ),
        (
            "examples/demo/run_train.sh",
            "#!/usr/bin/env bash\n# Reused from PR #858\necho ok\n",
            [(2, "CM005")],
        ),
        (
            "examples/demo/run_train.sh",
            '#!/usr/bin/env bash\n# Copyright 2026 Bytedance\n# Adapted from PR #123\necho "PR #858"\n'
            "cat <<'EOF'\n# PR #858\nEOF\n",
            [],
        ),
        (
            "examples/demo/run_train.sh",
            '#!/usr/bin/env bash\necho "<<EOF"\ncat <<< "PR #858"\n# Reused from PR #858\n',
            [(4, "CM005")],
        ),
    ],
)
def test_ai_comment_and_docstring_bloat_guards(path: str, source: str, expected: list[tuple[int, str]]) -> None:
    assert [(line, code) for line, code, _msg in findings(path, source.encode())] == expected
