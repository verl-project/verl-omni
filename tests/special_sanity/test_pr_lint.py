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

"""Verify Ruff rules and the combined checker on real PR diffs."""

import ast
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from check_ast_rules import findings as ast_findings
from check_comments import findings as comment_findings
from check_example_recipes import ACCEL_VARIANT_SCRIPT, delegates_to_base_script
from check_example_recipes import findings as example_findings
from check_pr_lint import _dead_function_candidates, _has_external_reference, _ruff_findings

CHECKER = Path(__file__).with_name("check_pr_lint.py").resolve()
CORE_PATH = "verl_omni/trainer/diffusion/example.py"
SHARED_IMPORT = "from verl_omni.pipelines.model_base import DiffusionModelBase"
ADAPTER_IMPORT = "from verl_omni.pipelines.qwen_image_flow_grpo import QwenImage"
QWEN_ROLLOUT = "verl_omni/pipelines/qwen_image_dpo/vllm_omni_rollout_adapter.py"
SDE_HELPER = "verl_omni/pipelines/example/common.py"
REQUEST_SEED = "generator.manual_seed(hash(request.request_id))\n"
MASK_LENGTH = "txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()\n"
SMOKE_PATH = "tests/workers/test_example.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo: Path) -> str:
    _git(repo, "add", "--all")
    _git(repo, "-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "-qm", "fixture")
    return _git(repo, "rev-parse", "HEAD")


def _branching_function(branches: int) -> str:
    body = "".join(f"    if number == {index}:\n        return {index}\n" for index in range(branches))
    return f"\ndef dispatch(number):\n{body}    return -1\n"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI check policy workflow supplies pinned Ruff.")
@pytest.mark.parametrize(
    ("base_source", "head_source", "expected_codes"),
    [
        ("import os\n", "\nimport os\n", set()),
        ("import os\n", "import os\nimport sys  # noqa: F401\n", {"F401"}),
        ("", f"{ADAPTER_IMPORT}\n", {"VO001", "F401"}),
        ("", f"{SHARED_IMPORT}\n\nTRAINER_BASE = DiffusionModelBase\n", set()),
        ("", _branching_function(15), {"C901"}),
        (_branching_function(15), _branching_function(16), {"C901"}),
        (
            "def first():\n    unused = 1\n\ndef second():\n    return 2\n",
            "def first():\n    return 1\n\ndef second():\n    unused = 1\n    return 2\n",
            {"F841"},
        ),
        ("", "def invalid(\n", {"E999"}),
        ("", "try:\n    setup()\nexcept Exception:\n    pass\n", {"S110"}),
        ("", "for item in items:\n    try:\n        setup(item)\n    except Exception:\n        continue\n", {"S112"}),
        ("", "try:\n    cleanup()\nexcept FileNotFoundError:\n    pass\n", set()),
        ("", "try:\n    setup()\nexcept Exception:\n    raise\n", set()),
    ],
)
def test_combined_checker_compares_base_and_head(
    tmp_path: Path, base_source: str, head_source: str, expected_codes: set[str]
) -> None:
    _git(tmp_path, "init", "-q")
    source_path = tmp_path / CORE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text(base_source, encoding="utf-8")
    base = _commit(tmp_path)
    source_path.write_text(head_source, encoding="utf-8")
    (tmp_path / "ruff.toml").write_text('[lint]\nignore = ["F401", "C901"]\n', encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == int(bool(expected_codes)), result.stdout + result.stderr
    assert {line.split()[1] for line in result.stdout.splitlines() if CORE_PATH in line} == expected_codes


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI check policy workflow supplies pinned Ruff.")
def test_check_without_refs_reads_the_index_not_the_worktree(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    source_path = tmp_path / CORE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text("", encoding="utf-8")
    _commit(tmp_path)
    source_path.write_text("import os\n", encoding="utf-8")
    _git(tmp_path, "add", CORE_PATH)
    source_path.write_text("", encoding="utf-8")
    result = subprocess.run([sys.executable, str(CHECKER)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "F401" in result.stdout


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI check policy workflow supplies pinned Ruff.")
def test_rollout_findings_are_independent_across_changed_files(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    legacy = tmp_path / QWEN_ROLLOUT.replace("qwen_image_dpo", "qwen_image_diffusion_nft")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(MASK_LENGTH, encoding="utf-8")
    base = _commit(tmp_path)

    rollout = tmp_path / QWEN_ROLLOUT
    helper = tmp_path / SDE_HELPER
    for target, source in ((rollout, MASK_LENGTH), (helper, REQUEST_SEED)):
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")
    legacy.write_text(f"\n{MASK_LENGTH}", encoding="utf-8")

    for expected, target, fix in (
        ({(SDE_HELPER, "VO004"), (QWEN_ROLLOUT, "VO005")}, helper, "generator.manual_seed(window_seed)\n"),
        ({(QWEN_ROLLOUT, "VO005")}, rollout, "txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)\n"),
        (set(), None, ""),
    ):
        head = _commit(tmp_path)
        result = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode == int(bool(expected)), result.stdout + result.stderr
        findings = [line.split(maxsplit=2) for line in result.stdout.splitlines() if line.startswith("'")]
        assert {(location.split("'")[1], code) for location, code, _message in findings} == expected
        if target is not None:
            target.write_text(fix, encoding="utf-8")


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI check policy workflow supplies pinned Ruff.")
def test_ci_lint_applies_distinct_rules_to_production_and_test_files(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    smoke_path = "tests/workers/test_example_cpu.py"
    smoke = tmp_path / smoke_path
    smoke.parent.mkdir(parents=True)
    smoke.with_name("test_example_on_cpu.py").write_text("", encoding="utf-8")
    base = _commit(tmp_path)
    production = tmp_path / CORE_PATH
    production.parent.mkdir(parents=True)
    production.write_text("try:\n    setup()\nexcept Exception:\n    pass\n", encoding="utf-8")
    smoke.write_text('import os\nimport pytest\n\nif __name__ == "__main__":\n    pytest.main()\n', encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"'{CORE_PATH}':3: S110" in result.stdout
    assert f"'{smoke_path}':5: VO006" in result.stdout
    assert f"'{smoke_path}':1: VO007" in result.stdout
    assert "F401" not in result.stdout


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("try:\n    work()\nexcept:\n    recover()\n", "E722"),
        ("def append(items=[]):\n    return items\n", "B006"),
        ("def work():\n    try:\n        risky()\n    finally:\n        return 1\n", "B012"),
        ("def work():\n    123\n", "B018"),
        ("callbacks = []\nfor number in range(3):\n    callbacks.append(lambda: number)\n", "B023"),
        ('__all__ = ["missing"]\n', "F822"),
        ('__all__ = ["missing", 3]\n', "PLE0604"),
        ('__all__ = "missing"\n', "PLE0605"),
        ("class Example:\n    def __init__(self):\n        return 2\n", "PLE0101"),
        ("value = value\n", "PLW0127"),
        ('import subprocess\nsubprocess.run(["true"])\n', "PLW1510"),
        ("class Example:\n    value = 1\n    value = 2\n", "PIE794"),
        ("def work(value, value):\n    return value\n", "E999"),
        ('handle = open("input.txt")\nhandle.close()\n', "SIM115"),
        ("import asyncio\nasyncio.create_task(work())\n", "RUF006"),
        ('result = eval("1 + 1")\n', "S307"),
        ('import subprocess\nsubprocess.run("echo value", shell=True, check=True)\n', "S602"),
        ("breakpoint()\n", "T100"),
        ("state = None\ndef work():\n    global state\n    state = 1\n", "PLW0603"),
        ("# result = legacy_compute()\nresult = compute()\n", "ERA001"),
    ],
)
def test_new_ruff_rules_reject_known_pitfalls(source: str, code: str) -> None:
    assert code in {finding.code for finding in _ruff_findings(CORE_PATH, source.encode())}


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
@pytest.mark.parametrize(
    "source",
    [
        "def append(items=None):\n    return [] if items is None else items\n",
        "def work():\n    try:\n        return risky()\n    finally:\n        cleanup()\n",
        "callbacks = []\nfor number in range(3):\n    callbacks.append(lambda number=number: number)\n",
        'import subprocess\nsubprocess.run(["true"], check=True)\n',
        'import subprocess\nstatus = subprocess.run(["true"], check=False).returncode\n',
        "import asyncio\nasync def run():\n    task = asyncio.create_task(work())\n    await task\n",
        'with open("input.txt") as handle:\n    data = handle.read()\n',
        'import ast\nvalue = ast.literal_eval("[1, 2]")\n',
        'import subprocess\nsubprocess.run(["echo", "value"], check=True)\n',
        "# Keep fp32 here to match the rollout scheduler contract.\nvalue = compute()\n",
        'def score():\n    """Return a scalar score; invalid media raises ValueError."""\n    return compute()\n',
    ],
)
def test_new_ruff_rules_accept_explicit_resource_and_error_handling(source: str) -> None:
    assert not list(_ruff_findings(CORE_PATH, source.encode()))


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_ruff_keeps_reward_cache_and_test_scopes_distinct() -> None:
    cache = b"state = None\ndef work():\n    global state\n    state = 1\n"
    assert not list(_ruff_findings("verl_omni/utils/reward_score/example.py", cache))
    assert "PLW0603" in {finding.code for finding in _ruff_findings(CORE_PATH, cache)}
    source = b"import os\ndef test_example():\n    assert False\n    breakpoint()\n"
    assert {finding.code for finding in _ruff_findings(SMOKE_PATH, source)} == {"PT015", "T100"}
    assert not list(_ruff_findings(SMOKE_PATH, b"def test_example():\n    assert work() == 1\n"))


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
@pytest.mark.parametrize(
    ("root_import", "expected"),
    [
        ("", {"VO010"}),
        ("from . import example as example\n", set()),
        ("from .example import Adapter as Adapter\n", set()),
        ('import verl_omni.pipelines.example as example\n__all__ = ["example"]\n', set()),
        ("from verl_omni.pipelines import example as example\n", set()),
        ("from .example.training import Adapter as Adapter\n", set()),
        ("try:\n    from . import example as example\nexcept ImportError:\n    pass\n", set()),
        ("def unused():\n    from . import example\n    return example\n", {"VO010"}),
        ("if TYPE_CHECKING:\n    from . import example as example\n", {"VO010"}),
        ('"from . import example"\n', {"VO010"}),
    ],
)
def test_pipeline_package_must_be_imported_from_entrypoint(
    tmp_path: Path, root_import: str, expected: set[str]
) -> None:
    _git(tmp_path, "init", "-q")
    pipelines = tmp_path / "verl_omni/pipelines"
    pipelines.mkdir(parents=True)
    entrypoint = pipelines / "__init__.py"
    entrypoint.write_text("", encoding="utf-8")
    base = _commit(tmp_path)
    package_init = pipelines / "example/__init__.py"
    package_init.parent.mkdir()
    package_init.write_text("", encoding="utf-8")
    entrypoint.write_text(root_import, encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == int(bool(expected)), result.stdout + result.stderr
    assert {line.split()[1] for line in result.stdout.splitlines() if "VO010" in line} == expected


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_pipeline_registration_checks_deletion_and_legacy_baselines(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    pipelines = tmp_path / "verl_omni/pipelines"
    for package in ("example", "legacy", "schedulers"):
        package_init = pipelines / package / "__init__.py"
        package_init.parent.mkdir(parents=True)
        package_init.write_text("", encoding="utf-8")
    entrypoint = pipelines / "__init__.py"
    entrypoint.write_text("from . import example\n", encoding="utf-8")
    base = _commit(tmp_path)
    entrypoint.write_text("", encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "VO010" in result.stdout and "/example/__init__.py" in result.stdout
    assert "/legacy/" not in result.stdout and "/schedulers/" not in result.stdout


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_pipeline_scan_never_imports_candidate_packages(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "base.txt").write_text("base", encoding="utf-8")
    base = _commit(tmp_path)
    pipelines = tmp_path / "verl_omni/pipelines"
    package = pipelines / "example"
    package.mkdir(parents=True)
    (pipelines / "__init__.py").write_text("from . import example as example\n", encoding="utf-8")
    (package / "__init__.py").write_text('from pathlib import Path\nPath("executed").touch()\n', encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "executed").exists()


def test_staged_example_is_checked_without_executing_head_code(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "base.txt").write_text("base", encoding="utf-8")
    _commit(tmp_path)
    recipe = tmp_path / "examples/demo/run_demo.sh"
    recipe.parent.mkdir(parents=True)
    recipe.write_text('touch executed\necho "no forwarding"\n', encoding="utf-8")
    _git(tmp_path, "add", "examples/")
    recipe.write_text('touch executed\ntrainer "$@"\n', encoding="utf-8")
    result = subprocess.run([sys.executable, str(CHECKER)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "EX002" in result.stdout
    assert not (tmp_path / "executed").exists()


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_one_pr_reports_all_layers_and_turns_green_after_fixes(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "base.txt").write_text("base", encoding="utf-8")
    base = _commit(tmp_path)
    changes = {
        CORE_PATH: ('if model == "qwen_image":\n    model.run()\n', "model.run()\n"),
        "verl_omni/utils/example.py": (
            "from verl_omni.tools.other import _helper\n_helper()\n",
            "from verl_omni.tools.other import helper\nhelper()\n",
        ),
        "verl_omni/pipelines/example/training.py": (
            "class Model(DiffusionModelBase):\n    def __init__(self):\n        pass\n",
            "class Model(DiffusionModelBase):\n    @classmethod\n    def forward(cls):\n        pass\n",
        ),
        "verl_omni/pipelines/example/__init__.py": ("", ""),
        "verl_omni/pipelines/__init__.py": ("", "from . import example as example\n"),
        "verl_omni/workers/config/example.py": (
            "@dataclass\nclass ExampleConfig:\n    pass\n",
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass\n',
        ),
        "verl_omni/reward_loop/reward_manager/example.py": (
            "import httpx\nhttpx.get(endpoint)\n",
            "score_batch()\n",
        ),
        "tests/workers/test_sample_on_cpu.py": (
            "import pytest\nimport torch\n"
            'pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")\n',
            'import pytest\npytestmark = pytest.mark.skipif(enabled_gpu_count() < 2, reason="two GPUs")\n',
        ),
        "examples/model/config.yaml": ('model: "/home/alice/weights"\n', "model: org/model\n"),
        "examples/model/run_train.sh": (
            "python trainer.py\n",
            '#!/usr/bin/env bash\nset -euo pipefail\npython trainer.py "$@"\n',
        ),
        "examples/model/eval.sh": (".venv/bin/python eval.py\n", "python3 eval.py\n"),
        "examples/model/setup.sh": ("if true; then\n", "true\n"),
        "examples/model/README_training.md": ("documentation", None),
        "examples/model/reward.png": ("image", None),
    }
    for name, (source, _fix) in changes.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    head = _commit(tmp_path)
    expected_codes = {"VO003", "VO008", "VO009", "VO010", "VO011", "VO012", "VO013", "VO014"}
    expected_codes.update({"EX001", "EX002", "EX003", "EX004", "EX005", "EX006"})
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert {line.split()[1] for line in result.stdout.splitlines()} == expected_codes

    for name, (_source, fix) in changes.items():
        target = tmp_path / name
        if fix is None:
            target.unlink()
        else:
            target.write_text(fix, encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No new code findings." in result.stdout


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_comment_checks_preserve_history_and_reject_new_noise_in_a_pr(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    target = tmp_path / CORE_PATH
    target.parent.mkdir(parents=True)
    legacy = "def legacy():\n    # Step 1: run legacy path\n    return 1\n"
    target.write_text(legacy, encoding="utf-8")
    base = _commit(tmp_path)
    noise = (
        "\ndef added():\n"
        "    # Added handling for the new field\n"
        "    # Here we compute latents\n"
        "    # See PR #942\n"
        "    # ======\n"
        "    # value = old_compute()\n"
        "    return 2\n"
    )
    target.write_text("\n" + legacy + noise, encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert {line.split()[1] for line in result.stdout.splitlines()} == {"CM001", "CM002", "CM005", "ERA001"}
    assert len(result.stdout.splitlines()) == 5

    clean = (
        '\ndef added():\n    """Return a scalar score; missing input raises ValueError."""\n'
        "    # Keep fp32 here to match the rollout scheduler contract.\n"
        "    return 2\n"
    )
    target.write_text("\n" + legacy + clean, encoding="utf-8")
    head = _commit(tmp_path)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No new code findings." in result.stdout


def test_live_repo_baseline_respects_precision_bounds() -> None:
    root = CHECKER.parents[2]
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "verl_omni", "examples"],
        text=True,
    ).splitlines()
    counts: Counter[str] = Counter()
    for rel in tracked:
        raw = (root / rel).read_bytes()
        if rel.startswith("examples/") or rel.endswith((".py", ".sh", ".yaml", ".yml")):
            counts.update(code for _ln, code, _msg in example_findings(rel, raw))
        if rel.startswith("examples/") and rel.endswith(".sh"):
            counts.update(code for _ln, code, _msg in comment_findings(rel, raw))
        if not rel.endswith(".py"):
            continue
        tree = ast.parse(raw)
        counts.update(code for _ln, code, _msg in ast_findings(rel, tree))
        counts.update(code for _ln, code, _msg in comment_findings(rel, raw, tree=tree))

    zero_false_positive_rules = {
        "VO002",
        "VO004",
        "VO006",
        "VO007",
        "VO008",
        "VO009",
        "VO012",
        "VO013",
        "VO014",
        "VO016",
        "CM004",
        "CM005",
        "EX001",
        "EX002",
        "EX003",
        "EX004",
        "EX007",
    }
    unexpected = {code: counts[code] for code in sorted(zero_false_positive_rules) if counts[code]}
    assert not unexpected, f"Unexpected live-repo hits: {unexpected}"
    assert counts["VO001"] <= 2
    assert counts["VO011"] <= 2
    assert counts["VO015"] <= 6
    assert counts["VO018"] <= 96
    assert counts["CM001"] <= 7
    assert counts["CM003"] <= 15
    assert counts["EX005"] <= 84
    assert counts["EX006"] <= 5
    assert counts["EX008"] <= 1
    dead_hits = []
    for rel in tracked:
        if not (rel.startswith("verl_omni/") and rel.endswith(".py") and not rel.endswith("/__init__.py")):
            continue
        for ln, name, kind in _dead_function_candidates((root / rel).read_bytes()):
            if not _has_external_reference("HEAD", rel, name, root=str(root)):
                dead_hits.append((rel, ln, name, kind))
    assert not dead_hits, f"Unexpected live-repo dead functions: {dead_hits}"
    tracked_set = set(tracked)
    variant_copies = 0
    for rel in tracked:
        rel_path = Path(rel)
        match = ACCEL_VARIANT_SCRIPT.match(rel_path.name)
        if not match:
            continue
        base_name = match.group(1) + (match.group(2) or "") + ".sh"
        base_rel = (rel_path.parent / base_name).as_posix()
        if base_rel in tracked_set and not delegates_to_base_script((root / rel).read_bytes(), base_name):
            variant_copies += 1
    assert variant_copies <= 17

    test_files = subprocess.check_output(["git", "-C", str(root), "ls-files", "tests"], text=True).splitlines()
    test_counts: Counter[str] = Counter()
    for rel in test_files:
        if not rel.endswith(".py"):
            continue
        raw = (root / rel).read_bytes()
        test_counts.update(code for _ln, code, _msg in example_findings(rel, raw))
        tree = ast.parse(raw)
        test_counts.update(code for _ln, code, _msg in ast_findings(rel, tree))
        test_counts.update(code for _ln, code, _msg in comment_findings(rel, raw, tree=tree))
    assert not {
        code: test_counts[code]
        for code in ("VO006", "VO007", "VO013", "VO014", "VO017", "CM001", "CM005")
        if test_counts[code]
    }
    assert test_counts["CM004"] <= 7
    assert test_counts["CM002"] <= 170


RULE_MATRIX: dict[str, tuple[str, bytes, str, bytes]] = {
    "VO001": (CORE_PATH, ADAPTER_IMPORT.encode(), CORE_PATH, SHARED_IMPORT.encode()),
    "VO002": (
        "verl_omni/workers/rollout/ex.py",
        b"from verl_omni.workers.engine.fsdp import Engine\n",
        "verl_omni/workers/rollout/ex.py",
        b"from verl_omni.pipelines.diffusion_rollout_output import DiffusionRolloutOutput\n",
    ),
    "VO003": (
        "verl_omni/workers/ex.py",
        b"from verl_omni.trainer.diffusion import _helper\n",
        "verl_omni/workers/ex.py",
        b"from verl_omni.trainer.diffusion import helper\n",
    ),
    "VO004": (
        "verl_omni/pipelines/ex/rollout.py",
        REQUEST_SEED.encode(),
        "verl_omni/pipelines/ex/rollout.py",
        b"generator.manual_seed(seed)\n",
    ),
    "VO005": (
        QWEN_ROLLOUT,
        MASK_LENGTH.encode(),
        QWEN_ROLLOUT,
        b"txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)\n",
    ),
    "VO006": (
        SMOKE_PATH,
        b"import pytest\npytest.main([__file__])\n",
        SMOKE_PATH,
        b"import pytest\nraise SystemExit(pytest.main([__file__]))\n",
    ),
    "VO007": (
        SMOKE_PATH,
        b"import pytest\npytestmark = pytest.mark.cpu\n",
        "tests/workers/test_example_on_cpu.py",
        b"import pytest\npytestmark = pytest.mark.cpu\n",
    ),
    "VO008": (CORE_PATH, b'if model == "qwen_image":\n    run()\n', CORE_PATH, b'if mode == "video":\n    run()\n'),
    "VO009": (
        "verl_omni/pipelines/ex/training_adapter.py",
        b"class Adapter(DiffusionModelBase):\n    def __init__(self):\n        pass\n",
        "verl_omni/pipelines/ex/training_adapter.py",
        b"class Adapter(DiffusionModelBase):\n    @classmethod\n    def step(cls):\n        pass\n",
    ),
    "VO010": (
        "verl_omni/pipelines/demo_pipe/__init__.py",
        b"value = 1\n",
        "verl_omni/pipelines/__init__.py",
        b"from . import demo_pipe as demo_pipe\n",
    ),
    "VO011": (
        "verl_omni/workers/config/diffusion/ex.py",
        b"@dataclass\nclass Cfg:\n    pass\n",
        "verl_omni/workers/config/diffusion/ex.py",
        b'__all__ = ["Cfg"]\n@dataclass\nclass Cfg(BaseConfig):\n    steps: int = 1\n',
    ),
    "VO012": (
        "verl_omni/reward_loop/reward_manager/ex.py",
        b"import aiohttp\n",
        "verl_omni/reward_loop/reward_manager/ex.py",
        b"from urllib.parse import urlparse\n",
    ),
    "VO013": (
        "tests/workers/test_example_on_cpu.py",
        b'import pytest, torch\npytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")\n',
        "tests/workers/test_example_on_cpu.py",
        b'import pytest\npytest.mark.skipif(enabled_gpu_count() < 1, reason="gpu")\n',
    ),
    "VO014": (
        "examples/demo/train.py",
        b'path = "/Users/alice/weights"\n',
        "examples/demo/train.py",
        b'path = "weights"\n',
    ),
    "VO015": (
        "verl_omni/utils/reward_score/custom.py",
        b"def compute_score_demo() -> float:\n    return 0.0\n",
        "verl_omni/utils/reward_score/custom.py",
        b'def compute_score_demo() -> float:\n    """Score demo sample."""\n    return 0.0\n',
    ),
    "VO016": (
        "verl_omni/utils/reward_score/custom.py",
        b"try:\n    run()\nexcept Exception:\n    return 0.0\n",
        "verl_omni/utils/reward_score/custom.py",
        b"try:\n    run()\nexcept ArtifactContractError:\n    raise\nexcept Exception:\n    return 0.0\n",
    ),
    "VO017": (
        SMOKE_PATH,
        b"from pathlib import Path\nADAPTER = Path('verl_omni/pipelines/ex/a.py')\n"
        b"def test_flag():\n    assert 'foo' in ADAPTER.read_text()\n",
        SMOKE_PATH,
        b"def test_flag():\n    assert build_flag() == 'foo'\n",
    ),
    "VO018": (
        "verl_omni/utils/custom.py",
        b'__all__ = ["scale"]\ndef scale(x):\n    """Scale input."""\n    return x\n',
        "verl_omni/utils/custom.py",
        b'__all__ = ["scale"]\ndef scale(x: float) -> float:\n    """Scale input."""\n    return x\n',
    ),
    "VO019": (
        "verl_omni/utils/demo.py",
        b"def _unused_helper() -> None:\n    pass\n",
        "verl_omni/utils/demo.py",
        b"def _unused_helper() -> None:\n    pass\n\ndef run() -> None:\n    _unused_helper()\n",
    ),
    "CM001": (
        CORE_PATH,
        b"# Added handling for input\nvalue = 1\n",
        CORE_PATH,
        b"# Keep fp32 for scheduler parity.\nvalue = 1\n",
    ),
    "CM002": (CORE_PATH, b"# ======\nvalue = 1\n", CORE_PATH, b"value = 1\n"),
    "CM003": (
        CORE_PATH,
        b"# Encode prompts\nprompt_embeds = encode_prompt(prompt)\n",
        CORE_PATH,
        b"prompt_embeds = encode_prompt(prompt)\n",
    ),
    "CM004": (
        CORE_PATH,
        b'def fn():\n    """Helper function to compute value."""\n    return 1\n',
        CORE_PATH,
        b'def fn():\n    """Return computed value."""\n    return 1\n',
    ),
    "CM005": (
        CORE_PATH,
        b"# Workaround for PR #942\nvalue = 1\n",
        CORE_PATH,
        b"# Workaround for issue #942\nvalue = 1\n",
    ),
    "EX001": (
        "examples/demo/run_train.sh",
        b"if [\n",
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\nexec python train.py "$@"\n',
    ),
    "EX002": (
        "examples/demo/run_train.sh",
        b"#!/usr/bin/env bash\nset -euo pipefail\npython train.py\n",
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\nexec python train.py "$@"\n',
    ),
    "EX003": ("examples/demo/sample.png", b"\x89PNG\r\n", "examples/demo/train.py", b"x = 1\n"),
    "EX004": ("examples/demo/README_en.md", b"# Demo\n", "examples/demo/README.md", b"# Demo\n"),
    "EX005": (
        "examples/demo/run_train.sh",
        b'python train.py "$@"\n',
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\npython train.py "$@"\n',
    ),
    "EX006": (
        "examples/demo/dataprocess/prepare.py",
        b"x = 1\n",
        "examples/demo/data_process/prepare.py",
        b"x = 1\n",
    ),
    "EX007": (
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\npython train.py trainer.total_epochs=1 trainer.total_epochs=2 "$@"\n',
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\npython train.py trainer.total_epochs=1 "$@"\n',
    ),
    "EX008": (
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\npkill -9 ray\npython train.py "$@"\n',
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\npython train.py "$@"\n',
    ),
    "EX009": (
        "examples/demo/run_train_npu.sh",
        b"#!/usr/bin/env bash\nset -euo pipefail\n"
        b'python3 -m verl_omni.trainer.main_diffusion trainer.device=npu "$@"\n',
        "examples/demo/run_train.sh",
        b'#!/usr/bin/env bash\nset -euo pipefail\npython3 -m verl_omni.trainer.main_diffusion "$@"\n',
    ),
}


def _declared_rule_codes() -> set[str]:
    codes: set[str] = set()
    for name in ("check_pr_lint.py", "check_ast_rules.py", "check_comments.py", "check_example_recipes.py"):
        text = CHECKER.with_name(name).read_text(encoding="utf-8")
        codes.update(re.findall(r'"((?:VO|CM|EX)\d{3})"', text))
    return codes


def test_rule_matrix_covers_every_declared_code() -> None:
    assert set(RULE_MATRIX) == _declared_rule_codes()


@pytest.mark.parametrize("code", sorted(RULE_MATRIX))
def test_every_declared_rule_code_triggers_and_clears_on_minimal_fix(tmp_path: Path, code: str) -> None:
    bad_path, bad_bytes, good_path, good_bytes = RULE_MATRIX[code]
    if code == "VO010":
        if shutil.which("ruff") is None:
            pytest.skip("The CI workflow supplies pinned Ruff.")
        _git(tmp_path, "init", "-q")
        init_file = tmp_path / good_path
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text("", encoding="utf-8")
        base = _commit(tmp_path)
        adapter = tmp_path / bad_path
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_bytes(bad_bytes)
        head = _commit(tmp_path)
        bad_run = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert bad_run.returncode == 1
        assert "VO010" in bad_run.stdout
        init_file.write_bytes(good_bytes)
        head = _commit(tmp_path)
        good_run = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert good_run.returncode == 0, good_run.stdout + good_run.stderr
        return
    if code == "VO019":
        if shutil.which("ruff") is None:
            pytest.skip("The CI workflow supplies pinned Ruff.")
        _git(tmp_path, "init", "-q")
        init_file = tmp_path / "verl_omni/utils/__init__.py"
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text("", encoding="utf-8")
        base = _commit(tmp_path)
        target = tmp_path / bad_path
        target.write_bytes(bad_bytes)
        head = _commit(tmp_path)
        bad_run = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert bad_run.returncode == 1, bad_run.stdout + bad_run.stderr
        assert "VO019" in bad_run.stdout
        target.write_bytes(good_bytes)
        head = _commit(tmp_path)
        good_run = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert good_run.returncode == 0, good_run.stdout + good_run.stderr
        return
    if code == "EX009":
        _git(tmp_path, "init", "-q")
        base_file = tmp_path / good_path
        base_file.parent.mkdir(parents=True, exist_ok=True)
        base_file.write_bytes(good_bytes)
        base = _commit(tmp_path)
        variant = tmp_path / bad_path
        variant.write_bytes(bad_bytes)
        head = _commit(tmp_path)
        bad_run = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert bad_run.returncode == 1
        assert "EX009" in bad_run.stdout
        variant.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"\n'
            'exec bash "$SCRIPT_DIR/run_train.sh" trainer.device=npu "$@"\n',
            encoding="utf-8",
        )
        head = _commit(tmp_path)
        good_run = subprocess.run(
            [sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True
        )
        assert good_run.returncode == 0, good_run.stdout + good_run.stderr
        return

    def collect(path: str, source: bytes) -> list[str]:
        found = [item_code for _ln, item_code, _msg in example_findings(path, source)]
        if path.endswith(".py"):
            tree = ast.parse(source)
            found.extend(item_code for _ln, item_code, _msg in ast_findings(path, tree))
            found.extend(item_code for _ln, item_code, _msg in comment_findings(path, source, tree=tree))
        return found

    assert code in collect(bad_path, bad_bytes)
    assert code not in collect(good_path, good_bytes)


def test_accelerator_variant_recipes_must_wrap_existing_gpu_base(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    demo = tmp_path / "examples/demo"
    demo.mkdir(parents=True)
    gpu_base = demo / "run_demo.sh"
    gpu_base.write_text('#!/usr/bin/env bash\nset -euo pipefail\npython3 -m verl_omni.trainer.main_diffusion "$@"\n')
    legacy_gpu = demo / "run_legacy.sh"
    legacy_gpu.write_text('#!/usr/bin/env bash\nset -euo pipefail\npython3 -m verl_omni.trainer.main_diffusion "$@"\n')
    legacy_npu = demo / "run_legacy_npu.sh"
    legacy_npu.write_text('#!/usr/bin/env bash\nset -euo pipefail\npython3 -m verl_omni.trainer.main_diffusion "$@"\n')
    base = _commit(tmp_path)

    for suffix in ("npu", "mlu", "amd", "rocm"):
        (demo / f"run_demo_{suffix}.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\npython3 -m verl_omni.trainer.main_diffusion "$@"\n',
            encoding="utf-8",
        )
    head = _commit(tmp_path)
    bad = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert bad.stdout.count("EX009") == 4
    assert "run_legacy_npu.sh" not in bad.stdout

    for suffix in ("npu", "mlu", "amd", "rocm"):
        (demo / f"run_demo_{suffix}.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f'exec bash "$SCRIPT_DIR/run_demo.sh" trainer.device={suffix} "$@"\n',
            encoding="utf-8",
        )
    (demo / "run_solo_npu.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\npython3 -m verl_omni.trainer.main_diffusion trainer.device=npu "$@"\n',
        encoding="utf-8",
    )
    head = _commit(tmp_path)
    good = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_baseline_preserves_line_shifted_duplicates_subdir_cwd_latin1_unborn_and_renames(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    sub = tmp_path / "tests"
    sub.mkdir(parents=True)
    # Staged mode works on an unborn HEAD before any commit exists.
    (sub / "test_unborn_cpu.py").write_text("", encoding="utf-8")
    _git(tmp_path, "add", "tests/test_unborn_cpu.py")
    unborn = subprocess.run([sys.executable, str(CHECKER)], cwd=sub, capture_output=True, text=True)
    assert unborn.returncode == 1
    assert "VO007" in unborn.stdout
    (sub / "test_unborn_cpu.py").unlink()
    _git(tmp_path, "reset", "-q")

    recipe = tmp_path / "examples/demo/run_dup.sh"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 t.py \\\n"
        '    trainer.total_epochs=1 \\\n    trainer.total_epochs=2 "$@"\n',
        encoding="utf-8",
    )
    legacy_npu = tmp_path / "examples/old/run_legacy_npu.sh"
    legacy_npu.parent.mkdir(parents=True)
    (tmp_path / "examples/old/run_legacy.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\npython3 t.py "$@"\n', encoding="utf-8"
    )
    legacy_npu.write_text('#!/usr/bin/env bash\nset -euo pipefail\npython3 t.py "$@"\n', encoding="utf-8")
    core = tmp_path / CORE_PATH
    core.parent.mkdir(parents=True)
    core.write_text("import os\nimport os\n", encoding="utf-8")
    base = _commit(tmp_path)

    # Line shifts above EX007/F811 preserve baseline; latin-1 yields E999 cleanly; renames keep legacy EX009 quiet.
    recipe.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n# comment shift\npython3 t.py \\\n"
        '    trainer.total_epochs=1 \\\n    trainer.total_epochs=2 "$@"\n',
        encoding="utf-8",
    )
    core.write_text("# shift\nimport os\nimport os\n", encoding="utf-8")
    moved = tmp_path / "examples/new/run_legacy_npu.sh"
    moved.parent.mkdir(parents=True)
    _git(tmp_path, "mv", "examples/old/run_legacy_npu.sh", "examples/new/run_legacy_npu.sh")
    latin1 = tmp_path / "verl_omni/trainer/diffusion/latin1_file.py"
    latin1.write_bytes(b"# -*- coding: latin-1 -*-\nx = 'caf\xe9'\n")
    head = _commit(tmp_path)

    # Running from subdirectory catches latin-1 E999 without spurious EX007/F811/EX009.
    res = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=sub, capture_output=True, text=True)
    assert res.returncode == 1, res.stdout + res.stderr
    lines = res.stdout.splitlines()
    assert len(lines) == 1 and "latin1_file.py" in lines[0] and "E999" in lines[0]


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_moved_decorated_skipif_is_not_masked_by_baseline(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    test_file = tmp_path / "tests/workers/test_dec_on_cpu.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "import pytest\nimport torch\n\n"
        '@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")\n'
        "def test_a():\n    pass\n\ndef test_b():\n    pass\n",
        encoding="utf-8",
    )
    base = _commit(tmp_path)
    test_file.write_text(
        "import pytest\nimport torch\n\ndef test_a():\n    pass\n\n"
        '@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")\n'
        "def test_b():\n    pass\n",
        encoding="utf-8",
    )
    head = _commit(tmp_path)
    res = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 1
    assert "VO013" in res.stdout


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_package_init_adapter_import_and_shell_pr_ref_caught_end_to_end(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    pipelines = tmp_path / "verl_omni/pipelines"
    demo = pipelines / "demo"
    demo.mkdir(parents=True)
    (pipelines / "__init__.py").write_text("from . import demo as demo\n", encoding="utf-8")
    (demo / "__init__.py").write_text("", encoding="utf-8")
    base = _commit(tmp_path)

    adapter = demo / "talker_adapter.py"
    adapter.write_text(
        '@OmniModelBase.register("Demo", stage="talker")\n'
        'class DemoTalkerAdapter(OmniModelBase):\n    """Talker adapter."""\n    pass\n',
        encoding="utf-8",
    )
    recipe = tmp_path / "examples/demo/run_train.sh"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n# Reused from PR #858\npython3 t.py "$@"\n',
        encoding="utf-8",
    )
    head = _commit(tmp_path)
    bad = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "VO010" in bad.stdout and "talker_adapter.py" in bad.stdout
    assert "CM005" in bad.stdout and "run_train.sh" in bad.stdout

    (demo / "__init__.py").write_text("from .talker_adapter import DemoTalkerAdapter as DemoTalkerAdapter\n")
    recipe.write_text('#!/usr/bin/env bash\nset -euo pipefail\npython3 t.py "$@"\n', encoding="utf-8")
    head = _commit(tmp_path)
    good = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_vo019_detects_unused_private_and_unexported_functions_across_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    pkg = tmp_path / "verl_omni/utils"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("def _init_helper() -> None:\n    pass\n", encoding="utf-8")
    base = _commit(tmp_path)

    mod = pkg / "parity.py"
    mod.write_text(
        '__all__ = ["kept"]\n'
        'def kept() -> None:\n    """Keep."""\n    pass\n'
        "def unused_unexported_helper() -> None:\n    pass\n"
        "def _dead_private() -> None:\n    pass\n"
        'def _self_rec(n: int) -> int:\n    """Recurse _self_rec."""\n    return n if n <= 0 else _self_rec(n - 1)\n'
        '@register("hook")\ndef _decorated() -> None:\n    pass\n'
        "def __custom_dunder__() -> None:\n    pass\n"
        "def main() -> None:\n    pass\n",
        encoding="utf-8",
    )
    meta_test = tmp_path / "tests/special_sanity/test_meta.py"
    meta_test.parent.mkdir(parents=True)
    meta_test.write_text('SAMPLE = "unused_unexported_helper _dead_private"\n', encoding="utf-8")
    open_mod = pkg / "open_api.py"
    open_mod.write_text("def unreferenced_public_without_all() -> None:\n    pass\n", encoding="utf-8")
    dyn_mod = pkg / "dyn.py"
    dyn_mod.write_text(
        "__all__ = list(other.__all__)\ndef unreferenced_with_dynamic_all() -> None:\n    pass\n",
        encoding="utf-8",
    )
    head = _commit(tmp_path)
    bad = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    lines = bad.stdout.splitlines()
    assert len(lines) == 3, bad.stdout
    assert any("VO019" in ln and "unused_unexported_helper" in ln for ln in lines)
    assert any("VO019" in ln and "_dead_private" in ln for ln in lines)
    assert any("VO019" in ln and "_self_rec" in ln for ln in lines)

    # Wiring callers locally, from tests/, or adding to __all__ clears VO019.
    mod.write_text(
        '__all__ = ["kept", "unused_unexported_helper"]\n'
        'def kept() -> None:\n    """Keep."""\n    _self_rec(1)\n'
        'def unused_unexported_helper() -> None:\n    """Export."""\n    pass\n'
        "def _dead_private() -> None:\n    pass\n"
        "def _self_rec(n: int) -> int:\n    return n if n <= 0 else _self_rec(n - 1)\n"
        '@register("hook")\ndef _decorated() -> None:\n    pass\n',
        encoding="utf-8",
    )
    test_file = tmp_path / "tests/utils/test_parity_on_cpu.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "from verl_omni.utils.parity import _dead_private\ndef test_dead():\n    _dead_private()\n",
        encoding="utf-8",
    )
    head = _commit(tmp_path)
    good = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr


@pytest.mark.skipif(shutil.which("ruff") is None, reason="The CI workflow supplies pinned Ruff.")
def test_vo019_baseline_rename_last_caller_removal_and_staged_mode(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    pkg = tmp_path / "verl_omni/utils"
    pkg.mkdir(parents=True)
    legacy = pkg / "legacy.py"
    legacy.write_text(
        "def _old_dead() -> None:\n    pass\ndef _orphaned() -> None:\n    pass\ndef run() -> None:\n    _orphaned()\n",
        encoding="utf-8",
    )
    base = _commit(tmp_path)

    # Unrelated edit + rename preserves suppression for _old_dead, while removing last caller catches _orphaned.
    moved = pkg / "renamed.py"
    _git(tmp_path, "mv", "verl_omni/utils/legacy.py", "verl_omni/utils/renamed.py")
    moved.write_text(
        "# shift comment\ndef _old_dead() -> None:\n    pass\n"
        "def _orphaned() -> None:\n    pass\n"
        "def run() -> None:\n    pass\n",
        encoding="utf-8",
    )
    head = _commit(tmp_path)
    res = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 1, res.stdout + res.stderr
    assert len(res.stdout.splitlines()) == 1
    assert "_orphaned" in res.stdout and "_old_dead" not in res.stdout

    # Staged mode (via write-tree) also detects newly added unused helpers.
    staged = pkg / "staged.py"
    staged.write_text("def _staged_dead() -> None:\n    pass\n", encoding="utf-8")
    _git(tmp_path, "add", "verl_omni/utils/staged.py")
    staged_res = subprocess.run([sys.executable, str(CHECKER)], cwd=tmp_path, capture_output=True, text=True)
    assert staged_res.returncode == 1, staged_res.stdout + staged_res.stderr
    assert "VO019" in staged_res.stdout and "_staged_dead" in staged_res.stdout
