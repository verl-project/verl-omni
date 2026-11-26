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

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_pr_lint import _ruff_findings

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


@pytest.mark.parametrize(
    ("root_import", "expected"),
    [
        ("", {"VO010"}),
        ("from . import example as example\n", set()),
        ("from .example import Adapter as Adapter\n", set()),
        ('import verl_omni.pipelines.example as example\n__all__ = ["example"]\n', set()),
        ("from verl_omni.pipelines import example as example\n", set()),
        ("from .example.training import Adapter as Adapter\n", set()),
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
            "@dataclass\nclass ExampleConfig(BaseConfig):\n    pass\n",
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
        "examples/model/run_train.sh": ("python trainer.py\n", 'python trainer.py "$@"\n'),
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
    expected_codes.update({"EX001", "EX002", "EX003", "EX004"})
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
    assert {line.split()[1] for line in result.stdout.splitlines()} == {"CM001", "CM002", "CM006", "ERA001"}
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
