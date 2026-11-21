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
"""Default lifecycle hooks inherited from the RewardScorer base."""

import pytest
import torch

from verl_omni.utils.reward_score.scorers.base import RewardScorer


class _Weights:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(torch.device(device))
        return self


class _MinimalScorer(RewardScorer):
    supports_cpu_offload = True

    def load(self, device):
        self.model = _Weights()
        self.device = torch.device(device)

    def score_batch(self, batch):
        return [{"score": 0.0} for _ in batch]


def test_default_hooks_move_declared_model_between_device_and_host():
    scorer = _MinimalScorer()
    scorer.load(torch.device("meta"))

    scorer.sleep()
    assert scorer.model.moves == [torch.device("cpu")]

    scorer.wake_up()
    assert scorer.model.moves == [torch.device("cpu"), torch.device("meta")]

    scorer.close()
    assert scorer.model is None


def test_offload_hook_without_declared_model_fails_fast():
    scorer = _MinimalScorer()
    with pytest.raises(RuntimeError, match="assign self.model"):
        scorer.sleep()


def test_constructor_rejects_invalid_declarations():
    from verl_omni.utils.reward_score.scorers.registry import register_scorer

    with pytest.raises(ValueError, match="max_batch_size"):

        @register_scorer("bad-bounds-scorer")
        class BadBatchSize(RewardScorer):
            max_batch_size = 0

            def score_batch(self, batch):
                return [{"score": 0.0} for _ in batch]

    with pytest.raises(ValueError, match="supports_cpu_offload"):

        @register_scorer("bad-offload-scorer")
        class BadOffload(RewardScorer):
            supports_cpu_offload = "yes"

            def score_batch(self, batch):
                return [{"score": 0.0} for _ in batch]


@pytest.fixture
def scorer_package(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from uuid import uuid4

    from verl_omni.utils.reward_score.scorers import registry

    name = "reward_package_" + uuid4().hex
    folder = tmp_path / name
    folder.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    source = folder / "scorer.py"
    source.write_text(
        "from . import state\n"
        "from verl_omni.utils.reward_score.scorers.base import RewardScorer\n"
        "from verl_omni.utils.reward_score.scorers.registry import register_scorer\n"
        "state.loads += 1\n"
        "state.entered.set()\n"
        "if not state.release.wait(5):\n"
        "    raise TimeoutError('package initialization was not released')\n"
        f"@register_scorer({name!r})\n"
        "class CustomScorer(RewardScorer):\n"
        "    def score_batch(self, batch):\n"
        "        return [{'score': 1} for _ in batch]\n"
        "def compute_score(**kwargs):\n"
        "    return {'score': 1}\n"
    )
    (folder / "state.py").write_text(
        "from threading import Event\nloads = 0\nentered = Event()\nrelease = Event()\nrelease.set()\n"
    )
    (folder / "__init__.py").write_text("")
    yield SimpleNamespace(name=name, folder=folder, source=source)
    module = registry._FILE_MODULES.pop(str(source.resolve()), None)
    if module is not None and module.__name__.startswith("_verl_omni_reward_"):
        sys.modules.pop(module.__name__, None)
    registry._REGISTERED.pop(name, None)
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            sys.modules.pop(key)


@pytest.mark.parametrize("file_first", [False, True])
def test_bundled_scorer_package_and_file_references_share_identity(file_first):
    from verl_omni.utils.reward_score.scorers import pickscore
    from verl_omni.utils.reward_score.scorers.registry import import_from_spec

    package_ref = "verl_omni.utils.reward_score.scorers.pickscore:PickScoreScorer"
    file_ref = f"{pickscore.__file__}:PickScoreScorer"
    first, second = (file_ref, package_ref) if file_first else (package_ref, file_ref)
    assert import_from_spec(first) is import_from_spec(second) is pickscore.PickScoreScorer


@pytest.mark.parametrize("file_first", [False, True])
@pytest.mark.parametrize("package_kind", ["regular", "eager", "namespace"])
def test_custom_scorer_package_and_file_orders_execute_once(scorer_package, file_first, package_kind):
    import importlib

    from verl_omni.utils.reward_score.scorers.registry import get_scorer_cls, import_from_spec

    package = scorer_package
    if package_kind == "eager":
        (package.folder / "__init__.py").write_text("from .scorer import CustomScorer\n")
    elif package_kind == "namespace":
        (package.folder / "__init__.py").unlink()
    file_ref = f"file://{package.source}:CustomScorer"
    if file_first:
        cls = import_from_spec(file_ref)
    else:
        cls = importlib.import_module(package.name + ".scorer").CustomScorer
    module = importlib.import_module(package.name + ".scorer")
    assert import_from_spec(file_ref) is module.CustomScorer is cls
    assert get_scorer_cls(package.name) is cls
    assert import_from_spec(f"{package.name}.scorer:compute_score") is module.compute_score
    alias = package.folder / "alias.py"
    alias.symlink_to(package.source)
    assert import_from_spec(f"{alias}:CustomScorer") is cls
    assert importlib.import_module(package.name + ".state").loads == 1
    other = package.folder / "different.py"
    other.write_text(package.source.read_text())
    with pytest.raises(ValueError, match="already been registered"):
        import_from_spec(f"{other}:CustomScorer")


def test_package_dependency_error_is_not_retried_as_a_standalone_file(scorer_package):
    from verl_omni.utils.reward_score.scorers.registry import import_from_spec

    (scorer_package.folder / "__init__.py").write_text(
        "raise ModuleNotFoundError('missing scorer dependency', name='missing_reward_dependency')\n"
    )
    with pytest.raises(ModuleNotFoundError) as caught:
        import_from_spec(f"{scorer_package.source}:CustomScorer")
    assert caught.value.name == "missing_reward_dependency"


def test_file_reference_waits_for_concurrent_package_initialization(scorer_package, monkeypatch):
    import importlib
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from verl_omni.utils.reward_score.scorers import registry
    from verl_omni.utils.reward_score.scorers.registry import import_from_spec

    observed = Event()
    find_source = registry._source_module

    def observe_source(path):
        module = find_source(path)
        observed.set()
        return module

    monkeypatch.setattr(registry, "_source_module", observe_source)
    package = scorer_package
    state = importlib.import_module(package.name + ".state")
    state.release.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        importing = pool.submit(importlib.import_module, package.name + ".scorer")
        try:
            assert state.entered.wait(5), "package import did not enter its source"
            resolving = pool.submit(import_from_spec, f"{package.source}:CustomScorer")
            assert observed.wait(5), "file lookup did not reach the initializing module"
            # A partial module must not escape while its class is still undefined.
            with pytest.raises(TimeoutError):
                resolving.result(timeout=0.05)
        finally:
            state.release.set()
        assert resolving.result(timeout=5) is importing.result(timeout=5).CustomScorer
    assert state.loads == 1


def test_explicit_file_does_not_initialize_an_unrelated_shadow_package(scorer_package, tmp_path, monkeypatch):
    import sys

    from verl_omni.utils.reward_score.scorers.registry import import_from_spec

    package = scorer_package
    shadow_root = tmp_path / "shadow"
    shadow = shadow_root / package.name
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise RuntimeError('unrelated package initializer ran')\n")
    monkeypatch.syspath_prepend(str(shadow_root))
    package.source.write_text(
        "from verl_omni.utils.reward_score.scorers.base import RewardScorer\n"
        "from verl_omni.utils.reward_score.scorers.registry import register_scorer\n"
        f"@register_scorer({package.name!r})\n"
        "class CustomScorer(RewardScorer):\n"
        "    def score_batch(self, batch):\n"
        "        return [{'score': 1} for _ in batch]\n"
    )
    cls = import_from_spec(f"{package.source}:CustomScorer")
    assert cls().score_batch([{}]) == [{"score": 1}]
    assert package.name not in sys.modules


@pytest.mark.parametrize("package_import", [False, True])
@pytest.mark.parametrize("alias_registration", [False, True])
def test_failed_plugin_registration_is_not_published_and_can_retry(
    tmp_path, monkeypatch, package_import, alias_registration
):
    import sys
    from uuid import uuid4

    from verl_omni.utils.reward_score.scorers import registry

    name = "retry_" + uuid4().hex
    good = "good_" + uuid4().hex
    missing = "dependency_" + uuid4().hex
    monkeypatch.syspath_prepend(str(tmp_path))
    template = (
        "from verl_omni.utils.reward_score.scorers.base import RewardScorer\n"
        "from verl_omni.utils.reward_score.scorers.registry import register_scorer\n"
        "@register_scorer({name!r})\n"
        "class Scorer(RewardScorer):\n"
        "    def score_batch(self, batch): return [{{'score': 1}} for _ in batch]\n"
    )
    (tmp_path / (good + ".py")).write_text(template.format(name=good))
    folder = tmp_path / (name if package_import else "not-a-package")
    folder.mkdir()
    if package_import:
        (folder / "__init__.py").write_text("")
    path = folder / "plugin.py"
    registration = (
        "from verl_omni.utils.reward_score.scorers.registry import register_scorer\n"
        f"from {good} import Scorer\n"
        f"register_scorer({name!r})(Scorer)\n"
        if alias_registration
        else template.format(name=name)
    )
    path.write_text(f"import {good}\n" + registration + f"import {missing}\n")
    ref = f"{name}.plugin:Scorer" if package_import else f"{path}:Scorer"
    try:
        with pytest.raises((ImportError, RuntimeError)):
            registry.import_from_spec(ref)
        assert name not in registry.peek_registered_scorers(), "failed imports must not expose a class"
        successful = registry.get_scorer_cls(good)
        with pytest.raises(ValueError, match="Unknown reward scorer"):
            registry.get_scorer_cls(name)
        (tmp_path / (missing + ".py")).write_text("ready = True\n")
        cls = registry.import_from_spec(ref)
        assert registry.get_scorer_cls(name) is cls
        assert registry.get_scorer_cls(good) is successful
    finally:
        registry.peek_registered_scorers()
        registry._REGISTERED.pop(name, None)
        registry._REGISTERED.pop(good, None)
        registry._FILE_MODULES.pop(str(path), None)
        for key in list(sys.modules):
            if key in (good, missing, name) or key.startswith(name + "."):
                sys.modules.pop(key, None)


def test_concurrent_duplicate_registration_has_one_winner(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from verl_omni.utils.reward_score.scorers import registry

    entered, release, second_started = Event(), Event(), Event()

    class First(_MinimalScorer):
        pass

    class Second(_MinimalScorer):
        pass

    class GatedRegistry(dict):
        def __setitem__(self, key, value):
            if key == "registration-race" and value is First:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("registration gate was not released")
            super().__setitem__(key, value)

    monkeypatch.setattr(registry, "_REGISTERED", GatedRegistry(registry.peek_registered_scorers()))

    def competing_registration():
        second_started.set()
        return registry.register_scorer("registration-race")(Second)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.register_scorer("registration-race"), First)
        try:
            assert entered.wait(5)
            second = pool.submit(competing_registration)
            assert second_started.wait(5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=5) is First
        with pytest.raises(ValueError, match="already been registered"):
            second.result(timeout=5)
    assert registry.register_scorer("registration-race")(First) is First
    assert registry.get_scorer_cls("registration-race") is First


@pytest.mark.parametrize("package_import", [False, True])
@pytest.mark.parametrize("successful_first", [False, True])
def test_successful_plugin_retains_a_registration_shared_with_a_failed_plugin(
    tmp_path, monkeypatch, package_import, successful_first
):
    import sys
    from uuid import uuid4

    from verl_omni.utils.reward_score.scorers import registry

    name = "shared_registration_" + uuid4().hex
    folder = tmp_path / name
    folder.mkdir()
    (folder / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    (folder / "shared.py").write_text(
        "from verl_omni.utils.reward_score.scorers.base import RewardScorer\n"
        "class Shared(RewardScorer):\n"
        "    def score_batch(self, batch): return [{'score': 1} for _ in batch]\n"
    )
    registration = (
        "from verl_omni.utils.reward_score.scorers.registry import register_scorer\n"
        f"from {name}.shared import Shared\n"
        f"register_scorer({name!r})(Shared)\n"
    )
    (folder / "successful.py").write_text(registration)
    if package_import:
        failing = folder / "failing.py"
        ref = f"{name}.failing:Shared"
    else:
        standalone = tmp_path / "standalone-files"
        standalone.mkdir()
        failing = standalone / "failing.py"
        ref = f"{failing}:Shared"
    successful_import = f"import {name}.successful\n"
    source = successful_import + registration if successful_first else registration + successful_import
    failing.write_text(source + "raise RuntimeError('first plugin failed')\n")
    try:
        with pytest.raises(RuntimeError) as caught:
            registry.import_from_spec(ref)
        assert str(caught.value.__cause__ or caught.value) == "first plugin failed"
        shared = sys.modules[name + ".shared"].Shared
        print("SHARED_REGISTRATION_ASSERT_ENTERED", package_import, successful_first, flush=True)
        assert registry.get_scorer_cls(name) is shared, "a successful owner must retain its shared registration"
        failing.write_text(source)
        assert registry.import_from_spec(ref) is shared
        assert registry.get_scorer_cls(name) is shared

        class Different(shared):
            pass

        with pytest.raises(ValueError, match="already been registered"):
            registry.register_scorer(name)(Different)
    finally:
        registry.peek_registered_scorers()
        registry._REGISTERED.pop(name, None)
        module = registry._FILE_MODULES.pop(str(failing), None)
        if module is not None:
            sys.modules.pop(module.__name__, None)
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("package_import", [False, True])
def test_concurrent_same_class_claims_survive_one_import_failure(tmp_path, monkeypatch, package_import):
    import importlib
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4

    from verl_omni.utils.reward_score.scorers import registry

    name = "concurrent_claim_" + uuid4().hex
    folder = tmp_path / name
    folder.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    (folder / "__init__.py").write_text("")
    (folder / "state.py").write_text(
        "from threading import Event\n"
        "first_entered = Event()\nsecond_entered = Event()\n"
        "release_failure = Event()\nrelease_success = Event()\n"
    )
    (folder / "shared.py").write_text(
        "from verl_omni.utils.reward_score.scorers.base import RewardScorer\n"
        "class Shared(RewardScorer):\n"
        "    def score_batch(self, batch): return [{'score': 1} for _ in batch]\n"
    )
    state = importlib.import_module(name + ".state")
    shared = importlib.import_module(name + ".shared").Shared
    registration = (
        "from verl_omni.utils.reward_score.scorers.registry import register_scorer\n"
        f"from {name}.shared import Shared\nfrom {name} import state\n"
        f"register_scorer({name!r})(Shared)\n"
    )
    sources = folder if package_import else tmp_path / "standalone-files"
    sources.mkdir(exist_ok=True)
    first, second = sources / "first.py", sources / "second.py"
    first.write_text(
        registration + "state.first_entered.set()\n"
        "assert state.release_failure.wait(5)\nraise RuntimeError('first plugin failed')\n"
    )
    second.write_text(registration + "state.second_entered.set()\nassert state.release_success.wait(5)\n")
    first_ref = f"{name}.first:Shared" if package_import else f"{first}:Shared"
    second_ref = f"{name}.second:Shared" if package_import else f"{second}:Shared"
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            failed = pool.submit(registry.import_from_spec, first_ref)
            try:
                assert state.first_entered.wait(5)
                succeeded = pool.submit(registry.import_from_spec, second_ref)
                assert state.second_entered.wait(5), "both pending owners must reach registration"
                state.release_failure.set()
                with pytest.raises(RuntimeError) as caught:
                    failed.result(timeout=5)
                assert str(caught.value.__cause__ or caught.value) == "first plugin failed"
                assert name not in registry.peek_registered_scorers(), "the surviving import is still initializing"
                state.release_success.set()
                assert succeeded.result(timeout=5) is shared
                print("CONCURRENT_REGISTRATION_ASSERT_ENTERED", package_import, flush=True)
                assert registry.get_scorer_cls(name) is shared, "the surviving owner must publish its claim"
            finally:
                state.release_failure.set()
                state.release_success.set()
        first.write_text(registration)
        assert registry.import_from_spec(first_ref) is shared
    finally:
        registry.peek_registered_scorers()
        registry._REGISTERED.pop(name, None)
        for path in (first, second):
            module = registry._FILE_MODULES.pop(str(path), None)
            if module is not None:
                sys.modules.pop(module.__name__, None)
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)


def test_standalone_import_does_not_block_a_different_file_from_package_init(tmp_path):
    import subprocess
    import sys
    import textwrap

    from verl.utils import import_utils

    from verl_omni.utils.reward_score.scorers import registry

    script = textwrap.dedent("""
        import ast
        import importlib
        import runpy
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from threading import Event
        from types import ModuleType

        # Exercise the actual registry and pinned file loader with only stdlib
        # dependencies. This probe uses no scorer/tensor operation; importing
        # the full accelerator stack would obscure the import-lock deadline.
        for name in ('verl', 'verl.utils', 'verl_omni', 'verl_omni.utils',
                     'verl_omni.utils.reward_score', 'verl_omni.utils.reward_score.scorers'):
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        helper = ModuleType('verl.utils.import_utils')
        helper.__dict__.update(runpy.run_path(sys.argv[3]))
        sys.modules['verl.utils.import_utils'] = helper
        name = 'verl_omni.utils.reward_score.scorers.registry'
        registry = ModuleType(name)
        registry.__file__ = sys.argv[2]
        registry.RewardScorer = object
        sys.modules[name] = registry
        tree = ast.parse(Path(sys.argv[2]).read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and
            node.module == 'verl_omni.utils.reward_score.scorers.base')]
        exec(compile(tree, sys.argv[2], 'exec'), registry.__dict__)
        import_from_spec = registry.import_from_spec

        root = Path(sys.argv[1])
        state = ModuleType('reward_import_gate')
        state.package_entered = Event()
        state.file_entered = Event()
        sys.modules[state.__name__] = state
        sys.path.insert(0, str(root))
        folder = root / 'standalone-files'
        folder.mkdir()
        other = folder / 'other.py'
        other.write_text('value = 9\\n')
        package = root / 'initializing_reward_package'
        package.mkdir()
        (package / '__init__.py').write_text(
            'import reward_import_gate as state\\n'
            'from verl_omni.utils.reward_score.scorers.registry import import_from_spec\\n'
            'state.package_entered.set()\\n'
            'assert state.file_entered.wait(5)\\n'
            f'value = import_from_spec({str(other) + ":value"!r})\\n'
        )
        source = folder / 'first.py'
        source.write_text(
            'import reward_import_gate as state\\n'
            'state.file_entered.set()\\n'
            'import initializing_reward_package\\n'
            'value = initializing_reward_package.value\\n'
        )
        print('IMPORT_LOCK_PROBE_ENTERED', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            package_future = pool.submit(importlib.import_module, 'initializing_reward_package')
            assert state.package_entered.wait(5)
            file_future = pool.submit(import_from_spec, str(source) + ':value')
            assert file_future.result(timeout=5) == 9
            assert package_future.result(timeout=5).value == 9
        print('IMPORT_LOCK_PROBE_PASSED', flush=True)
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), registry.__file__, import_utils.__file__],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        assert b"IMPORT_LOCK_PROBE_ENTERED" in (error.stdout or b"")
        pytest.fail("IMPORT_LOCK_PROBE_STALLED after entering the real import interleaving")
    assert "IMPORT_LOCK_PROBE_ENTERED" in result.stdout, result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_LOCK_PROBE_PASSED" in result.stdout


@pytest.mark.parametrize("retry_through_file", [False, True])
def test_failed_direct_import_retry_uses_the_new_module_owner(scorer_package, retry_through_file):
    import importlib
    import sys

    from verl_omni.utils.reward_score.scorers import registry

    name = scorer_package.name
    state = importlib.import_module(name + ".state")
    state.fail = True
    source = scorer_package.source
    source.write_text(source.read_text() + "\nif state.fail:\n    raise RuntimeError('direct import failed')\n")
    with pytest.raises(RuntimeError, match="direct import failed"):
        importlib.import_module(name + ".scorer")
    assert name + ".scorer" not in sys.modules
    state.fail = False
    print("DIRECT_IMPORT_RETRY_ASSERT_ENTERED", retry_through_file, flush=True)
    if retry_through_file:
        scorer = registry.load_reward_object(str(source), "CustomScorer")
    else:
        scorer = importlib.import_module(name + ".scorer").CustomScorer
    assert registry.get_scorer_cls(name) is scorer
    assert state.loads == 2


def test_file_lookup_rejects_package_path_redirection(scorer_package, tmp_path):
    from verl_omni.utils.reward_score.scorers import registry

    source = scorer_package.source
    source.write_text("value = 'requested'\n")
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    (redirected / "scorer.py").write_text("value = 'redirected'\n")
    (scorer_package.folder / "__init__.py").write_text(f"__path__ = [{str(redirected)!r}]\n")
    print("MODULE_ORIGIN_ASSERT_ENTERED", flush=True)
    with pytest.raises(ImportError, match="resolved to.*expected") as error:
        registry.load_reward_object(str(source), "value")
    assert str(source.resolve()) in str(error.value)
    assert str(redirected / "scorer.py") in str(error.value)
    assert str(source.resolve()) not in registry._FILE_MODULES
    assert registry.load_reward_object(str(source), "value") == "requested"


def test_documented_scorer_scaffold_parses_and_inherits_lifecycle():
    import ast
    import re
    from pathlib import Path

    doc = Path(__file__).resolve().parents[4] / "docs/contributing/integrating_a_reward_scorer.md"
    examples = [
        block for block in re.findall(r"```python\n(.*?)```", doc.read_text(), re.DOTALL) if "class MyScorer(" in block
    ]
    assert len(examples) == 1
    tree = ast.parse(examples[0])
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MyScorer")
    cls.decorator_list = []
    namespace = {"RewardScorer": RewardScorer}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(doc), "exec"), namespace)
    scorer = namespace["MyScorer"]()
    weights = _Weights()
    scorer.model = weights
    scorer.device = torch.device("meta")
    scorer.sleep()
    scorer.wake_up()
    assert weights.moves == [torch.device("cpu"), torch.device("meta")]
    scorer.close()
    assert scorer.model is None
