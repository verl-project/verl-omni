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
