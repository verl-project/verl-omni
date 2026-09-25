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
"""CPU tests for the stock reward-worker side of the answer-tag demotion.

The demotion itself (the tokenizer flags, the decode, the choice-reward score) is
covered by ``tests/models/test_minicpm_answer_tags_on_cpu.py``. What matters here
is the delivery the V1 omni trainer actually runs: verl's stock reward loop worker
resolves ``reward.reward_manager.name`` from the registry, so the fix has to arrive
as a registered manager — not as a patch on ``NaiveRewardManager``, and not on
``OmniRewardLoopWorker``, which only the diffusion trainers wire up.
"""

from __future__ import annotations

import subprocess
import sys

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

from verl_omni.pipelines.minicpm.reward_decode import MiniCPMNaiveRewardManager
from verl_omni.utils.reward_score.choice_reward import compute_score


def _tokenizer_with_special_answer_tags():
    """Scratch tokenizer mirroring MiniCPM-o 4.5's tag specialness."""
    # No vocab holes: a gap desyncs added-token id assignment.
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "A": 4, "B": 5, "[unk]": 3}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf.add_tokens(["<answer>", "</answer>"], special_tokens=True)  # the checkpoint's asymmetry
    return hf


def _plain_tokenizer():
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "[unk]": 3}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok)


def test_no_global_reward_manager_patch_is_installed():
    # The demotion must not be an import side effect: a class-level wrap would leak
    # into every run (and every process), not just the opt-in manager below.
    import verl_omni  # noqa: F401

    assert not getattr(NaiveRewardManager, "_minicpm_keeps_answer_tags", False)
    assert NaiveRewardManager.__init__ is not MiniCPMNaiveRewardManager.__init__


def test_stock_worker_path_resolves_the_manager_by_name():
    # The V1 omni trainer runs verl's stock RewardLoopWorker, which builds the
    # manager through get_reward_manager_cls(config.reward.reward_manager.name).
    import verl_omni.pipelines.minicpm  # noqa: F401  # registration side effect

    assert get_reward_manager_cls("minicpm_naive") is MiniCPMNaiveRewardManager
    assert issubclass(MiniCPMNaiveRewardManager, NaiveRewardManager)


def test_importing_verl_registers_the_manager(monkeypatch):
    # Delivery guarantee in the worker process: the recipe exports
    # VERL_USE_EXTERNAL_MODULES=verl_omni, and importing verl then imports
    # verl_omni, whose package inits register the name before any worker
    # resolves it. Proven in a fresh interpreter because this process already
    # holds the registration.
    monkeypatch.setenv("VERL_USE_EXTERNAL_MODULES", "verl_omni")
    probe = (
        "import verl; "
        "from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls; "
        "print(get_reward_manager_cls('minicpm_naive').__name__)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "MiniCPMNaiveRewardManager"


def test_manager_demotes_the_tags_it_decodes_with():
    tokenizer = _tokenizer_with_special_answer_tags()
    manager = MiniCPMNaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)
    ids = tokenizer.encode("<answer>B</answer>", add_special_tokens=False)

    decoded = manager.tokenizer.decode(ids, skip_special_tokens=True)
    # The whitespace pre-tokenizer emits "<answer> B </answer>"; extract_answer strips it.
    assert "B" in decoded and "<answer>" in decoded and "</answer>" in decoded
    assert compute_score(solution_str=decoded, ground_truth="<answer>B</answer>")["score"] == 1.0


def test_stock_manager_still_strips_the_tags():
    # The regression this file guards: with the stock manager the same ids decode
    # without the tags and the choice reward scores 0 while the answer is correct.
    tokenizer = _tokenizer_with_special_answer_tags()
    manager = NaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)
    ids = tokenizer.encode("<answer>B</answer>", add_special_tokens=False)

    decoded = manager.tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == "B"
    assert compute_score(solution_str=decoded, ground_truth="<answer>B</answer>")["score"] == 0.0


def test_manager_noops_on_a_tokenizer_without_the_tags():
    tokenizer = _plain_tokenizer()
    manager = MiniCPMNaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)
    assert manager.tokenizer is tokenizer
    assert manager.tokenizer.decode(tokenizer.encode("a b c"), skip_special_tokens=True) == "a b c"
