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
"""CPU tests for the reward-worker side of the answer-tag demotion.

The demotion itself (the tokenizer flags, the decode, the choice-reward score) is
covered by ``tests/models/test_minicpm_answer_tags_on_cpu.py``. What matters here
is the wiring: no import-time global patch, the worker demotes on the tokenizer
its own manager decodes with, and the actor gate applies or skips.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

from verl_omni.models.transformers.minicpm_o import (
    actor_registers_special_answer_tags,
)
from verl_omni.reward_loop.reward_loop import OmniRewardLoopWorker
from verl_omni.utils.reward_score.choice_reward import compute_score


def _minicpm_style_tokenizer():
    """Scratch tokenizer mirroring MiniCPM-o 4.5's tag specialness."""
    # No vocab holes: a gap desyncs added-token id assignment.
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "A": 4, "B": 5, "[unk]": 3}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf.add_tokens(["<answer>", "</answer>"], special_tokens=True)  # the checkpoint's asymmetry
    return hf


def _actor_config(architecture, model_path):
    """A trainer-config stand-in whose actor model path holds ``architecture``."""
    if architecture is not None:
        with open(f"{model_path}/config.json", "w") as f:
            json.dump({"architectures": [architecture]}, f)
    return SimpleNamespace(actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path=model_path)))


def test_no_global_reward_manager_patch_is_installed():
    # The demotion must not be an import side effect: a class-level wrap would leak
    # into every run (and every process), not just the omni reward worker.
    import verl_omni  # noqa: F401

    assert not getattr(NaiveRewardManager, "_minicpm_keeps_answer_tags", False)


def test_reward_worker_demotes_on_the_manager_it_constructed(monkeypatch, tmp_path):
    # _init_reward_fn is the install point: same process that built the tokenizer,
    # and the manager already holds the object it will decode with.
    seen = []
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.keep_answer_tags_when_decoding",
        lambda tokenizer: seen.append(tokenizer),
    )
    tokenizer = _minicpm_style_tokenizer()
    manager = NaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)

    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.RewardLoopWorker._init_reward_fn",
        lambda self: setattr(self, "reward_manager", manager),
    )
    worker = object.__new__(OmniRewardLoopWorker)
    worker.config = _actor_config("MiniCPMO", str(tmp_path))
    worker.reward_model_specs = {}
    worker.engine_reward_executors = {}
    worker.native_reward_executors = {}
    OmniRewardLoopWorker._init_reward_fn(worker)

    assert seen == [tokenizer]


def test_reward_worker_skips_the_demotion_for_other_models(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.keep_answer_tags_when_decoding",
        lambda tokenizer: seen.append(tokenizer),
    )
    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.RewardLoopWorker._init_reward_fn",
        lambda self: setattr(self, "reward_manager", SimpleNamespace(tokenizer=object())),
    )
    worker = object.__new__(OmniRewardLoopWorker)
    worker.config = _actor_config("Qwen3OmniMoeForConditionalGeneration", str(tmp_path))
    worker.reward_model_specs = {}
    worker.engine_reward_executors = {}
    worker.native_reward_executors = {}

    OmniRewardLoopWorker._init_reward_fn(worker)

    assert seen == []


def test_actor_gate_applies_for_minicpm_and_unknown_architectures(tmp_path):
    # Unknown counts as yes (the demotion no-ops unless the tags are special), so a
    # MiniCPM checkpoint under an unregistered architecture name still gets the fix.
    import verl_omni.pipelines  # noqa: F401  # register the adapters

    assert actor_registers_special_answer_tags(_actor_config("MiniCPMO", str(tmp_path))) is True
    assert actor_registers_special_answer_tags(_actor_config("SomeFutureOmni", str(tmp_path))) is True
    assert actor_registers_special_answer_tags(_actor_config(None, str(tmp_path))) is True  # no config.json
    assert actor_registers_special_answer_tags(None) is True  # no config at all


def test_actor_gate_skips_other_registered_omni_models(tmp_path):
    # A different registered omni model never registers the answer tags, so the worker
    # must not touch its tokenizer.
    import verl_omni.pipelines  # noqa: F401  # register the adapters

    assert (
        actor_registers_special_answer_tags(_actor_config("Qwen3OmniMoeForConditionalGeneration", str(tmp_path)))
        is False
    )
