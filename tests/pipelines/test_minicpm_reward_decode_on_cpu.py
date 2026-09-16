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
"""CPU tests for the MiniCPM answer-tag reward-decode demotion.

The reward worker builds its own tokenizer for the reward loop and decodes with
``skip_special_tokens=True``, which strips the checkpoint's special ``<answer>``
tags and zeroes every choice-reward score. ``OmniRewardLoopWorker._init_reward_fn``
demotes the two tags on the tokenizer its manager decodes with, so the fix lands on
that exact object in the process that scores — with no global reward-manager patch.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

from verl_omni.models.transformers.minicpm_o import (
    actor_registers_special_answer_tags,
    keep_answer_tags_when_decoding,
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


def test_stock_naive_manager_strips_answer_tags_without_the_demotion():
    tokenizer = _minicpm_style_tokenizer()
    ids = tokenizer.encode("a b c <answer>A</answer>", add_special_tokens=False)
    assert "<answer>" not in tokenizer.decode(ids, skip_special_tokens=True)  # the bug

    # A stock manager (the demotion not applied) scores 0 on a correct answer.
    manager = NaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)
    response_str = manager.tokenizer.decode(ids, skip_special_tokens=True)
    assert compute_score(data_source="avqa_r1_6k", solution_str=response_str, ground_truth="<answer>A</answer>") == {
        "score": 0.0,
        "accuracy": 0.0,
    }


def test_demotion_on_the_manager_tokenizer_restores_the_choice_reward():
    tokenizer = _minicpm_style_tokenizer()
    manager = NaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)
    ids = tokenizer.encode("a b c <answer>A</answer>", add_special_tokens=False)

    assert keep_answer_tags_when_decoding(manager.tokenizer) is True

    assert manager.tokenizer is tokenizer  # demoted in place on the manager's own tokenizer
    assert tokenizer.added_tokens_decoder[tokenizer.convert_tokens_to_ids("<answer>")].special is False
    response_str = tokenizer.decode(ids, skip_special_tokens=True)  # verl's decode path
    assert "<answer>" in response_str and "<" + "/answer>" in response_str
    assert compute_score(data_source="avqa_r1_6k", solution_str=response_str, ground_truth="<answer>A</answer>") == {
        "score": 1.0,
        "accuracy": 1.0,
    }


def test_demotion_is_a_noop_for_tagless_tokenizers():
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "[unk]": 1}, unk_token="[unk]"))
    plain = PreTrainedTokenizerFast(tokenizer_object=tok)
    plain.add_tokens(["<answer>", "</answer>"], special_tokens=False)

    manager = NaiveRewardManager(config={}, tokenizer=plain, compute_score=compute_score)

    assert keep_answer_tags_when_decoding(manager.tokenizer) is False
    assert manager.tokenizer is plain  # constructed unchanged, no raise


def test_reward_worker_demotes_on_the_manager_it_constructed(monkeypatch, tmp_path):
    # _init_reward_fn is the install point: same process that built the tokenizer, and
    # the manager is already constructed when it returns.
    seen = []
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.keep_answer_tags_when_decoding",
        lambda tokenizer: seen.append(tokenizer),
    )
    tokenizer = _minicpm_style_tokenizer()
    manager = NaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)

    def fake_super_init(self):
        self.reward_manager = manager

    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.RewardLoopWorker._init_reward_fn",
        fake_super_init,
    )
    worker = object.__new__(OmniRewardLoopWorker)
    worker.config = _actor_config("MiniCPMO", str(tmp_path))
    worker.reward_model_specs = {}
    worker.engine_reward_executors = {}
    worker.native_reward_executors = {}
    OmniRewardLoopWorker._init_reward_fn(worker)

    assert seen == [tokenizer]


def test_actor_gate_applies_for_minicpm_and_unknown_architectures(tmp_path):
    # Unknown counts as yes (the demotion no-ops unless the tags are special), so a
    # MiniCPM checkpoint under an unregistered architecture name still gets the fix.
    import verl_omni.pipelines  # noqa: F401  # register the adapters

    minicpm = _actor_config("MiniCPMO", str(tmp_path))
    assert actor_registers_special_answer_tags(minicpm) is True
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


def test_demotion_leaves_ids_and_encoding_untouched():
    tokenizer = _minicpm_style_tokenizer()
    answer_id = tokenizer.convert_tokens_to_ids("<answer>")
    close_id = tokenizer.convert_tokens_to_ids("</answer>")
    b_id = tokenizer.convert_tokens_to_ids("B")
    before = tokenizer.encode("<answer>B</answer>", add_special_tokens=False)

    keep_answer_tags_when_decoding(tokenizer)

    assert tokenizer.encode("<answer>B</answer>", add_special_tokens=False) == before == [answer_id, b_id, close_id]
