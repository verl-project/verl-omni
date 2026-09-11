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
"""CPU tests for the ``omni_naive`` reward manager.

verl's reward loop builds its own tokenizer (``hf_tokenizer`` on the model
path) that never passes through the model adapter's ``configure_tokenizer``.
The manager resolves the model adapter from ``actor_rollout_ref.model`` and
applies its ``prepare_reward_decode_tokenizer`` hook on the tokenizer it
decodes with — the default hook is a no-op (Qwen3-Omni keeps decoding fine
with the stock ``naive`` manager), MiniCPM's demotes the checkpoint's special
``<answer>`` tags so ``skip_special_tokens=True`` keeps them.
"""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from verl_omni.utils.reward_score.choice_reward import compute_score


def _minicpm_style_tokenizer():
    """Scratch tokenizer mirroring MiniCPM-o 4.5's tag specialness."""
    # No vocab holes: a gap desyncs added-token id assignment.
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "A": 4, "B": 5, "[unk]": 3}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf.add_tokens(["<answer>", "</answer>"], special_tokens=True)  # the checkpoint's asymmetry
    return hf


def _response_ids(tokenizer):
    return tokenizer.encode("a b c <answer>A</answer>", add_special_tokens=False)


def _trainer_config(**model_overrides):
    model = {"path": "/fake/model", "model_stage": "thinker", **model_overrides}
    return OmegaConf.create({"actor_rollout_ref": {"model": model}})


def test_minicpm_adapter_route_demotes_tags_on_the_reward_tokenizer():
    from verl_omni.reward_loop.reward_manager import OmniNaiveRewardManager

    tokenizer = _minicpm_style_tokenizer()
    ids = _response_ids(tokenizer)
    assert "<answer>" not in tokenizer.decode(ids, skip_special_tokens=True)  # the bug

    manager = OmniNaiveRewardManager(
        config=_trainer_config(architecture="MiniCPMO"), tokenizer=tokenizer, compute_score=compute_score
    )

    assert manager.tokenizer is tokenizer
    assert tokenizer.added_tokens_decoder[tokenizer.convert_tokens_to_ids("<answer>")].special is False
    response_str = tokenizer.decode(ids, skip_special_tokens=True)  # verl's decode path
    assert "<answer>" in response_str and "<" + "/answer>" in response_str
    assert compute_score(data_source="avqa_r1_6k", solution_str=response_str, ground_truth="<answer>A</answer>") == {
        "score": 1.0,
        "accuracy": 1.0,
    }


def test_default_adapter_hook_is_noop_for_tagless_tokenizers():
    # Qwen3-Omni's adapter does not override the hook: constructing the
    # manager against its architecture must not touch a tag-less tokenizer.
    from verl_omni.reward_loop.reward_manager import OmniNaiveRewardManager

    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "[unk]": 1}, unk_token="[unk]"))
    plain = PreTrainedTokenizerFast(tokenizer_object=tok)
    plain.add_tokens(["<answer>", "</answer>"], special_tokens=False)

    manager = OmniNaiveRewardManager(
        config=_trainer_config(architecture="Qwen3OmniMoeForConditionalGeneration"),
        tokenizer=plain,
        compute_score=compute_score,
    )

    assert manager.tokenizer is plain
    assert plain.added_tokens_decoder[plain.convert_tokens_to_ids("<answer>")].special is False


def test_architecture_autodetected_from_config_json(tmp_path, monkeypatch):
    from verl_omni.reward_loop.reward_manager import OmniNaiveRewardManager

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["MiniCPMO"]}))
    monkeypatch.setattr("verl_omni.reward_loop.reward_manager.omni_naive.copy_to_local", lambda path: str(tmp_path))

    tokenizer = _minicpm_style_tokenizer()
    OmniNaiveRewardManager(config=_trainer_config(), tokenizer=tokenizer, compute_score=compute_score)

    assert tokenizer.added_tokens_decoder[tokenizer.convert_tokens_to_ids("<answer>")].special is False


def test_resolution_fails_closed_without_architecture_or_path():
    from verl_omni.reward_loop.reward_manager import OmniNaiveRewardManager

    with pytest.raises(ValueError, match="architecture"):
        OmniNaiveRewardManager(
            config=OmegaConf.create({}), tokenizer=_minicpm_style_tokenizer(), compute_score=compute_score
        )


def test_resolution_fails_closed_for_unknown_architecture():
    from verl_omni.reward_loop.reward_manager import OmniNaiveRewardManager

    with pytest.raises(NotImplementedError, match="No omni model registered"):
        OmniNaiveRewardManager(
            config=_trainer_config(architecture="NoSuchArchitecture"),
            tokenizer=_minicpm_style_tokenizer(),
            compute_score=compute_score,
        )


def test_manager_registered_at_package_import():
    # The trainer resolves reward.reward_manager.name eagerly in the
    # task-runner process during _setup, before any reward worker imports
    # the custom reward fn — so registration must fire when the verl_omni
    # package (whose __init__ imports verl_omni.reward_loop) is imported.
    import importlib
    import sys

    from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

    from verl_omni.reward_loop.reward_manager import OmniNaiveRewardManager

    sys.modules.pop("verl_omni.reward_loop.reward_manager.omni_naive", None)
    importlib.import_module("verl_omni.reward_loop.reward_manager")

    assert issubclass(OmniNaiveRewardManager, NaiveRewardManager)
    assert get_reward_manager_cls("omni_naive") is OmniNaiveRewardManager
