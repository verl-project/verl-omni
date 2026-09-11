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
"""Opt-in naive reward manager that mirrors the adapter's tokenizer preparation."""

import json
import os

from omegaconf import OmegaConf
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.utils.fs import copy_to_local


@register("omni_naive")
class OmniNaiveRewardManager(NaiveRewardManager):
    """Naive reward manager with the model adapter's reward-decode preparation.

    Why (reward loop's second tokenizer vs the adapter's):
        verl's reward loop builds its own tokenizer (``hf_tokenizer`` on the
        model path) that never passes through the model adapter's
        ``configure_tokenizer``, so any adapter-side fix the decoded string
        depends on is missing in the reward worker — e.g. MiniCPM-o 4.5
        registers ``<answer>``/``</answer>`` as special tokens and the
        manager's ``skip_special_tokens=True`` decode strips them, zeroing
        every choice-reward score while the answers are correct. This manager
        resolves the model adapter from ``actor_rollout_ref.model`` (same
        architecture key ``OmniModelConfig`` uses) and applies its
        ``prepare_reward_decode_tokenizer`` hook on the exact tokenizer it
        decodes with. The hook defaults to a no-op, so recipes whose decodes
        are already correct (e.g. Qwen3-Omni) can keep the stock ``naive``
        manager, and models gain reward-decode fixes in their adapter instead
        of a per-model manager class.

    Registration must fire at ``verl_omni`` package-import time (this
    package's ``__init__``, imported by ``verl_omni/__init__``): the trainer
    resolves ``reward.reward_manager.name`` eagerly in the task-runner
    process (``RewardLoopManager.__init__`` during ``_setup``), before any
    reward worker imports the custom reward function — registering from the
    reward-fn module dies at launch with ``Unknown reward manager``.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(
            config,
            tokenizer,
            compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
        adapter_cls = self._resolve_adapter_cls(config)
        adapter_cls.prepare_reward_decode_tokenizer(self.tokenizer, self._model_config_group(config))

    @staticmethod
    def _model_config_group(config):
        if not OmegaConf.is_config(config):
            config = OmegaConf.create(config or {})
        return OmegaConf.select(config, "actor_rollout_ref.model") or OmegaConf.create({})

    @classmethod
    def _resolve_adapter_cls(cls, config):
        from verl_omni.pipelines.model_base import OmniModelBase

        model_cfg = cls._model_config_group(config)
        architecture = model_cfg.get("architecture")
        if not architecture:
            # Mirror OmniModelConfig's documented auto-detection.
            model_path = model_cfg.get("path")
            if not model_path:
                raise ValueError(
                    "OmniNaiveRewardManager needs actor_rollout_ref.model.architecture (or .path to "
                    "auto-detect it from config.json) to resolve the model adapter for reward-decode "
                    "tokenizer preparation."
                )
            config_path = os.path.join(copy_to_local(model_path), "config.json")
            try:
                with open(config_path) as fh:
                    architecture = json.load(fh)["architectures"][0]
            except (OSError, json.JSONDecodeError, KeyError, IndexError) as exc:
                raise ValueError(
                    f"OmniNaiveRewardManager failed to determine the model architecture from "
                    f"{config_path}: {exc}. Set actor_rollout_ref.model.architecture explicitly."
                ) from exc
        return OmniModelBase.get_class_by_name(
            architecture,
            model_cfg.get("model_stage") or "thinker",
            model_cfg.get("external_lib"),
        )
