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

"""FlowGRPO prompt, component-selection and LoRA-validation policies for H3."""

from typing import Any

import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    _LORA_TARGET_MAPPING,
    H3_LORA_TARGETS,
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    MiniMaxH3WeightSyncBase,
    _PromptTokenOverride,
    map_lora_tensors,
)
from verl_omni.pipelines.rollout_request import prompt_ids_from_payload


def _lora_target_suffix(target: str) -> str | None:
    return next((suffix for suffix in H3_LORA_TARGETS if target == suffix or target.endswith("." + suffix)), None)


class MiniMaxH3WeightSyncMixin(MiniMaxH3WeightSyncBase):
    """Retain FlowGRPO's combined-partition routing and token-ID-native prompts."""

    def _h3_weight_component_name(self) -> str:
        if getattr(self, "partition", None) == "combined" and hasattr(self, "transformers_ref"):
            return "transformers_ref"
        return "transformer"

    def encode_prompt(self, *, task: str, prompt: str, **kwargs):
        """Let upstream encode references while preserving Agent Loop prompt IDs."""
        prompt_ids = getattr(self, "_h3_prompt_ids", None)
        if prompt_ids is None:
            return super().encode_prompt(task=task, prompt=prompt, **kwargs)

        tokenizer = self.tokenizer
        self.tokenizer = _PromptTokenOverride(tokenizer, prompt, prompt_ids)
        try:
            return super().encode_prompt(task=task, prompt=prompt, **kwargs)
        finally:
            self.tokenizer = tokenizer

    def _ensure_prompt_text(self, request: Any) -> None:
        """Expose Agent Loop IDs while satisfying upstream's text check."""
        self._h3_prompt_ids = None
        prompts = getattr(request, "prompts", None)
        custom_prompt = prompts[0] if prompts and isinstance(prompts[0], dict) else getattr(request, "prompt", None)
        if not isinstance(custom_prompt, dict):
            return
        token_ids = prompt_ids_from_payload(custom_prompt)
        if token_ids is None:
            return
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        if extra_args.get(MINIMAX_H3_TOKEN_ID_NATIVE_KEY) is not True:
            raise ValueError(
                "MiniMax H3 token-ID-native rollout requires "
                "actor_rollout_ref.rollout.agent.default_agent_loop="
                "minimax_h3_diffusion_single_turn_agent."
            )
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        elif token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        self._h3_prompt_ids = torch.as_tensor([int(token) for token in token_ids], dtype=torch.long)
        if self._h3_prompt_ids.numel() == 0:
            raise ValueError("MiniMax H3 requires non-empty prompt_ids.")
        custom_prompt["prompt"] = "[pretokenized]"

    def map_lora_update_to_engine(
        self,
        tensors: dict[str, torch.Tensor],
        peft_config: dict,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Keep qualified target support, subset selection and strict payload checks."""
        target_modules = peft_config.get("target_modules") if peft_config is not None else None
        if isinstance(target_modules, str):
            requested_targets = {target_modules}
        elif isinstance(target_modules, list | tuple | set | frozenset):
            requested_targets = {str(target) for target in target_modules}
        else:
            raise ValueError(f"MiniMax H3 LoRA sync requires explicit target_modules, got {target_modules!r}.")

        target_suffixes = {target: _lora_target_suffix(target) for target in requested_targets}
        unsupported = sorted(target for target, suffix in target_suffixes.items() if suffix is None)
        if not requested_targets or unsupported:
            raise ValueError(
                "MiniMax H3 LoRA sync supports only attention Q/K/V/output and GEGLU projections; "
                f"unsupported targets: {unsupported or sorted(requested_targets)}."
            )

        component = self._h3_weight_component_name()
        mapped = map_lora_tensors(tensors, component, getattr(self, component).arch.ffn_hidden_size, strict=True)
        new_config = dict(peft_config)
        new_config["target_modules"] = sorted(
            {
                mapped
                for suffix in target_suffixes.values()
                if suffix is not None
                for mapped in _LORA_TARGET_MAPPING[suffix]
            }
        )
        return mapped, new_config


__all__ = ["H3_LORA_TARGETS", "MiniMaxH3WeightSyncMixin"]
