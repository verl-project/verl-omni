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

"""Map Diffusers MiniMax H3 weights to vLLM-Omni's fused DiT layout."""

from collections.abc import Iterable, Sequence

import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    H3_LORA_STACKED_PARAMS_MAPPING,
    H3PromptTokenOverride,
    diffusers_to_vllm_name,
    prepare_h3_token_id_prompt,
)

_LORA_TARGET_MAPPING = {
    "to_q": ("to_q",),
    "to_k": ("to_k",),
    "to_v": ("to_v",),
    "to_out.0": ("out_proj",),
    "ff.net.0.proj": ("fc1_0", "fc1_1"),
    "ff.net.2": ("fc2",),
}
H3_LORA_TARGETS = frozenset(_LORA_TARGET_MAPPING)
H3_VEOMNI_LORA_TARGETS = frozenset({"qkv_proj", "out_proj", "fc1", "fc2"})
_VEOMNI_LORA_TARGET_MAPPING = {
    "qkv_proj": ("to_q", "to_k", "to_v"),
    "out_proj": ("out_proj",),
    "fc1": ("fc1_0", "fc1_1"),
    "fc2": ("fc2",),
}


def _normalize_h3_lora_targets(target_modules: object) -> set[str]:
    if isinstance(target_modules, str):
        return {target_modules}
    if isinstance(target_modules, Sequence | set | frozenset):
        return {str(target) for target in target_modules}
    raise ValueError(f"MiniMax H3 LoRA requires explicit target_modules, got {target_modules!r}.")


def _target_suffix(target: str, supported_targets: frozenset[str]) -> str | None:
    return next(
        (suffix for suffix in supported_targets if target == suffix or target.endswith("." + suffix)),
        None,
    )


def resolve_h3_lora_target_layout(target_modules: object) -> tuple[str, set[str]]:
    """Normalize H3 LoRA targets and identify the Diffusers or VeOmni layout."""
    requested = _normalize_h3_lora_targets(target_modules)
    if requested and all(_target_suffix(target, H3_VEOMNI_LORA_TARGETS) is not None for target in requested):
        return "veomni", requested
    if requested and all(_target_suffix(target, H3_LORA_TARGETS) is not None for target in requested):
        return "diffusers", requested

    supported = sorted(H3_LORA_TARGETS | H3_VEOMNI_LORA_TARGETS)
    raise ValueError(
        "MiniMax H3 LoRA supports only one complete projection naming layout from "
        f"{supported}; got {sorted(requested)}. Other targets cannot be synchronized to the rollout model."
    )


def _split_lora_weight_name(name: str) -> tuple[str, str] | None:
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None


# TODO: Remove this MiniMax H3-specific mapping once vLLM-Omni natively
# supports syncing Diffusers full weights and LoRA updates into its fused
# QKV/GEGLU inference layout.
class MiniMaxH3WeightSyncMixin:
    """Translate Diffusers Actor weights and token-ID-native H3 prompts."""

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
        self.tokenizer = H3PromptTokenOverride(tokenizer, prompt, prompt_ids)
        try:
            return super().encode_prompt(task=task, prompt=prompt, **kwargs)
        finally:
            self.tokenizer = tokenizer

    def _ensure_prompt_text(self, request: object) -> None:
        """Expose Agent Loop IDs while satisfying upstream's text check."""
        self._h3_prompt_ids = prepare_h3_token_id_prompt(request)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load full weights through vLLM-Omni's TP-aware parameter loaders."""
        translated: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        component_params: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            component, separator, inner = name.partition(".")
            if separator != "." or component not in {"transformer", "transformers_ref"}:
                translated.append((name, tensor))
                continue
            target_component = self._h3_weight_component_name() if component == "transformer" else component

            # PEFT's merged full-weight path may retain ``base_layer`` in names.
            inner = inner.replace(".base_layer", "")
            if "lora_" in inner:
                continue

            if inner.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
                block, projection = inner.rsplit(".attn.to_", 1)
                target_name = f"{diffusers_to_vllm_name(block)}.attn.qkv_proj.weight"
                params = component_params.get(target_component)
                if params is None:
                    params = component_params[target_component] = dict(
                        getattr(self, target_component).named_parameters()
                    )
                param = params[target_name]

                # Keep Q/K/V separate. The fused parameter's native loader packs
                # the requested shard and applies the correct TP partition.
                param.weight_loader(param, tensor, projection[0])
                loaded.add(f"{component}.{target_name}")
                continue

            if inner.endswith(".ff.net.0.proj.weight"):
                target_name = diffusers_to_vllm_name(inner).replace(".ff.net.0.proj.", ".mlp.fc1.")
                params = component_params.get(target_component)
                if params is None:
                    params = component_params[target_component] = dict(
                        getattr(self, target_component).named_parameters()
                    )
                param = params[target_name]

                # Diffusers GEGLU stores [up, gate], while H3's fused fc1 expects
                # logical shards [gate, up]. Its loader performs the TP slicing.
                up, gate = tensor.chunk(2, dim=0)
                param.weight_loader(param, gate, 0)
                param.weight_loader(param, up, 1)
                loaded.add(f"{component}.{target_name}")
                continue

            translated.append((f"{target_component}.{diffusers_to_vllm_name(inner)}", tensor))

        # Native H3 loading still handles all parameters that only need renaming.
        if translated:
            loaded.update(super().load_weights(translated))
        return loaded

    def install_h3_lora_layout(self) -> None:
        """Expose H3's fused QKV and GEGLU layout to the LoRA manager."""
        transformer = getattr(self, self._h3_weight_component_name(), None)
        if transformer is None:
            return
        existing = list(getattr(transformer, "stacked_params_mapping", ()) or ())

        def _leaf_pair(item: tuple) -> tuple[str, str]:
            return tuple(str(name).strip(".").split(".")[-1] for name in item[:2])

        known = {_leaf_pair(item) for item in existing if len(item) >= 2}
        transformer.stacked_params_mapping = existing + [
            item for item in H3_LORA_STACKED_PARAMS_MAPPING if _leaf_pair(item) not in known
        ]

    def map_lora_update_to_engine(
        self,
        tensors: dict[str, torch.Tensor],
        peft_config: dict,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Map Diffusers LoRA tensors and targets to fused H3 modules."""
        target_layout, requested_targets = resolve_h3_lora_target_layout(
            peft_config.get("target_modules") if peft_config is not None else None
        )
        component = self._h3_weight_component_name()
        if target_layout == "veomni":
            transformer = getattr(self, component)
            heads = transformer.arch.num_attention_heads
            head_dim = transformer.arch.attention_head_dim
            ff_half = transformer.arch.ffn_hidden_size
            mapped = {}
            for name, tensor in tensors.items():
                prefix, separator, inner = name.partition(".")
                if separator and prefix == "transformer":
                    inner = inner.removeprefix("base_model.model.")
                    name = f"{component}.{diffusers_to_vllm_name(inner)}"
                lora_weight = _split_lora_weight_name(name)
                if lora_weight is None:
                    mapped[name] = tensor
                    continue
                module, suffix = lora_weight
                is_lora_a = suffix == ".lora_A.weight"
                if module.endswith(".attn.qkv_proj"):
                    base = module[: -len("qkv_proj")]
                    if is_lora_a:
                        projections = (tensor, tensor, tensor)
                    else:
                        expected = heads * 3 * head_dim
                        if tensor.shape[0] != expected:
                            raise ValueError(
                                f"MiniMax H3 qkv_proj LoRA B rows must be {expected}, got {tensor.shape[0]} for {name}."
                            )
                        grouped = tensor.view(heads, 3, head_dim, -1)
                        projections = tuple(grouped[:, index].reshape(heads * head_dim, -1) for index in range(3))
                    for target, projection in zip(("to_q", "to_k", "to_v"), projections, strict=True):
                        mapped[f"{base}{target}{suffix}"] = projection.contiguous()
                    continue
                if module.endswith(".mlp.fc1"):
                    base = module[: -len("fc1")]
                    if is_lora_a:
                        projections = (tensor, tensor)
                    else:
                        if tensor.shape[0] != 2 * ff_half:
                            raise ValueError(
                                f"MiniMax H3 fc1 LoRA B rows must be {2 * ff_half}, got {tensor.shape[0]} for {name}."
                            )
                        projections = tensor.chunk(2, dim=0)
                    for target, projection in zip(("fc1_0", "fc1_1"), projections, strict=True):
                        mapped[f"{base}{target}{suffix}"] = projection.contiguous()
                    continue
                mapped[name] = tensor
            new_config = dict(peft_config)
            new_config["target_modules"] = sorted(
                {
                    mapped_target
                    for target in requested_targets
                    for supported in (_target_suffix(target, H3_VEOMNI_LORA_TARGETS),)
                    if supported is not None
                    for mapped_target in _VEOMNI_LORA_TARGET_MAPPING[supported]
                }
            )
            return mapped, new_config

        target_suffixes = {
            target: suffix
            for target in requested_targets
            if (suffix := _target_suffix(target, H3_LORA_TARGETS)) is not None
        }
        ff_half = getattr(self, component).arch.ffn_hidden_size
        mapped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            lora_weight = _split_lora_weight_name(name)
            if lora_weight is None:
                mapped[name] = tensor
                continue

            module, suffix = lora_weight
            is_lora_b = suffix == ".lora_B.weight"
            anchors = [
                offset
                for offset in (module.find("transformer_blocks."), module.find("token_refiner.refiner_blocks."))
                if offset >= 0
            ]
            if not anchors:
                raise ValueError(f"MiniMax H3 cannot map LoRA tensor outside supported DiT blocks: {name}.")
            module = module[min(anchors) :]

            if ".ff.net.0.proj" in module:
                base = diffusers_to_vllm_name(module + ".")[:-1].replace(".ff.net.0.proj", ".mlp.fc1")
                if is_lora_b:
                    if tensor.shape[0] != 2 * ff_half:
                        raise ValueError(
                            f"MiniMax H3 fc1 LoRA B rows must be {2 * ff_half}, got {tensor.shape[0]} for {name}."
                        )

                    # A is shared by both logical FC1 slices; B carries the output
                    # rows and must be split and reordered from [up, gate].
                    up, gate = tensor.chunk(2, dim=0)
                    mapped[f"{component}.{base}_0{suffix}"] = gate.contiguous()
                    mapped[f"{component}.{base}_1{suffix}"] = up.contiguous()
                else:
                    mapped[f"{component}.{base}_0{suffix}"] = tensor
                    mapped[f"{component}.{base}_1{suffix}"] = tensor
                continue

            vllm_module = diffusers_to_vllm_name(module + ".")[:-1]
            mapped[f"{component}.{vllm_module}{suffix}"] = tensor

        # Configure only the fused submodules requested by the Actor recipe.
        new_config = dict(peft_config)
        new_config["target_modules"] = sorted(
            {mapped for suffix in target_suffixes.values() for mapped in _LORA_TARGET_MAPPING[suffix]}
        )
        return mapped, new_config

    @staticmethod
    def _validate_diffusion_lora_binding(*, lora_model, bound_lora_names) -> None:
        """Reject partial or no-op adapter binding in the rollout engine."""
        unbound = set(lora_model.loras) - set(bound_lora_names)
        if unbound:
            raise ValueError(
                f"MiniMax H3 LoRA has {len(unbound)} unbound modules; refusing a partial or no-op sync. "
                f"First unbound names: {sorted(unbound)[:5]}"
            )


__all__ = ["H3_LORA_TARGETS", "H3_VEOMNI_LORA_TARGETS", "MiniMaxH3WeightSyncMixin"]
