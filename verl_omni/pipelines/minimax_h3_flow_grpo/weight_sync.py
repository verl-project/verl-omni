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

from collections.abc import Iterable

import torch

# Diffusers and vLLM-Omni use different names for the same H3 modules. QKV
# and GEGLU projections also have different tensor layouts and are handled
# separately in ``load_weights`` below.
_TOPLEVEL_RENAMES = (
    ("audio_proj_in", "audio_patch_proj"),
    ("audio_proj_out", "final_layer.audio_out"),
    ("proj_in", "video_patch_proj"),
    ("proj_out", "final_layer.video_out"),
    ("context_embedder", "condition_proj"),
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"),
)

# These virtual sublayer names let vLLM-Omni load independent Diffusers LoRAs
# into its fused qkv_proj and fc1 modules.
_LORA_STACKED_PARAMS_MAPPING = [
    (".qkv_proj", ".to_q", "q"),
    (".qkv_proj", ".to_k", "k"),
    (".qkv_proj", ".to_v", "v"),
    (".fc1", ".fc1_0", "0"),
    (".fc1", ".fc1_1", "1"),
]
_LORA_TARGET_MAPPING = {
    "to_q": ("to_q",),
    "to_k": ("to_k",),
    "to_v": ("to_v",),
    "to_out.0": ("out_proj",),
    "ff.net.0.proj": ("fc1_0", "fc1_1"),
    "ff.net.2": ("fc2",),
}
H3_LORA_TARGETS = frozenset(_LORA_TARGET_MAPPING)


def _diffusers_to_vllm_name(name: str) -> str:
    """Rename an unfused Diffusers H3 parameter without changing its tensor."""
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    name = name.replace("transformer_blocks.", "blocks.")
    name = name.replace(".attn.norm_q.", ".attn.q_norm.")
    name = name.replace(".attn.norm_k.", ".attn.k_norm.")
    name = name.replace(".attn.to_out.0.", ".attn.out_proj.")
    name = name.replace(".ff.net.2.", ".mlp.fc2.")
    for source, target in _TOPLEVEL_RENAMES:
        if name.startswith(source + "."):
            return target + name[len(source) :]
    return name


def _lora_target_suffix(target: str) -> str | None:
    return next((suffix for suffix in H3_LORA_TARGETS if target == suffix or target.endswith("." + suffix)), None)


# TODO: Remove this MiniMax H3-specific mapping once vLLM-Omni natively
# supports syncing Diffusers full weights and LoRA updates into its fused
# QKV/GEGLU inference layout.
class MiniMaxH3WeightSyncMixin:
    """Translate Diffusers Actor weights before loading them into vLLM-Omni."""

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

            # PEFT's merged full-weight path may retain ``base_layer`` in names.
            inner = inner.replace(".base_layer", "")
            if "lora_" in inner:
                continue

            if inner.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
                block, projection = inner.rsplit(".attn.to_", 1)
                target_name = f"{_diffusers_to_vllm_name(block)}.attn.qkv_proj.weight"
                params = component_params.get(component)
                if params is None:
                    params = component_params[component] = dict(getattr(self, component).named_parameters())
                param = params[target_name]

                # Keep Q/K/V separate. The fused parameter's native loader packs
                # the requested shard and applies the correct TP partition.
                param.weight_loader(param, tensor, projection[0])
                loaded.add(f"{component}.{target_name}")
                continue

            if inner.endswith(".ff.net.0.proj.weight"):
                target_name = _diffusers_to_vllm_name(inner).replace(".ff.net.0.proj.", ".mlp.fc1.")
                params = component_params.get(component)
                if params is None:
                    params = component_params[component] = dict(getattr(self, component).named_parameters())
                param = params[target_name]

                # Diffusers GEGLU stores [up, gate], while H3's fused fc1 expects
                # logical shards [gate, up]. Its loader performs the TP slicing.
                up, gate = tensor.chunk(2, dim=0)
                param.weight_loader(param, gate, 0)
                param.weight_loader(param, up, 1)
                loaded.add(f"{component}.{target_name}")
                continue

            translated.append((f"{component}.{_diffusers_to_vllm_name(inner)}", tensor))

        # Native H3 loading still handles all parameters that only need renaming.
        if translated:
            loaded.update(super().load_weights(translated))
        return loaded

    def install_h3_lora_layout(self) -> None:
        """Expose H3's fused QKV and GEGLU layout to the LoRA manager."""
        transformer = getattr(self, "transformer", None)
        if transformer is not None and not getattr(transformer, "stacked_params_mapping", None):
            transformer.stacked_params_mapping = list(_LORA_STACKED_PARAMS_MAPPING)

    def map_lora_update_to_engine(
        self,
        tensors: dict[str, torch.Tensor],
        peft_config: dict,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Map Diffusers LoRA tensors and targets to fused H3 modules."""
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

        ff_half = self.transformer.arch.ffn_hidden_size
        mapped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            is_lora_a = name.endswith(".lora_A.weight")
            is_lora_b = name.endswith(".lora_B.weight")
            if not (is_lora_a or is_lora_b):
                mapped[name] = tensor
                continue

            suffix = ".lora_A.weight" if is_lora_a else ".lora_B.weight"
            module = name[: -len(suffix)]
            anchors = [
                offset
                for offset in (module.find("transformer_blocks."), module.find("token_refiner.refiner_blocks."))
                if offset >= 0
            ]
            if not anchors:
                raise ValueError(f"MiniMax H3 cannot map LoRA tensor outside supported DiT blocks: {name}.")
            module = module[min(anchors) :]

            if ".ff.net.0.proj" in module:
                base = _diffusers_to_vllm_name(module + ".")[:-1].replace(".ff.net.0.proj", ".mlp.fc1")
                if is_lora_b:
                    if tensor.shape[0] != 2 * ff_half:
                        raise ValueError(
                            f"MiniMax H3 fc1 LoRA B rows must be {2 * ff_half}, got {tensor.shape[0]} for {name}."
                        )

                    # A is shared by both logical FC1 slices; B carries the output
                    # rows and must be split and reordered from [up, gate].
                    up, gate = tensor.chunk(2, dim=0)
                    mapped[f"transformer.{base}_0{suffix}"] = gate.contiguous()
                    mapped[f"transformer.{base}_1{suffix}"] = up.contiguous()
                else:
                    mapped[f"transformer.{base}_0{suffix}"] = tensor
                    mapped[f"transformer.{base}_1{suffix}"] = tensor
                continue

            vllm_module = _diffusers_to_vllm_name(module + ".")[:-1]
            mapped[f"transformer.{vllm_module}{suffix}"] = tensor

        # Configure only the fused submodules requested by the Actor recipe.
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
