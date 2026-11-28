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

"""MiniMax-H3 T2VA rollout rows to VeOmni native DiT and back."""

from __future__ import annotations

import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import h3_ulysses_forward

FUSED_TARGETS = {
    "qkv_proj": ("to_q", "to_k", "to_v"),
    "out_proj": ("out_proj",),
    "fc1": ("fc1_0", "fc1_1"),
    "fc2": ("fc2",),
}


class MiniMaxH3VeOmniMixin:
    """VeOmni engine hooks for the registered H3 FlowGRPO adapter."""

    veomni_fp32_modules = (
        "dit.video_patch_proj",
        "dit.audio_patch_proj",
        "dit.time_embedder.proj_in",
        "dit.time_embedder.proj_out",
        "dit.final_layer.video_out",
        "dit.final_layer.audio_out",
    )

    @staticmethod
    def veomni_export_weight_keys(params: dict[str, torch.Tensor], module: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Strip the H3 training wrapper's leading ``dit.`` from rollout weight names."""
        return {name.removeprefix("dit."): tensor for name, tensor in params.items()}

    @staticmethod
    def _predict_velocities(
        module: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        *,
        training_backend: str | None,
        sp_size: int | None,
        target_only_trajectory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training_backend == "veomni":
            return predict_row_velocities(module, inputs, target_only_trajectory=target_only_trajectory)
        if training_backend in (None, "fsdp", "fsdp2"):
            return h3_ulysses_forward(module, inputs, sp_size)
        raise ValueError(f"Unsupported H3 FlowGRPO training backend: {training_backend!r}")


def predict_row_velocities(module, inputs, *, target_only_trajectory=False):
    """Replay trajectories with a shared layout in one wrapped DiT forward."""
    if target_only_trajectory:
        raise ValueError("VeOmni H3 T2VA replay does not support target-only trajectories.")

    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import pack_audio, patchify_video

    video, audio, text = (inputs[key] for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"))
    batch = video.size(0)
    if not batch or audio.size(0) != batch or text.size(0) != batch:
        raise ValueError("H3 velocity replay requires the same nonempty video, audio and text batch.")
    positions = inputs["position_ids"]
    length = len(positions)
    vi, ai, ti = (inputs[key].long() for key in ("video_indices", "audio_indices", "text_indices"))
    if (vi.numel(), ai.numel(), ti.numel()) != (video.size(1), audio.size(1), text.size(1)):
        raise ValueError("H3 velocity replay row counts disagree with the three modality index arrays.")

    def segment(n):
        bounds = (0, n)
        return {"cu_seqlens_q": vi.new_tensor(bounds, dtype=torch.int32), "cu_seqlens_host": bounds, "max_seqlen_q": n}

    video_rows = video.new_zeros(batch, length, video.size(-1)).index_copy_(1, vi, video)
    audio_rows = audio.new_zeros(batch, length, audio.size(-1)).index_copy_(1, ai, audio)
    # Lists select VeOmni's packed forward. It isolates attention and refiner context per trajectory.
    output = module(
        x=video_rows if batch == 1 else list(video_rows.split(1)),
        audio_x=audio_rows if batch == 1 else list(audio_rows.split(1)),
        prompt_embeds=text[0] if batch == 1 else list(text.unbind()),
        img_position_ids=positions.unsqueeze(0),
        unique_timesteps=inputs["timestep"],
        inverse_indices=inputs["timestep_indices"],
        token_tags=inputs["token_tags"],
        img_pos_info={"position_ids": vi},
        audio_pos_info={"position_ids": ai},
        text_pos_info={"position_ids": ti},
        img_pos_for_infer_output_info={"position_ids": vi},
        packed_seq_params=segment(length),
        refiner_packed_seq_params=segment(ti.numel()),
        # These shapes are consumed only by the final unpatchify/unpack. The actual
        # spatial and temporal coordinates went into the DiT through img_position_ids.
        video_latent_shape=(vi.numel(), 1, 1),
        audio_latent_shape=(1, ai.numel()),
        skip_mask_out_condition=True,
    )
    video_pred, audio_pred = output.predictions
    if batch == 1:
        return -patchify_video(video_pred).unsqueeze(0), -pack_audio(audio_pred).unsqueeze(0)
    return -torch.stack([patchify_video(v) for v in video_pred]), -torch.stack([pack_audio(a) for a in audio_pred])


def validate_veomni_lora_targets(requested: set[str]) -> None:
    """Check that native target modules can be synchronized to the rollout."""
    if not requested or not requested <= FUSED_TARGETS.keys():
        raise ValueError(f"VeOmni H3 LoRA supports {sorted(FUSED_TARGETS)}, got {sorted(requested)}.")


def map_fused_lora(tensors, targets, component, arch):
    """Convert native PEFT matrices into vLLM logical Q/K/V and gate/up slices."""
    mapped = {}
    pairs = {}
    for name, tensor in tensors.items():
        path, kind, weight = name.rsplit(".", 2)
        if weight != "weight" or kind not in ("lora_A", "lora_B"):
            raise ValueError(f"Unexpected H3 adapter tensor {name}")
        path = path.removeprefix("transformer.").removeprefix("base_model.model.").removeprefix("dit.")
        parent, leaf = path.rsplit(".", 1)
        if leaf not in targets or not parent.startswith(("blocks.", "token_refiner.blocks.")):
            raise ValueError(f"Unbound H3 LoRA target {name}")
        kinds = pairs.setdefault(path, set())
        if kind in kinds:
            raise ValueError(f"Duplicate H3 LoRA tensor for {path}.{kind}: {name}")
        kinds.add(kind)
        logical = FUSED_TARGETS[leaf]
        parts = (tensor,) * len(logical)
        if kind == "lora_B" and leaf == "qkv_proj":
            heads, dim = arch.num_attention_heads, arch.attention_head_dim
            if tensor.size(0) != heads * 3 * dim:
                raise ValueError(f"Unexpected H3 QKV LoRA B shape {tuple(tensor.shape)}")
            parts = [p.reshape(heads * dim, -1) for p in tensor.reshape(heads, 3, dim, -1).unbind(1)]
        elif kind == "lora_B" and leaf == "fc1":
            if tensor.size(0) != 2 * arch.ffn_hidden_size:
                raise ValueError(f"Unexpected H3 GEGLU LoRA B shape {tuple(tensor.shape)}")
            parts = tensor.chunk(2, dim=0)
        for target, part in zip(logical, parts, strict=True):
            mapped[f"{component}.{parent}.{target}.{kind}.weight"] = part.contiguous()
    if not mapped or any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("H3 adapter sync requires an A/B pair for each supplied module.")
    return mapped
