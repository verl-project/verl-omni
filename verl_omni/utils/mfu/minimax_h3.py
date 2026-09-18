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

"""FLOPs estimator for MiniMax-H3's packed multimodal DiT."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from verl_omni.utils.mfu.diffusion_flops_counter import (
    DiffusionModelFlops,
    register_diffusion_architecture,
    sum_seqlens,
)

__all__ = ["MiniMaxH3Flops"]


@register_diffusion_architecture("MiniMaxH3Pipeline")
class MiniMaxH3Flops(DiffusionModelFlops):
    """FLOPs estimator for ``MiniMaxH3Transformer3DModel``.

    MiniMax-H3 projects text, video, and audio rows into one packed sequence.
    The main DiT blocks run full self-attention over that sequence, while the
    token refiner runs only on text rows.
    """

    def _config_int(self, *names: str) -> int:
        for name in names:
            value = self.config.get(name)
            if value is not None:
                return int(value)
        raise KeyError(f"MiniMax-H3 FLOPs config requires one of {names!r}.")

    @staticmethod
    def _count_values(value: Any, batch_size: int) -> list[int] | None:
        if value is None:
            return [0] * batch_size
        if not isinstance(value, torch.Tensor):
            return None
        values = value.detach().reshape(-1)
        if values.numel() == 1:
            return [int(values.item())] * batch_size
        if values.numel() == batch_size:
            return [int(item) for item in values.tolist()]
        return None

    def _get_stream_seqlens(self, data: Any) -> tuple[list[int], list[int]]:
        video_rows = data.get("h3_video_rows")
        audio_rows = data.get("h3_audio_rows")
        if video_rows is not None and audio_rows is not None:
            if not isinstance(video_rows, torch.Tensor) or not isinstance(audio_rows, torch.Tensor):
                return [], []
            batch_size = int(video_rows.shape[0])
            video_counts = self._count_values(video_rows, batch_size)
            audio_counts = self._count_values(audio_rows, batch_size)
            return (video_counts, audio_counts) if video_counts is not None and audio_counts is not None else ([], [])

        latent_meta = data.get("latent_meta")
        if not isinstance(latent_meta, torch.Tensor) or latent_meta.ndim != 2 or latent_meta.shape[1] < 2:
            return [], []

        batch_size = int(latent_meta.shape[0])
        video_counts = [int(item) for item in latent_meta[:, 0].detach().tolist()]
        audio_counts = [int(item) for item in latent_meta[:, 1].detach().tolist()]
        condition_video = self._count_values(data.get("condition_video_row_count"), batch_size)
        condition_audio = self._count_values(data.get("condition_audio_row_count"), batch_size)
        if condition_video is None or condition_audio is None:
            return [], []
        return (
            [video + condition for video, condition in zip(video_counts, condition_video, strict=True)],
            [audio + condition for audio, condition in zip(audio_counts, condition_audio, strict=True)],
        )

    def get_latent_seqlens(self, data: Any) -> list[int]:
        """Return non-text packed rows: target plus optional reference rows."""
        video_seqlens, audio_seqlens = self._get_stream_seqlens(data)
        return [video + audio for video, audio in zip(video_seqlens, audio_seqlens, strict=True)]

    def _get_timestep_groups(self, data: Any) -> list[int]:
        """Distinct timestep embeddings each forward runs, per sample.

        H3 denoises a video and an audio target stream at (generally) distinct
        timesteps, and every present reference/condition modality adds one
        frozen timestep (``build_row_timesteps`` returns ``torch.unique`` of the
        row timesteps). AdaLN/timestep projections run once per distinct value.
        """
        latent_meta = data.get("latent_meta")
        if isinstance(latent_meta, torch.Tensor) and latent_meta.ndim == 2 and latent_meta.shape[1] >= 2:
            batch_size = int(latent_meta.shape[0])
            condition_video = self._count_values(data.get("condition_video_row_count"), batch_size)
            condition_audio = self._count_values(data.get("condition_audio_row_count"), batch_size)
            if condition_video is not None and condition_audio is not None:
                video = [int(item) for item in latent_meta[:, 0].detach().tolist()]
                audio = [int(item) for item in latent_meta[:, 1].detach().tolist()]
                return [
                    int(v > 0) + int(a > 0) + int(cv > 0) + int(ca > 0)
                    for v, a, cv, ca in zip(video, audio, condition_video, condition_audio, strict=True)
                ]

        # FlowGRPO packed layout does not expose the condition split; count the
        # video and audio target streams that are present.
        video_seqlens, audio_seqlens = self._get_stream_seqlens(data)
        return [int(video > 0) + int(audio > 0) for video, audio in zip(video_seqlens, audio_seqlens, strict=True)]

    def collect_meta(self, data: Any) -> dict[str, list[int]]:
        video_seqlens, audio_seqlens = self._get_stream_seqlens(data)
        return {
            "latent_seqlens": [video + audio for video, audio in zip(video_seqlens, audio_seqlens, strict=True)],
            "prompt_seqlens": list(self.get_prompt_seqlens(data)),
            "video_seqlens": video_seqlens,
            "audio_seqlens": audio_seqlens,
            "timestep_group_seqlens": self._get_timestep_groups(data),
        }

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        num_forward_passes: int,
        video_seqlens: Sequence[int] | None = None,
        audio_seqlens: Sequence[int] | None = None,
        timestep_group_seqlens: Sequence[int] | None = None,
    ) -> float:
        hidden_size = self._config_int("hidden_size")
        num_heads = self._config_int("num_attention_heads")
        head_dim = self._config_int("attention_head_dim")
        inner_dim = num_heads * head_dim
        num_layers = self._config_int("num_layers")
        num_refiner_layers = self._config_int("num_refiner_layers", "token_refiner_num_layers")
        ffn_dim = self._config_int("ffn_dim", "ffn_hidden_size")
        in_channels = self._config_int("in_channels", "latents_dim")
        audio_in_channels = self._config_int("audio_in_channels", "audio_latents_dim")
        text_dim = self._config_int("text_dim")
        freq_dim = self._config_int("freq_dim", "timestep_input_dim")
        time_embed_hidden_dim = self._config_int("time_embed_hidden_dim", "time_embed_hidden_size")
        time_embed_dim = self._config_int("time_embed_dim")
        patch_size = self.config.get("patch_size", (1, 2, 2))
        patch_volume = 1
        for size in patch_size:
            patch_volume *= int(size)
        video_patch_dim = in_channels * patch_volume

        latent_total = sum_seqlens(latent_seqlens)
        prompt_total = sum_seqlens(prompt_seqlens)
        if video_seqlens is None or audio_seqlens is None:
            video_seqlens = latent_seqlens
            audio_seqlens = [0] * len(latent_seqlens)
        if len(video_seqlens) != len(latent_seqlens) or len(audio_seqlens) != len(latent_seqlens):
            raise ValueError("MiniMax-H3 video/audio sequence lengths must match latent_seqlens.")
        video_total = sum_seqlens(video_seqlens)
        audio_total = sum_seqlens(audio_seqlens)
        if timestep_group_seqlens is None:
            timestep_groups = max(len(latent_seqlens), len(prompt_seqlens))
        elif len(timestep_group_seqlens) != len(latent_seqlens):
            raise ValueError("MiniMax-H3 timestep group counts must match latent_seqlens.")
        else:
            timestep_groups = sum_seqlens(timestep_group_seqlens)

        # QKV/out project hidden_size <-> inner_dim; SwiGLU uses two input
        # projections and one output projection.
        block_token_params = 4 * hidden_size * inner_dim + 3 * hidden_size * ffn_dim
        refiner_token_params = block_token_params

        main_dense = self.compute_dense_flops(num_layers * block_token_params, latent_total + prompt_total)
        refiner_dense = self.compute_dense_flops(num_refiner_layers * refiner_token_params, prompt_total)

        input_params = (
            video_patch_dim * hidden_size * video_total
            + audio_in_channels * hidden_size * audio_total
            + text_dim * hidden_size * prompt_total
        )
        # Both output heads run on every packed row before modality rows are selected.
        output_params = hidden_size * (video_patch_dim + audio_in_channels) * (latent_total + prompt_total)
        input_output_dense = self.compute_dense_flops(input_params + output_params, 1)

        # AdaLN/timestep projections run once per distinct timestep embedding.
        timestep_params = freq_dim * time_embed_hidden_dim + time_embed_hidden_dim * time_embed_dim
        adaln_params = num_layers * time_embed_dim * 18 * hidden_size + time_embed_dim * 2 * hidden_size
        time_dense = self.compute_dense_flops(timestep_params + adaln_params, timestep_groups)

        main_attention = self.compute_attention_flops(latent_seqlens, prompt_seqlens)
        refiner_attention = 12 * num_refiner_layers * inner_dim * sum(int(length) ** 2 for length in prompt_seqlens)

        flops_all_steps = (
            (main_dense + refiner_dense + input_output_dense + time_dense + main_attention + refiner_attention)
            * num_timesteps
            * num_forward_passes
        )
        return flops_all_steps / delta_time / 1e12
