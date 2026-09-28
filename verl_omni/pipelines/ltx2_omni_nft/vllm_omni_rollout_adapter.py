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

"""vLLM-Omni rollout adapter for LTX-2.3 OmniNFT."""

from __future__ import annotations

import os
from copy import copy
from dataclasses import replace

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXForwardContext
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

__all__ = ["LTX23OmniNFTPipeline"]


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="omni_nft")
class LTX23OmniNFTPipeline(LTX2Pipeline):
    """Sample one text-to-AV request and retain clean OmniNFT training tensors."""

    supports_request_batch = False
    support_image_input = False
    diffusion_io_spec = DiffusionIOSpec(
        primary=MediaSpec("video"),
        auxiliary=(MediaSpec("audio", sample_rate=24000),),
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        show_progress = os.environ.get("OMNIFT_ROLLOUT_PROGRESS", "").strip().lower() in {"1", "true", "yes"}
        self.set_progress_bar_config(desc="LTX denoise", disable=not show_progress)
        self._omni_nft_clean_state: LTXAVState | None = None
        self._omni_nft_forward_context: LTXForwardContext | None = None

    def _encode_prompt(
        self,
        token_ids: torch.Tensor | list[int],
        attention_mask: torch.Tensor | list[int] | None,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one tokenized prompt with LTX's left padding and hidden-layer layout."""
        token_ids = torch.as_tensor(token_ids, device=self.device, dtype=torch.long).reshape(1, -1)
        attention_mask = (
            torch.ones_like(token_ids, dtype=torch.bool)
            if attention_mask is None
            else torch.as_tensor(attention_mask, device=self.device, dtype=torch.bool).reshape(1, -1)
        )
        token_ids = token_ids[:, :max_sequence_length]
        attention_mask = attention_mask[:, :max_sequence_length]
        pad_length = max_sequence_length - token_ids.shape[1]
        if pad_length > 0:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            token_ids = torch.nn.functional.pad(token_ids, (pad_length, 0), value=pad_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (pad_length, 0), value=False)
        hidden_states = self.text_encoder(
            input_ids=token_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states
        embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=self.text_encoder.dtype)
        return embeds, attention_mask

    def _prepare_request(self, request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        """Prepare native tensor output and prompt embeddings without modifying the caller."""
        request = copy(request)
        request.sampling_params = copy(request.sampling_params)
        request.sampling_params.output_type = "pt"
        request.prompt = dict(request.prompt)
        # Native warmup may infer image support from the base pipeline.
        if OmniDiffusionRequest.is_dummy_run_request_id(request.request_id):
            multi_modal_data = dict(request.prompt.get("multi_modal_data") or {})
            multi_modal_data.pop("image", None)
            request.prompt["multi_modal_data"] = multi_modal_data

        max_length = request.sampling_params.max_sequence_length or self.tokenizer_max_length
        for prefix, token_key in (("", "prompt_token_ids"), ("negative_", "negative_prompt_ids")):
            token_ids = request.prompt.get(token_key)
            if token_ids is not None:
                embeds, mask = self._encode_prompt(token_ids, request.prompt.get(f"{prefix}prompt_mask"), max_length)
                # Native LTX stacks each request's unbatched embeddings and masks.
                request.prompt[f"{prefix}prompt_embeds"] = embeds[0]
                request.prompt[f"{prefix}prompt_attention_mask"] = mask[0]
        # Raw-text warmup prompts use the native text encoder path.
        return request

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture normalized clean tokens before native unpacking and decoding."""
        self._omni_nft_clean_state = LTXAVState(video=latents, audio=audio_latents)
        self._omni_nft_forward_context = forward_ctx
        return super()._unpack_and_denormalize_stage(forward_ctx, latents, audio_latents)

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Generate one request and attach replay data, releasing captured state on exit."""
        try:
            request = self._prepare_request(req.requests[0])
            output = super().forward(DiffusionRequestBatch(requests=[request]))
            state = self._omni_nft_clean_state
            context = self._omni_nft_forward_context
            prompt = context.prompt_context
            device = state.video.device
            # Sequence-parallel padding must not enter the training audio tokens.
            audio_latents = state.audio[:, : context.original_audio_num_frames]
            video, audio = output.output
            if video.ndim == 5:
                video = video[0]
            return with_rollout_data(
                replace(output, output=(video, audio)),
                media_key="video",
                prompt_embeddings={
                    "prompt_embeds": prompt.positive_connector_prompt_embeds,
                    "audio_prompt_embeds": prompt.positive_connector_audio_prompt_embeds,
                    "prompt_embeds_mask": prompt.positive_connector_attention_mask,
                    "negative_prompt_embeds": prompt.negative_connector_prompt_embeds,
                    "negative_audio_prompt_embeds": prompt.negative_connector_audio_prompt_embeds,
                    "negative_prompt_embeds_mask": prompt.negative_connector_attention_mask,
                },
                rl={
                    "audio": audio,
                    "video_latents_clean": state.video.float(),
                    "audio_latents_clean": audio_latents.float(),
                    "train_timesteps": context.timesteps.to(device=device, dtype=torch.float32).unsqueeze(0),
                    "video_seq_len": torch.tensor([state.video.shape[1]], device=device),
                    "fps": torch.tensor([context.request_inputs.frame_rate], device=device, dtype=torch.float32),
                    "audio_sample_rate": torch.tensor([self.vocoder.config.output_sampling_rate], device=device),
                },
                to_cpu=True,
            )
        finally:
            self._omni_nft_clean_state = None
            self._omni_nft_forward_context = None
