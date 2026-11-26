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

"""SD3.5 FlowGRPO rollout on vLLM-Omni's diffusers backend."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline

from verl_omni.pipelines.model_base import VllmOmniDiffusersPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.sd3_flow_grpo.common import SD3TokenIdPromptMixin
from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

__all__ = ["StableDiffusion3DiffusersPipelineWithLogProb"]

_TRANSFORMER_PREFIX = "transformer."


@VllmOmniDiffusersPipelineBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")
class StableDiffusion3DiffusersPipelineWithLogProb(SD3TokenIdPromptMixin, CFGParallelMixin, DiffusersAdapterPipeline):
    """SD3.5 FlowGRPO rollout that denoises with the diffusers modules the trainer optimizes.

    ``DiffusersAdapterPipeline`` loads the checkpoint with ``from_pretrained``;
    vLLM-Omni's server, sleep/wake and weight transfer are unchanged. Sampling
    and the returned trajectory are those of :class:`StableDiffusion3PipelineWithLogProb`,
    whose ``forward`` runs here on the diffusers ``SD3Transformer2DModel``.
    Selected with ``engine_kwargs.vllm_omni.diffusion_load_format=diffusers``.
    """

    always_encode_negative_prompt = False
    supports_request_batch = True
    emit_clean_latents = False
    diffusion_io_spec = StableDiffusion3PipelineWithLogProb.diffusion_io_spec

    forward = StableDiffusion3PipelineWithLogProb.forward
    diffuse = StableDiffusion3PipelineWithLogProb.diffuse
    _decode_latents = StableDiffusion3PipelineWithLogProb._decode_latents
    _model_dtype = StableDiffusion3PipelineWithLogProb._model_dtype
    prepare_latents = StableDiffusion3Pipeline.prepare_latents
    prepare_timesteps = StableDiffusion3Pipeline.prepare_timesteps
    interrupt = StableDiffusion3Pipeline.interrupt

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config)
        self.device = get_local_device()
        super().load_weights()
        pipe = self._pipeline
        # Registered as submodules so the loader's ``.to(device)`` and module discovery see them.
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.tokenizer_max_length = pipe.tokenizer_max_length
        self.vae_scale_factor = pipe.vae_scale_factor
        self.default_sample_size = pipe.default_sample_size
        self.output_type = od_config.output_type
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_config(pipe.scheduler.config)

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        prompt_3: str | list[str] | None = None,
        max_sequence_length: int = 256,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts (warm-up and serving requests) into SD3 sequence and pooled embeddings."""
        prompt_embeds, _, pooled_prompt_embeds, _ = self._pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        return prompt_embeds, pooled_prompt_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Copy synced transformer weights in place.

        Args:
            weights: ``(name, tensor)`` pairs named ``transformer.<diffusers parameter name>``,
                as the trainer exports them.

        Raises:
            KeyError: If a name is not a parameter or buffer of the transformer.
            ValueError: If a tensor's shape does not match its parameter.
        """
        targets = dict(self.transformer.named_parameters())
        targets.update(self.transformer.named_buffers())
        with torch.no_grad():
            for name, tensor in weights:
                target = targets.get(name[len(_TRANSFORMER_PREFIX) :]) if name.startswith(_TRANSFORMER_PREFIX) else None
                if target is None:
                    raise KeyError(f"Synced weight {name!r} has no matching transformer parameter.")
                if target.shape != tensor.shape:
                    raise ValueError(
                        f"Synced weight {name!r} has shape {tuple(tensor.shape)}, expected {tuple(target.shape)}."
                    )
                target.copy_(tensor)
