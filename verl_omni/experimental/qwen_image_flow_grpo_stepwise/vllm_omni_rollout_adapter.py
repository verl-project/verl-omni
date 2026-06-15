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

"""Experimental step-wise batching execution support for FlowGRPO.

This module provides :class:`QwenImagePipelineWithLogProbStepwise`, which
extends the standard FlowGRPO rollout pipeline with ``prepare_encode``,
``step_scheduler``, ``post_decode``, and ``denoise_step`` overrides that
enable per-step execution with SDE trajectory collection.
"""

import copy
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import build_img_shapes, coalesce_not_none, maybe_to_cpu
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

__all__ = ["QwenImagePipelineWithLogProbStepwise"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo_stepwise")
class QwenImagePipelineWithLogProbStepwise(QwenImagePipelineWithLogProb):
    """Experimental stepwise-execution variant of the FlowGRPO rollout pipeline.

    Extends :class:`QwenImagePipelineWithLogProb` with overrides that make
    the ``step_execution=True`` engine path (``prepare_encode`` →
    ``step_scheduler`` → ``post_decode``) equivalent to the full
    ``forward()`` path in terms of SDE noise injection, log-prob collection,
    and trajectory packaging.
    """

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        """Encode text prompt token IDs into dense embeddings.

        Args:
            prompt_ids (torch.Tensor): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (torch.Tensor, *optional*): Boolean mask of shape
                ``(B, L)`` for *prompt_ids*; inferred as all-ones when ``None``.
            num_images_per_prompt (int): Number of images to generate per prompt;
                embeddings are repeated accordingly.
            prompt_embeds (torch.Tensor, *optional*): Pre-computed embeddings;
                when provided *prompt_ids* is ignored.
            prompt_embeds_mask (torch.Tensor, *optional*): Attention mask for
                pre-computed *prompt_embeds*.
            max_sequence_length (int): Maximum sequence length; embeddings are
                truncated to this value.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair of
                ``(prompt_embeds, prompt_embeds_mask)`` tensors of shape
                ``(B * num_images_per_prompt, L, D)`` and
                ``(B * num_images_per_prompt, L)`` respectively.
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def _extract_prompt_ids(self, prompts):
        """Extract prompt_ids/mask and their negatives from the OmniCustomPrompt list.

        Falls back to tokenizing ``"prompt"`` / ``"negative_prompt"`` text fields
        when ``prompt_ids`` is not provided (e.g. during the engine's dummy
        warm-up run, which always submits a text prompt).
        """
        prompt_ids = None
        prompt_mask = None
        negative_prompt_ids = None
        negative_prompt_mask = None
        if prompts:
            p0 = prompts[0]
            if isinstance(p0, dict):
                prompt_ids = p0.get("prompt_token_ids", None)
                prompt_mask = p0.get("prompt_mask", None)
                negative_prompt_ids = p0.get("negative_prompt_ids", None)
                negative_prompt_mask = p0.get("negative_prompt_mask", None)

                # Fallback: tokenize raw text prompt (covers _dummy_run path).
                if prompt_ids is None and p0.get("prompt"):
                    prompt_ids, prompt_mask = self._tokenize_text_prompt(p0["prompt"])
                if negative_prompt_ids is None and p0.get("negative_prompt"):
                    negative_prompt_ids, negative_prompt_mask = self._tokenize_text_prompt(p0["negative_prompt"])
            elif isinstance(p0, str):
                prompt_ids, prompt_mask = self._tokenize_text_prompt(p0)
        return prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask

    def _tokenize_text_prompt(self, text: str | list[str]):
        """Tokenize a text prompt using the Qwen chat template (parent behavior)."""
        prompt = [text] if isinstance(text, str) else text
        txt = [self.prompt_template_encode.format(e) for e in prompt]
        tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return tokens.input_ids, tokens.attention_mask

    def prepare_encode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> "DiffusionRequestState":
        """Populate *state* with encoded prompts, latents, timesteps, and CFG config.

        Override of ``QwenImagePipeline.prepare_encode`` that accepts pre-tokenized
        ``prompt_ids`` (and optional ``prompt_mask``) instead of raw text prompts,
        matching the input contract of ``QwenImagePipelineWithLogProb``.
        """
        sampling = state.sampling
        prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask = self._extract_prompt_ids(
            state.prompts or []
        )

        # Normalize list inputs to tensors on device.
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        if prompt_ids is None:
            raise ValueError(
                "QwenImagePipelineWithLogProbStepwise.prepare_encode requires either "
                "'prompt_ids' or a text 'prompt' in state.prompts[0]."
            )

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling.num_inference_steps or 50
        sigmas = sampling.sigmas
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        true_cfg_scale = sampling.true_cfg_scale or 4.0
        max_sequence_length = sampling.max_sequence_length or self.tokenizer_max_length

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = kwargs.get("attention_kwargs") or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        else:
            batch_size = 1

        has_neg_prompt = negative_prompt_ids is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            self.device,
            generator,
            None,
        )

        img_shapes = build_img_shapes(height, width, batch_size, self.vae_scale_factor)

        timesteps, _ = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Set RoPE length from padded embed width (match diffusers text_seq_len),
        # not from mask.sum() (valid token count).  When Continuous Batching
        # pads requests to a shared target_seq_len, mask.sum() would give
        # different per-request RoPE lengths even though the embeddings have
        # been padded to a uniform width.
        txt_seq_lens = [int(prompt_embeds.shape[1])] * int(prompt_embeds.shape[0])
        if negative_prompt_embeds is not None:
            neg_seq_len = int(negative_prompt_embeds.shape[1])
            negative_txt_seq_lens = [neg_seq_len] * int(negative_prompt_embeds.shape[0])
        else:
            negative_txt_seq_lens = None

        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        # Resolve SDE / log-prob knobs from sampling extra_args so that the
        # step-execution path mirrors ``forward()``'s rollout behaviour.
        extra = sampling.extra_args or {}
        noise_level = coalesce_not_none(extra.get("noise_level", None), 0.7)
        sde_window_size = coalesce_not_none(extra.get("sde_window_size", None), None)
        sde_window_range = coalesce_not_none(extra.get("sde_window_range", None), (0, 5))
        sde_type = coalesce_not_none(extra.get("sde_type", None), "sde")
        logprobs = coalesce_not_none(extra.get("logprobs", None), True)
        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            sde_window = (start, start + sde_window_size)
        else:
            sde_window = (0, len(timesteps) - 1)

        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.negative_prompt_embeds = negative_prompt_embeds
        state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = do_true_cfg
        state.guidance = guidance
        state.img_shapes = img_shapes
        state.txt_seq_lens = txt_seq_lens
        state.negative_txt_seq_lens = negative_txt_seq_lens
        state.sampling.cfg_normalize = True
        # Persist the resolved generator so ``step_scheduler`` (executed
        # one step at a time by the step-execution engine) keeps drawing
        # from the same RNG stream as ``forward()``.
        state.sampling.generator = generator
        # Rollout / SDE state consumed by ``step_scheduler`` and packaged
        # into ``custom_output`` by ``post_decode``.
        state.sde_window = sde_window
        state.noise_level = noise_level
        state.sde_type = sde_type
        state.logprobs = logprobs
        state.all_latents = []
        state.all_log_probs = []
        state.all_timesteps = []

        return state

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One scheduler step that mirrors the per-iter body of :meth:`diffuse`.

        The default ``QwenImagePipeline.step_scheduler`` calls the standard
        scheduler.step without SDE noise / log-prob bookkeeping, which means
        ``step_execution=True`` would silently drop ``all_latents`` /
        ``all_log_probs`` / ``all_timesteps`` (and the
        ``prompt_embeds_mask`` consumer downstream would then receive a
        ``None`` value that turns into a non-tensor ``LinkedList`` inside the
        training ``TensorDict``).  Override here to keep the step-mode and
        request-mode trajectories equivalent.
        """
        del kwargs
        if self.interrupt:
            return

        i = state.step_index
        timestep_value = state.timesteps[i]
        sde_window = state.sde_window

        if i < sde_window[0]:
            cur_noise_level = 0.0
        elif i == sde_window[0]:
            cur_noise_level = state.noise_level
            state.all_latents.append(state.latents.to(torch.float32))
        elif i > sde_window[0] and i < sde_window[1]:
            cur_noise_level = state.noise_level
        else:
            cur_noise_level = 0.0

        new_latents, log_prob, _, _ = state.scheduler.step(
            noise_pred.to(torch.float32),
            timestep_value,
            state.latents.to(torch.float32),
            generator=state.sampling.generator,
            noise_level=cur_noise_level,
            sde_type=state.sde_type,
            return_logprobs=state.logprobs,
            return_dict=False,
        )
        # Save fp32 trajectory so the trainer later recomputes log-probs on
        # full-precision latents.
        if i >= sde_window[0] and i < sde_window[1]:
            state.all_latents.append(new_latents.to(torch.float32))
            state.all_log_probs.append(log_prob)
            state.all_timesteps.append(timestep_value)

        # Keep live state in fp32 for the whole trajectory. ``denoise_step``
        # already casts latents to the transformer dtype before the forward
        # pass, so storing model_dtype here is unnecessary. More importantly,
        # under continuous batching the engine gathers ``state.latents`` across
        # all in-flight requests: a freshly-added request still holds fp32
        # latents from ``prepare_encode`` while stepped requests would hold
        # model-dtype latents, producing a "Mixed dtypes in latents batch"
        # error. Keeping fp32 throughout makes the batch dtype consistent.
        state.latents = new_latents.to(torch.float32)

        state.step_index += 1

    def denoise_step(self, input_batch, **kwargs):
        del kwargs
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t
        x = input_batch.latents.to(self.transformer.img_in.weight.dtype)

        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=x,
            timestep=t,
            guidance=input_batch.guidance,
            prompt_embeds=input_batch.prompt_embeds,
            prompt_embeds_mask=input_batch.prompt_embeds_mask,
            img_shapes=input_batch.img_shapes,
            txt_seq_lens=input_batch.txt_seq_lens,
            do_true_cfg=input_batch.do_true_cfg,
            negative_prompt_embeds=input_batch.negative_prompt_embeds,
            negative_prompt_embeds_mask=input_batch.negative_prompt_embeds_mask,
            negative_txt_seq_lens=input_batch.negative_txt_seq_lens,
            extra_transformer_kwargs={"attention_kwargs": self.attention_kwargs, "return_dict": False},
        )
        noise_pred = self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
            output_slice,
        )
        return noise_pred.float()  # step_scheduler expects fp32 noise_pred

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents, package rollout trajectory, and move to CPU.

        In ``step_execution`` mode the worker ships the returned
        :class:`DiffusionOutput` across an inter-process MessageQueue to the
        ``vLLMOmniHttpServer`` actor.  We must (a) move tensors to CPU so the
        receiving process does not initialise a stray CUDA context on GPU 0,
        and (b) populate ``custom_output`` with the trajectory fields that
        :meth:`forward` produces, so downstream consumers
        (``vllm_omni_async_server.generate`` ->
        ``embeds_padding_2_no_padding``) receive real tensors rather than
        ``None`` (which becomes a non-tensor ``LinkedList`` in the
        ``TensorDict`` and breaks ``mask.shape[0]``).
        """
        output = super().post_decode(state, **kwargs)
        if not isinstance(output, DiffusionOutput):
            return output

        all_latents = state.all_latents
        all_log_probs = state.all_log_probs
        all_timesteps = state.all_timesteps

        stacked_latents = torch.stack(all_latents, dim=1) if all_latents else None
        stacked_log_probs = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        stacked_timesteps = (
            torch.stack(all_timesteps).unsqueeze(0).expand(state.latents.shape[0], -1) if all_timesteps else None
        )

        output.output = maybe_to_cpu(output.output)
        output.custom_output = {
            "all_latents": maybe_to_cpu(stacked_latents),
            "all_log_probs": maybe_to_cpu(stacked_log_probs),
            "all_timesteps": maybe_to_cpu(stacked_timesteps),
            "prompt_embeds": maybe_to_cpu(state.prompt_embeds),
            "prompt_embeds_mask": maybe_to_cpu(state.prompt_embeds_mask),
            "negative_prompt_embeds": maybe_to_cpu(state.negative_prompt_embeds),
            "negative_prompt_embeds_mask": maybe_to_cpu(state.negative_prompt_embeds_mask),
        }
        return output
