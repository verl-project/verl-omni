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
"""Qwen-Image 2.1 的 token-ID rollout 与 NFT 数据输出。"""

import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

from verl_omni.pipelines.diffusion_rollout_output import (
    install_rollout_postprocessor,
    wants_decoded_preview,
    with_rollout_data,
    with_visual_artifacts,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageLoRAMixin
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.rollout_request import prompt_ids_from_payload

install_rollout_postprocessor(pipeline_qwen_image_21, "get_qwen_image_21_post_process_func")


@VllmOmniPipelineBase.register("QwenImage21Pipeline", algorithm="diffusion_nft")
class QwenImage21DiffusionNFTPipeline(QwenImageLoRAMixin, QwenImage21Pipeline):
    """复用上游 VAE、scheduler 和权重 loader；训练请求只接收已分词的提示。"""

    supports_request_batch = True
    supports_step_execution = False
    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "LC"),
        }
    )

    def encode_prompt(
        self,
        prompt,
        images_per_prompt=None,
        num_images_per_prompt=1,
        max_sequence_length=None,
        prompt_name="prompt",
    ):
        """复用 agent 的 token IDs，只运行图像 processor 和冻结的 Qwen3-VL。"""
        if not all(isinstance(item, dict) for item in prompt):
            # 上游 dummy warm-up 没有 agent；真实请求在 forward 中拒绝此路径。
            return super().encode_prompt(
                prompt, images_per_prompt, num_images_per_prompt, max_sequence_length, prompt_name
            )
        ids = []
        for item in prompt:
            tokens = torch.as_tensor(item["ids"], dtype=torch.long, device=self.device)
            if tokens.ndim != 1:
                raise ValueError("Each tokenized request must contain a one-dimensional token sequence.")
            mask = item.get("mask")
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                if mask.shape != tokens.shape:
                    raise ValueError("Prompt token mask must match token IDs.")
                tokens = tokens[mask]
            if len(tokens) <= self._drop_idx:
                raise ValueError("Tokenized prompt is missing its system/user template.")
            ids.append(tokens)
        input_ids = pad_sequence(
            ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id, padding_side="left"
        )
        attention_mask = pad_sequence([torch.ones_like(item) for item in ids], batch_first=True, padding_side="left")
        image_mask = input_ids == self._img_token_id
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": image_mask.long(),
            "output_hidden_states": True,
        }
        images_per_prompt = images_per_prompt or [[] for _ in ids]
        if len(images_per_prompt) != len(ids):
            raise ValueError("Condition images must align with tokenized requests.")
        flat_images = []
        for images in images_per_prompt:
            for image in images or []:
                if image.mode == "RGBA":
                    white = Image.new("RGB", image.size, (255, 255, 255))
                    white.paste(image, mask=image.getchannel("A"))
                    image = white
                flat_images.append(image)
        if flat_images:
            image_inputs = self.processor.image_processor(images=flat_images, return_tensors="pt")
            tokens_per_image = image_inputs["image_grid_thw"].prod(-1) // self.processor.image_processor.merge_size**2
            tokens_per_request = torch.stack(
                [counts.sum() for counts in tokens_per_image.split([len(images or []) for images in images_per_prompt])]
            ).to(image_mask.device)
            # Encoder 只核对整批总数；逐请求核对，避免一条请求使用另一条的图像特征。
            if not torch.equal(image_mask.sum(-1), tokens_per_request):
                raise ValueError("Image slots must match the processor grid for each request.")
            kwargs["pixel_values"] = image_inputs["pixel_values"].to(self.device, self.text_encoder.dtype)
            kwargs["image_grid_thw"] = image_inputs["image_grid_thw"].to(self.device)
        elif image_mask.any():
            raise ValueError("Prompt contains image slots but no condition images were supplied.")

        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            hidden = self.text_encoder(**kwargs).hidden_states[-1]
        finally:
            handle.remove()
        hidden = [item[self._drop_idx :] for item in self._extract_masked_hidden(hidden, attention_mask)]
        tags = [item[self._drop_idx :] == self._img_token_id for item in ids]
        expected = [
            [
                (image.height // self.vae_scale_factor) * (image.width // self.vae_scale_factor) // 4
                for image in images or []
            ]
            for images in images_per_prompt
        ]
        sizes = [[image.size for image in images or []] for images in images_per_prompt]
        hidden, tags = self._downsample_image_pad_tokens(hidden, tags, expected, sizes)
        embeds = pad_sequence(hidden, batch_first=True).to(self.text_encoder.dtype)
        masks = pad_sequence(
            [torch.ones(item.shape[0], device=item.device, dtype=torch.long) for item in hidden], batch_first=True
        )
        tags = pad_sequence(tags, batch_first=True)
        return tuple(item.repeat_interleave(num_images_per_prompt, dim=0) for item in (embeds, masks, tags))

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch):
        """输出最终目标 latent、冻结条件和双端一致性所需的空间信息。"""
        if req.is_dummy_run():
            return super().forward(req)
        prompts = []
        negatives = []
        sampling = req.sampling_params_list[0]
        scale = sampling.true_cfg_scale if sampling.true_cfg_scale is not None else 1.0
        for item in req.prompts:
            ids = prompt_ids_from_payload(item) if isinstance(item, dict) else None
            if ids is None:
                raise ValueError("NFT rollout requires agent-tokenized prompt_ids.")
            prompts.append({"ids": ids, "mask": item.get("prompt_mask")})
            if scale > 1:
                negative = item.get("negative_prompt_ids")
                if negative is None:
                    raise ValueError("CFG requires agent-tokenized negative_prompt_ids.")
                negatives.append({"ids": negative, "mask": item.get("negative_prompt_mask")})

        height, width = normalize_min_aligned_size(
            sampling.height or 1024, sampling.width or 1024, self.vae_scale_factor * 2
        )
        repeats = max(sampling.num_outputs_per_prompt, 1)
        ctx = self._prepare_generation_context(
            prompt=prompts,
            negative_prompt=negatives or None,
            height=height,
            width=width,
            num_inference_steps=sampling.num_inference_steps or 40,
            sigmas=sampling.sigmas,
            num_images_per_prompt=repeats,
            generator=req.collate_request_generators(repeats, None),
            true_cfg_scale=scale,
            max_sequence_length=sampling.max_sequence_length or self._max_length,
            per_request_images=self._extract_request_images(req.prompts),
            latents=req.collate_request_tensors("latents", None),
            attention_kwargs={"attn_path": "mask"},
        )
        ctx["img_shapes"] = [shapes for shapes in ctx["img_shapes"] for _ in range(repeats)]
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            latents = self.diffuse(
                ctx["prompt_embeds"],
                ctx["prompt_embeds_mask"],
                ctx["negative_prompt_embeds"],
                ctx["negative_prompt_embeds_mask"],
                ctx["img_mask"],
                ctx["negative_img_mask"],
                ctx["latents"],
                ctx["image_latents"],
                ctx["img_shapes"],
                ctx["timesteps"],
                ctx["do_true_cfg"],
                scale,
                attention_kwargs=self.attention_kwargs,
                additional_transformer_kwargs={"return_dict": False, "kv_cache": None},
            )
        self._current_timestep = None
        output_type = sampling.output_type or "pil"
        decode = wants_decoded_preview(output_type, sampling)
        decoded = self._decode_latents(latents, height, width, "pil" if decode else "latent")
        outputs = split_diffusion_output_by_request(decoded, req, num_outputs_per_prompt=repeats)
        conditions = ctx["image_latents"]
        if conditions is None:
            conditions = latents.new_empty((latents.shape[0], 0, latents.shape[-1]))
        shape_tensor = torch.zeros((latents.shape[0], 4, 3), dtype=torch.long, device=latents.device)
        for index, shapes in enumerate(ctx["img_shapes"]):
            if len(shapes) > 1:
                shape_tensor[index, : len(shapes) - 1] = torch.tensor(shapes[:-1], device=latents.device)
        target_shapes = torch.tensor([shapes[-1] for shapes in ctx["img_shapes"]], device=latents.device)
        for index, output in enumerate(outputs):
            rows = slice(index * repeats, (index + 1) * repeats)
            embeds = {
                key: ctx[key][rows]
                for key in (
                    "prompt_embeds",
                    "prompt_embeds_mask",
                    "negative_prompt_embeds",
                    "negative_prompt_embeds_mask",
                )
                if ctx[key] is not None
            }
            rl = {
                "latents_clean": latents[rows].float(),
                "train_timesteps": ctx["timesteps"].unsqueeze(0).expand(repeats, -1),
                "condition_image_rows": conditions[rows],
                "condition_image_row_count": torch.full((repeats, 1), conditions.shape[1], device=latents.device),
                "condition_image_shapes": shape_tensor[rows],
                "target_image_shape": target_shapes[rows],
                "prompt_token_tags": ctx["image_pad_mask"][rows],
            }
            if ctx["negative_image_pad_mask"] is not None:
                rl["negative_prompt_token_tags"] = ctx["negative_image_pad_mask"][rows]
            result = with_rollout_data(output, prompt_embeddings=embeds, rl=rl, to_cpu=True)
            outputs[index] = with_visual_artifacts(
                result,
                decoded=output.output if decode else None,
                latents=latents[rows],
                latent_layout="LC",
                output_type=output_type,
                context=f"pipeline={type(self).__name__}, request_id={req.request_ids[index]}",
                requested=(req.sampling_params_list[index].extra_args or {}).get("requested_outputs"),
            )
        return outputs
