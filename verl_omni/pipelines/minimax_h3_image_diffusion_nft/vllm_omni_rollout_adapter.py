"""GPU rollout adapter for MiniMax H3 image DiffusionNFT (t2i and edit)."""

from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_image
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import (
    MiniMaxH3ImagePipeline,
    minimax_h3_image_run_sigmas,
)

from verl_omni.pipelines.diffusion_rollout_output import (
    install_rollout_postprocessor,
    wants_decoded_preview,
    with_rollout_data,
    with_visual_artifacts,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MiniMaxH3RolloutWeightSyncMixin,
    _PromptTokenOverride,
    serialize_ref_blocks,
)
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import (
    IMAGE_ROW_WIDTH,
    debug_dump,
    debug_enabled,
    image_latent_meta,
    pdd_train_timesteps,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

__all__ = ["MiniMaxH3ImageDiffusionNFTPipeline"]

install_rollout_postprocessor(pipeline_minimax_h3_image, "get_minimax_h3_image_post_process_func")


@VllmOmniPipelineBase.register("MiniMaxH3ImagePipeline", algorithm="diffusion_nft")
class MiniMaxH3ImageDiffusionNFTPipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3ImagePipeline):
    """Rollout pipeline for MiniMax H3 image used by DiffusionNFT.

    The official 6-step PDD sampler runs unchanged; ``encode_prompt`` and ``diffuse`` wrap its
    private text-encoding and denoise stages to keep the Actor's inputs.
    """

    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "LC"),
        }
    )

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        if "min_generation_area" not in (od_config.additional_config or {}):
            self.min_generation_area = None
        self._install_lora_layout()
        self._h3_prompt_ids: torch.Tensor | None = None
        self._nft_capture: dict[str, Any] | None = None

    def map_lora_update_to_engine(self, tensors: dict[str, torch.Tensor], peft_config: dict):
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"MiniMax H3 image rollout received a non-finite LoRA tensor: {name}.")
        if debug_enabled():
            from verl_omni.pipelines.minimax_h3_image_diffusion_nft.checkpoint import tensor_sha256

            self._nft_lora_updates = getattr(self, "_nft_lora_updates", 0) + 1
            debug_dump(
                "lora_recv",
                {
                    "update": self._nft_lora_updates,
                    "digests": {name: tensor_sha256(tensor.detach().cpu()) for name, tensor in tensors.items()},
                    "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
                    "abs_sum": {name: float(tensor.float().abs().sum()) for name, tensor in tensors.items()},
                    "peft_config": dict(peft_config or {}),
                },
            )
        return super().map_lora_update_to_engine(tensors, peft_config)

    def encode_prompt(self, prompt: str, references: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode Agent Loop token IDs verbatim while the native path builds reference vision spans."""
        prompt_ids = getattr(self, "_h3_prompt_ids", None)
        if prompt_ids is None:
            return super()._encode_text(prompt, references)
        tokenizer = self.tokenizer
        self.tokenizer = _PromptTokenOverride(tokenizer, prompt, prompt_ids)
        try:
            return super()._encode_text(prompt, references)
        finally:
            self.tokenizer = tokenizer

    def _encode_text(self, prompt: str, references: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_prompt(prompt, references)

    def diffuse(self, **kwargs: Any) -> torch.Tensor:
        """Run the official PDD denoiser and retain the Actor's condition state."""
        target_rows = super()._denoise(**kwargs)
        self._nft_capture = {
            "target_rows": target_rows.detach().to(device="cpu", dtype=torch.float32).clone(),
            "hidden": kwargs["hidden"],
            "text_tags": kwargs["text_tags"],
            "grids": [(int(height), int(width)) for height, width in kwargs["grids"]],
            "anchors": kwargs["anchors"].detach().to(device="cpu", dtype=torch.float32).clone(),
            "latent_h": int(kwargs["latent_h"]),
            "latent_w": int(kwargs["latent_w"]),
            "num_steps": int(kwargs["num_steps"]),
        }
        return target_rows

    def _denoise(self, **kwargs: Any) -> torch.Tensor:
        return self.diffuse(**kwargs)

    def forward(self, request: Any):
        """Generate one image and attach DiffusionNFT training tensors."""
        sampling = request.sampling_params
        if request.is_dummy_run():
            return super().forward(request)
        if sampling.seed is None:
            raise ValueError("MiniMax H3 image rollout requires a per-sample seed (the native default is seed 0).")
        if sampling.generator_device is None:
            sampling.generator_device = "cpu"
        extra_args = sampling.extra_args or {}
        output_type = extra_args.get("output_type", sampling.output_type) or "image"
        decode = wants_decoded_preview(output_type, sampling)

        self._ensure_prompt_text(request)
        self._nft_capture = None
        requested_output_type = sampling.output_type
        sampling.output_type = "pil" if decode else "latent"
        try:
            output = super().forward(request)
        finally:
            sampling.output_type = requested_output_type
            self._h3_prompt_ids = None
        capture = self._nft_capture
        self._nft_capture = None
        if capture is None or capture["num_steps"] != self.num_steps:
            raise RuntimeError("MiniMax H3 image rollout did not run the full PDD denoise loop.")

        target_rows = capture["target_rows"]
        num_rows = int(target_rows.shape[0])
        text_embeddings = capture["hidden"]
        prompt_embeds = text_embeddings.unsqueeze(0)
        run_sigmas = torch.tensor(minimax_h3_image_run_sigmas(self.pdd_sigmas)[:-1], dtype=torch.float32)
        anchors = capture["anchors"]
        rl = {
            "latents_clean": target_rows.reshape(1, num_rows * IMAGE_ROW_WIDTH),
            "train_timesteps": pdd_train_timesteps(run_sigmas).unsqueeze(0),
            "pdd_sigmas": run_sigmas.unsqueeze(0),
            "latent_meta": image_latent_meta(num_rows, capture["latent_h"], capture["latent_w"]),
            "prompt_token_tags": capture["text_tags"].unsqueeze(0),
            "condition_video_rows": anchors.unsqueeze(0),
            "condition_video_row_count": torch.tensor([[anchors.shape[0]]], dtype=torch.long),
        }
        if capture["grids"]:
            ref_block_meta, ref_block_count = serialize_ref_blocks(
                [{"kind": "image", "latent_h": height, "latent_w": width} for height, width in capture["grids"]]
            )
            rl["ref_block_meta"] = ref_block_meta.unsqueeze(0)
            rl["ref_block_count"] = torch.tensor([[ref_block_count]], dtype=torch.long)

        if debug_enabled():
            debug_dump(
                "rollout",
                {
                    "request_id": request.request_id,
                    "seed": int(sampling.seed),
                    "prompt": (request.prompts[0] or {}).get("prompt")
                    if isinstance(request.prompts[0], dict)
                    else None,
                    "lora_updates": getattr(self, "_nft_lora_updates", 0),
                    "latents": target_rows.clone(),
                    "latent_grid": (capture["latent_h"], capture["latent_w"]),
                    "preview": output.output.clone() if decode else None,
                    "prompt_embeds": text_embeddings.detach().cpu().clone(),
                    "text_tags": capture["text_tags"].detach().cpu().clone(),
                    "run_sigmas": run_sigmas.clone(),
                },
            )
        result = with_rollout_data(
            output,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": torch.ones(prompt_embeds.shape[:2], dtype=torch.long),
            },
            rl=rl,
            to_cpu=True,
        )
        return with_visual_artifacts(
            result,
            decoded=output.output.permute(0, 3, 1, 2) if decode else None,
            latents=target_rows.unsqueeze(0),
            latent_layout="LC",
            output_type=output_type,
            pixel_range="uint8",
            context=f"pipeline={type(self).__name__}, request_id={request.request_id}",
            requested=extra_args.get("requested_outputs"),
        )
