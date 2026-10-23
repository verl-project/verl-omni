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

"""DiffusionOPSD differentiable-reward target runtime.

DiffusionOPSD needs ``grad_y R(D(y), c)`` at the behaviour-policy clean-output anchor
``y0`` to build its bounded positive/negative targets. verl-omni's reward stack is
inference-only, so this module adds the one genuinely new runtime: a frozen VAE decoder
(kept in fp32 with gradient checkpointing) plus a frozen, differentiable reward model.
It deliberately owns *only* target construction; the policy update stays in
``OPSDLoss`` and never touches the reward/decoder graph.

The heavy imports (diffusers / transformers) are local to the loader so importing this
module in CPU contexts costs nothing. Target math itself lives in ``opsd_targets``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from verl_omni.trainer.diffusion.opsd_targets import build_positive_negative_targets, opsd_anchor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["OPSDTargetRuntime"]

#: CLIP normalization constants used by the PickScore image encoder.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_PICKSCORE_MODEL_PATH = "yuvalkirstain/PickScore_v1"
_PICKSCORE_PROCESSOR_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#: The reference implementation normalizes PickScore by this divisor before using it
#: as a differentiable training reward (see worldbench/DiffusionOPSD reward adapters).
_PICKSCORE_DIVISOR = 26.0
_CLIP_IMAGE_SIZE = 224


class OPSDTargetRuntime:
    """Build detached reward-gradient targets for DiffusionOPSD.

    Args:
        reward_of_latents: Callable ``(latents[B, C, H, W], prompts[list[str]]) -> scores[B]``
            that is differentiable w.r.t. ``latents``. Tests inject a light fake here; the
            production path is :meth:`from_model_config`.
        rho: Relative trust-region radius.
        n_ascent: Number of normalized reward-gradient steps per target.
        eta: Step-size multiplier inside the trust region.
        target_micro_batch: Trajectories per reward-gradient micro batch (bounds peak memory).
    """

    def __init__(
        self,
        *,
        reward_of_latents: Callable[[torch.Tensor, Sequence[str]], torch.Tensor],
        rho: float = 0.10,
        n_ascent: int = 2,
        eta: float = 1.0,
        target_micro_batch: int = 6,
    ):
        if target_micro_batch < 1:
            raise ValueError(f"target_micro_batch must be positive, got {target_micro_batch}.")
        self.reward_of_latents = reward_of_latents
        self.rho = float(rho)
        self.n_ascent = int(n_ascent)
        self.eta = float(eta)
        self.target_micro_batch = int(target_micro_batch)

    # ------------------------------------------------------------------
    # Production loader
    # ------------------------------------------------------------------
    @classmethod
    def from_model_config(cls, model_config, loss_config, device: torch.device) -> OPSDTargetRuntime:
        """Load the frozen VAE and the differentiable reward from the actor model config."""
        if loss_config.opsd_reward_kind != "pickscore":
            raise NotImplementedError(
                "The first DiffusionOPSD integration ships only the PickScore v1 differentiable reward, got "
                f"opsd_reward_kind={loss_config.opsd_reward_kind!r}. Add the scorer in opsd_target_runtime.py "
                "before enabling another reward."
            )
        model_path = getattr(model_config, "local_path", None) or model_config.path
        vae = _load_vae(model_path, device)
        tokenizer = _load_prompt_tokenizer(model_config)
        reward_path = loss_config.opsd_reward_model_path or _PICKSCORE_MODEL_PATH
        reward_model = _DifferentiablePickScore(
            model_path=reward_path,
            processor_path=_resolve_processor_path(reward_path),
            dtype=getattr(torch, loss_config.opsd_reward_dtype),
            device=device,
        )

        def reward_of_latents(latents: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
            prompts = _decode_prompts(tokenizer, prompt_ids)
            images = _decode_latents(vae, latents)
            return reward_model(images, prompts)

        return cls(
            reward_of_latents=reward_of_latents,
            rho=loss_config.opsd_rho,
            n_ascent=loss_config.opsd_n_ascent,
            eta=loss_config.opsd_eta,
            target_micro_batch=loss_config.opsd_target_micro_batch,
        )

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    def build_targets(
        self,
        *,
        query_latents: torch.Tensor,
        t_expanded: torch.Tensor,
        old_prediction: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        """Return detached OPSD targets and diagnostics for one micro batch.

        Args:
            query_latents: Behaviour-policy query state ``z_q``, shape ``(B, C, H, W)``.
            t_expanded: Query sigma broadcast to the latent shape, ``(B, 1, 1, 1)``.
            old_prediction: Frozen old-policy velocity at the query, ``(B, C, H, W)``.
            prompt_ids: Token ids of each row's prompt, ``(B, L)``.

        Returns:
            Dict with detached ``opsd_target_pos`` / ``opsd_target_neg`` tensors and the
            per-sample reward that built them. Diagnostics are plain floats for logging.
        """
        anchor = opsd_anchor(query_latents, t_expanded, old_prediction)
        batch = anchor.shape[0]
        target_pos = torch.empty_like(anchor)
        target_neg = torch.empty_like(anchor)
        # The target builder already reports the relative displacements; accumulate them
        # chunk-size weighted instead of recomputing norms over the assembled batch.
        displacement_pos_sum = 0.0
        displacement_neg_sum = 0.0

        with torch.enable_grad():
            for start in range(0, batch, self.target_micro_batch):
                end = min(start + self.target_micro_batch, batch)
                chunk_anchor = anchor[start:end]
                chunk_prompt_ids = prompt_ids[start:end]

                def reward_fn(latents: torch.Tensor, ids: torch.Tensor = chunk_prompt_ids) -> torch.Tensor:
                    return self.reward_of_latents(latents, ids)

                y_pos, y_neg, chunk_diagnostics = build_positive_negative_targets(
                    chunk_anchor,
                    reward_fn,
                    rho=self.rho,
                    n_ascent=self.n_ascent,
                    eta=self.eta,
                )
                target_pos[start:end] = y_pos
                target_neg[start:end] = y_neg
                chunk_rows = end - start
                displacement_pos_sum += chunk_diagnostics["displacement_pos_rel_mean"] * chunk_rows
                displacement_neg_sum += chunk_diagnostics["displacement_neg_rel_mean"] * chunk_rows

        rows = max(batch, 1)
        return {
            "opsd_target_pos": target_pos.detach(),
            "opsd_target_neg": target_neg.detach(),
            "opsd_anchor_latents": anchor.detach(),
            "opsd_target_displacement_rel": torch.tensor(
                displacement_pos_sum / rows, device=anchor.device, dtype=torch.float32
            ),
            "opsd_target_displacement_neg_rel": torch.tensor(
                displacement_neg_sum / rows, device=anchor.device, dtype=torch.float32
            ),
        }


# ----------------------------------------------------------------------
# Frozen model loading / differentiable forwards
# ----------------------------------------------------------------------
def _load_vae(model_path: str, device: torch.device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device=device, dtype=torch.float32)
    try:
        vae.enable_gradient_checkpointing()
    except Exception as exc:  # pragma: no cover - depends on the diffusers VAE version
        logger.warning(
            "OPSD: VAE gradient checkpointing unavailable (%s); target construction may be memory heavy.", exc
        )
    return vae


def _load_prompt_tokenizer(model_config):
    # The model config already built the tokenizer that encoded the rollout prompts, so
    # decoding with it round-trips the text the reward model should see.
    tokenizer = getattr(model_config, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    model_path = getattr(model_config, "local_path", None) or model_config.path
    return AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", trust_remote_code=bool(model_config.trust_remote_code)
    )


def _decode_prompts(tokenizer, prompt_ids: torch.Tensor) -> list[str]:
    """Decode prompt ids to text; the reward models are re-tokenized with their own tokenizer."""
    rows = prompt_ids.detach().to("cpu").tolist()
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in rows]


def _decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Differentiably decode latents to an image tensor in ``[0, 1]``.

    Parameters stay frozen, but the returned tensor is connected to ``latents`` so the
    reward gradient can flow back through the decoder.
    """
    scaled = latents / vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor is not None:
        scaled = scaled + shift_factor
    image = vae.decode(scaled.to(dtype=vae.dtype), return_dict=False)[0]
    return ((image / 2.0) + 0.5).clamp(0.0, 1.0).float()


def _unwrap_clip_features(features) -> torch.Tensor:
    """Return projected CLIP embeddings across transformers return-type versions.

    ``transformers>=5.13`` returns ``BaseModelOutputWithPooling`` from
    ``CLIPModel.get_text_features`` / ``get_image_features`` (the projected embedding lives
    in ``pooler_output``), while older versions return a plain tensor. This reuses the shared
    ``image_preference_utils.clip_feature_tensor`` helper so the differentiable target path
    unwraps exactly like the native PickScore reward path.
    """
    from verl_omni.utils.reward_score.image_preference_utils import clip_feature_tensor

    return clip_feature_tensor(features)


def _resolve_processor_path(reward_path: str) -> str:
    """Pick the CLIP processor that matches the reward checkpoint.

    A local checkpoint directory normally ships its own ``preprocessor_config.json``
    (real PickScore exports and the tiny e2e checkpoint both do). Using it keeps the
    differentiable preprocessing and the model's expected input size in sync; only when the
    checkpoint is a bare Hub id do we fall back to the canonical PickScore processor.

    ``CLIPProcessor`` needs both an image processor and a tokenizer, so a directory that
    carries only ``preprocessor_config.json`` is not self-sufficient and falls back as well.
    """
    if not os.path.isdir(reward_path):
        return _PICKSCORE_PROCESSOR_PATH
    if not os.path.isfile(os.path.join(reward_path, "preprocessor_config.json")):
        return _PICKSCORE_PROCESSOR_PATH
    if not _has_tokenizer_files(reward_path):
        return _PICKSCORE_PROCESSOR_PATH
    return reward_path


def _has_tokenizer_files(path: str) -> bool:
    """Return True when ``path`` holds tokenizer assets ``CLIPProcessor`` can load."""
    return any(
        os.path.isfile(os.path.join(path, name)) for name in ("tokenizer.json", "vocab.json", "tokenizer_config.json")
    )


class _DifferentiablePickScore:
    """Frozen PickScore v1 with a differentiable image preprocessing path.

    The upstream CLIP processor converts tensors to PIL images, which detaches the reward
    graph. This class reimplements the resize/center-crop/normalize step with torch ops so
    ``d reward / d latents`` reaches the VAE decoder, mirroring the reference implementation.
    """

    def __init__(self, *, model_path: str, processor_path: str, dtype: torch.dtype, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.dtype = dtype
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device=device, dtype=dtype)
        self.model.requires_grad_(False)
        self.mean = torch.tensor(_CLIP_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_CLIP_STD, dtype=dtype, device=device).view(1, 3, 1, 1)

    def preprocess(self, images01: torch.Tensor) -> torch.Tensor:
        """Resize the shorter side to 224, center-crop, and normalize, differentiably."""
        images = images01.to(dtype=self.dtype, device=self.device)
        height, width = images.shape[-2:]
        scale = _CLIP_IMAGE_SIZE / float(min(height, width))
        new_height = max(_CLIP_IMAGE_SIZE, int(round(height * scale)))
        new_width = max(_CLIP_IMAGE_SIZE, int(round(width * scale)))
        images = F.interpolate(
            images, size=(new_height, new_width), mode="bicubic", align_corners=False, antialias=True
        )
        top = (new_height - _CLIP_IMAGE_SIZE) // 2
        left = (new_width - _CLIP_IMAGE_SIZE) // 2
        images = images[:, :, top : top + _CLIP_IMAGE_SIZE, left : left + _CLIP_IMAGE_SIZE]
        return (images - self.mean) / self.std

    def __call__(self, images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        pixel_values = self.preprocess(images01)
        text_inputs = self.processor(
            text=list(prompts), padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        text_features = _unwrap_clip_features(self.model.get_text_features(**text_inputs))
        image_features = _unwrap_clip_features(self.model.get_image_features(pixel_values=pixel_values))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        scores = self.model.logit_scale.exp() * (text_features * image_features).sum(dim=-1)
        return (scores / _PICKSCORE_DIVISOR).float()
