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


"""Lance FlowGRPO training using BAGEL's MoT replay with Lance geometry and sigmas."""

from __future__ import annotations

import logging

import torch

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import get_flattened_position_ids
from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import BagelDiffusion
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import setup_lance_sigmas
from .lance_model import LanceForTraining

logger = logging.getLogger(__name__)


@DiffusionModelBase.register("OmniLanceForConditionalGeneration", algorithm="flow_grpo")
class LanceDiffusion(BagelDiffusion):
    """DiffusionModelBase wrapper for :class:`LanceForTraining`.

    The image and the video checkpoint share this adapter: nothing here needs to
    know which one is loaded, because ``_get_latent_pos_ids`` adds the temporal
    axis whenever the request itself carries more than one frame.
    """

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        """Load Lance through its own checkpoint path.

        Args:
            model_config: Model config carrying ``local_path``.
            torch_dtype: Target dtype.

        Returns:
            The loaded ``LanceForTraining`` module.
        """
        logger.info("Loading LanceForTraining from %s", model_config.local_path)
        return LanceForTraining.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
        )

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        """Apply Lance's sigma schedule.

        Lance's rollout shifts timesteps by 3.5 where BAGEL uses 3.0; the
        trainer has to use the same shift or the replayed log-probs are taken
        at different sigmas than the ones the rollout sampled.

        Args:
            scheduler: Scheduler to configure in place.
            model_config: Model config carrying ``pipeline.num_inference_steps``.
            device: Device for the scheduler buffers.
        """
        setup_lance_sigmas(scheduler, model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def _get_latent_pos_ids(cls, model_config: DiffusionModelConfig, module, device) -> torch.Tensor:
        """BAGEL's grid, extended with a temporal axis for video requests.

        Frames above one add the ``t * side**2`` rows the video position table is
        indexed with, so the trainer places the same rows
        ``LanceBagel._per_token_mrope_for_video_latent`` places on the rollout
        side.  A single frame is BAGEL's image grid unchanged.
        """
        config = module.config
        num_frames = int(getattr(model_config.pipeline, "num_frames", 1) or 1)
        if num_frames <= 1:
            return super()._get_latent_pos_ids(model_config, module, device)
        latent_ds = config.latent_patch_size * config.vae_downsample
        img_h = min(model_config.pipeline.height // latent_ds, config.max_latent_size)
        img_w = min(model_config.pipeline.width // latent_ds, config.max_latent_size)
        frames = (num_frames - 1) // int(config.vae_downsample_temporal) + 1
        if frames > config.max_num_frames:
            raise ValueError(f"Lance checkpoint supports {config.max_num_frames} latent frames, got {frames}.")
        pos_ids = get_flattened_position_ids(
            img_h * latent_ds,
            img_w * latent_ds,
            latent_ds,
            config.max_latent_size,
            num_frames=frames,
        )
        return pos_ids.to(device)
