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

"""BAGEL UniGRPO training and sampling hooks for the shared diffusion engine."""

import torch
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionEngineHooks


class BagelUniGRPOHooks(DiffusionEngineHooks):
    """Own algorithm state; the engine owns optimizer, scheduler and checkpoints."""

    def __init__(self, module, model_config, optimizer_config):
        self.module = module
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self._updater = None
        self._replica = None

    def _get_updater(self, loss_cfg=None):
        """Lazily build the joint updater over a native pipeline bound to the FSDP module.

        ``loss_cfg`` is the actor ``diffusion_loss`` config (threaded from the loss function in
        ``forward_backward_batch``); it makes ``clip_ratio``/``mse_weight``/``ratio_norm``/
        ``adv_clip_max`` recipe-controllable. Falls back to defaults matching the reference recipe.
        """
        if self._updater is None:
            from verl_omni.pipelines.bagel_unigrpo.joint_update import UniGRPOJointUpdater
            from verl_omni.pipelines.bagel_unigrpo.pipeline import BagelUniPipeline
            from verl_omni.pipelines.bagel_unigrpo.rollout import build_unigrpo_pipeline_kwargs

            loss = loss_cfg if loss_cfg is not None else getattr(self.model_config, "diffusion_loss", None)
            pipeline = BagelUniPipeline(self.module, **build_unigrpo_pipeline_kwargs(self.model_config, self.module))
            self._updater = UniGRPOJointUpdater(
                pipeline,
                param_group_lrs=dict(getattr(self.optimizer_config, "param_group_lrs", None) or {}),
                mse_weight=float(getattr(loss, "mse_weight", 1.5e-5)) if loss is not None else 1.5e-5,
                ratio_norm=bool(getattr(loss, "ratio_norm", True)) if loss is not None else True,
                image_clip_ratio=float(getattr(loss, "clip_ratio", 1e-6)) if loss is not None else 1e-6,
                adv_clip_max=float(getattr(loss, "adv_clip_max", 5.0)) if loss is not None else 5.0,
                max_grad_norm=float(self.optimizer_config.clip_grad),
                build_optimizer=False,
            )
        if loss_cfg is not None:
            # Rollout may create the updater before the actor loss config is available.
            self._updater.mse_weight = float(loss_cfg.mse_weight)
            self._updater.ratio_norm = bool(loss_cfg.ratio_norm)
            self._updater._loss_cfg.diffusion_loss.clip_ratio = float(loss_cfg.clip_ratio)
            self._updater._loss_cfg.diffusion_loss.adv_clip_max = float(loss_cfg.adv_clip_max)
            self._updater._loss_cfg.diffusion_loss.mse_weight = float(loss_cfg.mse_weight)
            self._updater._loss_cfg.diffusion_loss.ratio_norm = bool(loss_cfg.ratio_norm)
        return self._updater

    def generate(self, data):
        """Native rollout: sample thinking->image on a flat bf16 replica, then anchor ``old_logp``.

        The replica (a plain, frozen, full-param bf16 ``BagelForSFT``) is re-synced from the FSDP
        master each call and drives the collective-free per-rank AR decode + Flow-SDE image sampler.
        ``old_logp`` is then re-anchored to the FSDP training module (via ``record_old_logp``) so the
        on-policy ratio is 1 at update 0 despite sampling on a separate module. Requires
        param_offload=false (the standalone recipe runs with no CPU offload). Returns a TensorDict of
        per-sample ``responses`` (uint8 ``[n,3,H,W]`` for reward/validation) and ``unigrpo_samples``
        (a ``NonTensorStack`` of ``UniRolloutSample``) that ``update_actor`` feeds back to the joint update.
        """
        from verl.utils.device import get_device_id, get_device_name, get_torch_device

        from verl_omni.pipelines.bagel_unigrpo.pipeline import BagelUniPipeline
        from verl_omni.pipelines.bagel_unigrpo.rollout import (
            build_replica,
            build_unigrpo_pipeline_kwargs,
            sync_replica_from_master,
        )

        prompts = tu.get(data, "prompt_token_ids")
        assert prompts is not None, "UniGRPO generate expects data['prompt_token_ids'] (per-sample token ids)"
        device = torch.device(get_device_name(), get_device_id())
        model_path = self.model_config.local_path or self.model_config.path
        if self._replica is None:
            self._replica = build_replica(model_path, device)
        else:
            self._replica.to(device)

        was_training = self.module.training
        self.module.eval()
        try:
            # Refresh the flat replica from the FSDP master (a bounded all-gather per trainable DTensor).
            sync_replica_from_master(self._replica, self.module)
            pipeline = BagelUniPipeline(
                self._replica, **build_unigrpo_pipeline_kwargs(self.model_config, self._replica)
            )
            self._replica.eval()
            with torch.no_grad():
                samples = [pipeline.generate_one([int(t) for t in prompt]) for prompt in prompts]
            # Park the replica on CPU so record_old_logp + the update have GPU room, mirroring the standalone.
            del pipeline
            self._replica.to("cpu")
            import gc

            gc.collect()
            get_torch_device().empty_cache()
            # Anchor old_logp to the FSDP training module so update-0 ratio == 1.
            self._get_updater().record_old_logp(samples)
        finally:
            self.module.train(was_training)
        get_torch_device().empty_cache()

        images = torch.stack([s.image for s in samples], dim=0)
        return tu.get_tensordict({"responses": images, "unigrpo_samples": list(samples)}).cpu()

    def forward_backward_batch(self, data, loss_function, forward_only: bool = False):
        """Joint update over one mini-batch: AR backward + image backward accumulate grads on the
        module; the base ``train_batch`` closes the single ``optimizer_step``. Samples are carried
        as a non-tensor ``unigrpo_samples`` list; ``advantages`` is a per-sample tensor.

        Returns the dict shape ``{"loss": [...], "metrics": {...}, "model_output": {}}`` that
        ``BaseEngine.train_batch`` expects (it sets ``outputs["metrics"]["grad_norm"]`` and
        ``TrainingWorker._postprocess_output`` sums ``loss`` and DP-allgathers ``metrics``).
        """
        if forward_only:
            raise NotImplementedError(
                "BAGEL UniGRPO joint replay requires backward; use generate/evaluate for sampling"
            )
        loss_cfg = None
        if loss_function is not None and hasattr(loss_function, "keywords"):
            loss_cfg = getattr(loss_function.keywords.get("config"), "diffusion_loss", None)
        updater = self._get_updater(loss_cfg=loss_cfg)
        # NonTensorStack -> list[UniRolloutSample]; tu.get (not tu.get_non_tensor_data) unwraps the stack.
        samples = tu.get(data, "unigrpo_samples")
        assert samples is not None, "UniGRPO hooks expect data['unigrpo_samples'] (list[UniRolloutSample])"
        advantages = [float(a) for a in data["advantages"]]
        ar_met = updater._ar_backward(samples, advantages)
        img_met = updater._image_backward(samples, advantages)
        loss_val = float(ar_met.get("ar/loss", 0.0)) + float(img_met.get("image/loss", 0.0))
        metrics = {k: [float(v)] for k, v in {**ar_met, **img_met}.items()}
        return {"loss": [loss_val], "metrics": metrics, "model_output": {}}
