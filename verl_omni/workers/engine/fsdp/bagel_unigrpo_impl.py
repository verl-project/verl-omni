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

"""Custom FSDP2 engine for BAGEL UniGRPO -- joint AR-thinking + image, 2 backwards -> 1 step.

Subclasses ``PPODiffusersFSDPEngine`` but replaces the image-only single-backward train
path with the joint AR-thinking + image UniGRPO update (``UniGRPOJointUpdater``). It also
overrides three engine hooks to match the trainside recipe:

* ``_build_fsdp_module``: ``fully_shard`` EACH MoT layer and EACH root-level trainable leaf,
  never the root -- the model is driven functionally (``module.embed_tokens(...)`` etc. in the
  AR replay), so a root-only ``fully_shard`` would leave the root leaves as un-all-gathered
  ``DTensor``s.
* ``_build_optimizer``: per-expert LR param groups (name-substring match, e.g. ``moe_gen``).
* ``optimizer_step``: an FSDP2/DTensor-safe global grad-norm clip via a single ``all_gather``
  (a raw per-``DTensor`` ``clip_grad_norm_`` deadlocks cross-node against FSDP's own comm).
"""

from __future__ import annotations

import logging

import torch
from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import EngineRegistry

from .diffusers_impl import PPODiffusersFSDPEngine

logger = logging.getLogger(__name__)

# Root-level trainable leaves reached functionally (not via the root __call__), so each must be
# its own FSDP unit for the all-gather hook to fire during the AR replay.
_ROOT_TRAINABLE_SUBMODULES = ("embed_tokens", "lm_head", "norm", "norm_moe_gen", "time_embedder", "vae2llm", "llm2vae")


@EngineRegistry.register(model_type="diffusion_unigrpo_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class UniGRPODiffusersFSDPEngine(PPODiffusersFSDPEngine):
    """Diffusers FSDP engine whose train step is the joint AR + image UniGRPO update."""

    _updater = None
    _replica = None
    _report_tokenizer = None
    _report_scorer = None

    def _build_fsdp_module(self, module):
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
        from verl.utils.device import get_device_id
        from verl.utils.torch_dtypes import PrecisionType

        mp = self.engine_config.mixed_precision
        param_dtype = PrecisionType.to_dtype(mp.get("param_dtype", "bf16")) if mp is not None else torch.bfloat16
        reduce_dtype = PrecisionType.to_dtype(mp.get("reduce_dtype", "fp32")) if mp is not None else torch.float32
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
        shard_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": self.engine_config.reshard_after_forward,
        }
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            self._uses_fsdp2_cpu_offload_policy = True
            shard_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

        inner = getattr(module, "module", module)
        # Materialise the full model on this rank, then shard only the trainable units below.
        # fully_shard moves its own units to the device, but the frozen leaves it skips
        # (latent_pos_embed indexed in the image forward, the vision stack, the VAE) would
        # otherwise stay on CPU and crash the CUDA replay with a device-mismatch on indexing.
        module.to(get_device_id())
        for layer in inner.layers:
            fully_shard(layer, **shard_kwargs)
        for name in _ROOT_TRAINABLE_SUBMODULES:
            sub = getattr(inner, name, None)
            if sub is not None:
                fully_shard(sub, **shard_kwargs)
        return module

    def _build_optimizer(self, module):
        """AdamW with per-expert LR groups: a param whose name contains a ``param_group_lrs`` key
        uses that LR, the rest use the base ``lr``."""
        base_lr = float(self.optimizer_config.lr)
        weight_decay = float(getattr(self.optimizer_config, "weight_decay", 0.0))
        param_group_lrs = dict(getattr(self.optimizer_config, "param_group_lrs", None) or {"moe_gen": base_lr})
        keys = list(param_group_lrs.keys())
        buckets: dict[str, list] = {k: [] for k in keys}
        base_params: list = []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            matched = next((k for k in keys if k in name), None)
            (base_params if matched is None else buckets[matched]).append(p)
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": base_lr})
        for k in keys:
            if buckets[k]:
                groups.append({"params": buckets[k], "lr": float(param_group_lrs[k])})
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay, foreach=False)

    def _get_updater(self, loss_cfg=None):
        """Lazily build the joint updater over a trainside pipeline bound to the FSDP module.

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
        return self._updater

    def generate_rollout(self, data):
        """Trainside rollout: sample thinking->image on a flat bf16 replica, then anchor ``old_logp``.

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
        logger.info(
            "UniGRPO generate done: GPU alloc=%.1fGB reserved=%.1fGB (replica parked to CPU)",
            get_torch_device().memory_allocated() / 1e9,
            get_torch_device().memory_reserved() / 1e9,
        )

        images = torch.stack([s.image for s in samples], dim=0)
        return tu.get_tensordict({"responses": images, "unigrpo_samples": list(samples)}).cpu()

    def dump_report_samples(self, eval_prompts, eval_gts, out_dir, step, seed: int = 1234):
        """Fixed-prompt report samples at one checkpoint (official-CFG eval on the trainside replica).

        Mirrors the standalone ``dump_report_samples``: every rank first re-syncs the flat bf16
        replica from the FSDP master (an all-gather per trainable ``DTensor``, so ALL ranks must
        reach it), then only rank 0 samples on the local replica (collective-free) and writes each
        prompt's official-CFG image + COMPLETE thinking text + PickScore under
        ``<out_dir>/report_ff/step_<NNNN>/``. Rank-0 work is wrapped so a dump failure logs and
        returns instead of dead-locking the peers at the next collective. Returns rank-0's
        per-prompt PickScores (``None`` off rank 0 / on failure).
        """
        import os

        import torch.distributed as dist
        from verl.utils.device import get_device_id, get_device_name, get_torch_device

        from verl_omni.pipelines.bagel_unigrpo.pipeline import BagelUniPipeline
        from verl_omni.pipelines.bagel_unigrpo.rollout import (
            build_replica,
            build_unigrpo_pipeline_kwargs,
            sync_replica_from_master,
        )

        device = torch.device(get_device_name(), get_device_id())
        model_path = self.model_config.local_path or self.model_config.path
        if self._replica is None:
            self._replica = build_replica(model_path, device)
        else:
            self._replica.to(device)

        # Collective (all ranks): refresh the replica from the current FSDP master weights.
        was_training = self.module.training
        self.module.eval()
        try:
            sync_replica_from_master(self._replica, self.module)
        finally:
            self.module.train(was_training)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            self._replica.to("cpu")
            get_torch_device().empty_cache()
            return None

        # Rank-0-only past this point (no collectives): catch + log so a bad dump costs only its
        # samples instead of dead-locking the peers at the next collective.
        try:
            import json

            from PIL import Image

            from verl_omni.utils.reward_score.pickscore_reward import _PickScoreInferencer, _to_pil_hwc

            self._replica.eval()
            pipeline = BagelUniPipeline(
                self._replica, **build_unigrpo_pipeline_kwargs(self.model_config, self._replica)
            )
            sdir = os.path.join(out_dir, "report_ff", f"step_{int(step):04d}")
            os.makedirs(sdir, exist_ok=True)

            if self._report_tokenizer is None:
                from transformers import AutoTokenizer

                self._report_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = self._report_tokenizer

            images, texts = [], []
            for i, pids in enumerate(eval_prompts):
                torch.manual_seed(int(seed))
                get_torch_device().manual_seed_all(int(seed))
                gen = torch.Generator(device=device).manual_seed(int(seed))
                think_ids, image = pipeline.generate_eval([int(t) for t in pids], generator=gen)
                Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(os.path.join(sdir, f"p{i}.png"))
                text = tokenizer.decode(think_ids, skip_special_tokens=True) if tokenizer is not None else ""
                with open(os.path.join(sdir, f"p{i}.txt"), "w") as handle:
                    handle.write(text)
                images.append(image)
                texts.append(text)

            if self._report_scorer is None:
                self._report_scorer = _PickScoreInferencer(device=device)
            scores = self._report_scorer.score(list(eval_gts), [_to_pil_hwc(im) for im in images]).tolist()
            for i, score in enumerate(scores):
                with open(os.path.join(sdir, f"p{i}.json"), "w") as handle:
                    json.dump({"pickscore": float(score), "prompt": eval_gts[i]}, handle)
            logger.info(
                "UniGRPO report dump step %s: pickscore mean=%.4f -> %s",
                step,
                sum(scores) / max(len(scores), 1),
                sdir,
            )
            return [float(s) for s in scores]
        except Exception:
            import traceback

            logger.warning("UniGRPO report dump step %s failed (non-fatal):\n%s", step, traceback.format_exc())
            return None
        finally:
            import gc

            self._replica.to("cpu")
            if self._report_scorer is not None:
                del self._report_scorer
                self._report_scorer = None
            gc.collect()
            get_torch_device().empty_cache()

    def forward_backward_batch(self, data, loss_function, forward_only: bool = False):
        """Joint update over one mini-batch: AR backward + image backward accumulate grads on the
        module; the base ``train_batch`` closes the single ``optimizer_step``. Samples are carried
        as a non-tensor ``unigrpo_samples`` list; ``advantages`` is a per-sample tensor.

        Returns the dict shape ``{"loss": [...], "metrics": {...}, "model_output": {}}`` that
        ``BaseEngine.train_batch`` expects (it sets ``outputs["metrics"]["grad_norm"]`` and
        ``TrainingWorker._postprocess_output`` sums ``loss`` and DP-allgathers ``metrics``).
        """
        loss_cfg = None
        if loss_function is not None and hasattr(loss_function, "keywords"):
            loss_cfg = getattr(loss_function.keywords.get("config"), "diffusion_loss", None)
        updater = self._get_updater(loss_cfg=loss_cfg)
        # NonTensorStack -> list[UniRolloutSample]; tu.get (not tu.get_non_tensor_data) unwraps the stack.
        samples = tu.get(data, "unigrpo_samples")
        assert samples is not None, "UniGRPO engine expects data['unigrpo_samples'] (list[UniRolloutSample])"
        advantages = [float(a) for a in data["advantages"]]
        if forward_only:
            return {"loss": [0.0], "metrics": {}, "model_output": {}}
        from verl.utils.device import get_torch_device as _gtd

        logger.info(
            "UniGRPO update entry: GPU alloc=%.1fGB reserved=%.1fGB replica_on_cpu=%s",
            _gtd().memory_allocated() / 1e9,
            _gtd().memory_reserved() / 1e9,
            (self._replica is None) or (next(self._replica.parameters()).device.type == "cpu"),
        )
        ar_met = updater._ar_backward(samples, advantages)
        img_met = updater._image_backward(samples, advantages)
        loss_val = float(ar_met.get("ar/loss", 0.0)) + float(img_met.get("image/loss", 0.0))
        metrics = {k: [float(v)] for k, v in {**ar_met, **img_met}.items()}
        return {"loss": [loss_val], "metrics": metrics, "model_output": {}}

    def optimizer_step(self):
        """FSDP2/DTensor-safe global grad-norm clip (single all_gather) + one AdamW step.

        Reuses the collision-free reduction the trainside updater established: a raw per-DTensor
        ``clip_grad_norm_`` deadlocks cross-node against FSDP's own communicator, so we sum local
        shard squared-norms, ``all_gather`` the per-rank scalars over FSDP's mesh group, sum
        locally, and rescale each shard in place.
        """
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor

        grads = [p.grad for p in self.module.parameters() if p.requires_grad and p.grad is not None]
        if not grads:
            return 0.0
        mesh_group = None
        for g in grads:
            if isinstance(g, DTensor):
                mesh_group = g.device_mesh.get_group()
                break
        first = grads[0].to_local() if isinstance(grads[0], DTensor) else grads[0]
        local_sq = torch.zeros(1, device=first.device, dtype=torch.float32)
        for g in grads:
            gl = g.to_local() if isinstance(g, DTensor) else g
            local_sq = local_sq + gl.detach().float().pow(2).sum().reshape(1)
        ws = (
            dist.get_world_size(mesh_group)
            if mesh_group is not None
            else (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
        )
        if ws > 1:
            gathered = [torch.empty_like(local_sq) for _ in range(ws)]
            dist.all_gather(gathered, local_sq, group=mesh_group)
            total_sq = torch.stack(gathered).sum()
        else:
            total_sq = local_sq.sum()
        total_norm = float(total_sq.sqrt())
        max_norm = float(self.optimizer_config.clip_grad)
        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
        if total_norm == total_norm and total_norm != float("inf"):
            self.optimizer.step()
        return total_norm


__all__ = ["UniGRPODiffusersFSDPEngine"]
