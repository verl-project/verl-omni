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
"""UniGRPO joint update: AR-GRPO backward + UniGRPO image backward -> ONE optimizer step.

Single-process port of UniRL ``train/unified_model_stack.py`` (2-backwards->1-step,
per-expert LR) onto verl-omni's committed native pieces (``BagelUniPipeline`` rollout,
``bagel_ar_thinking.replay_thinking_logprobs``, ``BagelDiffusion`` per-step replay,
``UniGRPOLoss``). No vLLM. The AR (understanding) and image (generation) tracks share
the ONE MoT transformer; their two backward passes accumulate into the same grads and a
single optimizer step is taken per update. ``num_updates_per_batch`` disjoint mini-batches
make the 2nd+ updates off-policy so RatioNorm/clipping engage.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from verl.utils.device import get_device_id, get_device_name, is_cuda_available


def ar_grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages_per_token: torch.Tensor,
    clip_range: float,
    *,
    loss_agg_mode: str = "token-mean",
    clip_range_high: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level clipped policy-gradient loss (UniRL grpo.py ``_grpo_clip_loss``)."""
    old_logp = old_logp.to(dtype=new_logp.dtype, device=new_logp.device)
    adv = advantages_per_token.to(dtype=new_logp.dtype, device=new_logp.device)
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    low = 1.0 - clip_range
    high = 1.0 + (clip_range if clip_range_high is None else clip_range_high)
    unclipped = -adv * ratio
    clipped = -adv * ratio.clamp(low, high)
    per_elem = torch.maximum(unclipped, clipped)
    if loss_agg_mode == "token-mean":
        loss = per_elem.mean()
    elif loss_agg_mode == "seq-mean-token-mean":
        loss = per_elem.mean()
    else:
        loss = per_elem.mean()
    with torch.no_grad():
        metrics = {
            "ar/ppo_kl": float((-log_ratio).mean().item()),
            "ar/ratio_mean": float(ratio.mean().item()),
            "ar/clipfrac": float(((ratio - 1.0).abs() > clip_range).float().mean().item()),
        }
    return loss, metrics


def _build_expert_optimizer(module, base_lr: float, param_group_lrs: dict[str, float], weight_decay: float):
    """AdamW with per-expert LR groups: name-substring match wins, else base lr."""
    keys = list(param_group_lrs.keys())
    buckets: dict[str, list] = {k: [] for k in keys}
    base_params, counts, base_count = [], {k: 0 for k in keys}, 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        matched = next((k for k in keys if k in name), None)
        if matched is None:
            base_params.append(p)
            base_count += p.numel()
        else:
            buckets[matched].append(p)
            counts[matched] += p.numel()
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    for k in keys:
        if buckets[k]:
            groups.append({"params": buckets[k], "lr": float(param_group_lrs[k])})
    opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay, foreach=False)
    summary = {"base": (base_lr, base_count), **{k: (float(param_group_lrs[k]), counts[k]) for k in keys}}
    return opt, summary


class _LossCfg:
    """Minimal stand-in for ``config`` that ``UniGRPOLoss`` reads (``config.diffusion_loss.*``)."""

    def __init__(self, *, clip_ratio, adv_clip_max, mse_weight, ratio_norm):
        from types import SimpleNamespace

        self.diffusion_loss = SimpleNamespace(
            clip_ratio=clip_ratio, adv_clip_max=adv_clip_max, mse_weight=mse_weight, ratio_norm=ratio_norm
        )


class UniGRPOJointUpdater:
    """Joint AR + image update over ONE MoT transformer: 2 backwards -> 1 optimizer step."""

    def __init__(
        self,
        pipeline,
        *,
        base_lr: float = 1e-6,
        param_group_lrs: dict[str, float] | None = None,
        num_updates_per_batch: int = 2,
        max_grad_norm: float = 1.0,
        mse_weight: float = 1.5e-5,
        ratio_norm: bool = True,
        ar_clip_range: float = 1e-2,
        image_clip_ratio: float = 1e-6,
        adv_clip_max: float = 5.0,
        weight_decay: float = 0.0,
        build_optimizer: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.module = pipeline.module
        self.num_updates_per_batch = int(num_updates_per_batch)
        self.max_grad_norm = float(max_grad_norm)
        self.mse_weight = float(mse_weight)
        self.ratio_norm = bool(ratio_norm)
        self.ar_clip_range = float(ar_clip_range)
        self.temperature = float(pipeline.temperature)
        param_group_lrs = param_group_lrs or {"moe_gen": 3e-5}
        # When an engine drives the optimizer step (native verl-omni path), the updater only
        # supplies the two backward passes; it does not own an optimizer.
        if build_optimizer:
            self.optimizer, self.lr_summary = _build_expert_optimizer(
                self.module, base_lr, param_group_lrs, weight_decay
            )
        else:
            self.optimizer, self.lr_summary = None, None
        self._loss_cfg = _LossCfg(
            clip_ratio=image_clip_ratio, adv_clip_max=adv_clip_max, mse_weight=mse_weight, ratio_norm=ratio_norm
        )
        self._ref_snapshot: list[torch.Tensor] | None = None
        from verl_omni.trainer.diffusion.diffusion_algos import UniGRPOLoss

        self._image_loss = UniGRPOLoss()

    @contextmanager
    def _reference_weights(self):
        """Swap the one-time frozen (initial-policy) trainable params in for a v_ref forward."""
        live = [p for p in self.module.parameters() if p.requires_grad]
        if self._ref_snapshot is None:
            # Snapshot in the params' own (compute) dtype: a bf16 v_ref target halves the
            # persistent reference memory versus fp32, which matters for full-FT on 96 GB GPUs.
            self._ref_snapshot = [p.detach().clone() for p in live]
        stash = [p.detach().clone() for p in live]
        for p, ref in zip(live, self._ref_snapshot, strict=False):
            p.data.copy_(ref.to(p.dtype))
        try:
            yield
        finally:
            for p, saved in zip(live, stash, strict=False):
                p.data.copy_(saved)

    def _new_scheduler(self, device):
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        from ..bagel_flow_grpo.common import setup_bagel_sigmas

        sch = FlowMatchSDEDiscreteScheduler()
        setup_bagel_sigmas(sch, self.pipeline.num_inference_steps, shift=self.pipeline.shift, device=str(device))
        sch._step_index = None
        return sch

    def _replay_sample(self, sample, *, want_logp: bool):
        """Recompute per-SDE-step velocity (grad) and, if wanted, log_prob/mean on the recorded trajectory."""
        from ..bagel_flow_grpo.bagel_model import BagelForTraining
        from .pipeline import _latent_pos_ids

        module = self.module
        cfg = module.config
        # Compute is on CUDA even when FSDP offloads the (sharded) params to CPU.
        device = (
            torch.device(get_device_name(), get_device_id()) if is_cuda_available else next(module.parameters()).device
        )
        gh, gw = sample.latent_grid
        pos_ids = _latent_pos_ids(cfg, gh, gw, device)[None]
        cond = torch.tensor(sample.cond_token_ids, dtype=torch.long, device=device)[None]
        cond_mask = torch.ones_like(cond, dtype=torch.bool)
        model_dtype = module.vae2llm.weight.dtype
        sch = self._new_scheduler(device) if want_logp else None
        latents = sample.all_latents.to(device)
        tsteps = sample.all_timesteps.to(device)
        vel, logp, mean, std, sqrtdt = [], [], [], [], []
        for step in sample.sde_step_indices:
            t = tsteps[step].reshape(1)
            x_t = latents[step][None].float()
            velocity = BagelForTraining.forward(
                module,
                hidden_states=x_t.to(model_dtype),
                timestep=t.to(model_dtype),
                text_token_ids=cond,
                latent_pos_ids=pos_ids,
                text_attention_mask=cond_mask,
            )[0]
            vel.append(velocity[0])
            if want_logp:
                prev = latents[step + 1][None].float()
                _, lp, mn, sdt, sq = sch.sample_previous_step(
                    sample=x_t,
                    model_output=velocity.float(),
                    timestep=t,
                    noise_level=self.pipeline.eta,
                    prev_sample=prev,
                    sde_type="sde",
                    return_logprobs=True,
                    return_sqrt_dt=True,
                    include_logprob_normalizer=False,
                )
                logp.append(lp.mean(dim=tuple(range(1, lp.ndim))).squeeze(0) if lp.ndim > 1 else lp.reshape(-1)[0])
                mean.append(mn[0])
                std.append(torch.as_tensor(sdt, device=device).float().reshape(-1)[0])
                sqrtdt.append(torch.as_tensor(sq, device=device).float().reshape(-1)[0])
        out = {"velocity": torch.stack(vel)}
        if want_logp:
            out.update(
                log_probs=torch.stack(logp),
                prev_sample_mean=torch.stack(mean),
                std_dev_t=torch.stack(std),
                sqrt_dt=torch.stack(sqrtdt),
            )
        return out

    def _ar_backward(self, samples, advantages) -> dict[str, float]:
        """Backward the AR-GRPO loss (thinking track) over the mini-batch.

        Backward is done per sample (scaled by 1/n and accumulated) rather than stacking every
        sample's loss and calling one backward: the mean-reduced gradient is identical, but only
        one sample's (up to 1024-token) activation graph is alive at a time, which is what keeps
        full-FT of the 14.61B model inside 96 GB.
        """
        from .bagel_ar_thinking import replay_thinking_logprobs

        n = len(samples)
        mets, total = [], 0.0
        for s, adv in zip(samples, advantages, strict=False):
            think_prompt = self.pipeline.system_token_ids + s.prompt_token_ids
            new_logp = replay_thinking_logprobs(
                self.module, think_prompt, s.thinking_token_ids, temperature=self.temperature
            )
            old_logp = s.thinking_logprobs.to(new_logp.device)
            adv_tok = torch.full_like(new_logp, float(adv))
            loss, m = ar_grpo_loss(new_logp, old_logp, adv_tok, self.ar_clip_range)
            (loss / n).backward()
            total += float(loss.detach().item())
            mets.append(m)
        avg = {k: float(sum(d[k] for d in mets) / len(mets)) for k in mets[0]}
        avg["ar/loss"] = total / n
        return avg

    def _image_backward(self, samples, advantages) -> dict[str, float]:
        """Backward the UniGRPO image loss (GRPO-Guard PG + velocity-MSE) over the mini-batch.

        Backward is per sample (scaled 1/n, accumulated) so only one sample's SDE-step velocity
        graph is alive at once — the batched version held all ``n * num_sde_steps`` forwards and
        was the full-FT memory wall. This is exact, not an approximation: GRPO-Guard's only
        cross-sample terms are ``std_dev_t.mean()`` / ``sqrt_dt.mean()``, which are per-SDE-step
        schedule constants identical for every sample, and the final ``mean`` over equal-length
        samples equals the average of the per-sample means. The frozen-reference velocities are
        still computed under a SINGLE ``_reference_weights`` swap for the whole chunk (the swap
        copies every trainable param, so it must not run per sample).
        """
        n = len(samples)
        ref_vels = None
        if self.mse_weight > 0.0:
            with torch.no_grad(), self._reference_weights():
                ref_vels = [self._replay_sample(s, want_logp=False)["velocity"].detach() for s in samples]
        mets, total = [], 0.0
        for i, (s, adv) in enumerate(zip(samples, advantages, strict=False)):
            live = self._replay_sample(s, want_logp=True)
            device = live["log_probs"].device
            model_output = {
                "log_probs": live["log_probs"],
                "prev_sample_mean": live["prev_sample_mean"],
                "std_dev_t": live["std_dev_t"],
                "sqrt_dt": live["sqrt_dt"],
            }
            data: dict[str, Any] = {
                "old_log_probs": s.sde_logp.to(device),
                "advantages": torch.full((len(s.sde_step_indices),), float(adv), device=device),
                "old_prev_sample_mean": s.sde_means.to(device),
            }
            if ref_vels is not None:
                model_output["velocity"] = live["velocity"]
                data["ref_velocity"] = ref_vels[i]
            result = self._image_loss(config=self._loss_cfg, model_output=model_output, data=data)
            (result.loss / n).backward()
            total += float(result.loss.detach().item())
            mets.append({f"image/{k.split('/')[-1]}": float(v) for k, v in result.metrics.items()})
        avg = {k: float(sum(d[k] for d in mets) / len(mets)) for k in mets[0]}
        avg["image/loss"] = total / n
        return avg

    @torch.no_grad()
    def record_old_logp(self, samples) -> None:
        """Anchor each sample's old-policy logp to THIS updater's module (the trainable model).

        When rollout runs on a separate flat bf16 replica, the logp it records is not bit-identical
        to a replay on the FSDP model (different module, dtype/all-gather path), so the tight image
        clip (``image_clip_ratio``) would clip everything at update 0. Recomputing ``old_logp`` here
        with the exact replay code the update uses makes ``old_logp`` and update-0's ``new_logp`` come
        from the identical forward -> ratio == 1 at update 0. Cheap: one teacher-forced AR forward
        plus a few SDE-step forwards per sample, versus the thousands of steps the decode itself took.
        """
        from .bagel_ar_thinking import replay_thinking_logprobs

        for s in samples:
            think_prompt = self.pipeline.system_token_ids + s.prompt_token_ids
            s.thinking_logprobs = (
                replay_thinking_logprobs(self.module, think_prompt, s.thinking_token_ids, temperature=self.temperature)
                .detach()
                .cpu()
            )
            rep = self._replay_sample(s, want_logp=True)
            s.sde_logp = rep["log_probs"].detach().cpu()
            s.sde_means = rep["prev_sample_mean"].detach().cpu()
            s.std_dev_t = rep["std_dev_t"].detach().cpu()
            s.sqrt_dt = rep["sqrt_dt"].detach().cpu()

    def _clip_grad_norm(self) -> float:
        """FSDP2/DTensor-safe global grad-norm clip using a SINGLE cross-rank all_gather.

        ``torch.nn.utils.clip_grad_norm_`` on FSDP2 ``DTensor`` grads issues per-parameter DTensor
        reductions; across two nodes those many small collectives deadlock. Instead sum the local
        shard squared-norms, do one ``all_gather`` (then local sum) over the world, then rescale each
        grad in place (a local op on the shard). Equivalent global norm because the shards partition
        the full gradient.
        """
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor

        grads = [p.grad for p in self.module.parameters() if p.requires_grad and p.grad is not None]
        if not grads:
            return 0.0
        # Reduce over FSDP's OWN device-mesh process group, NOT the default group. Opening a second
        # NCCL communicator (the default group) alongside FSDP's while both span two nodes deadlocks
        # cross-node; reusing FSDP's group keeps a single communicator.
        mesh_group = None
        for g in grads:
            if isinstance(g, DTensor):
                mesh_group = g.device_mesh.get_group()
                break
        first = grads[0].to_local() if isinstance(grads[0], DTensor) else grads[0]
        local_sq = torch.zeros(1, device=first.device, dtype=torch.float32)  # 1-elem (not 0-dim) for NCCL
        for g in grads:
            gl = g.to_local() if isinstance(g, DTensor) else g
            local_sq = local_sq + gl.detach().float().pow(2).sum().reshape(1)
        # Cross-rank reduction via all_gather + local sum. A raw all_reduce hangs cross-node on this
        # IB fabric (observed on both the default and the FSDP-mesh group), while the all_gather that
        # FSDP itself relies on works — so gather each rank's scalar and sum locally.
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
        clip_coef = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
        return total_norm

    def joint_train_step(self, samples, advantages) -> list[dict[str, Any]]:
        """num_updates_per_batch disjoint mini-batches; each: zero_grad -> AR bwd -> image bwd -> ONE step."""
        n = len(samples)
        idx = list(range(n))
        chunks = [idx[i :: self.num_updates_per_batch] for i in range(self.num_updates_per_batch)]
        reports = []
        for u, chunk in enumerate(chunks):
            if not chunk:
                continue
            mb = [samples[i] for i in chunk]
            mb_adv = [advantages[i] for i in chunk]
            self.optimizer.zero_grad(set_to_none=True)
            ar_met = self._ar_backward(mb, mb_adv)
            img_met = self._image_backward(mb, mb_adv)
            gn = self._clip_grad_norm()
            stepped = False
            if gn == gn and gn != float("inf"):
                self.optimizer.step()
                stepped = True
            report = {"update": u, "grad_norm": gn, "optimizer_stepped": stepped, **ar_met, **img_met}
            report["lr_groups"] = [g["lr"] for g in self.optimizer.param_groups]
            reports.append(report)
        return reports


__all__ = ["ar_grpo_loss", "UniGRPOJointUpdater"]
