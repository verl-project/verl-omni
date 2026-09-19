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
"""Trainside UniGRPO diffusion trainer (BAGEL joint AR-thinking + image, no vLLM).

Reuses ``PolicyGradientRayTrainer``'s worker init, reward, advantage, checkpoint, and
logging helpers, but replaces the vLLM/AgentLoop rollout with a trainside worker
``generate`` call that samples on the live FSDP actor module (via a flat bf16 replica) and
re-anchors ``old_logp`` there. Selected by ``algorithm.trainer_type=unigrpo`` from
``main_diffusion._get_trainer_cls``.
"""

from __future__ import annotations

import uuid
from pprint import pprint

import numpy as np
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_reward_extra_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl_omni.trainer.diffusion.diffusion_trainer_utils import NoOpCheckpointManager
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    PolicyGradientRayTrainer,
    compute_advantage,
)
from verl_omni.workers.config.reward import reward_role_required

REPORT_PROMPTS = [
    "A curious cat exploring a haunted mansion",
    "a spanish water dog breed as arthur morgan from red dead redemption",
    "A close-up photograph of a fat orange cat with lasagna in its mouth. Shot on Leica M6.",
    "toilet design toilet in style of dodge charger toilet, black, photo",
    "an attractive young woman rolling her eyes",
]
"""Five fixed report prompts (verbatim from the reference run) re-sampled at each report
checkpoint so only the weights vary across steps -- continuity with the prior report."""


class UniGRPORayTrainer(PolicyGradientRayTrainer):
    """Trainside UniGRPO trainer: worker ``generate`` -> reward -> flow_grpo advantage -> joint update."""

    def init_workers(self):
        """Colocated actor workers + a standalone reward loop; no vLLM rollout / checkpoint sync."""
        actor_rollout_resource_pool = self._init_colocated_workers()
        from verl_omni.reward_loop import OmniRewardLoopManager

        reward_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if reward_role_required(self.config)
            else None
        )
        # PickScore etc. run as a colocated reward loop over the actor pool; no reward-model server.
        self.reward_loop_manager = OmniRewardLoopManager(
            config=self.config,
            rm_resource_pool=reward_pool,
            accelerator_resource_pool=actor_rollout_resource_pool,
        )
        # Trainside samples on the live FSDP module, so there is no vLLM server, no async rollout
        # manager, no streaming reward, and nothing to sync weights to.
        self.enable_agent_reward_loop = False
        self.llm_server_manager = None
        self.async_rollout_manager = None
        self.checkpoint_manager = NoOpCheckpointManager()

    def _validate(self):
        """Trainside has no vLLM rollout to validate through; skip with a marker.

        The reference recipe runs its evaluation/report generation as a separate offline pass
        (see the session report generator), so in-loop validation is intentionally a no-op here.
        """
        return {"val/trainside/skipped": 1.0}

    def _generate_trainside(self, gen_batch_output: DataProto) -> DataProto:
        """Run the trainside worker ``generate`` and return a DataProto of images + rollout samples."""
        from verl.utils import tensordict_utils as tu

        # The gen batch's only field is a variable-length ``prompt_token_ids`` stored as non-tensor
        # data, so ``DataProto.batch`` is None and ``DataProto.to_tensordict()`` (which dereferences
        # ``self.batch.to_dict()``) would raise ``AttributeError: 'NoneType' object has no attribute
        # 'to_dict'``. Build the dispatch TensorDict directly from the tensor + non-tensor batch;
        # ``get_tensordict`` wraps the lists as NonTensorStacks and infers the batch size.
        td_dict: dict = {}
        if gen_batch_output.batch is not None:
            td_dict.update(gen_batch_output.batch.to_dict())
        for key, val in gen_batch_output.non_tensor_batch.items():
            td_dict[key] = list(val)
        gen_td = tu.get_tensordict(td_dict)
        gen_output = self.actor_rollout_wg.generate(gen_td)
        return DataProto.from_tensordict(gen_output)

    def _init_report(self):
        """Set up periodic report dumping when ``UNIGRPO_REPORT_DIR`` is set (else a no-op).

        Loads the five fixed report prompts' token ids + ground-truth text from the training
        parquet (matched by text, so row order is irrelevant), records the run settings for the
        report env table, resets the per-step curve log, and writes the initial manifest. The
        report dir lives on shared storage both the driver and rank-0 worker can see.
        """
        import os

        self._report_dir = os.environ.get("UNIGRPO_REPORT_DIR")
        self._eval_prompts, self._eval_gts = [], []
        if not self._report_dir:
            return
        self._report_seed = int(os.environ.get("UNIGRPO_REPORT_SEED", "1234"))
        self._report_freq = max(int(os.environ.get("UNIGRPO_REPORT_FREQ", "5")), 1)
        try:
            import pandas as pd

            train_files = self.config.data.train_files
            if isinstance(train_files, list | tuple):
                train_files = train_files[0]
            df = pd.read_parquet(train_files)

            def _prompt_text(row):
                prompt = row["prompt"]
                try:
                    return prompt[0]["content"]
                except Exception:
                    return str(prompt)

            texts = [_prompt_text(df.iloc[i]) for i in range(len(df))]
            for report_prompt in REPORT_PROMPTS:
                if report_prompt in texts:
                    row = df.iloc[texts.index(report_prompt)]
                    self._eval_prompts.append([int(t) for t in row["prompt_token_ids"]])
                    self._eval_gts.append(report_prompt)
        except Exception:
            import traceback

            print(f"[unigrpo report] prompt load failed, dumping disabled:\n{traceback.format_exc()}", flush=True)
            self._report_dir = None
            return
        if not self._eval_prompts:
            print("[unigrpo report] none of the fixed prompts found in train_files; dumping disabled", flush=True)
            self._report_dir = None
            return
        os.makedirs(os.path.join(self._report_dir, "report_ff"), exist_ok=True)
        self._report_curve_path = os.path.join(self._report_dir, "train_rank0.jsonl")
        open(self._report_curve_path, "w").close()
        ar = self.config.actor_rollout_ref
        pipe = ar.rollout.pipeline
        self._report_settings = {
            "model": str(ar.model.path),
            "G_samples_per_prompt": int(ar.rollout.n),
            "num_inference_steps(train)": int(pipe.num_inference_steps),
            "image_size": f"{int(pipe.height)}x{int(pipe.width)}",
            "base_lr(und)": ar.actor.optim.lr,
            "moe_gen_lr": dict(getattr(ar.actor.optim, "param_group_lrs", None) or {}),
            "mse_weight": ar.actor.diffusion_loss.mse_weight,
            "image_clip_ratio": ar.actor.diffusion_loss.clip_ratio,
            "ratio_norm": bool(ar.actor.diffusion_loss.ratio_norm),
            "sde_noise_level": ar.rollout.algo.noise_level,
            "sde_window_size": ar.rollout.algo.sde_window_size,
            "rollout": str(ar.rollout.name),
            "nodes_x_gpus": f"{int(self.config.trainer.nnodes)} x {int(self.config.trainer.n_gpus_per_node)}",
            "eval_setting": "official CFG=4, 50 steps, global renorm",
        }
        self._write_report_manifest("running")
        print(
            f"[unigrpo report] enabled -> {self._report_dir} "
            f"({len(self._eval_prompts)} prompts, seed {self._report_seed}, every {self._report_freq} steps)",
            flush=True,
        )

    def _report_step_list(self):
        """Report checkpoints (update counts) the run will dump: 0, freq, 2*freq, ..., total."""
        freq = getattr(self, "_report_freq", 5)
        total = int(self.total_training_steps)
        return sorted({0, total, *range(freq, total + 1, freq)})

    def _write_report_manifest(self, status):
        import json
        import os

        if not getattr(self, "_report_dir", None):
            return
        manifest = dict(self._report_settings)
        manifest.update(
            {
                "prompts": list(self._eval_gts),
                "seed": self._report_seed,
                "steps": self._report_step_list(),
                "status": status,
            }
        )
        with open(os.path.join(self._report_dir, "report_ff", "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)

    def _dump_report(self, step, status="running"):
        """Trigger the trainside worker report dump at ``step`` (update count) and refresh manifest."""
        if not getattr(self, "_report_dir", None):
            return
        try:
            self.actor_rollout_wg.dump_report_samples(
                self._eval_prompts, self._eval_gts, self._report_dir, int(step), self._report_seed
            )
        except Exception:
            import traceback

            print(f"[unigrpo report] dump at step {step} failed (non-fatal):\n{traceback.format_exc()}", flush=True)
        self._write_report_manifest(status)

    def _log_report_curve(self, step, batch, reward_tensor):
        """Append ``{step, rollout_reward_mean, avg_think_len}`` for this step to the curve jsonl."""
        import json

        if not getattr(self, "_report_dir", None):
            return
        try:
            reward_mean = float(reward_tensor.float().mean().item())
        except Exception:
            reward_mean = float("nan")
        avg_think_len = float("nan")
        samples = batch.non_tensor_batch.get("unigrpo_samples")
        if samples is not None and len(samples) > 0:
            lens = [len(getattr(s, "thinking_token_ids", [])) for s in samples]
            if lens:
                avg_think_len = float(sum(lens) / len(lens))
        with open(self._report_curve_path, "a") as handle:
            handle.write(
                json.dumps({"step": int(step), "rollout_reward_mean": reward_mean, "avg_think_len": avg_think_len})
                + "\n"
            )

    def fit(self):
        """Trainside training loop: worker generate -> reward -> flow_grpo advantage -> joint update."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self._init_report()

        self.global_steps = 0
        self._load_checkpoint()

        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.global_steps == 0:
            self._dump_report(0, status="running")

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        adv_estimator = self.config.algorithm.adv_estimator
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict = {}
                timing_raw: dict = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                # The gen batch carries only non-tensor prompt data (variable-length prompt_token_ids),
                # so DataProto.batch is None. Seed an empty tensor batch of the right size so union()
                # with the generated responses (and every downstream tensor op) has a batch to merge into.
                if gen_batch_output.batch is None:
                    from tensordict import TensorDict

                    gen_batch_output.batch = TensorDict({}, batch_size=[len(gen_batch_output)])

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Trainside rollout on the live FSDP actor module (no vLLM).
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_output = self._generate_trainside(gen_batch_output)
                    batch = gen_batch_output.union(gen_output)

                    # Reward is always computed here (trainside never streams it during rollout).
                    with marked_timer("reward", timing_raw, color="yellow"):
                        batch_reward = self._compute_reward_colocate(batch)
                        batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["sample_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        # One advantage per sample (num_timesteps == 1): the joint update reads a
                        # per-sample scalar, not a per-denoise-step vector like image-only flow_grpo.
                        sample_level_scores = batch.batch["sample_level_scores"]
                        batch.batch["sample_level_rewards"] = (
                            sample_level_scores if sample_level_scores.ndim > 1 else sample_level_scores.unsqueeze(-1)
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=adv_estimator,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            global_std=self.config.algorithm.global_std,
                            config=self.config.algorithm,
                        )

                    self._log_report_curve(self.global_steps, batch, reward_tensor)

                    # Joint AR + image update on the FSDP module (record_old_logp already anchored
                    # old_logp inside generate). num_updates_per_batch == number of framework
                    # mini-batches (ppo_mini_batch_size vs the per-step sample count).
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    save_freq = self.config.trainer.get("rollout_data_save_freq", 1)
                    if rollout_data_dir and save_freq > 0 and self.global_steps % save_freq == 0:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics_diffusion(batch=batch))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                num_images = batch.batch["advantages"].shape[0]
                metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
                metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics.update(compute_reward_extra_metrics_diffusion(reward_extra_infos_dict))
                logger.log(data=metrics, step=self.global_steps)

                if getattr(self, "_report_dir", None) and (self.global_steps % self._report_freq == 0 or is_last_step):
                    self._dump_report(self.global_steps, status="done" if is_last_step else "running")

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


__all__ = ["UniGRPORayTrainer"]
